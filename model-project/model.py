"""
model.py
~~~~~~~~~

定义用于商品图片检索与水印分类的多任务神经网络模型。模型基于预训练 ResNet50，
通过共享骨干提取通用特征，并在此之上分别构建嵌入分支和水印分类分支。

嵌入分支输出归一化的特征向量，用于 Triplet/对比损失训练；
分类分支输出单个 logits，用于判断图片是否带水印。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class MultiTaskModel(nn.Module):
    """多任务模型：商品嵌入 + 水印分类。

    参数：
        embedding_dim: 嵌入向量维度。
        backbone: 预训练骨干模型名，默认为 'resnet50'。
        pretrained: 是否加载 ImageNet 预训练权重。
        input_channels: 输入通道数，默认 4（RGB+Mask）。
        use_mask_gating: 是否启用掩模 gating（将掩模区域像素抑制为 0）。

    输出：
        embeddings: 经过 L2 归一化的特征向量。
        watermark_logits: 水印分类的原始 logits（未经过 sigmoid）。
    """

    def __init__(
            self,
            backbone: str = "resnet50",
            embedding_dim: int = 256,
            input_channels: int = 4,
            use_mask_gating: bool = True,
            pretrained: bool = True,
    ):
        super(MultiTaskModel, self).__init__()
        self.use_mask_gating = use_mask_gating

        # 新式 weights 写法，兼容 torchvision >= 0.13
        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = resnet50(weights=weights)

            # 若输入通道不是3，则扩展第一层卷积
            if input_channels != 3:
                old_conv = resnet.conv1
                new_conv = nn.Conv2d(
                    input_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=(old_conv.bias is not None),
                )
                with torch.no_grad():
                    # 复制前三个通道的预训练权重
                    new_conv.weight[:, :3, :, :] = old_conv.weight
                    # 其余通道置零
                    if input_channels > 3:
                        new_conv.weight[:, 3:, :, :] = 0.0
                    if old_conv.bias is not None:
                        new_conv.bias = old_conv.bias
                resnet.conv1 = new_conv

            modules = list(resnet.children())[:-1]
            self.feature_extractor = nn.Sequential(*modules)
            in_features = resnet.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 嵌入分支 + 水印分类分支（单 logit，配合 BCEWithLogitsLoss）
        self.fc = nn.Linear(in_features, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor):
        # 掩模 gating：对 RGB 三通道在 mask=1 的区域置零，保留第四通道作为提示
        if self.use_mask_gating and x.dim() == 4 and x.size(1) >= 4:
            rgb = x[:, :3, :, :]
            mask = x[:, 3:4, :, :].clamp(0, 1)
            rgb = rgb * (1.0 - mask)
            x = torch.cat([rgb, mask], dim=1)

        x = self.feature_extractor(x)      # [N, C, 1, 1]
        x = x.flatten(1)                   # [N, C]
        embeddings = self.fc(x)            # [N, D]
        embeddings = F.normalize(embeddings, dim=1)  # L2 归一化，便于检索
        logits = self.classifier(embeddings).squeeze(1)  # [N]
        return embeddings, logits
