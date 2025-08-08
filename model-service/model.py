"""
model.py
~~~~~~~~

该模块定义用于商品图片向量化的深度学习模型结构。模型与训练阶段保持一致，
基于预训练的 ResNet50 骨干，包含嵌入分支和水印分类分支，并支持掩模 gating
以降低水印干扰。嵌入向量维度默认为 256，输出通过 L2 归一化。

使用示例：

    from vectorization_model_service.model import MultiTaskModel, load_model
    model = load_model('model_final.pth', device='cuda')
    model.eval()

"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiTaskModel(nn.Module):
    """多任务模型：商品嵌入 + 水印分类。

    参数：
        embedding_dim: 嵌入向量维度。
        backbone: 预训练骨干模型名，默认为 'resnet50'。
        pretrained: 是否加载 ImageNet 预训练权重。
        input_channels: 输入通道数，默认 4（RGB+掩模）。
        use_mask_gating: 是否启用掩模 gating，通过 (1-mask) 抑制水印区域。
    """

    def __init__(self,
                 embedding_dim: int = 256,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 input_channels: int = 4,
                 use_mask_gating: bool = True) -> None:
        super().__init__()
        self.use_mask_gating = use_mask_gating
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            # 修改第一层卷积以适应 4 通道输入
            if input_channels != 3:
                old_conv = resnet.conv1
                new_conv = nn.Conv2d(input_channels, old_conv.out_channels,
                                     kernel_size=old_conv.kernel_size,
                                     stride=old_conv.stride,
                                     padding=old_conv.padding,
                                     bias=old_conv.bias is not None)
                with torch.no_grad():
                    # 复制前三个通道权重
                    new_conv.weight[:, :3, :, :] = old_conv.weight
                    # 多余的通道权重置 0
                    if input_channels > 3:
                        for c in range(3, input_channels):
                            new_conv.weight[:, c:c+1, :, :] = 0.0
                    if old_conv.bias is not None:
                        new_conv.bias = old_conv.bias
                resnet.conv1 = new_conv
            modules = list(resnet.children())[:-1]
            self.feature_extractor = nn.Sequential(*modules)
            in_features = resnet.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        # 嵌入分支
        self.embedding_fc = nn.Linear(in_features, embedding_dim)
        # 分类分支
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x: 输入张量，形状为 [B, C, H, W]，其中 C>=4 时包含 RGB 和掩模通道。

        Returns:
            embeddings: 归一化后的嵌入向量。
            logits: 水印分类 logits。
        """
        # 掩模 gating：将水印区域的像素置零
        if self.use_mask_gating and x.dim() == 4 and x.size(1) >= 4:
            mask = x[:, 3:4, :, :]
            x = x.clone()
            x[:, :3, :, :] = x[:, :3, :, :] * (1.0 - mask)
        # 特征提取
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding_fc(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        logits = self.classifier(features).squeeze(1)
        return embeddings, logits


def load_model(weights_path: str,
               device: str = "cpu",
               embedding_dim: int = 256,
               use_mask_gating: bool = True) -> MultiTaskModel:
    """加载模型并载入权重。

    Args:
        weights_path: 训练过程中保存的 `.pth` 文件路径。
        device: 部署设备，如 "cpu" 或 "cuda"。
        embedding_dim: 嵌入维度。
        use_mask_gating: 是否启用掩模 gating。

    Returns:
        初始化并加载权重的 `MultiTaskModel` 实例。
    """
    model = MultiTaskModel(embedding_dim=embedding_dim,
                           input_channels=4,
                           use_mask_gating=use_mask_gating)
    ckpt = torch.load(weights_path, map_location=device)
    # 支持包含在训练脚本中保存的字典
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model