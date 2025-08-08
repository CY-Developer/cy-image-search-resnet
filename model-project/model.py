"""
model.py
~~~~~~~~~

定义用于商品图片检索与水印分类的多任务神经网络模型。模型基于预训练 ResNet50，
通过共享骨干提取通用特征，并在此之上分别构建嵌入分支和水印分类分支。

嵌入分支输出归一化的特征向量，用于 Triplet/对比损失训练；
分类分支输出单个 logits，用于判断图片是否带水印。

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

    输出：
        embeddings: 经过 L2 归一化的特征向量。
        watermark_logits: 水印分类的原始 logits（未经过 sigmoid）。
    """

    def __init__(self,
                 embedding_dim: int = 256,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 input_channels: int = 4,
                 use_mask_gating: bool = True) -> None:
        super().__init__()
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            # 修改第一层卷积使其能够接受更多通道，例如 4 通道：RGB+掩模
            if input_channels != 3:
                # 复制预训练权重，并在第四通道上填充 0
                old_conv = resnet.conv1
                # 创建新的卷积层
                new_conv = nn.Conv2d(input_channels, old_conv.out_channels,
                                     kernel_size=old_conv.kernel_size,
                                     stride=old_conv.stride,
                                     padding=old_conv.padding,
                                     bias=old_conv.bias is not None)
                # 初始化新卷积权重
                with torch.no_grad():
                    # 复制已有权重的前三通道
                    new_conv.weight[:, :3, :, :] = old_conv.weight
                    # 对额外通道赋零
                    if input_channels > 3:
                        for c in range(3, input_channels):
                            new_conv.weight[:, c:c+1, :, :] = 0.0
                    # 如果有偏置，复制
                    if old_conv.bias is not None:
                        new_conv.bias = old_conv.bias
                resnet.conv1 = new_conv
            # 去掉最后的全连接层，保留到全局平均池化之前
            modules = list(resnet.children())[:-1]
            self.feature_extractor = nn.Sequential(*modules)
            in_features = resnet.fc.in_features
        else:
            raise ValueError(f"暂不支持的骨干网络: {backbone}")
        # 嵌入分支
        self.embedding_fc = nn.Linear(in_features, embedding_dim)
        # 分类分支
        self.classifier = nn.Linear(in_features, 1)
        # 是否启用掩模 gating：通过 (1 - mask) 抑制水印区域
        self.use_mask_gating = use_mask_gating

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入形状：[batch_size, C, H, W]
        # 如果启用了掩模 gating 并且输入包含掩模通道，先对前三通道施加 (1 - mask) 权重
        if self.use_mask_gating and x.dim() == 4 and x.size(1) >= 4:
            # mask 形状 [B,1,H,W]
            mask = x[:, 3:4, :, :]
            # 克隆以避免修改原输入
            x = x.clone()
            x[:, :3, :, :] = x[:, :3, :, :] * (1.0 - mask)
        # 提取基础特征
        features = self.feature_extractor(x)  # [B, C, 1, 1]
        features = features.view(features.size(0), -1)  # [B, C]
        # 嵌入向量
        embeddings = self.embedding_fc(features)
        # L2 归一化，使向量分布在单位球面，有利于度量学习
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # 水印分类 logits
        watermark_logits = self.classifier(features).squeeze(1)
        return embeddings, watermark_logits
