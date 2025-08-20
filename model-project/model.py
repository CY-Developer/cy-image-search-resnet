"""
Multi-task neural network model for e-commerce image retrieval with watermark detection.

This module defines the ``MultiTaskResNet`` class, which wraps an
ImageNet-pretrained ResNet-50 backbone and adds three heads:

1. **Embedding head**: Projects backbone features into a low-dimensional
   space suitable for metric learning. A BatchNorm layer follows the
   linear projection. L2 normalization can be applied optionally.
2. **Category classification head**: Outputs logits for product categories.
3. **Watermark detection head**: Outputs a single logit indicating whether
   the input image contains a watermark. Binary cross-entropy with logits
   should be used as the corresponding loss.

The model returns a tuple ``(embeddings, category_logits, watermark_logits)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class MultiTaskResNet(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_categories: int,
        pretrained: bool = True,
        l2_norm: bool = True,
    ):
        """
        Initialize the multi-task ResNet model.

        Parameters
        ----------
        embed_dim: int
            Dimensionality of the output embedding vector.
        num_categories: int
            Number of product categories.
        pretrained: bool
            If True, load ImageNet-pretrained weights for the ResNet backbone.
        l2_norm: bool
            If True, apply L2 normalization to the embedding vectors.
        """
        super().__init__()
        # Load ResNet-50 backbone
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.l2_norm = l2_norm

        # Embedding head: Linear + BatchNorm
        self.embed_head = nn.Sequential(
            nn.Linear(in_features, embed_dim, bias=True),
            nn.BatchNorm1d(embed_dim),
        )
        # Category classification head
        self.category_head = nn.Linear(in_features, num_categories)
        # Watermark detection head (binary classification)
        self.watermark_head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the model.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        embeddings: torch.Tensor
            Tensor of shape (B, embed_dim). Normalized if ``l2_norm`` is True.
        cat_logits: torch.Tensor
            Tensor of shape (B, num_categories) with raw class scores.
        wm_logits: torch.Tensor
            Tensor of shape (B, 1) with raw scores for watermark detection.
        """
        feats = self.backbone(x)
        # Embedding
        emb = self.embed_head(feats)
        if self.l2_norm:
            emb = nn.functional.normalize(emb, dim=1)
        # Category logits
        cat_logits = self.category_head(feats)
        # Watermark logits
        wm_logits = self.watermark_head(feats)
        return emb, cat_logits, wm_logits.squeeze(-1)