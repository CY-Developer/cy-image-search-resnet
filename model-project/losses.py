"""
Loss functions used in the e-commerce image retrieval project.

This module defines the BatchHardTripletLoss for metric learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard Triplet Loss implementation.

    For each anchor in the batch, this loss selects the hardest positive (the
    sample with the largest distance within the same class) and the hardest
    negative (the sample with the smallest distance in a different class),
    then computes the margin-based triplet loss. If ``margin`` is ``None``,
    a softplus variant is used.

    Parameters
    ----------
    margin: float or None
        The margin for the triplet loss. If ``None``, use softplus instead
        of hinge.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        self.softplus = nn.Softplus()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, D), labels: (B)
        dist = torch.cdist(embeddings, embeddings, p=2)  # pairwise distances
        labels = labels.unsqueeze(1)
        same = (labels == labels.t())
        pos_mask = same ^ torch.eye(same.size(0), dtype=torch.bool, device=same.device)
        neg_mask = ~same
        # Hardest positive: largest distance within the same class
        hardest_pos = (dist * pos_mask.float()).max(dim=1).values
        # Hardest negative: smallest distance to different class
        dist_pos_inf = dist.clone()
        dist_pos_inf[~neg_mask] = float('inf')
        hardest_neg = dist_pos_inf.min(dim=1).values
        if self.margin is None:
            loss = self.softplus(hardest_pos - hardest_neg).mean()
        else:
            loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0).mean()
        return loss