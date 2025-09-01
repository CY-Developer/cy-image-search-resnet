"""
model.py
~~~~~~~~

This module defines the multi‑task neural network used to embed product
images.  It closely mirrors the architecture used during training and
supports two parallel outputs: a normalised embedding vector and a
watermark classification logit.  The network is based on a ResNet50
backbone with its first convolution adapted to accept four input
channels (RGB + mask).

The `load_model` helper will construct a ``MultiTaskModel`` and load
weights from a checkpoint file.  When ``strict=False`` the loader will
ignore any missing keys allowing you to drop unused heads (such as the
category classifier) when deploying the model for inference.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiTaskModel(nn.Module):
    """A multi‑task network for image embedding and watermark detection.

    Parameters
    ----------
    embedding_dim: int
        Dimensionality of the output embedding vector.  A smaller
        dimensionality reduces index storage and retrieval time but may
        reduce discriminative power.

    backbone: str
        Backbone architecture.  Only ``resnet50`` is supported at
        present because it offers a good trade‑off between performance
        and throughput.

    pretrained: bool
        If ``True`` the backbone will be initialised with ImageNet
        weights.  Pretraining helps the network converge quickly on
        limited amounts of domain specific data.

    input_channels: int
        Number of channels passed into the backbone.  Use 4 when
        supplying an RGB image concatenated with a single channel mask.

    use_mask_gating: bool
        When ``True`` the forward method will multiply the RGB portion
        of the input by ``(1‑mask)`` before passing it through the
        network.  This effectively removes watermark pixels from the
        activations.
    """

    def __init__(self,
                 embedding_dim: int = 256,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 input_channels: int = 4,
                 use_mask_gating: bool = True) -> None:
        super().__init__()
        self.use_mask_gating = use_mask_gating
        if backbone != "resnet50":
            raise ValueError(f"Unsupported backbone: {backbone}")
        # Load a pretrained ResNet50 and adapt the first conv layer
        resnet = models.resnet50(pretrained=pretrained)
        # Replace the first convolution to accept the specified number of input channels
        if input_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(input_channels,
                                 old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=old_conv.bias is not None)
            with torch.no_grad():
                # Copy weights for the existing RGB channels
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # Initialise any additional channels to zero
                if input_channels > 3:
                    for c in range(3, input_channels):
                        new_conv.weight[:, c:c+1, :, :] = 0.0
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            resnet.conv1 = new_conv
        # Drop the classification head
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        in_features = resnet.fc.in_features
        # Projection head for embedding
        self.embedding_fc = nn.Linear(in_features, embedding_dim)
        # Binary classifier for watermark presence
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, C, H, W)`` where ``C >= 4``.  If
            ``use_mask_gating`` is ``True`` and ``C >= 4`` then channel
            ``3`` is treated as the mask and used to zero‑out the RGB
            channels.

        Returns
        -------
        embeddings: torch.Tensor
            L2 normalised embedding of shape ``(B, embedding_dim)``.

        logits: torch.Tensor
            Raw logits for the watermark classifier of shape ``(B,)``.
        """
        if self.use_mask_gating and x.dim() == 4 and x.size(1) >= 4:
            mask = x[:, 3:4, :, :]
            # Clone to avoid modifying the caller's tensor
            x = x.clone()
            x[:, :3, :, :] = x[:, :3, :, :] * (1.0 - mask)
        # Extract global features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        # Compute embedding and normalise
        embeddings = self.embedding_fc(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # Binary watermark classification
        logits = self.classifier(features).squeeze(1)
        return embeddings, logits


def load_model(weights_path: str,
               device: str = "cpu",
               embedding_dim: int = 256,
               use_mask_gating: bool = True) -> MultiTaskModel:
    """Instantiate a ``MultiTaskModel`` and load weights from disk.

    Parameters
    ----------
    weights_path: str
        Location on disk of the ``.pth`` file to load.  The checkpoint may
        contain either the raw ``state_dict`` or a dictionary with a
        ``'model_state_dict'`` key.

    device: str
        Device identifier on which to place the model (e.g. ``'cpu'`` or
        ``'cuda'``).  When deploying with GPU support you should pass
        ``'cuda'``.

    embedding_dim: int
        Dimensionality of the embedding head.  Must match the size
        specified when the checkpoint was created.

    use_mask_gating: bool
        Whether to apply mask gating during inference.

    Returns
    -------
    MultiTaskModel
        The initialised and weight loaded model in evaluation mode.
    """
    model = MultiTaskModel(embedding_dim=embedding_dim,
                           input_channels=4,
                           use_mask_gating=use_mask_gating)
    ckpt = torch.load(weights_path, map_location=device)
    # Check for wrapped dictionary from the training script
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model