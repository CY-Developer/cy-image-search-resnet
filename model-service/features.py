"""
features.py
~~~~~~~~~~~

This module defines a lightweight wrapper around the trained multi-task
ResNet backbone and associated linear adapter used by the vectorisation
service.  It exposes utility functions for loading the model and
adapter weights, preprocessing images and converting them into
2048‑dimensional backbone features as well as aligned vectors in the
search space.  A cosine similarity helper is also provided.

Usage
-----

Before calling any of the public functions you must call
``init_model`` with the paths to your checkpoint (.pth) and adapter
(.npz) files.  This loads the model and adapter into globally scoped
variables and keeps them on the appropriate device.  Subsequent
invocations of ``vectorize_pil`` or ``feats2048`` will reuse the
loaded model.

The alignment step applies a linear transformation and mean
subtraction as defined in your adapter file.  The result is then L2
normalised.  This mirrors the behaviour of the Java ``search_one.py``
script that uses ``adapter_linear.npz`` to align features with the
Milvus index.

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
from PIL import Image
from typing import Tuple

import os
from config import Config
IMG_SIZE     = getattr(Config, 'IMG_SIZE', 224)
CKPT_PATH    = os.getenv('CKPT_PATH', getattr(Config, 'CKPT_PATH', 'stage2_best.pth'))
ADAPTER_PATH = os.getenv('ADAPTER_PATH', getattr(Config, 'ADAPTER_PATH', 'adapter_linear.npz'))



class MultiTaskResNet(nn.Module):
    """Replicates the training architecture used for the 2048‑dimensional
    backbone.  The network returns both the backbone features and the
    downstream embedding heads but only the backbone output is used
    for the new vectorisation pipeline.

    Parameters
    ----------
    embed_dim: int
        Dimensionality of the smaller projection head (unused here but
        preserved for compatibility with checkpoint loading).
    num_categories: int
        Number of categories for the classification head (unused here).
    pretrained: bool, optional
        Whether to initialise from ImageNet weights (unused for
        inference).  The weights from ``ckpt_path`` will overwrite
        anything loaded here.
    l2_norm: bool, optional
        When ``True`` the embedding head outputs are normalised
        (unused here).
    """

    def __init__(self, embed_dim: int, num_categories: int, pretrained: bool = False, l2_norm: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        # Keep only convolutional trunk; remove final linear layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embed_head = nn.Sequential(
            nn.Linear(in_features, embed_dim, bias=True),
            nn.BatchNorm1d(embed_dim),
        )
        self.category_head = nn.Linear(in_features, num_categories)
        self.watermark_head = nn.Linear(in_features, 1)
        self.l2_norm = l2_norm

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)  # (B,2048)
        emb = self.embed_head(feats)  # (B, embed_dim)
        if self.l2_norm:
            emb = F.normalize(emb, dim=1)
        cat = self.category_head(feats)
        wm = self.watermark_head(feats).squeeze(-1)
        return feats, emb, cat, wm


# Global state to avoid reloading on every request
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: MultiTaskResNet | None = None
_W: torch.Tensor | None = None
_mean_q: torch.Tensor | None = None
_mean_g: torch.Tensor | None = None


def _load_ckpt(path: str) -> MultiTaskResNet:
    """Load a checkpoint file into a ``MultiTaskResNet`` instance.

    The checkpoint is expected to be a dictionary with a ``model`` key
    containing the state dict.  The metadata keys ``embed_dim`` and
    ``num_categories`` are read when available to initialise the
    correct layer sizes.
    """
    ckpt = torch.load(path, map_location=_device)
    meta = ckpt if isinstance(ckpt, dict) else {}
    sd = meta.get("model", meta)
    embed_dim = int(meta.get("embed_dim", 512))
    num_cat = int(meta.get("num_categories", 1))
    model = MultiTaskResNet(embed_dim=embed_dim, num_categories=num_cat, pretrained=False).to(_device).eval()
    # Remove any "module." prefix that might appear from DataParallel
    cleaned = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    return model


def _load_adapter(path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the linear adapter parameters from a .npz file.

    The file is expected to contain a matrix ``W`` of shape (D, D)
    along with two mean vectors ``mean_q`` and ``mean_g`` each of
    length D.  These are converted into torch tensors on the
    appropriate device.
    """
    data = np.load(path)
    # Compatible key names; choose the first that exists
    def _first_key(opts):
        for k in opts:
            if k in data:
                return k
        return None
    W_key = _first_key(["W", "A", "weight", "adapter_W", "linear_W"])
    mq_key = _first_key(["mean_q", "mq", "meanQ"])
    mg_key = _first_key(["mean_g", "mg", "meanG"])
    if W_key is None or mq_key is None or mg_key is None:
        raise KeyError(f"Adapter npz file missing expected keys: {data.keys()}")
    W = torch.as_tensor(data[W_key], dtype=torch.float32, device=_device)
    mq = torch.as_tensor(data[mq_key], dtype=torch.float32, device=_device)
    mg = torch.as_tensor(data[mg_key], dtype=torch.float32, device=_device)
    return W, mq, mg


def init_model(ckpt_path: str | None = None, adapter_path: str | None = None) -> None:
    """Load the model and adapter once on module import or app startup.

    This function populates the global variables ``_model``, ``_W``,
    ``_mean_q`` and ``_mean_g``.  Subsequent calls have no effect.  If
    ``ckpt_path`` or ``adapter_path`` are omitted then the values from
    ``config.py`` are used.
    """
    global _model, _W, _mean_q, _mean_g
    if _model is None:
        _model = _load_ckpt(ckpt_path or CKPT_PATH)
    if _W is None:
        _W, _mean_q, _mean_g = _load_adapter(adapter_path or ADAPTER_PATH)


def _preprocess(pil: Image.Image) -> torch.Tensor:
    """Convert a PIL image into a normalised tensor for the model."""
    tfm = T.Compose([
        T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(pil.convert("RGB")).unsqueeze(0)


def feats2048(pil: Image.Image) -> torch.Tensor:
    """Extract the 2048‑dimensional backbone features from a PIL image.

    The model must have been initialised via ``init_model``.  The
    returned tensor is on the same device as the model and detached
    from the computational graph.
    """
    if _model is None:
        init_model()
    x = _preprocess(pil).to(_device)
    feats, _, _, _ = _model(x)
    return feats[0]


def to_aligned_vec(q2048: torch.Tensor) -> np.ndarray:
    """Transform a backbone feature vector into the aligned search space.

    The transformation applies mean subtraction and a linear mapping
    followed by L2 normalisation.  This mirrors the behaviour of
    ``search_one.py`` in the Java service.
    """
    if _W is None or _mean_q is None:
        init_model()
    q = q2048 - _mean_q
    v = _W @ q
    v = F.normalize(v, dim=0)
    return v.detach().cpu().numpy().astype("float32")


def vectorize_pil(pil: Image.Image) -> np.ndarray:
    """Compute an aligned vector from a PIL image.

    This convenience function wraps ``feats2048`` and ``to_aligned_vec``.
    """
    return to_aligned_vec(feats2048(pil))


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two numpy vectors."""
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))
