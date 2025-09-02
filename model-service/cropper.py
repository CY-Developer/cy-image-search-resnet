"""
cropper.py
~~~~~~~~~~

This module implements the sliding window search used to extract a
compact watch crop from noisy detail images.  Rather than relying on
Grad‑CAM (which can respond to unrelated regions in busy scenes), the
approach here iterates over a set of candidate boxes, computes a
vector for each and selects the one that is most similar to a set of
reference vectors or to the full image embedding.  This heuristic
works well for product detail pages where the object of interest
appears somewhere on the lower half of the image but may be
surrounded by text and other UI elements.

The ``crop_watch_and_vec`` function returns both the aligned vector
for the best crop and the bounding box itself, along with the full
image vector.  If no suitable crop is found (for example when no
windows are generated) ``v_crop`` is ``None`` and the caller should
fall back to using the full vector.

"""

from __future__ import annotations

import numpy as np
from PIL import Image
# import cv2
import torch
from typing import Tuple, List, Optional
from features import feats2048, to_aligned_vec, _device, init_model, CKPT_PATH, ADAPTER_PATH
import features as Fs

from  config import Config

import torchvision.transforms as T


# Preprocessing for candidate windows
_TFM = T.Compose([
    T.Resize(Config.IMG_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(Config.IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _gen_windows(W: int, H: int,
                 scales: Tuple[float, ...] = (0.18, 0.24, 0.32, 0.40),
                 stride_ratio: float = 0.18,
                 ar_range: Tuple[float, float] = (0.85, 1.15),
                 limit: int = 160) -> List[Tuple[int, int, int, int]]:
    """Generate a list of candidate bounding boxes for cropping.

    Parameters
    ----------
    W, H : int
        Width and height of the original image.
    scales : tuple of float
        Relative sizes of the windows as a fraction of the smaller image
        dimension.  Larger values produce bigger windows which may
        overlap multiple objects.
    stride_ratio : float
        Stride as a fraction of the window size.  Smaller values
        produce more overlapping windows.  Values too small will
        drastically increase computation time.
    ar_range : tuple of float
        Allowed range of aspect ratios (w/h) to consider.  Values
        outside of this range are discarded.
    limit : int
        Maximum number of windows to return.  The list is truncated
        deterministically.
    """
    boxes: List[Tuple[int, int, int, int]] = []
    for s in scales:
        base = int(min(W, H) * s)
        for r in np.linspace(ar_range[0], ar_range[1], 3):
            ww = int(base * r)
            hh = int(base / r)
            sx = max(1, int(ww * stride_ratio))
            sy = max(1, int(hh * stride_ratio))
            for y in range(0, H - hh + 1, sy):
                for x in range(0, W - ww + 1, sx):
                    # Heuristic: ignore top quarter; focus on lower portions
                    if y < int(0.25 * H):
                        continue
                    boxes.append((x, y, x + ww, y + hh))
    # Wrist priors: additional boxes centred at typical wrist positions
    for fx, fy in [(0.30, 0.70), (0.50, 0.70), (0.70, 0.70)]:
        size = int(min(W, H) * 0.30)
        x1 = max(0, int(W * fx - size / 2))
        y1 = max(0, int(H * fy - size / 2))
        boxes.append((x1, y1, min(W, x1 + size), min(H, y1 + size)))
    # Deduplicate and truncate
    uniq = list(dict.fromkeys(boxes))
    return uniq[:limit]


@torch.no_grad()
def crop_watch_and_vec(pil: Image.Image,
                       batch: int = 32,
                       ref_vecs: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], np.ndarray]:
    """Locate and vectorise the watch region in a detail image.

    This function uses a sliding window to generate candidate crops and
    selects the one with the highest cosine similarity to either the
    provided reference vectors or to the full image vector if no
    references are given.

    Parameters
    ----------
    pil : PIL.Image
        The image to analyse.
    batch : int, optional
        Batch size used when computing candidate embeddings.  Larger
        values offer better throughput on GPUs but may increase memory
        usage.
    ref_vecs : ndarray of shape (N, D), optional
        Reference vectors to compare against.  When provided the
        candidate with the highest maximum similarity to this set is
        returned.  If ``None``, the similarity is computed against
        the vector of the full image.

    Returns
    -------
    v_crop : Optional[np.ndarray]
        The aligned vector for the best crop.  ``None`` if no crops
        were generated (e.g. image too small).
    box : Optional[Tuple[int,int,int,int]]
        The (x1, y1, x2, y2) coordinates of the best crop in the
        original image.  ``None`` if no crops were generated.
    v_full : np.ndarray
        The aligned vector for the full image (always computed).
    """

    if Fs._model is None:
        init_model(CKPT_PATH, ADAPTER_PATH)
    W0, H0 = pil.size
    # Always compute full image vector
    v_full = to_aligned_vec(feats2048(pil))
    boxes = _gen_windows(W0, H0)
    if not boxes:
        return None, None, v_full

    # If no references provided, use the full vector as the reference
    if ref_vecs is None or len(ref_vecs) == 0:
        ref = v_full[None, :]
    else:
        ref = ref_vecs
    ref_norm = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12

    # Process candidates in batches
    sims: List[float] = []
    # Preprocess candidate crops into a tensor batch
    import torch.nn.functional as F
    crops = []
    for (x1, y1, x2, y2) in boxes:
        crop = pil.crop((x1, y1, x2, y2))
        crops.append(_TFM(crop))
    X = torch.stack(crops, 0).to(_device)
    idx = 0
    scores: List[float] = []
    for start in range(0, len(X), batch):
        end = min(len(X), start + batch)
        feats, _, _, _ = Fs._model(X[start:end])  # (n,2048)
        from  features import _W, _mean_q
        q = feats - _mean_q  # (n,2048)
        v = (_W @ q.T).T     # (n,2048)
        v = F.normalize(v, dim=1)
        v_np = v.detach().cpu().numpy()
        # Cosine similarity against each reference, take the max per candidate
        sim = (v_np @ ref.T) / (np.linalg.norm(v_np, axis=1, keepdims=True) + 1e-12) / ref_norm.T
        max_sim = sim.max(axis=1)
        scores.extend(max_sim.tolist())
    # Pick best candidate
    best_idx = int(np.argmax(np.array(scores)))
    best_box = boxes[best_idx]
    # Recompute aligned vector for best crop
    best_crop_vec = to_aligned_vec(feats2048(pil.crop(best_box)))
    return best_crop_vec, best_box, v_full
