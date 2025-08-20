"""
Utility functions for watermark mask processing and PSD handling.

This module provides helpers to generate binary masks from PNG or PSD files.

``load_global_mask`` loads a watermark image with transparency and
converts the alpha channel (or grayscale intensity) to a binary mask
based on a given threshold. The resulting mask is a PIL Image in mode
"L" with pixel values 0 (no watermark) or 1 (watermark).

For PSD files, Pillow does not support direct alpha extraction. If
``psd-tools`` is installed, ``load_psd_mask`` can be used to read the
alpha channel from a PSD. Otherwise, users should convert the PSD to
PNG offline.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

try:
    from psd_tools import PSDImage  # optional
    _PSD_AVAILABLE = True
except ImportError:
    _PSD_AVAILABLE = False


def load_global_mask(path: str, alpha_threshold: float = 0.5) -> Image.Image:
    """Load a global watermark image and produce a binary mask.

    Parameters
    ----------
    path: str
        Path to the watermark image. Supported formats include PNG with
        transparency (RGBA or LA) or standard image formats (RGB, L).
    alpha_threshold: float, optional
        Alpha threshold in [0,1]. Pixels with alpha greater than this
        threshold are considered watermark (mask value 1). If the image
        lacks an alpha channel, grayscale intensity is used instead.

    Returns
    -------
    PIL.Image.Image
        A grayscale image (mode "L") with values 0 or 1. 1 indicates
        watermark region.
    """
    img = Image.open(path)
    # Extract alpha or grayscale
    if img.mode in ("RGBA", "LA"):
        alpha = img.getchannel("A")
    elif img.mode == "P":
        # Palette mode may have transparency info in info['transparency']
        if "transparency" in img.info:
            transparency = img.info["transparency"]
            # convert palette to RGBA then extract alpha
            img_rgba = img.convert("RGBA")
            alpha = img_rgba.getchannel("A")
        else:
            # no transparency; treat full intensity as watermark
            alpha = img.convert("L")
    else:
        # Fallback: convert to grayscale and treat intensity as alpha
        alpha = img.convert("L")
    # Normalize alpha to [0,1]
    alpha_f = alpha.point(lambda p: p / 255.0)
    # Apply threshold
    mask = alpha_f.point(lambda p: 1 if p > alpha_threshold else 0)
    mask = mask.convert("L")
    return mask


def load_psd_mask(path: str, alpha_threshold: float = 0.5) -> Image.Image:
    """Extract a watermark mask from a PSD file, if psd-tools is available.

    Parameters
    ----------
    path: str
        Path to the PSD file.
    alpha_threshold: float
        Threshold for alpha channel.

    Returns
    -------
    PIL.Image.Image
        A grayscale mask image (0/1) of the same size as the PSD.

    Raises
    ------
    ImportError
        If psd-tools is not installed.
    RuntimeError
        If PSD file has no alpha channel.
    """
    if not _PSD_AVAILABLE:
        raise ImportError(
            "psd-tools is required to load PSD files; install psd-tools or convert the PSD to PNG."
        )
    psd = PSDImage.open(path)
    # Attempt to find a layer containing transparency (alpha)
    # This is a heuristic: we search for the first layer with a non-zero alpha channel.
    for layer in psd:
        if layer.has_transparency():
            # layer.topil returns RGBA
            img_rgba = layer.topil()
            alpha = img_rgba.getchannel("A")
            alpha_f = alpha.point(lambda p: p / 255.0)
            mask = alpha_f.point(lambda p: 1 if p > alpha_threshold else 0).convert("L")
            return mask
    raise RuntimeError("No layer with transparency found in PSD")
