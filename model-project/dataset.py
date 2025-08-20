"""
Dataset for e-commerce product image retrieval and watermark detection.

The ``EcommerceDataset`` class extends ``torch.utils.data.Dataset`` to load
images and associated labels (product id, category, watermark label) from
a CSV file. It optionally applies water mark gating using local mask files
and/or a global watermark mask. Gating zeros out watermark regions,
reducing noise in the embedding space.

CSV columns expected by default:

* ``image_path``: path to the image file (relative to ``image_root`` or absolute).
* ``product_id``: identifier grouping images of the same product.
* ``category``: the category/class of the product.
* ``mask_path`` (optional): path to a binary mask image. White (255) denotes
  watermark regions and black (0) denotes clean regions. If missing or empty,
  the sample is considered free of watermark (unless global_mask is used).

The dataset returns a tuple ``(image_tensor, product_label, category_label,
watermark_label)``. ``watermark_label`` is ``1`` if ``mask_path`` exists and
``0`` otherwise. If ``global_watermark_path`` is provided, gating will still
use the global mask, but ``watermark_label`` remains based on ``mask_path``.

Gating can be disabled via ``mask_gating=False``, in which case the mask is
ignored and the original image is returned (while watermark labels are still
provided). Global masks are loaded via ``utils.load_global_mask``.
"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from .utils import load_global_mask


class EcommerceDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_root: str = "",
        image_col: str = "image_path",
        product_col: str = "product_id",
        category_col: str = "category",
        mask_col: str = "mask_path",
        mask_suffix: str = "_mask.png",
        global_watermark_path: Optional[str] = None,
        alpha_threshold: float = 0.5,
        mask_gating: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        csv_path: str
            Path to the CSV file containing dataset annotations.
        image_root: str
            Root directory used to resolve relative image paths.
        image_col: str
            Name of the CSV column containing image paths.
        product_col: str
            Name of the column containing product identifiers.
        category_col: str
            Name of the column containing category labels.
        mask_col: str
            Name of the column containing paths to local watermark masks. If
            empty or missing in a row, the sample is treated as having no
            watermark (``watermark_label=0``) and ``mask_path`` is ignored.
        mask_suffix: str
            Suffix appended to ``image_path`` to infer mask path when
            ``mask_col`` is not present or empty. For example, if
            ``image_path`` is ``abc.jpg`` and ``mask_suffix`` is ``_mask.png``,
            the inferred mask path is ``abc_mask.png`` in the same directory.
        global_watermark_path: Optional[str]
            Path to a global watermark PNG (or PSD converted to PNG). If
            provided, a binary mask will be generated from its alpha channel
            using ``alpha_threshold``. The global mask is combined with
            local masks using logical OR during gating. ``watermark_label``
            remains based solely on ``mask_col``.
        alpha_threshold: float
            Threshold for converting the global watermark alpha channel into a
            binary mask. Range [0,1]. Higher values produce smaller masks.
        mask_gating: bool
            If True, apply gating by zeroing out pixels in watermark regions.
        transform: Optional[transforms.Compose]
            Transform applied to the images before returning. If None,
            a default ImageNet-style transform (Resize->CenterCrop->ToTensor->Normalize)
            is used.
        """
        super().__init__()
        self.image_root = image_root
        self.image_col = image_col
        self.product_col = product_col
        self.category_col = category_col
        self.mask_col = mask_col
        self.mask_suffix = mask_suffix
        self.mask_gating = mask_gating

        self.samples: List[Tuple[str, str, str, Optional[str], int]] = []
        # Read CSV and populate samples (image_path, product_id, category, mask_path, watermark_label)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if self.image_col not in reader.fieldnames:
                raise ValueError(f"Column {self.image_col} not found in CSV {csv_path}")
            for field in (self.product_col, self.category_col):
                if field not in reader.fieldnames:
                    raise ValueError(f"Column {field} not found in CSV {csv_path}")
            # mask_col may be missing entirely; we handle per-row
            for row in reader:
                img_rel = row[self.image_col].strip()
                prod = row[self.product_col].strip()
                cat = row[self.category_col].strip()
                # Determine mask path
                mask_path: Optional[str] = None
                # If mask_col exists and row has entry, use it
                if self.mask_col in row and row[self.mask_col]:
                    candidate = row[self.mask_col].strip()
                    mask_path = candidate if candidate else None
                # Fallback: infer mask by suffix
                if not mask_path and self.mask_suffix:
                    # Append suffix to base filename without extension
                    base, ext = os.path.splitext(img_rel)
                    candidate = f"{base}{self.mask_suffix}"
                    mask_path = candidate
                # Compute watermark_label: 1 if mask exists and is file, else 0
                resolved_mask_path = None
                if mask_path:
                    resolved_mask_path = self._resolve_path(mask_path)
                    if not os.path.isfile(resolved_mask_path):
                        resolved_mask_path = None
                wm_label = 1 if resolved_mask_path else 0
                # Resolve image path
                img_path = self._resolve_path(img_rel)
                if not os.path.isfile(img_path):
                    # Skip if image does not exist
                    continue
                self.samples.append((img_path, prod, cat, resolved_mask_path, wm_label))

        # Build product and category mapping to indices
        product_ids = sorted({s[1] for s in self.samples})
        self.prod2label = {pid: i for i, pid in enumerate(product_ids)}
        categories = sorted({s[2] for s in self.samples})
        self.cat2label = {cat: i for i, cat in enumerate(categories)}

        # Create transforms for images and masks
        if transform is None:
            # Default transforms: Resize -> CenterCrop -> ToTensor -> Normalize
            self.transform_img = transforms.Compose([
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform_img = transform
        # Mask transform: Resize and CenterCrop, nearest interpolation, to tensor (0/1 values)
        self.transform_mask = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Load global mask if provided
        if global_watermark_path:
            if not os.path.isfile(global_watermark_path):
                raise FileNotFoundError(f"Global watermark file not found: {global_watermark_path}")
            gmask_img = load_global_mask(global_watermark_path, alpha_threshold=alpha_threshold)
            # Apply mask transform and threshold to get values 0/1 (Tensor shape [1, H, W])
            gmask_tensor = self.transform_mask(gmask_img)
            # Ensure binary
            gmask_tensor = (gmask_tensor > 0.5).float()
            self.global_mask = gmask_tensor  # shape [1,224,224]
        else:
            self.global_mask = None

    def _resolve_path(self, rel_path: str) -> str:
        """Resolve a path relative to image_root if not absolute."""
        if os.path.isabs(rel_path) or not self.image_root:
            return rel_path
        return os.path.join(self.image_root, rel_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, prod, cat, mask_path, wm_label = self.samples[idx]
        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        img_tensor = self.transform_img(img)  # shape [3,224,224]
        # Prepare gating mask if required
        if self.mask_gating:
            # Start with zeros (no gating)
            mask_combined = None
            # Local mask
            if mask_path:
                with Image.open(mask_path) as mimg:
                    mimg = mimg.convert("L")
                local_mask = self.transform_mask(mimg)
                local_mask = (local_mask > 0.5).float()  # binary [1,224,224]
                mask_combined = local_mask if mask_combined is None else torch.max(mask_combined, local_mask)
            # Global mask
            if self.global_mask is not None:
                mask_combined = self.global_mask if mask_combined is None else torch.max(mask_combined, self.global_mask)
            # If mask exists, apply gating: zero out masked regions
            if mask_combined is not None:
                # Expand mask to match channels
                mask_inv = 1.0 - mask_combined  # shape [1,H,W]
                img_tensor = img_tensor * mask_inv
        # Convert labels to indices
        prod_label = self.prod2label[prod]
        cat_label = self.cat2label[cat]
        return img_tensor, prod_label, cat_label, wm_label

    @property
    def num_products(self) -> int:
        return len(self.prod2label)

    @property
    def num_categories(self) -> int:
        return len(self.cat2label)

    @property
    def labels(self) -> List[int]:
        """Return list of product labels for sampling."""
        return [self.prod2label[s[1]] for s in self.samples]