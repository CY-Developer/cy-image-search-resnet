"""
Dataset for e-commerce product image retrieval and watermark detection.

The ``EcommerceDataset`` class extends ``torch.utils.data.Dataset`` to load
images and associated labels (product id, category, watermark label) from
a CSV file. It optionally applies water mark gating using local mask files
and/or a global watermark mask. Gating zeros out watermark regions,
reducing noise in the embedding space.

CSV columns expected by default:

* ``image_path``: path to the clean (non-watermarked) image file (relative to
  ``image_root`` or absolute). If empty, no clean sample is generated from
  this row.
* ``product_id``: identifier grouping images of the same product.
* ``category``: the category/class of the product.
* ``mask_path`` (optional): path to a binary mask image for the *clean* image.
  White (255) denotes watermark regions and black (0) denotes clean regions.
  If missing or empty, gating falls back to the global watermark mask only.
* ``wm_mask_path`` (optional; configurable via ``wm_col``): **path to the
  watermarked version of the image**, not a mask. When populated, the
  dataset will generate an additional sample for this row marked as
  watermarked.  The confusing name stems from historical reasons; ``wm_mask_path``
  actually holds the *watermarked image path*, while ``mask_path`` remains
  the location of a local binary mask for the clean image.

The dataset returns a tuple ``(image_tensor, product_label, category_label,
watermark_label)``. ``watermark_label`` is ``1`` for samples generated from
``wm_mask_path`` (the watermarked image) and ``0`` for samples generated
from ``image_path`` (the clean image).  The presence or absence of a local
mask file (``mask_path``) does **not** change ``watermark_label``.  If
``global_watermark_path`` is provided, gating still uses the global mask,
but ``watermark_label`` remains based solely on whether the sample is a
clean or watermarked version.

Gating can be disabled via ``mask_gating=False``, in which case the mask is
ignored and the original image is returned (while watermark labels are still
provided). Global masks are loaded via ``utils.load_global_mask``.
"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple, Optional, Set, Dict

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
        wm_col: Optional[str] = None,
        mask_suffix: str = "_mask.png",
        global_watermark_path: Optional[str] = None,
        alpha_threshold: float = 0.5,
        mask_gating: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):

        super().__init__()
        self.image_root = image_root
        self.image_col = image_col
        self.product_col = product_col
        self.category_col = category_col
        self.mask_col = mask_col
        # Optional name of column containing watermarked image paths. When
        # provided, the dataset will generate two samples from a single row
        # whenever both ``image_col`` and ``wm_col`` are populated. If
        # ``wm_col`` is None, only the clean image is considered.
        self.wm_col = wm_col
        self.mask_suffix = mask_suffix
        self.mask_gating = mask_gating

        self.samples: List[Tuple[str, str, str, Optional[str], int]] = []
        # Read CSV and populate samples. Each row may yield up to two samples:
        # a clean image (image_col) and/or a watermarked image (wm_col). The
        # tuple structure is (image_path, product_id, category, mask_path, wm_label)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Validate required columns
            if self.image_col and self.image_col not in reader.fieldnames:
                raise ValueError(f"Column {self.image_col} not found in CSV {csv_path}")
            if self.wm_col and self.wm_col not in reader.fieldnames:
                raise ValueError(f"Column {self.wm_col} not found in CSV {csv_path}")
            for field in (self.product_col, self.category_col):
                if field not in reader.fieldnames:
                    raise ValueError(f"Column {field} not found in CSV {csv_path}")
            # Iterate rows to create samples
            for row in reader:
                prod = row[self.product_col].strip()
                cat = row[self.category_col].strip()
                # Fetch local mask path from mask_col, if provided
                mask_path_value: Optional[str] = None
                if self.mask_col and self.mask_col in row and row[self.mask_col]:
                    candidate = row[self.mask_col].strip()
                    mask_path_value = candidate if candidate else None
                # Helper to resolve a given mask candidate using suffix inference
                def resolve_mask_for(image_rel: str) -> Optional[str]:
                    """Resolve mask path for a given image relative path."""
                    # Start from explicit mask_path_value
                    local_candidate = mask_path_value
                    # If no explicit mask for this row, try suffix-based inference
                    if not local_candidate and self.mask_suffix:
                        base, ext = os.path.splitext(image_rel)
                        candidate = f"{base}{self.mask_suffix}"
                        local_candidate = candidate
                    if local_candidate:
                        resolved = self._resolve_path(local_candidate)
                        return resolved if os.path.isfile(resolved) else None
                    return None
                # Clean image sample
                if self.image_col:
                    img_rel = row[self.image_col].strip()
                    if img_rel:
                        img_path = self._resolve_path(img_rel)
                        if os.path.isfile(img_path):
                            # Resolve mask for clean image
                            resolved_mask = resolve_mask_for(img_rel)
                            # For clean image, wm_label=0
                            self.samples.append((img_path, prod, cat, resolved_mask, 0))
                # Watermarked image sample
                if self.wm_col:
                    wm_rel = row[self.wm_col].strip() if self.wm_col in row else ""
                    if wm_rel:
                        wm_path = self._resolve_path(wm_rel)
                        if os.path.isfile(wm_path):
                            # Resolve mask for watermarked image using the same mask inference
                            resolved_wm_mask = resolve_mask_for(wm_rel)
                            # For watermarked image, wm_label=1
                            self.samples.append((wm_path, prod, cat, resolved_wm_mask, 1))

        # Build product and category mapping to indices
        product_ids = sorted({s[1] for s in self.samples})
        self.prod2label = {pid: i for i, pid in enumerate(product_ids)}
        categories = sorted({s[2] for s in self.samples})
        self.cat2label = {cat: i for i, cat in enumerate(categories)}

        # Identify S0/S1/S01 cohorts
        # S0: products with only clean images (wm_label=0)
        # S1: products with only watermarked images (wm_label=1)
        # S01: products that have both clean and watermarked images
        # Build a mapping of product -> set of watermark labels observed
        prod_wm: Dict[str, Set[int]] = {}
        for _, prod, _, _, wm_label in self.samples:
            prod_wm.setdefault(prod, set()).add(wm_label)
        # Products with both 0 and 1 labels are S01
        self.s01_products: Set[str] = {p for p, wset in prod_wm.items() if 0 in wset and 1 in wset}
        # Convert to the corresponding product label indices
        self.s01_label_set: Set[int] = {self.prod2label[p] for p in self.s01_products}

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