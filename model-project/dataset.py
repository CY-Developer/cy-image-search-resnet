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
# 替换 ecommerce_image_retrieval/dataset.py 中的整个 EcommerceDataset 类

import csv, os
from typing import List, Tuple, Optional, Set, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
# from .utils import load_global_mask
from utils import load_global_mask

class EcommerceDataset(Dataset):
    """
    支持的 CSV 列（向后兼容）：
      - image_path         : 干净/普通图片（可为空；为空则本行不生成 wm=0 样本）
      - product_id         : 商品ID
      - category           : 类目
      - mask_path          : 对应 image_path 的局部掩模（可空，用于 gating，但不决定 wm_label）
      - wm_image_path      : 水印版图片路径（新增，推荐列名；若你还在用 'wm_mask_path' 存水印图，也能自动识别）
      - wm_local_mask      : 对应 wm_image_path 的局部掩模（新增，可空；为空则仅用全局掩模）

    展开规则（每行最多生成两条样本）：
      1) image_path 存在 → 生成一条样本，wm_label=0；若 mask_path 有值则用于 gating
      2) wm_image_path 存在 → 再生成一条样本，wm_label=1；优先用 wm_local_mask gating，否则退回全局掩模
    最终自动按商品聚合得到 S0 / S1 / S01 三类，用于 S01 优先采样和跨水印一致性学习。
    """

    def __init__(
            self,
            csv_path: str,
            image_root: str = "",
            image_col: str = "image_path",
            product_col: str = "product_id",
            category_col: str = "category",
            mask_col: str = "mask_path",
            # 推荐使用 wm_image_path / wm_local_mask；若仍使用旧列名 wm_mask_path 表示水印图，也能自动识别
            wm_col: str = "wm_image_path",
            wm_mask_col: str = "wm_local_mask",
            mask_suffix: Optional[str] = None,
            global_watermark_path: Optional[str] = None,
            alpha_threshold: float = 0.5,
            mask_gating: bool = True,
            transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.root_dir = image_root
        self.image_col, self.product_col, self.category_col = image_col, product_col, category_col
        self.mask_col, self.wm_col, self.wm_mask_col = mask_col, wm_col, wm_mask_col
        self.mask_suffix = mask_suffix
        self.mask_gating = mask_gating

        # 读取 CSV
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            rows = list(rd)
            fieldnames = set(rd.fieldnames or [])

        # 兼容旧列名：若没有 wm_image_path 而存在 wm_mask_path，则把它当成“水印图列”
        if (self.wm_col not in fieldnames) and ("wm_mask_path" in fieldnames):
            self.wm_col = "wm_mask_path"
        # wm_local_mask 缺失时就当作没有逐图水印掩模
        if self.wm_mask_col not in fieldnames:
            self.wm_mask_col = None

        # 样本：(img_path, product_id, category, wm_label, local_mask_path)
        self.samples: List[Tuple[str, str, str, int, Optional[str]]] = []

        for r in rows:
            pid = (r.get(self.product_col) or "").strip()
            cat = (r.get(self.category_col) or "").strip()

            # 1) 干净样本（wm=0）
            img_rel = (r.get(self.image_col) or "").strip()
            if img_rel:
                img_path = self._resolve(img_rel)
                if os.path.isfile(img_path):
                    m_local = None
                    if self.mask_col in fieldnames:
                        m_rel = (r.get(self.mask_col) or "").strip()
                        if m_rel:
                            m_local = self._resolve(m_rel)
                    # 可选根据后缀猜掩模（例如 _mask.png）
                    if (not m_local) and self.mask_suffix:
                        guess = os.path.splitext(img_path)[0] + self.mask_suffix
                        if os.path.isfile(guess):
                            m_local = guess
                    self.samples.append((img_path, pid, cat, 0, m_local))

            # 2) 水印样本（wm=1）
            if self.wm_col:
                wm_img_rel = (r.get(self.wm_col) or "").strip()
                if wm_img_rel:
                    wm_img_path = self._resolve(wm_img_rel)
                    if os.path.isfile(wm_img_path):
                        wm_local = None
                        if self.wm_mask_col:
                            wmm_rel = (r.get(self.wm_mask_col) or "").strip()
                            if wmm_rel:
                                wm_local = self._resolve(wmm_rel)
                        self.samples.append((wm_img_path, pid, cat, 1, wm_local))

        # 编码标签
        products = sorted({p for _, p, _, _, _ in self.samples})
        categories = sorted({c for _, _, c, _, _ in self.samples})
        self.prod2label = {p: i for i, p in enumerate(products)}
        self.cat2label = {c: i for i, c in enumerate(categories)}

        # 标记 S01 商品集合
        prod2wm: Dict[str, Set[int]] = {}
        for _, p, _, w, _ in self.samples:
            prod2wm.setdefault(p, set()).add(w)
        self.s01_products: Set[str] = {p for p, s in prod2wm.items() if 0 in s and 1 in s}
        self.s01_label_set: Set[int] = {self.prod2label[p] for p in self.s01_products}

        # 变换
        if transform is None:
            self.transform_img = transforms.Compose([
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform_img = transform

        self.transform_mask = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # 全局水印掩模（可选）
        self.global_mask = None
        if global_watermark_path:
            if not os.path.isfile(global_watermark_path):
                raise FileNotFoundError(f"Global watermark not found: {global_watermark_path}")
            g = load_global_mask(global_watermark_path, alpha_threshold)
            self.global_mask = (self.transform_mask(g) > 0.5).float()  # [1,H,W]

    def _resolve(self, rel: str) -> str:
        rel = rel.replace("\\", "/").strip()
        return rel if os.path.isabs(rel) or not self.root_dir else os.path.join(self.root_dir, rel)

    def __len__(self) -> int:
        return len(self.samples)

    def _open_rgb   (self,path: str) -> Image.Image:
        im = Image.open(path)
        # P 模式且带透明度 → 先 RGBA，铺白底后回到 RGB
        if im.mode == "P" and "transparency" in im.info:
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im).convert("RGB")
        else:
            im = im.convert("RGB")
        return im
    def _open_mask(self,path: str) -> Image.Image:
        im = Image.open(path)
        # 若自带 alpha（RGBA/LA），优先用 alpha 作为掩模
        if im.mode in ("RGBA", "LA"):
            return im.split()[-1].convert("L")
        # P 模式且有透明信息：转 RGBA 后取 alpha
        if im.mode == "P" and "transparency" in im.info:
            im = im.convert("RGBA")
            return im.split()[-1].convert("L")
        # 普通灰度
        return im.convert("L")
    def __getitem__(self, idx: int):
        path, pid, cat, wm, local_mask = self.samples[idx]
        img = self._open_rgb(path).convert("RGB")
        x = self.transform_img(img)

        # gating：优先本地掩模，其次全局掩模（干净/水印样本都可用，有就抑制）
        if self.mask_gating:
            m = None
            if local_mask and os.path.isfile(local_mask):
                try:
                    m_img = self._open_mask(local_mask).convert("L")
                    m = (self.transform_mask(m_img) > 0.5).float()
                except Exception:
                    m = None
            if (m is None) and (self.global_mask is not None):
                m = self.global_mask
            if m is not None:
                x = x * (1.0 - m)

        return x, self.prod2label[pid], self.cat2label[cat], wm

    @property
    def num_products(self) -> int:
        return len(self.prod2label)

    @property
    def num_categories(self) -> int:
        return len(self.cat2label)

    @property
    def labels(self) -> List[int]:
        return [self.prod2label[p] for _, p, _, _, _ in self.samples]
