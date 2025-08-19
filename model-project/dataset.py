"""
dataset.py
~~~~~~~~~~~~~~~~

此模块定义了用于电商商品图像识别任务的数据集类以及三元组采样逻辑。数据集根据提供的 CSV 文件读取图片路径、商品编号和水印标签，并支持在线随机生成 anchor-positive-negative 组合。

主要类：

* ``ProductDataset``——从 CSV 加载单张图片及其标签，提供基本的图像读取与预处理。
* ``TripletDataset``——包装 ``ProductDataset``，在 ``__getitem__`` 时随机选取正样本和负样本，返回三张图片以及对应的标签，用于 Triplet Loss 训练。

使用示例见 ``train.py``。

"""

import os
import random
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd  # ← 按你的要求改为 pandas 读表


class ProductDataset(Dataset):
    """基础商品数据集。

    参数:
        csv_path: 包含图像路径、商品编号、类别和水印标签的 CSV 文件。
        image_root: 图片根目录，若 ``image_path`` 为相对路径，则在此目录下查找。
        transform: 可选的图像预处理函数/组合，例如 ``torchvision.transforms``。

    返回:
        image: 经过预处理的图像张量（此处为 4 通道：RGB + Mask）。
        product_id: 商品编号（字符串）。
        category: 商品类别（字符串）。
        is_watermark: 是否有水印（布尔）——为兼容旧训练脚本仍返回，但掩模加载不再依赖该字段。
    """

    def __init__(
            self,
            csv_path: str,
            image_root: str = "",
            transform=None,
            mask_suffix: str = "_mask.png",
            mask_transform=None,
            global_watermark_path: Optional[str] = None,
            alpha_threshold: float = 0.5,
    ) -> None:
        """
        初始化数据集。

        参数：
            csv_path: CSV 标注文件路径。推荐列名：
                      image_path, product_id, category[, is_watermark][, mask_path]
            image_root: 图片根目录，当图像路径和掩模路径为相对路径时，在该目录下查找。
            transform: 用于处理 RGB 图像的 torchvision 变换，通常包括 Resize、ToTensor、Normalize。
            mask_suffix: 自动猜测单张图片掩模文件的默认后缀，例如 ``"_mask.png"``。
            mask_transform: 掩模所需的变换，若为 None，则从 ``transform`` 复制除 Normalize 之外的变换。
            global_watermark_path: 全局水印 PNG 或带透明通道的图片路径。若提供，则从该文件生成统一的水印掩模，
                用于所有图片；此掩模会与单独的 ``mask_path`` 或 ``mask_suffix`` 推断的掩模进行合并。
            alpha_threshold: 当 ``global_watermark_path`` 为 PNG 时，提取其 alpha/灰度通道大于该阈值的位置视为水印区域（0~1）。

        说明：
            现在“是否含水印”的布尔列仅作为分类标签使用；是否加载掩模不再依赖该列。
            掩模优先顺序：CSV.mask_path > 同名文件 + mask_suffix > 全零掩模；全局掩模与之取 max 合并。
        """
        super().__init__()
        self.image_root = image_root
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.mask_transform = mask_transform
        self.alpha_threshold = alpha_threshold

        # 掩模变换：若未提供，则复制 transform 去掉 Normalize
        if self.mask_transform is None and self.transform is not None and hasattr(self.transform, "transforms"):
            self.mask_transform = transforms.Compose(
                [t for t in self.transform.transforms if not isinstance(t, transforms.Normalize)]
            )

        # —— 用 pandas 读取 CSV，自动嗅探分隔符与编码，并支持注释行 ——
        df = None
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                df = pd.read_csv(csv_path, engine="python", sep=None, encoding=enc, comment="#")
                break
            except Exception:
                continue
        if df is None:
            raise RuntimeError(f"无法读取 CSV：{csv_path}")

        # 列名标准化
        df.columns = [str(c).strip().lower() for c in df.columns]
        required = {"image_path", "product_id", "category"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"CSV 缺少必须列：{required}；当前列：{list(df.columns)}")

        if "mask_path" not in df.columns:
            df["mask_path"] = ""
        if "is_watermark" not in df.columns:
            # 为兼容旧训练逻辑，默认 False；但掩模加载不再依赖此列
            df["is_watermark"] = False

        # 组装记录
        self.records: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            self.records.append(
                {
                    "image_path": str(row["image_path"]).strip(),
                    "product_id": str(row["product_id"]).strip(),
                    "category": str(row["category"]).strip(),
                    "is_watermark": str(row["is_watermark"]).strip().lower() in ("1", "true", "yes"),
                    "mask_path": (str(row["mask_path"]).strip() if pd.notna(row["mask_path"]) else ""),
                }
            )

        # 建立产品编号到索引的映射，便于采样正样本
        self.label_to_indices: Dict[str, List[int]] = {}
        for idx, record in enumerate(self.records):
            pid = record["product_id"]
            self.label_to_indices.setdefault(pid, []).append(idx)

        # 初始化全局水印掩模（可选）
        self.global_mask: Optional[torch.Tensor] = None
        if global_watermark_path:
            abs_path = global_watermark_path
            if not os.path.isabs(abs_path) and self.image_root:
                abs_path = os.path.join(self.image_root, global_watermark_path)
            try:
                with Image.open(abs_path) as wm_img:
                    if wm_img.mode in ("RGBA", "LA"):
                        alpha = wm_img.split()[-1]
                        mask_img = alpha
                    else:
                        mask_img = wm_img.convert("L")
                    if self.mask_transform:
                        mask_tensor = self.mask_transform(mask_img)
                    else:
                        mask_tensor = transforms.ToTensor()(mask_img)
                    self.global_mask = (mask_tensor > self.alpha_threshold).float()
            except Exception as e:
                print(f"警告：无法加载 global_watermark_path {global_watermark_path}: {e}")
                self.global_mask = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, str, bool]:
        """
        返回包括掩模在内的图像张量。输出张量形状为 [4, H, W]，前三个通道为 RGB，第四个通道为掩模。
        掩模加载顺序：CSV.mask_path → 同名 + mask_suffix → 全零；随后与全局掩模 max 合并。
        """
        record = self.records[index]
        img_path = record["image_path"]

        # 构造绝对路径
        if not os.path.isabs(img_path) and self.image_root:
            img_path = os.path.join(self.image_root, img_path)

        # 读取 RGB 图像（若损坏则随机回退）
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except (FileNotFoundError, OSError):
            return self.__getitem__(random.randint(0, len(self.records) - 1))

        img_tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)

        # 读取掩模：不再依赖 is_watermark，直接按优先级尝试
        mask_tensor: Optional[torch.Tensor] = None

        # 1) CSV 指定 mask_path
        m_path = record.get("mask_path") or ""
        if m_path:
            if not os.path.isabs(m_path) and self.image_root:
                m_path = os.path.join(self.image_root, m_path)
            if os.path.exists(m_path):
                with Image.open(m_path) as m:
                    m = m.convert("L")
                    mask_tensor = (self.mask_transform(m) if self.mask_transform else transforms.ToTensor()(m))
                    mask_tensor = (mask_tensor > 0.5).float()

        # 2) 猜测同名掩模（即使 is_watermark=False 也尝试）
        if mask_tensor is None:
            guessed_path = os.path.splitext(img_path)[0] + self.mask_suffix
            if os.path.exists(guessed_path):
                with Image.open(guessed_path) as m:
                    m = m.convert("L")
                    mask_tensor = (self.mask_transform(m) if self.mask_transform else transforms.ToTensor()(m))
                    mask_tensor = (mask_tensor > 0.5).float()

        # 3) 仍无则全零掩模
        if mask_tensor is None:
            mask_tensor = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32)

        # 4) 合并全局掩模
        if self.global_mask is not None:
            gm = self.global_mask
            if gm.shape[1:] != mask_tensor.shape[1:]:
                gm = torch.nn.functional.interpolate(gm.unsqueeze(0), size=mask_tensor.shape[1:], mode="nearest").squeeze(0)
            mask_tensor = torch.max(mask_tensor, gm)

        # 拼接通道：[3,H,W] + [1,H,W] -> [4,H,W]
        img_with_mask = torch.cat([img_tensor, mask_tensor], dim=0)
        return img_with_mask, record["product_id"], record["category"], record["is_watermark"]


class TripletDataset(Dataset):
    """三元组数据集生成器。

    给定基础 ``ProductDataset``，本类在每次索引时随机采样同一商品的正样本和不同商品的负样本，
    用于 Triplet Loss 训练。返回值包括三张图片和相应的水印标签，便于多任务训练。

    返回:
        A tuple of:
            images: (anchor_img, positive_img, negative_img)
            watermarks: (anchor_is_watermark, positive_is_watermark, negative_is_watermark)
    """

    def __init__(self, base_dataset: ProductDataset) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.label_to_indices = base_dataset.label_to_indices
        self.labels = list(self.label_to_indices.keys())

    def __len__(self) -> int:
        # 长度与基础数据集相同
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        # Anchor
        anchor_img, anchor_pid, _, anchor_is_wm = self.base_dataset[index]

        # Positive：同一 product_id
        positive_indices = self.label_to_indices[anchor_pid]
        pos_index = index
        if len(positive_indices) > 1:
            while pos_index == index:
                pos_index = random.choice(positive_indices)
        positive_img, _, _, positive_is_wm = self.base_dataset[pos_index]

        # Negative：优先同类别不同商品
        anchor_cate = self.base_dataset.records[index]["category"]
        attempt = 0
        negative_pid = anchor_pid
        while negative_pid == anchor_pid and attempt < 10:
            cand_idx = random.randint(0, len(self.base_dataset.records) - 1)
            cand_rec = self.base_dataset.records[cand_idx]
            if cand_rec["product_id"] != anchor_pid and cand_rec["category"] == anchor_cate:
                negative_pid = cand_rec["product_id"]
                negative_index = cand_idx
            attempt += 1
        if negative_pid == anchor_pid:
            negative_pid = random.choice([pid for pid in self.labels if pid != anchor_pid])
            negative_index = random.choice(self.label_to_indices[negative_pid])

        negative_img, _, _, negative_is_wm = self.base_dataset[negative_index]
        return (anchor_img, positive_img, negative_img), (anchor_is_wm, positive_is_wm, negative_is_wm)
