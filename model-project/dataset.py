"""
dataset.py
~~~~~~~~~~~~~~~~

此模块定义了用于电商商品图像识别任务的数据集类以及三元组采样逻辑。数据集根据提供的 CSV 文件读取图片路径、商品编号和水印标签，并支持在线随机生成 anchor‑positive‑negative 组合。

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


class ProductDataset(Dataset):
    """基础商品数据集。

    参数:
        csv_path: 包含图像路径、商品编号、类别和水印标签的 CSV 文件。
        image_root: 图片根目录，若 ``image_path`` 为相对路径，则在此目录下查找。
        transform: 可选的图像预处理函数/组合，例如 ``torchvision.transforms``。

    返回:
        image: 经过预处理的图像张量。
        product_id: 商品编号（字符串）。
        category: 商品类别（字符串）。
        is_watermark: 是否有水印（布尔）。
    """

    def __init__(self,
                 csv_path: str,
                 image_root: str = "",
                 transform=None,
                 mask_suffix: str = "_mask.png",
                 mask_transform=None,
                 global_watermark_path: Optional[str] = None,
                 alpha_threshold: float = 0.5) -> None:
        """
        初始化数据集。

        参数：
            csv_path: CSV 标注文件路径，每行格式为 ``<image_path>,<product_id>,<category>,<is_watermark>[,<mask_path>]``。
            image_root: 图片根目录，当图像路径和掩模路径为相对路径时，在该目录下查找。
            transform: 用于处理 RGB 图像的 torchvision 变换，通常包括 Resize、ToTensor、Normalize。
            mask_suffix: 自动猜测单张图片掩模文件的默认后缀，例如 ``"_mask.png"``。
            mask_transform: 掩模所需的变换，若为 None，则从 ``transform`` 复制除 Normalize 之外的变换。
            global_watermark_path: 全局水印 PNG 或带透明通道的图片路径。若提供，则从该文件生成统一的水印掩模，
                用于所有包含水印的图片。此掩模会与单独的 ``mask_path`` 或 ``mask_suffix`` 推断的掩模进行合并。
            alpha_threshold: 当 ``global_watermark_path`` 为 PNG 时，提取其 alpha 通道或灰度通道大于该阈值的位置
                视为水印区域；取值范围为 0~1。

        说明：
            通过 ``global_watermark_path``，可以利用已有的水印源文件（如 PSD 导出为 PNG）来生成统一的水印掩模。
            这样模型在训练时能清楚知道水印的具体形状和位置，对应地在推理阶段自动抑制水印区域的影响。
        """
        super().__init__()
        self.records: List[Dict[str, str]] = []
        # 读取 CSV 标注，支持注释行（以 # 开头）
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    raise ValueError(f"CSV 格式错误: {line}")
                image_path, product_id, category, is_watermark = parts[:4]
                mask_path = None
                if len(parts) >= 5:
                    mask_path = parts[4] or None
                self.records.append({
                    "image_path": image_path,
                    "product_id": product_id,
                    "category": category,
                    "is_watermark": is_watermark.lower() in ("1", "true", "yes"),
                    "mask_path": mask_path
                })
        self.image_root = image_root
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.mask_transform = mask_transform
        # 如果未提供掩模变换，则沿用图像变换（去掉归一化）
        if self.mask_transform is None and self.transform is not None:
            self.mask_transform = transforms.Compose([
                t for t in self.transform.transforms if not isinstance(t, transforms.Normalize)
            ]) if hasattr(self.transform, 'transforms') else None
        # 初始化全局水印掩模
        self.global_mask: Optional[torch.Tensor] = None
        self.alpha_threshold = alpha_threshold
        if global_watermark_path:
            # 解析全局水印 PNG：读取 alpha 通道或灰度值，生成 0/1 掩模
            abs_path = global_watermark_path
            if not os.path.isabs(abs_path) and self.image_root:
                abs_path = os.path.join(self.image_root, global_watermark_path)
            try:
                with Image.open(abs_path) as wm_img:
                    # 若图像包含透明通道，则使用 alpha 通道；否则转为灰度
                    if wm_img.mode in ("RGBA", "LA"):
                        alpha = wm_img.split()[-1]
                        mask_img = alpha
                    else:
                        mask_img = wm_img.convert("L")
                    # 转为 [1,H,W]，并应用相同几何变换
                    if self.mask_transform:
                        mask_tensor = self.mask_transform(mask_img)
                    else:
                        mask_tensor = transforms.ToTensor()(mask_img)
                    # 二值化：大于阈值的像素视为水印区域
                    self.global_mask = (mask_tensor > self.alpha_threshold).float()
            except Exception as e:
                print(f"警告：无法加载 global_watermark_path {global_watermark_path}: {e}")
                self.global_mask = None
        # 建立产品编号到索引的映射，便于采样正样本
        self.label_to_indices: Dict[str, List[int]] = {}
        for idx, record in enumerate(self.records):
            pid = record["product_id"]
            self.label_to_indices.setdefault(pid, []).append(idx)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, str, bool]:
        """
        返回包括掩模在内的图像张量。若该图片带水印，则尝试在同目录下加载对应的掩模文件（图片名加 ``mask_suffix``），
        掩模为单通道灰度图，像素值为 1 表示水印区域，0 表示非水印区域。如未找到掩模，则默认为零矩阵。
        输出张量形状为 [4, H, W]，前三个通道为 RGB，第四个通道为掩模。
        """
        record = self.records[index]
        img_path = record["image_path"]
        # 构造绝对路径
        if not os.path.isabs(img_path) and self.image_root:
            img_path = os.path.join(self.image_root, img_path)
        # 读取 RGB 图像
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except (FileNotFoundError, OSError):
            # 跳过损坏或缺失图片：随机换一条样本
            return self.__getitem__(random.randint(0, len(self.records) - 1))

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        # 读取掩模（针对该图片的水印区）。由于掩模和图片经相同的 ``transform`` 处理，因此输出尺寸一致。
        mask: Optional[torch.Tensor] = None
        # 1. 如果 CSV 指定了 mask_path，则尝试加载该掩模
        mask_path = record.get("mask_path")
        if mask_path:
            resolved_mask_path = mask_path
            if not os.path.isabs(mask_path) and self.image_root:
                resolved_mask_path = os.path.join(self.image_root, mask_path)
            if os.path.exists(resolved_mask_path):
                with Image.open(resolved_mask_path) as m:
                    m = m.convert("L")
                    if self.mask_transform:
                        mask_tensor = self.mask_transform(m)
                    else:
                        mask_tensor = transforms.ToTensor()(m)
                    mask = (mask_tensor > 0.5).float()
        # 2. 若未指定 mask_path 且标记为含水印图片，则尝试按后缀猜测掩模路径
        if mask is None and record["is_watermark"]:
            guessed_path = os.path.splitext(img_path)[0] + self.mask_suffix
            if os.path.exists(guessed_path):
                with Image.open(guessed_path) as m:
                    m = m.convert("L")
                    if self.mask_transform:
                        mask_tensor = self.mask_transform(m)
                    else:
                        mask_tensor = transforms.ToTensor()(m)
                    mask = (mask_tensor > 0.5).float()
        # 3. 若没有单独掩模，则使用全零掩模（即认为图片上没有水印或无需遮挡）
        if mask is None:
            mask = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32)
        # 4. 如果存在全局水印掩模，则与当前掩模取最大值，确保任何一个位置标记为水印都会被遮挡
        if self.global_mask is not None:
            # 全局掩模尺寸可能与掩模尺寸不匹配（例如数据集中多种尺寸）。这里假设 transform 已经把图像和掩模缩放到统一大小，
            # 因此直接广播即可。若大小不一致，需插值调整。
            if self.global_mask.shape[1:] != mask.shape[1:]:
                # 使用最近邻插值调整尺寸
                resized_global = torch.nn.functional.interpolate(self.global_mask.unsqueeze(0), size=mask.shape[1:],
                                                                 mode='nearest').squeeze(0)
            else:
                resized_global = self.global_mask
            mask = torch.max(mask, resized_global)
        # 拼接通道：[3,H,W] + [1,H,W] -> [4,H,W]
        img_with_mask = torch.cat([img_tensor, mask], dim=0)
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

    def __getitem__(self, index: int) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[bool, bool, bool]]:
        # Anchor 样本
        anchor_img, anchor_pid, _, anchor_is_wm = self.base_dataset[index]
        # 正样本：从同一 product_id 中随机挑选另一个不同的索引
        positive_indices = self.label_to_indices[anchor_pid]
        # 如果商品只有一张图，则正样本为自身；实际部署时应保证每个商品至少两张图
        pos_index = index
        if len(positive_indices) > 1:
            while pos_index == index:
                pos_index = random.choice(positive_indices)
        positive_img, _, _, positive_is_wm = self.base_dataset[pos_index]

        # 负样本：同类别不同商品→更具区分度
        # —— 负样本采样（优先：同类别 + 不同商品） ——————————————
        # 先拿到 anchor 的完整标注，便于获取类别
        anchor_record = self.base_dataset.records[index]
        anchor_cate   = anchor_record["category"]

        attempt      = 0
        negative_pid = anchor_pid          # 初始化为相同商品，方便进入循环

        # 尝试最多 10 次，找“同类别 • 不同商品”的负样本
        while negative_pid == anchor_pid and attempt < 10:
            cand_idx  = random.randint(0, len(self.base_dataset.records) - 1)
            cand_rec  = self.base_dataset.records[cand_idx]
            if cand_rec["product_id"] != anchor_pid and cand_rec["category"] == anchor_cate:
                negative_pid   = cand_rec["product_id"]
                negative_index = cand_idx               # 记录索引，后面直接取图
            attempt += 1

        # 如果仍然没找到，就退而求其次：任意“不同商品”的负样本
        if negative_pid == anchor_pid:
            negative_pid  = random.choice([pid for pid in self.labels if pid != anchor_pid])
            negative_index = random.choice(self.label_to_indices[negative_pid])

        negative_img, _, _, negative_is_wm = self.base_dataset[negative_index]


        return (anchor_img, positive_img, negative_img), (anchor_is_wm, positive_is_wm, negative_is_wm)





