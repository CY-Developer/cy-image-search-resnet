"""
preprocess.py
~~~~~~~~~~~~~~

该模块提供图片预处理功能，在特征提取前尽量过滤掉不相关的噪声，例如模特人像、复杂背景等。
主要包含：
    - 使用 TorchVision 的 Faster R‑CNN 模型检测人类目标并遮挡；
    - 根据商品类别裁剪特定区域（例如鞋子保留底部 2/3）；
    - 简单的中心裁剪函数。

实际业务中可以根据需求替换为更强的实例分割模型，如 Mask R‑CNN 或自定义的人体检测模型。
官方文档说明预训练检测模型需要输入 list[Tensor[C, H, W]]，并通过 weights.transforms() 进行预处理【89920199025526†L2104-L2112】【89920199025526†L2135-L2143】。
"""

from typing import Tuple, List, Optional
import warnings

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights


class Preprocessor:
    """预处理器：提供多种方法去除噪声并裁剪关注区域。"""

    def __init__(self,
                 device: str = "cpu",
                 person_score_thresh: float = 0.7) -> None:
        self.device = device
        self.person_score_thresh = person_score_thresh
        # 按需加载检测模型，这里使用 Faster R‑CNN
        try:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=person_score_thresh)
            self.detector.to(device)
            self.detector.eval()
            self.preprocess_transform = weights.transforms()
            self.categories = weights.meta.get("categories", [])
        except Exception as e:
            warnings.warn(f"载入人像检测模型失败: {e}, 将禁用人物抑制功能")
            self.detector = None
            self.preprocess_transform = None
            self.categories = []

    def remove_person(self, image: Image.Image) -> Image.Image:
        """检测并遮挡图片中的人物区域。返回处理后的 PIL 图像。

        如果检测模型未成功加载，则直接返回原图。
        """
        if self.detector is None or self.preprocess_transform is None:
            return image
        # 图像转 Tensor
        img_tensor = self.preprocess_transform(image)
        # 预测
        with torch.no_grad():
            preds = self.detector([img_tensor.to(self.device)])[0]
        boxes = preds['boxes'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        # 创建遮罩
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for box, label, score in zip(boxes, labels, scores):
            # label=1 通常表示 person 类别，过滤分数低的检测
            if label == 1 and score >= self.person_score_thresh:
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)
                mask[y1:y2, x1:x2] = 1
        # 用白色覆盖人物区域
        img_arr = np.array(image).copy()
        img_arr[mask == 1] = 255  # 白色
        return Image.fromarray(img_arr)

    def crop_center(self, image: Image.Image, ratio: float = 0.8) -> Image.Image:
        """裁剪图片中心区域，保留主商品。

        Args:
            ratio: 保留面积的比例，0~1。默认 0.8 表示保留 80% 的宽高。
        Returns:
            裁剪后的 PIL 图像。
        """
        w, h = image.size
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h
        return image.crop((left, top, right, bottom))

    def crop_by_category(self, image: Image.Image, category: str) -> Image.Image:
        """根据商品类别裁剪局部区域。

        示例策略：
            - 鞋子（包含 "shoe"）：保留底部 2/3；
            - 包包（包含 "bag"）：返回原图；
            - 手表（包含 "watch"）：放大中心 50% 区域；
            - 珠宝（包含 "jewelry"、"bracelet"、"ring"）：放大中心 60% 区域；
            - 其他类别：中心裁剪 80% 区域。

        Args:
            image: 原始 PIL 图片。
            category: 商品类别名称。
        Returns:
            裁剪后的图片。
        """
        category_lower = category.lower() if category else ""
        w, h = image.size
        if "shoe" in category_lower:
            # 鞋子：重点关注鞋底和主体，裁掉上部 20%
            y_start = int(h * 0.2)
            return image.crop((0, y_start, w, h))
        elif "bag" in category_lower:
            # 包包：保持全图，或可根据 logo 区域裁剪
            return image
        elif "watch" in category_lower:
            # 手表：放大中心 50% 区域
            return self.crop_center(image, ratio=0.5)
        elif any(key in category_lower for key in ["jewelry", "bracelet", "ring"]):
            # 珠宝：放大中心 60% 区域
            return self.crop_center(image, ratio=0.6)
        else:
            # 默认中心裁剪 80%
            return self.crop_center(image, ratio=0.8)