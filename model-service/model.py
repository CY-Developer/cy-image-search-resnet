import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class OpenCLIPFeatureExtractor:
    CATEGORY_WEIGHTS = {
        "Shoes": 1.2,
        "Bag": 1.0,
        "Watches": 1.3,
        "Jewelry": 1.2
    }
    MAIN_IMAGE_WEIGHT = 1.4

    def __init__(self, device="cpu", category="default"):
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.category = category.lower()  # 标准化

    def extract(self, img: Image.Image, is_main: bool = False) -> np.ndarray:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        vector = outputs[0].cpu().numpy()

        # 特殊类目权重加成
        if self.category in self.CATEGORY_WEIGHTS:
            vector *= self.CATEGORY_WEIGHTS[self.category]

        # 主图权重加成
        if is_main:
            vector *= self.MAIN_IMAGE_WEIGHT

        # 归一化
        return vector / (np.linalg.norm(vector) + 1e-8)
