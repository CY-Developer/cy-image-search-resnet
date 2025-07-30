import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import timm
from torchvision import transforms

class MobileViTFeatureExtractor:
    def __init__(self, device="cpu", category: str = "default"):
        self.device = torch.device(device)
        self.model = timm.create_model("mobilevit_s", pretrained=True, features_only=True)
        self.model.eval().to(self.device)
        self.category = category
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, img: Image.Image) -> np.ndarray:
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model(x)
        pooled = [F.adaptive_avg_pool2d(f, (1, 1)).squeeze() for f in feats[-3:]]
        feat = torch.cat(pooled, dim=-1)
        arr = feat.cpu().numpy()

        # 对特定类别（鞋子、包等）加强权重融合
        if self.category == "shoe":
            arr *= 1.1  # 增强鞋子的特征
        elif self.category == "bag":
            arr *= 1.05  # 增强包包的特征
        elif self.category == "watch":
            arr *= 1.2  # 增强手表的特征
        elif self.category == "jewelry":
            arr *= 1.15  # 增强珠宝的特征

        return arr / (np.linalg.norm(arr) + 1e-8)  # L2 归一化
