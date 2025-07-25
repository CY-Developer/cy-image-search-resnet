# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import timm
from torchvision import transforms

class MobileViTFeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        # 使用 timm 的 MobileViT small
        self.model = timm.create_model("mobilevit_small_100", pretrained=True, features_only=True)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def extract(self, img: Image.Image) -> np.ndarray:
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast(), torch.no_grad():
            feats = self.model(x)  # list of feature maps
        # 取最后3层池化后拼接
        pooled = [F.adaptive_avg_pool2d(f, (1,1)).squeeze() for f in feats[-3:]]
        feat = torch.cat(pooled, dim=-1)
        arr = feat.cpu().numpy()
        norm = np.linalg.norm(arr)
        return arr/norm if norm>0 else arr
