# model.py
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

class FeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        resnet = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def extract(self, img):
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(x).squeeze().cpu().numpy()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat
