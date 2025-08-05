import torch
import random
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class OpenCLIPFeatureExtractor:
    def __init__(self):
        fix_seed(42)
        self.device = "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, image: Image.Image):
        preprocessed = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**preprocessed)
        embedding = outputs[0].cpu().numpy()
        norm = np.linalg.norm(embedding)
        return (embedding / norm).flatten().tolist()
