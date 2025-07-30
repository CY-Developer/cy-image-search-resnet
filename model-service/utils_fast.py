import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# 加载目标检测模型（例如，Faster R-CNN）
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 图像目标检测函数
def detect_objects_in_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)

    detected_items = []
    for element in range(len(prediction[0]['boxes'])):
        box = prediction[0]['boxes'][element].cpu().numpy()
        score = prediction[0]['scores'][element].cpu().numpy()
        if score > 0.5:  # 假设 score > 0.5 为有效检测
            detected_items.append({
                'box': box,
                'score': score
            })

    return detected_items

def crop_image_from_detection(image: Image.Image, item: dict) -> Image.Image:
    x_min, y_min, x_max, y_max = item['box']
    img_array = np.array(image)

    # 裁剪出商品图像区域
    cropped_img = img_array[int(y_min):int(y_max), int(x_min):int(x_max)]

    # 转回PIL Image格式
    return Image.fromarray(cropped_img)

def fuse_item_vectors(item_vectors: list) -> np.ndarray:
    mat = np.stack(item_vectors)
    sim = mat @ mat.T
    weights = np.exp(np.clip(sim.mean(1), 1e-5, None))
    weights /= weights.sum()

    # 加权融合
    fused = np.sum([w * v for w, v in zip(weights, item_vectors)], axis=0)
    fused /= np.linalg.norm(fused) + 1e-8  # L2归一化

    return fused
