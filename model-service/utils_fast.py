import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# 加载目标检测模型（Faster R-CNN）
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def detect_objects_in_image(image: Image.Image):
    """
    检测图片中的物体，返回每个目标的box信息和置信分数
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor)
    detected_items = []
    for element in range(len(prediction[0]['boxes'])):
        box = prediction[0]['boxes'][element].cpu().numpy()
        score = float(prediction[0]['scores'][element].cpu().numpy())
        if score > 0.5:  # 分数阈值可调
            detected_items.append({
                'box': box,
                'score': score
            })
    return detected_items

def crop_image_from_detection(image: Image.Image, item: dict) -> Image.Image:
    """
    根据目标检测的box裁剪图片区域，异常时返回原图
    """
    try:
        x_min, y_min, x_max, y_max = item['box']
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        # 边界保护
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(w, int(x_max))
        y_max = min(h, int(y_max))
        if x_max - x_min < 10 or y_max - y_min < 10:
            return image  # 区域太小直接返回全图
        cropped_img = img_array[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0:
            return image
        return Image.fromarray(cropped_img)
    except Exception:
        return image  # 任何异常都 fallback

def fuse_item_vectors(item_vectors: list) -> np.ndarray:
    """
    对多组向量加权融合，归一化
    """
    mat = np.stack(item_vectors)
    sim = mat @ mat.T
    weights = np.exp(np.clip(sim.mean(1), 1e-5, None))
    weights /= weights.sum()
    fused = np.sum([w * v for w, v in zip(weights, item_vectors)], axis=0)
    fused /= np.linalg.norm(fused) + 1e-8
    return fused

def multi_scale_preprocess(image_bytes: bytes, category: str = "default") -> Image.Image:
    """
    多尺度预处理，支持类别增强+前景裁剪。解码失败/异常直接抛出。
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image.")

    # 自动兼容大写小写
    cat = category.lower()
    if cat in ["Shoes", "Shoes"]:
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    elif cat == "Bag":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif cat == "Watches":
        blur = cv2.GaussianBlur(img, (0,0), 10)
        img = cv2.addWeighted(img, 2.5, blur, -1.5, 128)
    elif cat == "Jewelry":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 灰度+自适应阈值提取主体
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        # 防止全黑/全白或异常小区域
        if w > 10 and h > 10:
            crop = img[y:y+h, x:x+w]
            if crop.size > 0:
                return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
