import cv2
import numpy as np
from PIL import Image

def preprocess_light(image_bytes: bytes, category: str = "default") -> Image.Image:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    orig = img.copy()  # 保留原图用于裁剪

    # 选择不同类目增强策略
    if category == "shoe":
        # 增强鞋子图像的颜色和轮廓
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        mask = cv2.inRange(s, 20, 255)  # 保留高饱和度区域
    elif category == "bag":
        # 包的图像在光泽度上的要求
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        mask = cv2.inRange(s, 0, 255)  # 保留亮度较强的部分
    elif category == "watch":
        # 手表类目，突出反射、金属质感
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    elif category == "jewelry":
        # 珠宝类目，突出金属光泽与透明度
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        mask = cv2.inRange(v, 0, 255)  # 保留高光区域
    else:
        # 默认处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 查找最大外接轮廓（排除背景）
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        crop = orig[y:y+h, x:x+w]
    else:
        crop = orig

    # 返回 RGB 图像用于后续模型处理
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
