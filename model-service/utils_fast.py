import numpy as np
import cv2
from PIL import Image

def preprocess_light(image_bytes: bytes) -> Image.Image:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(thresh, edges)
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    crop = img[y:y+h, x:x+w]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))