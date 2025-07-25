# utils.py
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# 预加载 TorchScript MiDaS，减少冷启动
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_midas = torch.jit.load("midas_v3_small.pt", map_location=device).eval()
_tfm = torch.hub.load("intel-isl/MiDaS","transforms").default_transform

def preprocess_image_v2(image_bytes: bytes) -> Image.Image:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h,w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    highlight = (gray > 245)
    white_mask = (v>200)&(s<30)&(~highlight)
    if white_mask.mean()>0.7:
        mask = np.zeros((h,w), np.uint8)
        rect = (int(w*0.05),int(h*0.05),int(w*0.9),int(h*0.9))
        bgd,fgd = np.zeros((1,65),np.float64),np.zeros((1,65),np.float64)
        cv2.grabCut(img,mask,rect,bgd,fgd,5,cv2.GC_INIT_WITH_RECT)
        m2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
        img = img*m2[:,:,None]
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,11,2)
    cnts,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x,y,wi,hi = cv2.boundingRect(max(cnts,key=cv2.contourArea))
        crop = img[y:y+hi, x:x+wi]
        return Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def detect_and_mask_watermark(image_bytes: bytes) -> Image.Image:
    arr = cv2.imdecode(np.frombuffer(image_bytes,np.uint8),cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((5,5),np.uint8),1)
    try:
        res = cv2.inpaint(arr,mask,3,cv2.INPAINT_TELEA)
    except:
        res = arr
    return Image.fromarray(cv2.cvtColor(res,cv2.COLOR_BGR2RGB))

def apply_fallback_method(target: Image.Image) -> Image.Image:
    arr = cv2.cvtColor(np.array(target),cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(arr,50,150)
    cnts,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts,key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        crop = np.array(target)[y:y+h, x:x+w]
        if crop.size:
            return Image.fromarray(crop)
    return target

def crop_using_sift(template: Image.Image, target: Image.Image) -> Image.Image:
    # 下采样大图以加速
    if template.width*template.height > 1_000_000:
        template = template.resize((800,800))
        target   = target.resize((800,800))
    tpl = cv2.cvtColor(np.array(template),cv2.COLOR_RGB2GRAY)
    tgt = cv2.cvtColor(np.array(target),cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp1,des1 = sift.detectAndCompute(tpl,None)
    kp2,des2 = sift.detectAndCompute(tgt,None)
    if des1 is None or des2 is None:
        return apply_fallback_method(target)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = [m for m,n in matches if m.distance<0.75*n.distance]
    if len(good)<4:
        return apply_fallback_method(target)
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H,_ = cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    h,w = tpl.shape
    pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(pts,H)
    xs,ys = warped[:,0,0], warped[:,0,1]
    x1,y1,x2,y2 = map(int,(xs.min(),ys.min(),xs.max(),ys.max()))
    arr = np.array(target)
    crop = arr[max(0,y1):min(arr.shape[0],y2),
               max(0,x1):min(arr.shape[1],x2)]
    if crop.size:
        return Image.fromarray(crop)
    return target

def refine_roi_with_depth(template: Image.Image, roi: Image.Image) -> Image.Image:
    # 多尺度深度（复用 _midas, _tfm）
    global _midas,_tfm
    scales = [0.5,1.0,1.5]
    maps=[]
    for s in scales:
        sz=(int(roi.height*s),int(roi.width*s))
        tmp=roi.resize(sz,Image.BICUBIC)
        inp=_tfm(tmp).to(device).unsqueeze(0)
        with torch.no_grad():
            d=_midas(inp).squeeze().cpu().numpy()
        maps.append(cv2.resize(d,(roi.width,roi.height)))
    depth=np.mean(np.stack(maps),axis=0)
    m=(depth>depth.mean()).astype(np.uint8)
    mask=np.stack([m]*3,axis=2)
    arr=np.array(roi)
    return Image.fromarray(arr*mask)

def adaptive_weighting(vectors: list[np.ndarray]) -> np.ndarray:
    mat=np.stack(vectors)
    sim=mat@mat.T
    scores=np.clip(sim.mean(axis=1),1e-5,None)
    exp=np.exp(scores)
    return exp/exp.sum()
