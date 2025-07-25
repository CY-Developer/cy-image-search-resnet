# app.py
import os
import uuid
import pickle
import base64
import numpy as np
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from PIL import Image
from model import MobileViTFeatureExtractor
from utils import (
    preprocess_image_v2,
    detect_and_mask_watermark,
    crop_using_sift,
    refine_roi_with_depth,
    adaptive_weighting,
)

import redis
import concurrent.futures

app = FastAPI(title="Image Search Service")

# API Key (Java 客户端需设置 X-API-Key)
API_KEY = os.getenv("API_KEY", "93c1240be757f04a38c2aeb7e5cd7178")

# Redis 用于存储任务和结果
redis_client = redis.Redis(host=os.getenv("REDIS_HOST","redis"),
                           port=int(os.getenv("REDIS_PORT","6379")),
                           db=0)

extractor = MobileViTFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu")

# Pydantic 模型
class BatchRequestItem(BaseModel):
    product_id: str
    main_image: str
    additional_images: List[str] = []
    detail_images: List[str] = []

# 全局异常捕获与 API Key 中间件
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path.startswith(("/extract", "/extract-batch", "/batch-result")):
        key = request.headers.get("X-API-Key")
        if key != API_KEY:
            return JSONResponse(status_code=403, content={"error":"Invalid API Key"})
    return await call_next(request)

@app.post("/extract")
async def extract_vector(file: bytes):
    """同步单图向量化"""
    try:
        img = preprocess_image_v2(file)
        vec = extractor.extract(img)
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")
    return {"vector": vec.tolist()}

@app.post("/extract-batch")
async def extract_batch(
    items: List[BatchRequestItem],
    bg: BackgroundTasks
):
    """
    异步批量处理：
    1) 立即返回 task_id
    2) 后台通过 ThreadPool 并行处理存量数据
    """
    task_id = str(uuid.uuid4())
    # 序列化请求参数，1h 过期
    redis_client.setex(f"task:{task_id}:req", 3600, pickle.dumps(items))
    bg.add_task(process_batch, task_id)
    return {"task_id": task_id}

@app.get("/batch-result")
async def batch_result(task_id: str):
    """
    获取批量处理结果。
    返回 {"recognized": [...], "failed": [...]}
    """
    data = redis_client.get(f"task:{task_id}:res")
    if not data:
        raise HTTPException(404, "Result not ready")
    return pickle.loads(data)

def process_batch(task_id: str):
    """后台任务函数：并行处理，结果写回 Redis"""
    raw = redis_client.get(f"task:{task_id}:req")
    if not raw:
        return
    items = pickle.loads(raw)
    results = {"recognized": [], "failed": []}

    def _process_item(it: BatchRequestItem):
        # 主图
        try:
            main_bytes = base64.b64decode(it.main_image)
            main_img = preprocess_image_v2(main_bytes)
            main_vec = extractor.extract(main_img)
        except:
            return ("fail", it.product_id)
        vecs = [main_vec]
        for b64 in it.additional_images + it.detail_images:
            try:
                tb = base64.b64decode(b64)
                clean = detect_and_mask_watermark(tb)
                roi = crop_using_sift(main_img, clean)
                roi = refine_roi_with_depth(main_img, roi)
                vecs.append(extractor.extract(roi))
            except:
                continue
        if len(vecs) < 2:
            return ("fail", it.product_id)
        weights = adaptive_weighting(vecs)
        fused = np.sum([w*v for w,v in zip(weights, vecs)],axis=0)
        if np.linalg.norm(fused)>0:
            fused = fused/np.linalg.norm(fused)
        return ("ok", {
            "product_id": it.product_id,
            "vector": fused.tolist(),
            "vectors": [v.tolist() for v in vecs]
        })

    with concurrent.futures.ThreadPoolExecutor() as pool:
        for status, payload in pool.map(_process_item, items):
            if status == "ok":
                results["recognized"].append(payload)
            else:
                results["failed"].append(payload)

    # 写结果，1h 过期
    redis_client.setex(f"task:{task_id}:res", 3600, pickle.dumps(results))
    # 清理请求缓存
    redis_client.delete(f"task:{task_id}:req")
