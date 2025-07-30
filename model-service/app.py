import os
import uuid
import json
import numpy as np
import torch
import redis
import pickle
import logging

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List

from model import MobileViTFeatureExtractor
from utils_fast import multi_scale_preprocess, detect_objects_in_image, crop_image_from_detection, fuse_item_vectors

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY", "93c1240be757f04a38c2aeb7e5cd7178")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

app = FastAPI(title="Image Search Service")
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)
extractor = MobileViTFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu")

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path.startswith(("/extract", "/extract-batch", "/batch-result")):
        if request.headers.get("X-API-Key") != API_KEY:
            return JSONResponse(status_code=403, content={"error": "Invalid API Key"})
    return await call_next(request)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/extract")
async def extract_vector(file: UploadFile = File(...), category: str = Form(...)):
    try:
        # 读取并预处理图片
        data = await file.read()
        img = multi_scale_preprocess(data, category)  # 使用多尺度预处理

        # 目标检测：检测图片中的商品并返回每个商品的区域
        detected_items = detect_objects_in_image(img)

        # 针对每个检测到的商品区域，提取特征
        item_vectors = []
        for item in detected_items:
            item_image = crop_image_from_detection(img, item)  # 从检测到的区域裁剪出商品图像
            vector = extractor.extract(item_image, is_main=False)
            item_vectors.append(vector)

        # 如果没有检测到商品，返回错误
        if len(item_vectors) == 0:
            raise HTTPException(status_code=400, detail="No items detected in the image")

        # 这里可以进行进一步的向量融合，例如对多个商品向量进行加权融合
        final_vector = fuse_item_vectors(item_vectors)

        # 返回最终的向量
        logger.info("Successfully extracted and fused vectors.")
        return {"vector": final_vector.tolist()}

    except Exception as e:
        logger.error(f"Error extracting vector: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-batch")
async def extract_batch(
        product_id: str = Form(...),
        main_image: UploadFile = File(...),
        additional_images: List[UploadFile] = File([]),
        detail_images: List[UploadFile] = File([]),
        category: str = Form(...),
        bg: BackgroundTasks = None
):
    task_id = str(uuid.uuid4())
    payload = {
        "product_id": product_id,
        "main": await main_image.read(),
        "additional": [await f.read() for f in additional_images],
        "detail": [await f.read() for f in detail_images],
        "category": category
    }
    redis_client.setex(f"task:{task_id}:req", 3600, pickle.dumps(payload))
    bg.add_task(process_batch, task_id)
    return {"task_id": task_id}

@app.get("/batch-result")
async def batch_result(task_id: str):
    try:
        data = redis_client.get(f"task:{task_id}:res")
        if not data:
            raise HTTPException(status_code=404, detail="Result not ready")

        decoded = data.decode('utf-8')
        return json.loads(decoded)
    except Exception as e:
        logger.error(f"Error fetching batch result: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching batch result")

def process_batch(task_id: str):
    try:
        raw = redis_client.get(f"task:{task_id}:req")
        if not raw:
            logger.error(f"Task {task_id} not found in Redis")
            return
        payload = pickle.loads(raw)
        results = {"recognized": [], "failed": []}

        # 进行图像处理，提取特征
        main_img = multi_scale_preprocess(payload["main"], category=payload["category"])
        vecs = [extractor.extract(main_img, is_main=True)]  # 主图

        # 处理附图和详情图
        for b in payload["additional"] + payload["detail"]:
            img = multi_scale_preprocess(b, category=payload["category"])
            vecs.append(extractor.extract(img))

        # 检查是否有足够的有效图像
        if len(vecs) < 2:
            raise Exception("Too few valid images")

        # 计算相似度并融合向量
        mat = np.stack(vecs)
        sim = mat @ mat.T
        weights = np.exp(np.clip(sim.mean(1), 1e-5, None))
        weights /= weights.sum()
        fused = np.sum([w * v for w, v in zip(weights, vecs)], axis=0)
        fused /= np.linalg.norm(fused) + 1e-8

        # 将结果存储到Redis
        results["recognized"].append({
            "product_id": payload["product_id"],
            "vector": fused.tolist(),
            "vectors": [v.tolist() for v in vecs]
        })

        # 保存结果
        json_data = json.dumps(results)
        redis_client.setex(f"task:{task_id}:res", 3600, json_data)
        redis_client.delete(f"task:{task_id}:req")
        logger.info(f"Successfully processed task {task_id}")

    except Exception as e:
        logger.error(f"Error processing batch {task_id}: {str(e)}")
