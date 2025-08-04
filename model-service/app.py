# app.py
import os
import uuid
import json
import numpy as np
import torch
import redis
import pickle
import logging
import traceback

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List

from model import OpenCLIPFeatureExtractor  # 替换模型类
from utils_fast import multi_scale_preprocess, detect_objects_in_image, crop_image_from_detection, fuse_item_vectors

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY", "93c1240be757f04a38c2aeb7e5cd7178")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

app = FastAPI(title="Image Search Service")
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)
extractor = OpenCLIPFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu")

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
async def extract_vector(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img = multi_scale_preprocess(data, "default")
        detected_items = detect_objects_in_image(img)

        item_vectors = []
        for item in detected_items:
            item_image = crop_image_from_detection(img, item)
            if getattr(item_image, "width", 0) < 10 or getattr(item_image, "height", 0) < 10:
                item_image = img
            vector = extractor.extract(item_image, is_main=False)
            item_vectors.append(vector)

        if len(item_vectors) == 0:
            vector = extractor.extract(img, is_main=False)
            item_vectors.append(vector)

        final_vector = fuse_item_vectors(item_vectors)
        logger.info("Successfully extracted and fused vectors.")
        return {"vector": final_vector.tolist()}
    except Exception as e:
        logger.error(f"Error extracting vector: {str(e)}\n{traceback.format_exc()}")
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
            # 处理不到 req，写个失败的 res，防止查不到
            redis_client.setex(f"task:{task_id}:res", 3600, json.dumps({
                "status": "fail",
                "msg": "task req not found in redis"
            }))
            return

        payload = pickle.loads(raw)
        results = {"recognized": [], "failed": []}

        try:
            # 主流程（图片预处理、特征提取、融合等）
            vecs = []
            weights = []
            main_img = multi_scale_preprocess(payload["main"], category=payload["category"])
            vec_main = extractor.extract(main_img, is_main=True)
            vecs.append(vec_main)
            weights.append(1.45)
            for b in payload["additional"]:
                try:
                    img = multi_scale_preprocess(b, category=payload["category"])
                    v = extractor.extract(img)
                    vecs.append(v)
                    weights.append(1.0)
                except Exception as ex:
                    logger.warning(f"skip one additional image due to error: {ex}")

            for b in payload["detail"]:
                try:
                    img = multi_scale_preprocess(b, category=payload["category"])
                    v = extractor.extract(img)
                    vecs.append(v)
                    weights.append(1.0)
                except Exception as ex:
                    logger.warning(f"skip one detail image due to error: {ex}")

            if len(vecs) < 2:
                raise Exception("Too few valid images")

            # 融合权重并归一化
            mat = np.stack(vecs)
            weight_array = np.array(weights).reshape(-1, 1)
            fused = (mat * weight_array).sum(axis=0)
            fused /= np.linalg.norm(fused) + 1e-8

            results["recognized"].append({
                "product_id": payload["product_id"],
                "vector": fused.tolist(),
                "vectors": [v.tolist() for v in vecs]
            })
            # 正常处理，写入 res
            redis_client.setex(f"task:{task_id}:res", 3600, json.dumps({
                "status": "success",
                "result": results
            }))
            logger.info(f"Successfully processed task {task_id}")

        except Exception as proc_e:
            # 处理出错也写入 res，带失败原因
            logger.error(f"Processing error for task {task_id}: {str(proc_e)}")
            redis_client.setex(f"task:{task_id}:res", 3600, json.dumps({
                "status": "fail",
                "msg": str(proc_e)
            }))

    except Exception as e:
        # 大 try 捕获“拿 req 本身”都出错的情况
        logger.error(f"Fatal error for task {task_id}: {str(e)}")
        redis_client.setex(f"task:{task_id}:res", 3600, json.dumps({
            "status": "fail",
            "msg": f"fatal error: {str(e)}"
        }))

    finally:
        # 无论成败都删掉 req，避免死信队列堆积
        try:
            redis_client.delete(f"task:{task_id}:req")
        except Exception as del_e:
            logger.error(f"Failed to delete req for task {task_id}: {str(del_e)}")

