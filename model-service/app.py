import os
import uuid
import json
import io
import logging
import traceback
import pickle
import redis

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image

from model import OpenCLIPFeatureExtractor
from utils_fast import multi_scale_preprocess

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 环境变量配置
API_KEY = os.getenv("API_KEY", "93c1240be757f04a38c2aeb7e5cd7178")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

app = FastAPI(title="Image Vectorization Service")
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)
extractor = OpenCLIPFeatureExtractor()  # 使用 CPU，默认加载

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
        img = Image.open(io.BytesIO(data)).convert("RGB")
        preprocessed = multi_scale_preprocess(img)
        vectors = [extractor(im) for im in preprocessed]
        return {
            "status": "success",
            "vector": vectors[0],
            "vectors": vectors
        }
    except Exception as e:
        logger.error(f"Extract error: {e}\n{traceback.format_exc()}")
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
        "detail": [await f.read() for f in detail_images]
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
        return json.loads(data.decode('utf-8'))
    except Exception as e:
        logger.error(f"Batch result error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch result")

def process_batch(task_id: str):
    try:
        raw = redis_client.get(f"task:{task_id}:req")
        if not raw:
            raise Exception("No task data in Redis")
        payload = pickle.loads(raw)
        product_id = payload["product_id"]

        vecs = []
        images = [("main", 0, payload["main"])] + \
                 [("additional", i, b) for i, b in enumerate(payload["additional"])] + \
                 [("detail", i, b) for i, b in enumerate(payload["detail"])]

        for img_type, index, image_bytes in images:
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                preprocessed = multi_scale_preprocess(img)
                for im in preprocessed:
                    vec = extractor(im)
                    vecs.append(vec)
            except Exception as e:
                logger.warning(f"{img_type} image {index} failed: {e}")

        if not vecs:
            raise Exception("No vectors extracted")

        redis_client.setex(f"task:{task_id}:res", 3600, json.dumps({
            "status": "success",
            "result": {
                "recognized": [{
                    "product_id": product_id,
                    "vector": vecs[0],
                    "vectors": vecs
                }]
            }
        }))
        logger.info(f"Task {task_id} completed with {len(vecs)} vectors")

    except Exception as e:
        logger.error(f"Fatal error in batch: {e}\n{traceback.format_exc()}")
        redis_client.setex(f"task:{task_id}:res", 3600, json.dumps({
            "status": "fail",
            "msg": str(e)
        }))
    finally:
        redis_client.delete(f"task:{task_id}:req")
