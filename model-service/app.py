    import os
    import uuid
    import json  # 新增json模块
    import numpy as np
    import torch
    import redis
    import pickle

    from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from typing import List

    from model import MobileViTFeatureExtractor
    from utils_fast import preprocess_light

    API_KEY = os.getenv("API_KEY", "93c1240be757f04a38c2aeb7e5cd7178")
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

    app = FastAPI(title="Image Search Service")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)  # 确保二进制安全[8](@ref)
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
    async def extract_vector(file: UploadFile = File(...)):
        try:
            data = await file.read()
            img = preprocess_light(data)
            vec = extractor.extract(img)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"vector": vec.tolist()}  # 保持直接返回列表格式

    @app.post("/extract-batch")
    async def extract_batch(
            product_id: str = Form(...),
            main_image: UploadFile = File(...),
            additional_images: List[UploadFile] = File([]),
            detail_images: List[UploadFile] = File([]),
            bg: BackgroundTasks = None
    ):
        task_id = str(uuid.uuid4())
        payload = {
            "product_id": product_id,
            "main": await main_image.read(),
            "additional": [await f.read() for f in additional_images],
            "detail": [await f.read() for f in detail_images]
        }
        # 保持请求数据使用pickle（含二进制图像）
        redis_client.setex(f"task:{task_id}:req", 3600, pickle.dumps(payload))
        bg.add_task(process_batch, task_id)
        return {"task_id": task_id}

    @app.get("/batch-result")
    async def batch_result(task_id: str):
        data = redis_client.get(f"task:{task_id}:res")
        if not data:
            raise HTTPException(status_code=404, detail="Result not ready")

        # 关键修改：二进制解码+JSON解析[7,8](@ref)
        try:
            decoded = data.decode('utf-8')  # 字节转字符串
            return json.loads(decoded)  # 解析JSON
        except (UnicodeDecodeError, json.JSONDecodeError):
            # 兼容旧版pickle数据（过渡期）[4](@ref)
            try:
                return pickle.loads(data)
            except:
                raise HTTPException(
                    status_code=500,
                    detail="Result data format invalid"
                )

    def process_batch(task_id: str):
        raw = redis_client.get(f"task:{task_id}:req")
        if not raw:
            return
        payload = pickle.loads(raw)  # 请求数据保持pickle解析
        results = {"recognized": [], "failed": []}

        try:
            main_img = preprocess_light(payload["main"])
            vecs = [extractor.extract(main_img)]

            for b in payload["additional"] + payload["detail"]:
                try:
                    img = preprocess_light(b)
                    vecs.append(extractor.extract(img))
                except:
                    continue

            if len(vecs) < 2:
                raise Exception("Too few valid images")

            mat = np.stack(vecs)
            sim = mat @ mat.T
            weights = np.exp(np.clip(sim.mean(1), 1e-5, None))
            weights /= weights.sum()
            fused = np.sum([w * v for w, v in zip(weights, vecs)], axis=0)
            fused /= np.linalg.norm(fused) + 1e-8

            # 关键修改：确保向量存储为JSON兼容格式[6](@ref)
            results["recognized"].append({
                "product_id": payload["product_id"],
                "vector": fused.tolist(),  # numpy数组转Python列表
                "vectors": [v.tolist() for v in vecs]  # 嵌套列表转换
            })
        except Exception as e:
            results["failed"].append(payload["product_id"])
            print(f"Batch processing failed: {str(e)}")

        # 关键修改：结果存储为JSON字符串[8](@ref)
        json_data = json.dumps(results)  # 序列化为JSON字符串
        redis_client.setex(f"task:{task_id}:res", 3600, json_data)  # 直接存储字符串
        redis_client.delete(f"task:{task_id}:req")