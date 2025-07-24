import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import JSONResponse
from PIL import Image
from model import FeatureExtractor
from utils import preprocess_image

app = FastAPI(title="Image Feature Extraction Service")
extractor = FeatureExtractor(device="cpu")  # or "cuda"

@app.post("/extract")  # 单图提取（兼容旧版）
async def extract(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    data = await file.read()
    img = preprocess_image(data)
    vector = extractor.extract(img)
    return JSONResponse(content={"vector": vector.tolist()})

@app.post("/extract-multi")  # 多图提取：接受多张图片，返回多个向量
async def extract_multi(files: list[UploadFile] = File(...)):
    vectors = []
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        data = await file.read()
        img = preprocess_image(data)
        vectors.append(extractor.extract(img).tolist())
    if not vectors:
        raise HTTPException(status_code=400, detail="No valid images provided.")
    return JSONResponse(content={"vectors": vectors})