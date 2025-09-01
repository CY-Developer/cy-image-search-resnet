"""
server.py
~~~~~~~~~

使用 FastAPI 构建向量化服务接口。提供以下端点：

* `GET /ping`：心跳检测，返回服务运行状态。
* `POST /embed`：上传单张图片，返回其嵌入向量。
* `POST /add`：上传图片并插入向量库，返回插入后的向量 ID。
* `POST /search`：上传图片并在向量库中检索相似项，返回 id-score 列表。

注意：此服务示例未包含鉴权与限流，请根据实际部署需求补充。
"""

import io
from typing import List, Tuple

from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image

from model_service import ImageEmbeddingService
from config import Config


app = FastAPI(title="Image Vectorization Service", version="2.0")

# 初始化嵌入服务实例
embedding_service = ImageEmbeddingService(Config())


class SearchResponse(BaseModel):
    results: List[Tuple[int, float]]


@app.get("/ping")
def ping() -> dict:
    """健康检查接口。"""
    return {"status": "ok"}


@app.post("/embed")
async def embed_image(file: UploadFile = File(...), category: str = Form("")) -> dict:
    """向量化单张图片。返回嵌入列表。"""
    content = await file.read()
    with Image.open(io.BytesIO(content)) as img:
        img = img.convert("RGB")
        embedding = embedding_service.embed_image(img, category)
        return {"embedding": embedding.tolist()}


@app.post("/add")
async def add_image(file: UploadFile = File(...), category: str = Form("")) -> dict:
    """向量化图片并插入 Milvus，返回向量 ID。"""
    content = await file.read()
    with Image.open(io.BytesIO(content)) as img:
        img = img.convert("RGB")
        embedding = embedding_service.embed_image(img, category)
        vid = embedding_service.insert_embedding(embedding)
        return {"id": vid}


@app.post("/search")
async def search_image(file: UploadFile = File(...), category: str = Form(""), top_k: int = Form(10)) -> SearchResponse:
    """向量化图片并在 Milvus 中查询相似向量。返回 id-score 列表。"""
    content = await file.read()
    with Image.open(io.BytesIO(content)) as img:
        img = img.convert("RGB")
        embedding = embedding_service.embed_image(img, category)
        matches = embedding_service.search_embedding(embedding, top_k=top_k)
        return SearchResponse(results=matches)