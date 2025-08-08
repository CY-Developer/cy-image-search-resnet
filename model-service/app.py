"""
app.py
~~~~~~

FastAPI 应用入口，为外部提供向量化接口。该服务只负责生成图片向量，不直接
与 Milvus 交互，保持与现有 Java→Python→Milvus 流程兼容。使用 `ImageEmbeddingService`
处理图片，并利用 Redis 缓存结果。

提供的主要端点：

* `GET /ping`：健康检查，返回 pong。
* `POST /v1/feature`：输入 Base64 图片、可选类别，返回向量和水印概率。

请求必须包含有效的 `apiKey`，否则返回 401。若配置未设置 API_KEYS 或为空，则跳过验证。
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List

from .config import Config
from .service import ImageEmbeddingService


app = FastAPI(title="Vectorization Model Service", version="2.0")

# 初始化服务实例
config = Config()
service = ImageEmbeddingService(config=config)


class ImageFeatureRequest(BaseModel):
    """请求体定义。"""
    apiKey: str = Field(..., description="API 密钥")
    imageBase64: str = Field(..., description="Base64 编码的图片数据")
    category: Optional[str] = Field("", description="商品类别，可用于裁剪策略")


class ImageFeatureResponse(BaseModel):
    code: int
    message: str
    vector: List[float]
    watermark_prob: float


def check_api_key(api_key: str) -> None:
    """校验 API 密钥。如果配置中没有设置，则允许任意键。"""
    if config.API_KEYS and api_key not in config.API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid apiKey")


@app.get("/ping")
def ping() -> dict:
    """健康检查接口。"""
    return {"message": "pong"}


@app.post("/v1/feature", response_model=ImageFeatureResponse)
def get_feature(req: ImageFeatureRequest) -> ImageFeatureResponse:
    """生成图片向量并返回。"""
    # 校验 API 密钥
    check_api_key(req.apiKey)
    try:
        emb, prob = service.embed_from_base64(req.imageBase64, req.category or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部错误: {e}")
    return ImageFeatureResponse(
        code=0,
        message="success",
        vector=emb.astype(float).tolist(),
        watermark_prob=float(prob)
    )