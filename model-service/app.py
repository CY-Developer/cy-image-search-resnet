"""
FastAPI entry point for the improved vectorisation service.

This application exposes several endpoints that mirror the existing API
surface used by the Java application but with enhanced functionality
under the hood.  Authentication is enforced via an API key passed in
either the request body (for JSON payloads) or the ``X-API-Key``
header.  When no API keys are configured all requests are permitted.

Endpoints
---------

* **GET /ping** – health check; returns ``{"message":"pong"}``.
* **POST /extract** – accept an uploaded image file and optional
  category; returns the embedding and watermark probability.
* **POST /extract-batch** – process multiple uploaded images for a
  single product; returns a task identifier and logs any errors.
* **POST /v1/feature** – accept a Base64 encoded image in the body
  (along with optional category) and return the embedding and
  watermark probability.

The ``/extract`` and ``/extract-batch`` endpoints are intended to
retain compatibility with the existing Java client which uses
multipart form uploads.  The ``/v1/feature`` endpoint offers a more
JSON friendly alternative for clients that prefer Base64 encoded
payloads.
"""

from __future__ import annotations

from typing import List, Optional

import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
import uuid
from pydantic import BaseModel, Field

from  config import Config
from PIL import Image

# ---------------------------------------------------------------------------
# Enhanced vectorisation imports
#
# We load the trained ResNet backbone and linear adapter at module import
# time so that subsequent requests can reuse the same objects.  The
# ``features`` module exposes helper functions to convert a PIL image into
# an aligned 2048‑dimensional vector using the adapter (see features.py).
# The ``cropper`` module exposes a sliding‑window heuristic for locating
# the watch region in noisy detail images.  Both modules honour the
# configuration values defined in config.py.
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from typing import Optional, List, Dict, Any
from PIL import Image
import io, os, json, time
import numpy as np
import redis

from config import Config
from features import init_model, vectorize_pil, cosine_np, IMG_SIZE, CKPT_PATH, ADAPTER_PATH
from cropper import crop_watch_and_vec

app = FastAPI(title="Enterprise Vector Service")

# --- 固定 API-Key（你已写死的话保留这一段） ---
FIXED_API_KEY = "dev-key"
def _validate_api_key(api_key: Optional[str]) -> None:
    if api_key != FIXED_API_KEY:
        raise HTTPException(401, "Invalid apiKey")

config = Config()

# --- Redis：连不上也不崩，统一 JSON ---
REDIS_URL = os.getenv("REDIS_URL", getattr(Config, "REDIS_URL", "redis://cy-redis:6379/0"))
try:
    rdb = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    rdb.ping()
    REDIS_OK = True
    print(f"[REDIS] connected: {REDIS_URL}")
except Exception as e:
    print(f"[REDIS] disabled: {e}")
    rdb, REDIS_OK = None, False

def _setex_json(key: str, ttl: int, obj: Dict[str, Any]) -> bool:
    if rdb is None:
        print(f"[REDIS] skip setex {key} (no redis)")
        return False
    try:
        rdb.setex(key, ttl, json.dumps(obj))
        return True
    except Exception as e:
        print(f"[REDIS] setex failed: {e}")
        return False

# --- 启动即初始化并 warmup，避免首请求慢 ---
@app.on_event("startup")
def _startup():
    init_model(CKPT_PATH, ADAPTER_PATH)
    dummy = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0,0,0))
    _ = vectorize_pil(dummy)
    print("[MODEL] inited & warmed up")
# Initialise model and adapter once.  This is executed on first import.
init_model(config.CKPT_PATH, config.ADAPTER_PATH)

def _dump_vec(v: np.ndarray) -> str:
    import base64
    return base64.b64encode(v.astype("float32").tobytes()).decode("ascii")

def _load_vec(s: str) -> np.ndarray:
    import base64
    return np.frombuffer(base64.b64decode(s.encode("ascii")), dtype="float32")


FIXED_API_KEY = "93c1240be757f04a38c2aeb7e5cd7178"



# ---------------------------------------------------------------------------
# User query endpoint
#
# Accept an uploaded image and return both the aligned vector for the
# entire image and the aligned vector for an automatically cropped
# region (if one can be identified).  This allows the downstream
# search logic to fuse the two similarities as described in the
# project requirements.
@app.post("/vectorize-query")
async def vectorize_query(image: UploadFile = File(...),
                          category: str = Form(""),
                          apiKey: Optional[str] = Header(None, alias="X-API-Key")) -> dict:
    """Vectorise a user uploaded image with and without cropping.

    Returns both the full‑image vector and a crop vector (when
    detected) along with the bounding box.  The crop vector may be
    ``None`` if no suitable region was found.  Callers can combine
    the two similarities with ``score = max(sim_crop, ALPHA_FUSE *
    sim_full)``.
    """
    _validate_api_key(apiKey)
    # Read file into memory
    try:
        data = await image.read()
        from io import BytesIO
        pil = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    # Compute crop and full vectors
    v_crop, box, v_full = crop_watch_and_vec(pil)
    out = {
        "vec_full": _dump_vec(v_full),
        "vec_crop": _dump_vec(v_crop) if v_crop is not None else None,
        "box": box,
        "alpha_fuse": config.ALPHA_FUSE,
        "notes": "score = max(cos(q_crop,v), ALPHA_FUSE * cos(q_full,v))"
    }
    return out


@app.get("/ping")
def ping() -> dict:
    """Simple health check endpoint."""
    return {"message": "pong"}


class FeatureRequest(BaseModel):
    apiKey: str = Field(..., description="API key for authentication")
    imageBase64: str = Field(..., description="Image encoded as a Base64 string")
    category: Optional[str] = Field("", description="Optional category name")


class FeatureResponse(BaseModel):
    code: int
    message: str
    vector: List[float]
    watermark_prob: float


@app.post("/v1/feature", response_model=FeatureResponse)
def v1_feature(req: FeatureRequest) -> FeatureResponse:
    """Generate an embedding from a Base64 encoded image."""
    # Validate API key from request body
    _validate_api_key(req.apiKey)
    # Decode the Base64 string into bytes
    import base64
    try:
        data = base64.b64decode(req.imageBase64.encode("ascii"))
        from io import BytesIO
        pil = Image.open(BytesIO(data)).convert("RGB")  # type: ignore[arg-type]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 image")
    # Compute the aligned 2048‑dimensional vector.  Watermark
    # probability is no longer returned in the enhanced service so
    # populate it with 0.0 for backward compatibility.
    vec = vectorize_pil(pil)
    return FeatureResponse(code=0,
                           message="success",
                           vector=vec.astype(float).tolist(),
                           watermark_prob=0.0)

@app.post("/embed-query")
async def embed_query(
        image: UploadFile = File(...),
        apiKey: Optional[str] = Header(None, alias="X-API-Key")
):
    if apiKey != FIXED_API_KEY:
        return {"status": "fail", "msg":"Invalid apiKey"}
    pil = Image.open(io.BytesIO(await image.read())).convert("RGB")
    # 先裁剪 -> 再向量（crop + full）
    v_crop, box, v_full = crop_watch_and_vec(pil)  # v_crop 可能为 None
    return {
        "status": "ok",
        "vector_full": (v_full.tolist() if v_full is not None else None),
        "vector_crop": (v_crop.tolist() if v_crop is not None else None)
    }

@app.post("/extract")
async def extract(file: UploadFile = File(...),
                  category: str = Form(""),
                  apiKey: Optional[str] = Header(None, alias="X-API-Key"),
                  product_id: Optional[str] = Form(None)) -> dict:
    """Compute an aligned vector for an uploaded file.

    This endpoint retains compatibility with the original API but now
    leverages the new 2048‑feature backbone and adapter.  Only the
    vector is returned; watermark probability is no longer
    applicable.
    """
    _validate_api_key(apiKey)
    # Read the file contents into a BytesIO.  ``UploadFile.read`` returns
    # bytes but must be awaited.  We avoid writing to disk.
    try:
        data = await file.read()
        from io import BytesIO
        pil = Image.open(BytesIO(data)).convert("RGB")  # type: ignore[arg-type]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")
    # Compute the aligned vector using the enhanced backbone and adapter.
    vec = vectorize_pil(pil)
    return {"code": 0, "message": "success", "vector": vec.tolist()}


@app.post("/extract-batch")
async def extract_batch(
        product_id: str = Form(...),
        category: str = Form(""),
        main_image: Optional[UploadFile] = File(None),
        additional_images: Optional[List[UploadFile]] = File(None),
        detail_images: Optional[List[UploadFile]] = File(None),
        apiKey: Optional[str] = Header(None, alias="X-API-Key"),
        return_vec: int = Form(0),
        FUSE_DETAILS: int = Form(0),
):
    t0 = time.time()
    _validate_api_key(apiKey)

    TH_MATCH  = float(getattr(Config, "TH_MATCH", 0.38))
    WEIGHT_LO = float(getattr(Config, "WEIGHT_LOW_DETAIL", 0.60))
    key_res   = f"task:{product_id}:res"     # 按你 Java 消费端的 key 规范

    def read_pil(up: UploadFile) -> Image.Image:
        return Image.open(io.BytesIO(up.file.read())).convert("RGB")

    # 1) 主图/附图 → 先向量化，作为 ref
    ref_vecs : list[np.ndarray] = []
    main_vec : np.ndarray | None = None
    extra_vecs: list[np.ndarray] = []     # 这里先放附图，后面把详情图也追加进来
    meta: list[Dict[str, Any]] = []       # 记录每个 extra 向量的来源/信息

    if main_image:
        try:
            pil = read_pil(main_image)
            main_vec = vectorize_pil(pil)
            ref_vecs.append(main_vec)
        except Exception as e:
            return {"ok": False, "msg": f"main_image failed: {e}"}

    if additional_images:
        for i, up in enumerate(additional_images):
            try:
                pil = read_pil(up)
                v = vectorize_pil(pil)
                extra_vecs.append(v)
                ref_vecs.append(v)
                meta.append({"source": f"additional_{i}"})
            except Exception as e:
                meta.append({"source": f"additional_{i}", "error": str(e)})

    # 2) 详情图：先剪裁，再与 ref 比对，决定是用 crop 还是 full（一定先剪裁，再匹配）
    if detail_images:
        for i, up in enumerate(detail_images):
            try:
                pil = read_pil(up)
                v_crop, box, v_full = crop_watch_and_vec(pil)
                # 计算与 ref 的最大相似度（没有 ref → None）
                max_sim = None
                if ref_vecs and (v_crop is not None):
                    sims = [cosine_np(v_crop, r) for r in ref_vecs]
                    max_sim = float(max(sims)) if sims else None

                if v_crop is None:
                    # 剪裁失败，降权保存原图
                    extra_vecs.append(v_full)
                    meta.append({
                        "source": "detail_full_lowconf",
                        "box": box, "sim": None, "weight": WEIGHT_LO
                    })
                elif (max_sim is None) or (max_sim < TH_MATCH):
                    # 无 ref 或低于阈值 → 仍用原图，降权
                    extra_vecs.append(v_full)
                    meta.append({
                        "source": "detail_full_lowconf",
                        "box": box, "sim": max_sim, "weight": WEIGHT_LO
                    })
                else:
                    # 通过阈值 → 采用裁剪向量
                    extra_vecs.append(v_crop)
                    meta.append({
                        "source": "detail_crop",
                        "box": box, "sim": max_sim, "weight": 1.0
                    })
            except Exception as e:
                meta.append({"source": "detail_error", "error": str(e)})

    # 3)（可选）融合详情图（默认关闭）
    fused_detail: np.ndarray | None = None
    if FUSE_DETAILS and extra_vecs:
        fused_detail = np.mean(np.stack(extra_vecs, axis=0), axis=0)

    # 4) 组织与 Java 消费端完全一致的最小结构并写入 Redis
    res_payload: Dict[str, Any] = {
        "status": "success",
        "result": {
            "recognized": [{
                "product_id": product_id,
                "vector": (main_vec.tolist() if main_vec is not None else None),
                "vectors": [v.tolist() for v in extra_vecs],  # 附图 + 处理后的详情图
            }]
        }
    }
    _setex_json(f"task:{product_id}:res", 3600, res_payload)

    # 5) HTTP 响应：默认不回 vector，返回统计和性能
    resp: Dict[str, Any] = {
        "ok": True,
        "product_id": product_id,
        "category": category,
        "num_ref": len(ref_vecs)
    }
    if return_vec:
        resp["preview"] = {
            "main_dim": (len(main_vec) if main_vec is not None else None),
            "extra_dims": [len(v) for v in extra_vecs],
            "meta": meta
        }
    return resp
