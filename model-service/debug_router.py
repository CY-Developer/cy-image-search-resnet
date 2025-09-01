# -*- coding: utf-8 -*-
# model-service/debug_router.py
from fastapi import APIRouter
import hashlib, os

from config import Config
from service import ImageEmbeddingService

router = APIRouter()
_svc = ImageEmbeddingService(Config())

def _md5_of_file(path: str, chunk: int = 1 << 20) -> str:
    m = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            m.update(b)
    return m.hexdigest()

@router.get("/debug/model")
def debug_model():
    model_path = getattr(Config, "MODEL_PATH", None) or os.getenv("MODEL_PATH")
    info = {
        "model_path": model_path,
        "model_path_exists": bool(model_path and os.path.exists(model_path)),
        "model_md5": _md5_of_file(model_path) if model_path and os.path.exists(model_path) else None,
        "device": getattr(_svc, "device", "cpu"),
        "embedding_dim": getattr(_svc, "embedding_dim", None),
        "has_embed_from_pil": hasattr(_svc, "embed_from_pil"),
    }
    return info
