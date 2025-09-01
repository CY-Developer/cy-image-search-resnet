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
from  service import ImageEmbeddingService


app = FastAPI(title="Improved Vectorisation Service", version="2.0")

# Instantiate global config and service
config = Config()
service = ImageEmbeddingService(config=config)


def _validate_api_key(api_key: Optional[str]) -> None:
    """Validate an API key against the configured whitelist.

    Raises an HTTPException if the key is invalid.
    """
    if config.API_KEYS:
        if not api_key or api_key not in config.API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid apiKey")


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
    try:
        emb, prob = service.embed_from_base64(req.imageBase64, req.category or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    return FeatureResponse(code=0,
                           message="success",
                           vector=emb.astype(float).tolist(),
                           watermark_prob=float(prob))


@app.post("/extract")
async def extract(file: UploadFile = File(...),
                  category: str = Form(""),
                  apiKey: Optional[str] = Header(None, alias="X-API-Key"),
                  product_id: Optional[str] = Form(None)) -> dict:
    """Compute an embedding for an uploaded file.

    Accepts a single file uploaded as multipart/form data.  A category
    can be supplied either via the form body or as a query parameter.
    The API key should be passed in the ``X-API-Key`` header.  Returns
    a JSON object containing the embedding and watermark probability.
    """
    _validate_api_key(apiKey)
    # Read file contents into bytes
    data = await file.read()
    try:
        emb, prob = service._embed_from_bytes(data, category, cache=True, product_id=product_id, file_path=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    return {"code": 0,
            "message": "success",
            "vector": emb.astype(float).tolist(),
            "watermark_prob": float(prob)}


@app.post("/extract-batch")
async def extract_batch(product_id: str = Form(...),
                        category: str = Form(""),
                        main_image: Optional[UploadFile] = File(None),
                        additional_images: Optional[List[UploadFile]] = File(None),
                        detail_images: Optional[List[UploadFile]] = File(None),
                        apiKey: Optional[str] = Header(None, alias="X-API-Key")) -> dict:
    """Process multiple images for a single product.

    This endpoint accepts a product ID, an optional category and three
    groups of images (main, additional and detail).  All provided
    images will be embedded and cached.  A task ID is returned for
    correlation.  Any failures will be logged to Redis under the
    configured error list key.
    """
    _validate_api_key(apiKey)
    # Persist uploaded files temporarily in memory and process them
    file_paths: List[str] = []
    # Collect the bytes for each file and embed directly without writing to disk
    upload_groups = []
    if main_image:
        upload_groups.append(main_image)
    if additional_images:
        upload_groups.extend(additional_images)
    if detail_images:
        upload_groups.extend(detail_images)
    for upload in upload_groups:
        try:
            data = await upload.read()
            service._embed_from_bytes(data, category, cache=True, product_id=product_id, file_path=upload.filename)
        except Exception:
            # Errors are logged inside _embed_from_bytes via _log_error
            continue
    # Return a task id for tracking
    task_id = str(uuid.uuid4())
    return {"code": 0, "message": "success", "task_id": task_id}