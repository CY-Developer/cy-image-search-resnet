"""
service.py
~~~~~~~~~~~

This module exposes the ``ImageEmbeddingService`` class responsible for
turning arbitrary images into stable embedding vectors using the
``MultiTaskModel`` defined in :mod:`model`.  It handles all
preprocessing, caching, error logging and model inference.  In
addition to single image inference the service also supports batch
processing of multiple images for a given product.

The methods in this class are designed to be called from FastAPI
handlers defined in :mod:`app`.  They can also be used directly from
other Python code if desired.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import uuid
from typing import List, Optional, Tuple

import numpy as np
import redis
from PIL import Image
import torch
from torchvision import transforms

from  config import Config
from  model import load_model
from  preprocess import Preprocessor


class ImageEmbeddingService:
    """Encapsulates all logic for computing image embeddings.

    Parameters
    ----------
    config: Config
        Configuration object controlling model paths, Redis settings and
        preprocessing behaviour.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # Lazily initialise Redis connection
        self.redis = redis.Redis(host=config.REDIS_HOST,
                                 port=config.REDIS_PORT,
                                 db=config.REDIS_DB,
                                 password=config.REDIS_PASSWORD,
                                 decode_responses=False)
        # Determine compute device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model weights
        self.model = load_model(config.MODEL_PATH,
                                device=self.device,
                                embedding_dim=config.EMBEDDING_DIM,
                                use_mask_gating=config.USE_MASK_GATING)
        # Build preprocessor
        self.preprocessor = Preprocessor(device=self.device)
        # Construct image transforms matching those used during training
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        # Load global watermark mask if provided
        self.global_mask: Optional[torch.Tensor] = None
        if config.GLOBAL_WATERMARK_PATH:
            try:
                wm_img = Image.open(config.GLOBAL_WATERMARK_PATH).convert("RGBA")
                alpha = np.array(wm_img.split()[-1], dtype=np.float32) / 255.0
                mask = (alpha > config.ALPHA_THRESHOLD).astype(np.float32)
                # Resize mask to IMAGE_SIZE and keep single channel
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_resized = mask_img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
                mask_arr = np.array(mask_resized, dtype=np.float32) / 255.0
                # Shape (1, H, W)
                self.global_mask = torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(0)
            except Exception as e:
                # Silently ignore mask loading errors
                print(f"Warning: failed to load global watermark mask: {e}")
                self.global_mask = None

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _hash_bytes(self, data: bytes) -> str:
        """Compute a stable hash of the input bytes for caching.

        Parameters
        ----------
        data: bytes
            Raw image data.

        Returns
        -------
        str
            Hexadecimal SHA256 digest.
        """
        return hashlib.sha256(data).hexdigest()

    def _prepare_image(self, image: Image.Image, category: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply preprocessing and return tensors ready for the model.

        This method performs person removal and category based cropping if
        configured.  It returns a 3‑channel RGB tensor and a single
        channel mask tensor.  If a global watermark mask is configured
        it is returned, otherwise a zero tensor is used.

        Parameters
        ----------
        image: Image.Image
            Input PIL image.

        category: str
            Category name used for category specific cropping.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            A tuple consisting of the RGB image tensor of shape ``(3, H, W)``
            and a mask tensor of shape ``(1, H, W)``.  The mask will be
            all zeros if no global watermark mask is configured.
        """
        # Remove persons if a detector is available
        img = self.preprocessor.remove_person(image) if self.config.CROPPING_ENABLED else image
        # Apply category specific cropping if enabled
        if self.config.CROPPING_ENABLED and category:
            img = self.preprocessor.crop_by_category(img, category)
        # Convert to RGB just in case and apply transforms
        img = img.convert("RGB")
        img_tensor = self.transform(img)
        # Prepare mask tensor
        if self.global_mask is not None:
            mask_tensor = self.global_mask.clone()
        else:
            # zeros of shape (1, H, W)
            mask_tensor = torch.zeros((1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), dtype=torch.float32)
        return img_tensor, mask_tensor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_from_base64(self, image_base64: str, category: str = "") -> Tuple[np.ndarray, float]:
        """Compute an embedding from a Base64 encoded image.

        The service first checks Redis for a cached embedding based on the
        hash of the raw image bytes.  If no cached value exists the image
        is decoded, preprocessed and passed through the model.  The
        resulting embedding and watermark probability are cached and
        returned.

        Parameters
        ----------
        image_base64: str
            The image encoded as a Base64 string.

        category: str, optional
            Optional category label to inform category specific cropping.

        Returns
        -------
        (np.ndarray, float)
            A tuple containing the normalised embedding vector and the
            watermark probability.
        """
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid Base64 input: {e}")
        return self._embed_from_bytes(image_bytes, category, cache=True)

    def embed_from_file(self, file_path: str, category: str = "", product_id: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """Compute an embedding from an image file on disk.

        Parameters
        ----------
        file_path: str
            Path to the image file.

        category: str, optional
            Optional category label.

        product_id: Optional[str]
            When provided this value will be recorded along with any
            processing errors in Redis.

        Returns
        -------
        (np.ndarray, float)
            A tuple containing the embedding and watermark probability.
        """
        try:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            # Log error to Redis and propagate
            self._log_error(product_id, file_path, f"Failed to read file: {e}")
            raise
        return self._embed_from_bytes(image_bytes, category, cache=True, product_id=product_id, file_path=file_path)

    def _embed_from_bytes(self,
                           image_bytes: bytes,
                           category: str,
                           cache: bool = True,
                           product_id: Optional[str] = None,
                           file_path: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """Internal helper to compute an embedding from raw bytes.

        This method encapsulates caching, preprocessing, model inference
        and error handling.  When ``cache`` is ``True`` the computed
        embedding will be stored in Redis under a key derived from the
        hash of ``image_bytes``.

        Parameters
        ----------
        image_bytes: bytes
            The raw image data.

        category: str
            Category name used for cropping.

        cache: bool
            Whether to cache the result in Redis.

        product_id: Optional[str]
            Product ID associated with this image (for error logging).

        file_path: Optional[str]
            Original file path (for error logging).

        Returns
        -------
        (np.ndarray, float)
            Embedding and watermark probability.
        """
        # Compute cache key
        digest = self._hash_bytes(image_bytes)
        redis_key = f"emb:{digest}"
        # Check cache
        if cache:
            cached = self.redis.get(redis_key)
            if cached:
                try:
                    obj = json.loads(cached.decode('utf-8'))
                    emb = np.array(obj['embedding'], dtype=np.float32)
                    prob = float(obj['watermark_prob'])
                    return emb, prob
                except Exception:
                    # Corrupted cache entry; ignore and recompute
                    pass
        # Decode image
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            # Log and propagate
            self._log_error(product_id, file_path or 'base64', f"Failed to decode image: {e}")
            raise ValueError(f"Failed to decode image: {e}")
        # Preprocess to tensors
        try:
            img_tensor, mask_tensor = self._prepare_image(pil_image, category)
        except Exception as e:
            self._log_error(product_id, file_path or 'base64', f"Preprocessing error: {e}")
            raise
        # Combine channels: shape (4, H, W)
        inp = torch.cat([img_tensor, mask_tensor], dim=0).unsqueeze(0).to(self.device)
        # Inference
        try:
            with torch.no_grad():
                embedding, logits = self.model(inp)
            embedding_np = embedding.cpu().squeeze(0).numpy().astype(np.float32)
            # Convert logits to probability via sigmoid
            prob = torch.sigmoid(logits).cpu().item()
        except Exception as e:
            self._log_error(product_id, file_path or 'base64', f"Model inference error: {e}")
            raise
        # Cache result
        if cache:
            try:
                obj = {'embedding': embedding_np.tolist(), 'watermark_prob': prob}
                self.redis.set(redis_key, json.dumps(obj))
            except Exception:
                # Ignore cache write failures
                pass
        return embedding_np, prob

    def embed_batch(self,
                    product_id: str,
                    category: str,
                    file_paths: List[str]) -> str:
        """Process multiple images for a single product.

        The embeddings are cached in Redis and not returned directly.  A
        task identifier is returned which can be used by the caller to
        correlate logs.  Any errors encountered will be appended to the
        error list in Redis with details of the offending file.

        Parameters
        ----------
        product_id: str
            Unique identifier for the product whose images are being
            processed.  This value will be stored alongside error
            records.

        category: str
            Category name used for cropping.

        file_paths: List[str]
            List of absolute or relative paths to the image files.

        Returns
        -------
        str
            A UUID representing this batch operation.  Can be used as a
            trace identifier in logs or downstream processing.
        """
        task_id = str(uuid.uuid4())
        for path in file_paths:
            if not path:
                continue
            try:
                self.embed_from_file(path, category, product_id=product_id)
            except Exception:
                # embed_from_file already logs errors
                continue
        return task_id

    def _log_error(self, product_id: Optional[str], file_path: str, reason: str) -> None:
        """Record an error in Redis for later inspection.

        Each error is appended to a list stored under ``config.ERROR_LIST_KEY``.
        The payload is a JSON object containing the product ID (if
        supplied), the file path and a reason string.
        """
        error_record = {
            'product_id': product_id or '',
            'file_path': file_path,
            'reason': reason
        }
        try:
            self.redis.rpush(self.config.ERROR_LIST_KEY, json.dumps(error_record))
        except Exception:
            # Do not let logging failures interfere with the main flow
            pass