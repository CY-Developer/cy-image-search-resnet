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

from typing import Dict



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
    # Internal helpers for category‑focused cropping
    # ------------------------------------------------------------------
    def _apply_focus_crop(self, image: Image.Image, category: str) -> Image.Image:
        """Apply a Grad‑CAM based crop to isolate the relevant object.

        When ``config.CATEGORY_FOCUSED_CROP`` is enabled and the supplied
        ``category`` matches an entry in ``config.FOCUS_ENABLED_CLASSES``
        the model is used to generate a heatmap highlighting the most
        discriminative region of the image.  Pixels outside of the
        resulting bounding box are painted white to reduce noise.  If
        anything goes wrong or the category does not match then the
        original image is returned unchanged.

        Parameters
        ----------
        image: Image.Image
            The original PIL image.

        category: str
            Category string provided by the caller (e.g. "watch").

        Returns
        -------
        Image.Image
            Image with only the relevant region retained and the rest
            whitened.  When disabled or unmatched, the input image is
            returned.
        """
        # Bail early if the feature is disabled
        if not self.config.CATEGORY_FOCUSED_CROP:
            return image
        # Normalise category string
        cat = category.lower() if category else ""
        # Determine whether this category should use focus cropping
        enabled = False
        for key, flag in getattr(self.config, "FOCUS_ENABLED_CLASSES", {}).items():
            if flag and key in cat:
                enabled = True
                break
        if not enabled:
            return image
        try:
            # Build a transform for Grad‑CAM with a larger size to retain detail
            focus_size = getattr(self.config, "FOCUS_IMG_SIZE", 448)
            focus_transform = transforms.Compose([
                transforms.Resize((focus_size, focus_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            # Prepare the image tensor and dummy mask for the model
            rgb = image.convert("RGB")
            img_tensor = focus_transform(rgb).to(self.device)  # (3, H, W)
            # Zero mask channel
            mask_tensor = torch.zeros((1, focus_size, focus_size), dtype=torch.float32, device=self.device)
            inp = torch.cat([img_tensor, mask_tensor], dim=0).unsqueeze(0)
            # Hook to capture the activations of the last convolutional block
            features: list = []
            def forward_hook(module, input, output):
                # Ensure gradients are available on the output for CAM
                output.retain_grad()
                features.append(output)
            # Access layer4 from the ResNet backbone
            try:
                last_conv = self.model.feature_extractor[7]
            except Exception:
                last_conv = None
            if last_conv is None:
                return image
            handle = last_conv.register_forward_hook(forward_hook)
            # Forward pass
            inp.requires_grad_(True)
            embedding, _ = self.model(inp)
            # Use the L2 norm of the embedding as the optimisation target
            loss = embedding.norm(p=2)
            self.model.zero_grad()
            loss.backward()
            handle.remove()
            # If the hook did not fire, skip cropping
            if not features:
                return image
            # Extract activations and their gradients
            act = features[0][0].detach().cpu().numpy()  # (C, H, W)
            grad = features[0].grad[0].detach().cpu().numpy()  # (C, H, W)
            # Compute channel weights as global average of gradients
            weights = grad.mean(axis=(1, 2))  # (C,)
            # Compute the raw heatmap
            cam = np.maximum((act * weights[:, None, None]).sum(axis=0), 0)
            if cam.max() > 0:
                cam = cam / (cam.max() + 1e-6)
            # Resize the heatmap back to the original image resolution
            cam_img = Image.fromarray((cam * 255).astype(np.uint8))
            cam_resized = cam_img.resize((image.width, image.height), resample=Image.BILINEAR)
            cam_arr = np.array(cam_resized, dtype=np.float32) / 255.0
            # Determine a threshold keeping only the top percentage of activations
            top_pct = getattr(self.config, "FOCUS_TOP_PERCENT", 0.15)
            flat = cam_arr.flatten()
            if top_pct <= 0.0 or top_pct >= 1.0:
                thr = 0.0
            else:
                thr = np.quantile(flat, 1.0 - top_pct)
            mask = cam_arr >= thr
            # If no activations exceed the threshold, skip cropping
            if not mask.any():
                return image
            # Compute the bounding box of the selected region
            ys, xs = np.where(mask)
            y1 = int(ys.min())
            y2 = int(ys.max()) + 1
            x1 = int(xs.min())
            x2 = int(xs.max()) + 1
            # Expand the box by a small ratio
            pad_ratio = getattr(self.config, "FOCUS_PAD_RATIO", 0.05)
            pad_x = int((x2 - x1) * pad_ratio)
            pad_y = int((y2 - y1) * pad_ratio)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(image.width, x2 + pad_x)
            y2 = min(image.height, y2 + pad_y)
            # Build a mask to retain only the region inside the bounding box
            img_arr = np.array(rgb).copy()
            keep_mask = np.zeros((image.height, image.width), dtype=bool)
            keep_mask[y1:y2, x1:x2] = True
            img_arr[~keep_mask] = 255
            return Image.fromarray(img_arr)
        except Exception:
            # Fail gracefully by returning the original image
            return image

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
        # ------------------------------------------------------------------
        # Apply a series of preprocessing steps in the following order:
        #   1) Category‑focused cropping via Grad‑CAM (whitens background)
        #   2) Person suppression (whitens people) if enabled
        #   3) Heuristic category cropping (e.g. zoom into shoes)
        # These operations operate on PIL images and return a new PIL image.

        # Step 1: Use the model to isolate the relevant object when enabled
        img = self._apply_focus_crop(image, category)
        # Step 2: Mask out persons if person suppression is enabled.  The
        # ``remove_person`` method will return the original image when
        # ``Config.PERSON_MASK_ENABLED`` is False.
        img = self.preprocessor.remove_person(img)
        # Step 3: Apply category specific cropping (heuristic) if requested
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