"""
config.py
~~~~~~~~~~

This module defines a configuration class for the vectorisation service.  The
`Config` class exposes a number of parameters that control how the service
behaves at runtime including the path to the trained model weights, Redis
connection details, API key validation and various preprocessing options.

The default values here are reasonable for a local development environment.
They can be overridden via environment variables or by editing this file.
"""

from __future__ import annotations

from typing import Optional, List


class Config:
    """Collects all runtime configuration for the model service.

    Attributes
    ----------
    MODEL_PATH: str
        Path to the `.pth` file containing the trained model weights.  This
        should point to the best checkpoint produced by your training pipeline.

    REDIS_HOST: str
        Hostname for connecting to the Redis instance used for caching
        embeddings and error information.

    REDIS_PORT: int
        Port number for the Redis instance.

    REDIS_DB: int
        Database index on the Redis instance to use.  Using a dedicated DB
        avoids colliding with other applications.

    REDIS_PASSWORD: Optional[str]
        Password for authenticating against Redis if one is required.  Leave
        ``None`` for no password.

    API_KEYS: List[str]
        A list of valid API keys.  If this list is empty or ``None`` then
        API key validation is skipped entirely.  For security in production
        deployments you should specify one or more keys here and require
        callers to present them in the ``X-API-Key`` header or request body.

    IMAGE_SIZE: int
        Side length (in pixels) of images fed into the model.  All uploaded
        images will be resized to ``(IMAGE_SIZE, IMAGE_SIZE)`` prior to
        inference.  This should match the size used during training.

    USE_MASK_GATING: bool
        When ``True`` the service will zero‑out regions of the image marked by
        the watermark mask before passing them to the model.  This reduces
        the influence of watermarks on the learned embedding.  Disable to
        forego this behaviour.

    GLOBAL_WATERMARK_PATH: Optional[str]
        Path to a PNG file containing a global watermark pattern with an
        alpha channel.  When provided the alpha channel is used to create a
        mask which is then applied to all images.  When ``None`` no global
        mask is used.

    ALPHA_THRESHOLD: float
        Threshold for converting the alpha channel in ``GLOBAL_WATERMARK_PATH``
        into a binary mask.  Pixels with ``alpha > ALPHA_THRESHOLD`` are
        considered part of the watermark.

    CROPPING_ENABLED: bool
        When ``True`` the service will attempt to crop images based on the
        reported category (e.g. shoes, bags) before feeding them into the
        network.  Cropping strategies are implemented in ``preprocess.py``.  If
        set to ``False`` the entire image will be used.

    EMBEDDING_DIM: int
        Dimensionality of the output embedding.  This should match the
        ``embedding_dim`` parameter used when training the model.  The
        default of 256 corresponds to the values in the provided training
        scripts.

    ERROR_LIST_KEY: str
        Redis key used to store details about failures that occur during
        preprocessing or inference.  Each entry in this list will be a JSON
        encoded dictionary containing the product ID (if available), the
        offending file path and a textual error message.
    """

    # Path to the model checkpoint.  Replace with your trained weight file.
    MODEL_PATH: str = "model_final.pth"

    # Redis connection settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # API key validation.  Provide one or more keys here to enable
    # authentication.  Leaving this empty will disable authentication.
    API_KEYS: List[str] = ["your-secret-key"]

    # Preprocessing settings
    IMAGE_SIZE: int = 224
    USE_MASK_GATING: bool = True
    GLOBAL_WATERMARK_PATH: Optional[str] = None
    ALPHA_THRESHOLD: float = 0.5
    CROPPING_ENABLED: bool = True

    # Embedding configuration.  Must match the value used during training.
    EMBEDDING_DIM: int = 256

    # Key for storing error information in Redis.  The service will append
    # failure records to this list.  A consuming process can monitor this
    # list for troubleshooting.
    ERROR_LIST_KEY: str = "error_ids"