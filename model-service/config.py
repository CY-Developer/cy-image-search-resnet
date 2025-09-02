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
    # ---- defaults for vectorisation pipeline (MUST exist) ----
    def __init__(self):
        pass

    IMG_SIZE = 224

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

    # ------------------------------------------------------------------
    # New configuration for enhanced vectorisation
    # ------------------------------------------------------------------
    # Path to the 2048‑feature backbone checkpoint (.pth).  This file
    # should correspond to the model trained in your offline pipeline
    # (e.g. model_final.pth).  You can override this via the
    # ``CKPT_PATH`` environment variable.
    CKPT_PATH: str = "model_final.pth"

    # Path to the linear adapter (.npz) used to align raw 2048‑dimensional
    # features with the search space.  The adapter file must contain
    # ``W`` (or equivalent), ``mean_q`` and ``mean_g`` arrays.  See
    # ``features._load_adapter`` for details.  Override via
    # ``ADAPTER_PATH`` environment variable.
    ADAPTER_PATH: str = "adapter_linear.npz"

    # Directory containing your Milvus index snapshot.  This is used to
    # locate reference prototype vectors (gf.npy or other files) if
    # required.  Override via the ``INDEX_DIR`` environment variable.
    INDEX_DIR: str = "index_s0_snapshot"

    # Similarity threshold used when deciding whether to accept a
    # cropped detail vector as matching the product.  When the cosine
    # similarity between the crop vector and any of the main/appendix
    # vectors exceeds this value the crop is stored with full weight.
    # Lower values are more permissive and may admit noisy crops.
    TH_MATCH: float = 0.38

    # Weight assigned to detail vectors when they fail the match
    # threshold.  These vectors are still stored to improve recall but
    # contribute less to the overall ranking when scoring user queries.
    WEIGHT_LOW_DETAIL: float = 0.60

    # Weight used to fuse the full and crop query vectors during
    # retrieval.  The Java ``ImageSearchService`` can use this to
    # combine both similarities with ``score = max(sim_crop, ALPHA_FUSE
    # * sim_full)``.  Values less than 1 favour the crop vector.
    ALPHA_FUSE: float = 0.85

    # Redis configuration for storing embeddings.  You can override
    # these via environment variables (e.g. REDIS_URL).
    REDIS_URL = "redis://host.docker.internal:6379/0"
    REDIS_KEY_PREFIX: str = "vecsvc:"

    # Key for storing error information in Redis.  The service will append
    # failure records to this list.  A consuming process can monitor this
    # list for troubleshooting.
    ERROR_LIST_KEY: str = "error_ids"

    # ------------------------------------------------------------------
    # New configuration toggles for advanced preprocessing
    # ------------------------------------------------------------------
    # Enable category‑focused cropping.  When set to True the service will
    # attempt to locate the region of interest for certain categories using
    # Grad‑CAM and remove the rest of the image by painting it white.  This
    # helps the model focus on the relevant object (e.g. a watch) even in
    # cluttered scenes.  See ``service.py`` for implementation details.
    CATEGORY_FOCUSED_CROP: bool = True

    # Specify which high‑level categories should use the focused crop.  The
    # keys should be lower‑case substrings that might appear in the category
    # string supplied by clients.  If the value is True and the substring
    # appears in the category then the cropping heuristic is applied.  If no
    # entry matches or the value is False the image will be processed as
    # normal.  You can enable additional classes once your model is trained
    # to recognise them.
    FOCUS_ENABLED_CLASSES: dict = {
        "watch": True,
        "bag": False,
        "shoe": False,
        "jewelry": False,
    }

    # Image size used when computing the Grad‑CAM heatmap.  This can be
    # larger than ``IMAGE_SIZE`` to preserve more spatial detail.  The
    # default of 448 offers a good compromise between speed and fidelity.
    FOCUS_IMG_SIZE: int = 448

    # Proportion of the heatmap to retain when determining the bounding box.
    # For example, a value of 0.15 keeps the top 15% of activation pixels and
    # discards the rest.  Smaller values yield tighter crops at the risk of
    # missing part of the object, whereas larger values yield looser crops.
    FOCUS_TOP_PERCENT: float = 0.15

    # Amount of padding to add (as a fraction of the bounding box size) when
    # expanding the crop box.  This can help ensure the entire object is
    # captured even if the heatmap is conservative.
    FOCUS_PAD_RATIO: float = 0.05

    # Enable person masking.  When ``True`` the service uses a pre‑trained
    # detector to find people in the image and paints them white before
    # feeding the image into the model.  Set to ``False`` to disable this
    # behaviour entirely.  Disabling person masking is useful when the
    # detector might erroneously mask the target object (e.g. watches on
    # wrists).  The default is ``False`` to favour object preservation.
    PERSON_MASK_ENABLED: bool = False