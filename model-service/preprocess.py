"""
preprocess.py
~~~~~~~~~~~~~~

This module implements a handful of preprocessing functions to prepare
product images for embedding.  Preprocessing serves two purposes:

* **Noise reduction:**  detect and mask out people in the scene.  In many
  e‑commerce photos a model may be present and can dominate the
  representation if left unchecked.  We employ a pre‑trained Faster
  R‑CNN model from `torchvision` to locate person instances and then
  whiten those regions.

* **Category specific cropping:**  a simple cropping strategy based on
  the category name helps the network focus on the relevant portion of
  the image.  For example shoes often occupy the bottom of the frame,
  whereas watches are centred.

If the detection model fails to load (for example due to missing
weights) the service will gracefully degrade and skip person removal.
"""

from __future__ import annotations

from typing import Tuple, List, Optional
import warnings

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

# Import the global configuration so that person masking can be toggled at
# runtime.  This avoids unnecessarily loading the heavyweight detector
# when person suppression is disabled via ``Config.PERSON_MASK_ENABLED``.
from  config import Config


class Preprocessor:
    """Wraps a person detector and category based cropping logic.

    Parameters
    ----------
    device: str
        The device on which the detector should run.  If a GPU is
        available you should supply ``'cuda'`` to accelerate detection.

    person_score_thresh: float
        Confidence threshold for retaining person detections.  Only
        detections with a score greater than or equal to this threshold
        will be used to mask the image.
    """

    def __init__(self, device: str = "cpu", person_score_thresh: float = 0.7) -> None:
        self.device = device
        self.person_score_thresh = person_score_thresh
        # Decide up front whether person masking is enabled.  If it is
        # disabled, avoid loading the expensive detector to save memory and
        # startup time.  The detector will remain ``None`` and calls to
        # ``remove_person`` will simply return the original image.
        if not Config.PERSON_MASK_ENABLED:
            self.detector = None
            self.preprocess_transform = None
            self.categories = []
            return
        # Attempt to load the detection model.  If it fails we set
        # ``self.detector`` to ``None`` so that calls to ``remove_person``
        # simply return the original image.
        try:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.detector = fasterrcnn_resnet50_fpn_v2(weights=weights,
                                                       box_score_thresh=person_score_thresh)
            self.detector.to(device)
            self.detector.eval()
            self.preprocess_transform = weights.transforms()
            # Grab the category names for reference
            self.categories = weights.meta.get("categories", [])
        except Exception as e:
            warnings.warn(f"Failed to load person detection model: {e}.  Person suppression disabled.")
            self.detector = None
            self.preprocess_transform = None
            self.categories = []

    def remove_person(self, image: Image.Image) -> Image.Image:
        """Detect people in the image and paint them white.

        If the detector failed to initialise this simply returns the
        original image.

        Parameters
        ----------
        image: Image.Image
            The input PIL image.

        Returns
        -------
        Image.Image
            A new PIL image with detected people painted white.
        """
        # If person masking is disabled or the detector failed to load,
        # simply return the original image.  See ``Config.PERSON_MASK_ENABLED``.
        if (not Config.PERSON_MASK_ENABLED) or self.detector is None or self.preprocess_transform is None:
            return image
        # Convert image to tensor using the builtin transform from the weights
        img_tensor = self.preprocess_transform(image)
        with torch.no_grad():
            preds = self.detector([img_tensor.to(self.device)])[0]
        boxes = preds['boxes'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        # Build a mask of detected person regions
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for box, label, score in zip(boxes, labels, scores):
            # In COCO dataset the person class has label ID 1
            if label == 1 and score >= self.person_score_thresh:
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)
                mask[y1:y2, x1:x2] = 1
        img_arr = np.array(image).copy()
        img_arr[mask == 1] = 255
        return Image.fromarray(img_arr)

    def crop_center(self, image: Image.Image, ratio: float = 0.8) -> Image.Image:
        """Crop a centred region from the image.

        Parameters
        ----------
        image: Image.Image
            Input image.

        ratio: float, optional
            Fraction of the width and height to retain.  For example a
            ``ratio`` of ``0.8`` keeps 80% of the width and height.  Must
            lie between 0 and 1.

        Returns
        -------
        Image.Image
            The cropped image.
        """
        w, h = image.size
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h
        return image.crop((left, top, right, bottom))

    def crop_by_category(self, image: Image.Image, category: str) -> Image.Image:
        """Apply a category specific cropping heuristic.

        The heuristics here are intentionally simple and easy to tweak.  They
        were derived empirically for common e‑commerce categories.  Feel free
        to extend or replace them with more sophisticated object detectors.

        Parameters
        ----------
        image: Image.Image
            The input image.

        category: str
            The category name.  Matching is case insensitive and only
            checks whether certain keywords appear in the category string.

        Returns
        -------
        Image.Image
            The cropped image.
        """
        category_lower = category.lower() if category else ""
        w, h = image.size
        if "shoe" in category_lower:
            # Keep the bottom 80% for shoes
            y_start = int(h * 0.2)
            return image.crop((0, y_start, w, h))
        elif "bag" in category_lower:
            # Bags are often centred; return full image for now
            return image
        elif "watch" in category_lower:
            # Zoom into the centre for watches
            return self.crop_center(image, ratio=0.5)
        elif any(key in category_lower for key in ["jewelry", "bracelet", "ring"]):
            # Jewellery is small; zoom a bit less aggressively
            return self.crop_center(image, ratio=0.6)
        else:
            # Default behaviour retains 80% of the image centre
            return self.crop_center(image, ratio=0.8)