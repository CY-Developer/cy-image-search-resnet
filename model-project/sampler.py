"""
Sampler for metric learning: P-K sampler.

The ``PKSampler`` yields indices such that each mini-batch contains exactly
``P`` distinct classes (products) and ``K`` samples per class. This is
useful when training models with triplet or batch-hard losses.
"""

import random
from typing import Iterator, List, Optional, Set, Dict

from torch.utils.data import Sampler


class PKSampler(Sampler[int]):
    """
    P-K sampler with optional preferred class subset.

    Each mini-batch contains exactly P distinct classes and K samples per class. When
    ``prefer_labels`` is provided, a portion of the P classes are drawn from this
    subset to encourage representation of certain cohorts (e.g., products that
    contain both clean and watermarked images). ``prefer_ratio`` controls the
    fraction of classes in each batch that should be sampled from ``prefer_labels``.
    """

    def __init__(
        self,
        labels: List[int],
        P: int,
        K: int,
        shuffle: bool = True,
        prefer_labels: Optional[Set[int]] = None,
        prefer_ratio: float = 0.5,
    ):
        if not isinstance(labels, list) or not all(isinstance(l, int) for l in labels):
            raise ValueError("labels must be a list of integers")
        if P <= 0 or K <= 0:
            raise ValueError("P and K must be positive integers")
        self.labels = labels
        self.P = P
        self.K = K
        self.shuffle = shuffle
        self.prefer_labels = set(prefer_labels) if prefer_labels else set()
        # Clamp prefer_ratio to [0,1]
        self.prefer_ratio = max(0.0, min(1.0, float(prefer_ratio)))
        # Build mapping from label to indices
        self.label_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            self.label_to_indices.setdefault(label, []).append(idx)
        self.classes = list(self.label_to_indices.keys())
        # Separate preferred and other classes
        self.preferred_classes = [c for c in self.classes if c in self.prefer_labels]
        self.other_classes = [c for c in self.classes if c not in self.prefer_labels]

    def __iter__(self) -> Iterator[int]:
        # Copy lists as we may shuffle them
        pref_classes = self.preferred_classes[:]
        other_classes = self.other_classes[:]
        if self.shuffle:
            random.shuffle(pref_classes)
            random.shuffle(other_classes)
        # Determine number of preferred and other classes per batch
        num_pref = int(round(self.P * self.prefer_ratio))
        num_other = max(0, self.P - num_pref)
        pref_idx = 0
        other_idx = 0
        batch: List[int] = []
        while pref_idx < len(pref_classes) or other_idx < len(other_classes):
            # Choose classes for this segment
            chosen_classes: List[int] = []
            # Fill preferred portion
            for _ in range(num_pref):
                if pref_idx < len(pref_classes):
                    chosen_classes.append(pref_classes[pref_idx])
                    pref_idx += 1
            # Fill remaining with other classes
            for _ in range(num_other):
                if other_idx < len(other_classes):
                    chosen_classes.append(other_classes[other_idx])
                    other_idx += 1
            if not chosen_classes:
                break
            # Sample K indices per class
            for cls in chosen_classes:
                indices = self.label_to_indices[cls]
                if len(indices) >= self.K:
                    sampled = random.sample(indices, self.K)
                else:
                    sampled = [random.choice(indices) for _ in range(self.K)]
                batch.extend(sampled)
            # Yield full batches of size P*K
            while len(batch) >= self.P * self.K:
                for idx in batch[: self.P * self.K]:
                    yield idx
                batch = batch[self.P * self.K :]
        # Yield any remaining samples
        for idx in batch:
            yield idx

    def __len__(self) -> int:
        return len(self.labels)