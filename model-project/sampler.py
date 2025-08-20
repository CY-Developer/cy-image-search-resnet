"""
Sampler for metric learning: P-K sampler.

The ``PKSampler`` yields indices such that each mini-batch contains exactly
``P`` distinct classes (products) and ``K`` samples per class. This is
useful when training models with triplet or batch-hard losses.
"""

import random
from typing import Iterator, List

from torch.utils.data import Sampler


class PKSampler(Sampler[int]):
    def __init__(self, labels: List[int], P: int, K: int, shuffle: bool = True):
        if not isinstance(labels, list) or not all(isinstance(l, int) for l in labels):
            raise ValueError("labels must be a list of integers")
        if P <= 0 or K <= 0:
            raise ValueError("P and K must be positive integers")
        self.labels = labels
        self.P = P
        self.K = K
        self.shuffle = shuffle
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            self.label_to_indices.setdefault(label, []).append(idx)
        self.classes = list(self.label_to_indices.keys())

    def __iter__(self) -> Iterator[int]:
        classes = self.classes[:]
        if self.shuffle:
            random.shuffle(classes)
        batch: List[int] = []
        for cls in classes:
            indices = self.label_to_indices[cls]
            if len(indices) >= self.K:
                sampled = random.sample(indices, self.K)
            else:
                sampled = [random.choice(indices) for _ in range(self.K)]
            batch.extend(sampled)
            if len(batch) == self.P * self.K:
                for idx in batch:
                    yield idx
                batch = []
        # yield remaining
        for idx in batch:
            yield idx

    def __len__(self) -> int:
        return len(self.labels)