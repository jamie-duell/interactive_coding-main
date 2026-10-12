"""Official CIFAR-10 test split, in its original order and planar RGB layout."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import get_settings

CLASS_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
N_CLASSES = 10
N_TEST = 10_000
IMAGE_SHAPE = (3, 32, 32)


@lru_cache
def hidden_test_set() -> tuple[np.ndarray, np.ndarray]:
    archive = get_settings().cifar10_archive
    if not archive.is_file():
        raise RuntimeError(
            "CIFAR-10 test data is missing. Run python -m app.scripts.build_cifar10 "
            "or rebuild the Docker app image."
        )
    with np.load(archive, allow_pickle=False) as z:
        X, y = z["X_test"], z["y_test"]
    if X.shape != (N_TEST, 3072) or X.dtype != np.uint8:
        raise ValueError("CIFAR-10 test pixels must be a (10000, 3072) uint8 array.")
    if y.shape != (N_TEST,) or np.bincount(y, minlength=N_CLASSES).tolist() != [1000] * 10:
        raise ValueError("CIFAR-10 test labels must contain 1000 examples of each class.")
    return X, y.astype(np.int64)
