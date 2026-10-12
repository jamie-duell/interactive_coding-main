"""The official 28x28 PneumoniaMNIST test split, preserving its published order."""

from functools import lru_cache

import numpy as np

from app.config import get_settings

CLASS_LABELS = ["normal", "pneumonia"]
N_CLASSES = 2
N_TEST = 624
IMAGE_SHAPE = (1, 28, 28)


@lru_cache
def hidden_test_set() -> tuple[np.ndarray, np.ndarray]:
    archive = get_settings().pneumoniamnist_archive
    if not archive.is_file():
        raise RuntimeError(
            "PneumoniaMNIST test data is missing. Run python -m app.scripts.build_pneumoniamnist "
            "or rebuild the Docker app image."
        )
    with np.load(archive, allow_pickle=False) as z:
        X, y = z["X_test"], z["y_test"]
    if X.shape != (N_TEST, 784) or X.dtype != np.uint8:
        raise ValueError("PneumoniaMNIST test pixels must be a (624, 784) uint8 array.")
    if (
        y.shape != (N_TEST,)
        or not np.issubdtype(y.dtype, np.integer)
        or np.bincount(y, minlength=N_CLASSES).tolist() != [234, 390]
    ):
        raise ValueError(
            "PneumoniaMNIST test labels must contain 234 normal and 390 pneumonia cases."
        )
    return X, y.astype(np.int64)
