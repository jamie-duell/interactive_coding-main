"""Frozen classifier and deletion benchmark, shared with the notebook via NumPy.

The server evaluates maps with this implementation; it never runs student code.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MODEL_PATH = HERE / "reference_model.npz"
DATA_PATH = HERE / "explanation_data.npz"
N_IMAGES = 64
N_PIXELS = 784
N_STEPS = 28
CLASS_LABELS = ["normal", "pneumonia"]


def model_sha256() -> str:
    return hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()


@lru_cache
def weights() -> dict[str, np.ndarray]:
    with np.load(MODEL_PATH, allow_pickle=False) as archive:
        return {key: archive[key] for key in ("w0", "b0", "w1", "b1", "w2", "b2")}


@lru_cache
def datasets() -> dict[str, np.ndarray]:
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def predict(raw_pixels: np.ndarray, parameters: dict | None = None) -> np.ndarray:
    """Same 784 -> 64 -> 32 -> 2 ReLU network supplied to Captum in the notebook."""
    w = weights() if parameters is None else parameters
    x = (np.asarray(raw_pixels, dtype=np.float32) / 255.0 - 0.5) / 0.5
    x = np.maximum(x @ w["w0"] + w["b0"], 0)
    x = np.maximum(x @ w["w1"] + w["b1"], 0)
    logits = x @ w["w2"] + w["b2"]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def deletion_curves(images: np.ndarray, maps: np.ndarray, forward=predict) -> tuple:
    """Delete 28 ranked pixels per step; retain each image's original predicted class."""
    images = np.asarray(images, dtype=np.float32)
    maps = np.asarray(maps, dtype=np.float64)
    if images.ndim != 2 or images.shape != maps.shape or images.shape[1] != N_PIXELS:
        raise ValueError("Expected matching (N, 784) images and attribution maps.")
    if not np.all(np.isfinite(maps)):
        raise ValueError("Attributions must be finite.")
    targets = forward(images).argmax(axis=1)
    order = np.argsort(-maps, axis=1, kind="stable")
    fractions = np.arange(N_STEPS + 1, dtype=np.float64) / N_STEPS
    curves = np.empty((len(images), N_STEPS + 1), dtype=np.float64)
    deleted = images.copy()
    rows = np.arange(len(images))
    for step in range(N_STEPS + 1):
        if step:
            indexes = order[:, (step - 1) * 28 : step * 28]
            deleted[rows[:, None], indexes] = 0.0  # black raw pixels, or -1 after normalisation
        curves[:, step] = forward(deleted)[rows, targets]
    areas = np.sum((curves[:, :-1] + curves[:, 1:]) * 0.5, axis=1) / N_STEPS
    return fractions, curves, areas, targets
