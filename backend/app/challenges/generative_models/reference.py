"""Fixed domain-feature RBF MMD² benchmark; self-contained for Colab and server use."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ENCODER_PATH = HERE / "encoder_model.npz"
REFERENCE_PATH = HERE / "reference_distribution.npz"
N_SAMPLES = 256
FAMILIES = {"vae": "Variational Autoencoder", "diffusion": "Diffusion Model"}


def benchmark_sha256():
    return hashlib.sha256(ENCODER_PATH.read_bytes() + REFERENCE_PATH.read_bytes()).hexdigest()


@lru_cache
def parameters():
    with np.load(ENCODER_PATH, allow_pickle=False) as z:
        return {key: z[key] for key in ("w0", "b0", "w1", "b1")}


@lru_cache
def distribution():
    with np.load(REFERENCE_PATH, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def encode(images):
    x = np.asarray(images, dtype=np.float32).reshape(-1, 784) * 2 - 1
    p = parameters()
    x = np.maximum(x @ p["w0"] + p["b0"], 0)
    return np.maximum(x @ p["w1"] + p["b1"], 0).astype(np.float64)


def squared_distances(x, y):
    return np.maximum((x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T, 0)


def score(images):
    d = distribution()
    features = (encode(images) - d["feature_mean"]) / d["feature_scale"]
    real = d["reference_features"]
    bandwidth = float(d["bandwidth_squared"])

    def kernel(x, y):
        return np.exp(-squared_distances(x, y) / (2 * bandwidth))

    # Biased V-statistic: include diagonal entries, fixed sample count, non-negative MMD².
    mmd2 = max(
        0.0,
        float(
            kernel(features, features).mean()
            + kernel(real, real).mean()
            - 2 * kernel(features, real).mean()
        ),
    )
    diversity = float(np.sqrt(np.var(features, axis=0).mean()))
    return mmd2, diversity
