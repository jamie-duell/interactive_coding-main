"""Prepare the fixed generation benchmark once; commit the assets, do not retrain on startup."""

import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from app.challenges.captum_deletion.reference import MODEL_PATH
from app.challenges.generative_models import reference
from app.scripts.build_pneumoniamnist import MD5, URL


def main():
    reference.HERE.mkdir(exist_ok=True)
    shutil.copyfile(MODEL_PATH, reference.ENCODER_PATH)
    with tempfile.TemporaryDirectory() as directory:
        src = Path(directory) / "pneumoniamnist.npz"
        urllib.request.urlretrieve(URL, src)
        assert hashlib.md5(src.read_bytes()).hexdigest() == MD5
        with np.load(src, allow_pickle=False) as z:
            train = z["train_images"].astype(np.float32) / 255
            test = z["test_images"].astype(np.float32) / 255
    train_features = reference.encode(train)
    mean, scale = train_features.mean(0), np.maximum(train_features.std(0), 1e-3)
    indexes = np.random.default_rng(42).choice(len(test), 256, replace=False)
    features = (reference.encode(test[indexes]) - mean) / scale
    distances = reference.squared_distances(features, features)
    bandwidth = float(np.median(distances[np.triu_indices(256, 1)]))
    np.savez_compressed(
        reference.REFERENCE_PATH,
        reference_features=features,
        feature_mean=mean,
        feature_scale=scale,
        bandwidth_squared=np.array(bandwidth),
    )
    public = Path(__file__).resolve().parents[3] / "examples/data"
    for source, name in (
        (reference.ENCODER_PATH, "generative_encoder_model.npz"),
        (reference.REFERENCE_PATH, "generative_reference_distribution.npz"),
    ):
        shutil.copyfile(source, public / name)
    print("Generation benchmark SHA256:", reference.benchmark_sha256())


if __name__ == "__main__":
    main()
