"""Prepare the standard 28x28 PneumoniaMNIST test split for offline server evaluation.

python -m app.scripts.build_pneumoniamnist [--src pneumoniamnist.npz] [--out path.npz]
The official MedMNIST archive is verified against its published checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
MD5 = "28209eda62fecd6e6a2d98b1501bb15f"
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent / "challenges/pneumoniamnist/pneumoniamnist_test.npz"
)


def build(src: Path, out: Path) -> None:
    with src.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != MD5:
        raise ValueError(
            "PneumoniaMNIST archive checksum mismatch; use the official 28x28 archive."
        )
    with np.load(src, allow_pickle=False) as z:
        images, labels = z["test_images"], z["test_labels"]
    if images.shape != (624, 28, 28) or images.dtype != np.uint8:
        raise ValueError("Expected 624 grayscale test images of size 28x28, with uint8 pixels.")
    if (
        labels.shape != (624, 1)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.bincount(labels[:, 0], minlength=2).tolist() != [234, 390]
    ):
        raise ValueError("Expected 234 normal and 390 pneumonia test labels.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".npz", delete=False) as temp:
        temporary = Path(temp.name)
        np.savez_compressed(temp, X_test=images.reshape(624, 784), y_test=labels[:, 0])
    temporary.replace(out)
    print(f"Prepared {out}: 624 grayscale test images, 784 raw pixels per image.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="Existing official pneumoniamnist.npz (optional)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.src:
        build(args.src, args.out)
    else:
        with tempfile.TemporaryDirectory(prefix="pneumoniamnist-") as directory:
            archive = Path(directory) / "pneumoniamnist.npz"
            print(f"Downloading {URL}...", flush=True)
            with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as dest:
                while block := response.read(1024 * 1024):
                    dest.write(block)
            build(archive, args.out)


if __name__ == "__main__":
    main()
