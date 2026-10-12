"""Prepare the official test split without pickle or extracting a tar to disk.

python -m app.scripts.build_cifar10 [--src cifar-10-binary.tar.gz] [--out path.npz]
The downloaded archive is verified against the dataset publisher's checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "challenges/cifar10/cifar10_test.npz"


def build(src: Path, out: Path) -> None:
    with src.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != MD5:
        raise ValueError("CIFAR-10 archive checksum mismatch; use the official binary archive.")
    with tarfile.open(src, "r:gz") as tar:
        member = tar.getmember("cifar-10-batches-bin/test_batch.bin")
        if not member.isfile() or member.size != 10_000 * 3073:
            raise ValueError("Unexpected CIFAR-10 test batch size or type.")
        with tar.extractfile(member) as stream:
            rows = np.frombuffer(stream.read(), dtype=np.uint8).reshape(10_000, 3073)
    if np.bincount(rows[:, 0], minlength=10).tolist() != [1000] * 10:
        raise ValueError("Unexpected CIFAR-10 test class counts.")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement, so interrupted preparation cannot leave a partial data file.
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".npz", delete=False) as temp:
        temporary = Path(temp.name)
        np.savez_compressed(temp, X_test=rows[:, 1:], y_test=rows[:, 0])
    temporary.replace(out)
    print(f"Prepared {out}: 10000 test images, planar RGB, 3072 raw pixels per image.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="Existing official binary tar.gz (optional)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.src:
        build(args.src, args.out)
    else:
        with tempfile.TemporaryDirectory(prefix="cifar10-") as directory:
            archive = Path(directory) / "cifar-10-binary.tar.gz"
            print(f"Downloading {URL} (about 162 MB)...", flush=True)
            with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as dest:
                while block := response.read(1024 * 1024):
                    dest.write(block)
            build(archive, args.out)


if __name__ == "__main__":
    main()
