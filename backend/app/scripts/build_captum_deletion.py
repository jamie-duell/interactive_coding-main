"""Generate a frozen reference model once, then commit its weights and public images.

python -m app.scripts.build_captum_deletion [--src official-pneumoniamnist.npz]
This is a maintainer tool, not a Docker startup/build step. Existing sessions depend
on the committed model digest; regenerating it requires a new classroom session.
"""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from app.scripts.build_pneumoniamnist import MD5, URL

HERE = Path(__file__).resolve().parents[1] / "challenges/captum_deletion"
PUBLIC = Path(__file__).resolve().parents[3] / "examples/data"


def build(src, out=HERE, public=PUBLIC):
    with Path(src).open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != MD5:
            raise ValueError("Use the official 28x28 PneumoniaMNIST archive.")
    with np.load(src, allow_pickle=False) as z:
        train = z["train_images"].reshape(-1, 784).astype(np.float32) / 127.5 - 1
        labels = z["train_labels"].reshape(-1)
        val, test = z["val_images"], z["test_images"]
        val_y, test_y = z["val_labels"].reshape(-1), z["test_labels"].reshape(-1)
    classifier = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        random_state=42,
        max_iter=200,
        batch_size=128,
        early_stopping=True,
        n_iter_no_change=15,
    )
    classifier.fit(train, labels)
    val_p = classifier.predict_proba(val.reshape(-1, 784).astype(np.float32) / 127.5 - 1)
    val_acc, val_auc = accuracy_score(val_y, val_p.argmax(1)), roc_auc_score(val_y, val_p[:, 1])
    out, public = Path(out), Path(public)
    out.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    # sklearn's binary final layer is one sigmoid; [0, logit] gives equivalent softmax outputs.
    np.savez_compressed(
        out / "reference_model.npz",
        w0=classifier.coefs_[0].astype(np.float32),
        b0=classifier.intercepts_[0].astype(np.float32),
        w1=classifier.coefs_[1].astype(np.float32),
        b1=classifier.intercepts_[1].astype(np.float32),
        w2=np.column_stack([np.zeros(32), classifier.coefs_[2][:, 0]]).astype(np.float32),
        b2=np.array([0, classifier.intercepts_[2][0]], dtype=np.float32),
        validation_accuracy=np.array(val_acc),
        validation_auc=np.array(val_auc),
    )
    rng = np.random.default_rng(42)

    def sample(images, targets, per_class):
        indexes = np.concatenate(
            [rng.choice(np.flatnonzero(targets == c), per_class, replace=False) for c in (0, 1)]
        )
        rng.shuffle(indexes)
        return images[indexes].reshape(-1, 784)

    np.savez_compressed(
        out / "explanation_data.npz",
        ids=np.arange(64),
        images=sample(test, test_y, 32),
        practice_images=sample(val, val_y, 16),
    )
    for source, name in (
        ("reference_model.npz", "captum_reference_model.npz"),
        ("explanation_data.npz", "captum_explanation_data.npz"),
    ):
        shutil.copyfile(out / source, public / name)
    print(
        f"Frozen MLP: validation accuracy {val_acc:.3f}, AUROC {val_auc:.3f}; "
        f"{classifier.n_iter_} epochs."
    )
    print("Model SHA256:", hashlib.sha256((out / "reference_model.npz").read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path)
    args = parser.parse_args()
    if args.src:
        build(args.src)
    else:
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "pneumoniamnist.npz"
            urllib.request.urlretrieve(URL, src)
            build(src)


if __name__ == "__main__":
    main()
