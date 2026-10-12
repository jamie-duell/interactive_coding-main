"""Export with model.py and weights.pt; MedMNIST is needed only during Colab training."""

from pathlib import Path

import numpy as np
import torch
from model import CNN

_ROOT = Path(__file__).resolve().parent
_model = CNN().cpu()
_model.load_state_dict(torch.load(_ROOT / "weights.pt", map_location="cpu", weights_only=True))
_model.eval()


@torch.inference_mode()
def predict(X):
    out = []
    for start in range(0, len(X), 128):
        batch = np.array(X[start : start + 128], dtype=np.float32, copy=True)
        images = torch.from_numpy(batch.reshape(-1, 1, 28, 28)) / 255.0
        images = (images - 0.5) / 0.5
        out.append(torch.softmax(_model(images), dim=1).numpy())
    return np.concatenate(out, axis=0)
