"""MedMNIST image classification exercise, using the standard train/val/test splits."""

from pathlib import Path

import numpy as np

from app.challenges.pneumoniamnist import data
from app.core.challenge import Challenge, MetricSpec, Resource
from app.core.submission_adapter import InputDataset

_HERE = Path(__file__).parent


def _inputs() -> dict[str, InputDataset]:
    X, _ = data.hidden_test_set()
    return {"test": InputDataset("test", np.arange(len(X)), X)}


def _hidden_targets() -> dict[str, np.ndarray]:
    return {"test": data.hidden_test_set()[1]}


CHALLENGE = Challenge(
    id="pneumoniamnist",
    title="PneumoniaMNIST — Classify Chest X-rays",
    short_description=(
        "Design and train a CNN in Colab using MedMNIST's 28×28 chest X-rays. "
        "Classify normal versus pneumonia; ranked by accuracy on the 624 official test images."
    ),
    instructions_md=(_HERE / "instructions.md").read_text(encoding="utf-8"),
    submission_instructions_md={
        "python_bundle": (_HERE / "instructions_python_bundle.md").read_text(encoding="utf-8"),
    },
    submission_type="python_bundle",
    evaluator="binary_class_probabilities",
    visualization="accuracy_leaderboard",
    primary_metric=MetricSpec("accuracy", "test accuracy", format="%"),
    secondary_metrics=[
        MetricSpec("auc", "AUROC", format=".3f"),
        MetricSpec("balanced_accuracy", "balanced accuracy", format="%"),
        MetricSpec("log_loss", "log loss", higher_is_better=False),
    ],
    inputs=_inputs,
    hidden_targets=_hidden_targets,
    display_data=lambda: {"class_labels": data.CLASS_LABELS, "n_test": data.N_TEST},
    resources=[
        Resource(
            "submission_template.py",
            "CPU inference and preprocessing used by the notebook",
            lambda: (_HERE / "submission_template.py").read_bytes(),
            content_type="text/x-python",
            submission_types=("python_bundle",),
        ),
    ],
    adapter_config={
        "allowed_submission_types": ["python_bundle"],
        "output": {"kind": "class_probabilities", "n_classes": data.N_CLASSES},
    },
    evaluator_config={"dataset": "test"},
)
