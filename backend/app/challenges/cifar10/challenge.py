"""CNN architecture exercise, ranked by accuracy on the official CIFAR-10 test split."""

from pathlib import Path

import numpy as np

from app.challenges.cifar10 import data
from app.core.challenge import Challenge, MetricSpec, Resource
from app.core.submission_adapter import InputDataset

_HERE = Path(__file__).parent


def _inputs() -> dict[str, InputDataset]:
    X, _ = data.hidden_test_set()
    return {"test": InputDataset("test", np.arange(len(X)), X)}


def _hidden_targets() -> dict[str, np.ndarray]:
    return {"test": data.hidden_test_set()[1]}


CHALLENGE = Challenge(
    id="cifar10",
    title="CIFAR-10 — Design a CNN",
    short_description=(
        "Develop your own convolutional network in Google Colab. Train on CIFAR-10 "
        "and compete on test accuracy across 10,000 colour images."
    ),
    instructions_md=(_HERE / "instructions.md").read_text(encoding="utf-8"),
    submission_instructions_md={
        "python_bundle": (_HERE / "instructions_python_bundle.md").read_text(encoding="utf-8"),
    },
    submission_type="python_bundle",
    evaluator="multiclass_classification",
    visualization="accuracy_leaderboard",
    primary_metric=MetricSpec("accuracy", "test accuracy", format="%"),
    secondary_metrics=[MetricSpec("log_loss", "log loss", higher_is_better=False)],
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
