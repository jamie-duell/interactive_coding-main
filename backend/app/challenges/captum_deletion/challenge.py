"""Compare Captum explanations of one frozen PneumoniaMNIST classifier."""

from pathlib import Path

from app.challenges.captum_deletion import reference
from app.core.challenge import Challenge, MetricSpec, Resource
from app.core.submission_adapter import InputDataset

_HERE = Path(__file__).parent


def _inputs() -> dict[str, InputDataset]:
    data = reference.datasets()
    return {"attributions": InputDataset("attributions", data["ids"], data["images"])}


CHALLENGE = Challenge(
    id="captum_deletion",
    title="Explainable AI — Captum Deletion Challenge",
    short_description=(
        "Find the Captum attribution method and settings that minimise deletion AUC. "
        "Explain a frozen PneumoniaMNIST classifier; lower deletion score wins."
    ),
    instructions_md=(_HERE / "instructions.md").read_text(encoding="utf-8"),
    submission_instructions_md={
        "attribution_npz": (_HERE / "instructions_attribution_npz.md").read_text(encoding="utf-8")
    },
    submission_type="attribution_npz",
    evaluator="deletion_auc",
    visualization="deletion_arena",
    primary_metric=MetricSpec(
        "deletion_auc", "deletion score ↓", higher_is_better=False, format=".4f"
    ),
    secondary_metrics=[
        MetricSpec(
            "confidence_at_25pct",
            "confidence after 25% deletion",
            higher_is_better=False,
            format="%",
        ),
        MetricSpec("n_explanations", "explanations", format="d"),
    ],
    inputs=_inputs,
    hidden_targets=lambda: {},  # deletion measures the original prediction
    display_data=lambda: {
        "n_images": reference.N_IMAGES,
        "class_labels": reference.CLASS_LABELS,
        "image_shape": [28, 28],
        "n_steps": reference.N_STEPS,
    },
    resources=[
        Resource(
            "reference_model.npz",
            "Frozen classifier weights; do not retrain",
            reference.MODEL_PATH.read_bytes,
            content_type="application/octet-stream",
        ),
        Resource(
            "explanation_data.npz",
            "64 scoring images and 32 separate practice images",
            reference.DATA_PATH.read_bytes,
            content_type="application/octet-stream",
        ),
    ],
    adapter_config={"allowed_submission_types": ["attribution_npz"]},
)
