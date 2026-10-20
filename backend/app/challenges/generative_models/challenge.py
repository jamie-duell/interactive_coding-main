from pathlib import Path

from app.challenges.generative_models import reference
from app.core.challenge import Challenge, MetricSpec, Resource

HERE = Path(__file__).parent

CHALLENGE = Challenge(
    id="generative_models",
    title="Generative Models — VAE and Diffusion",
    short_description=(
        "Train a VAE or diffusion model on PneumoniaMNIST. "
        "Two independent leaderboards, one shared classroom screen."
    ),
    instructions_md=(HERE / "instructions.md").read_text(encoding="utf-8"),
    submission_instructions_md={
        "generated_samples": (HERE / "instructions_generated_samples.md").read_text(
            encoding="utf-8"
        )
    },
    submission_type="generated_samples",
    evaluator="generative_mmd",
    visualization="generative_arena",
    primary_metric=MetricSpec("mmd2", "MMD² ↓", higher_is_better=False, format=".4f"),
    secondary_metrics=[
        MetricSpec("feature_diversity", "diversity", format=".3f"),
        MetricSpec("n_samples", "samples", format="d"),
    ],
    inputs=lambda: {},
    hidden_targets=lambda: {},
    leaderboard_tracks=reference.FAMILIES,
    starter_notebooks=[
        {"filename": "pneumoniamnist_vae_colab.ipynb", "label": "VAE starter notebook"},
        {"filename": "pneumoniamnist_diffusion_colab.ipynb", "label": "Diffusion starter notebook"},
    ],
    resources=[
        Resource(
            "encoder_model.npz",
            "Frozen benchmark feature encoder",
            reference.ENCODER_PATH.read_bytes,
            "application/octet-stream",
        ),
        Resource(
            "reference_distribution.npz",
            "Fixed benchmark feature statistics",
            reference.REFERENCE_PATH.read_bytes,
            "application/octet-stream",
        ),
    ],
    adapter_config={"allowed_submission_types": ["generated_samples"]},
    display_data=lambda: {"n_samples": reference.N_SAMPLES, "families": reference.FAMILIES},
)
