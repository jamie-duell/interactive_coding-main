"""Evaluate data-only attribution maps against the fixed classroom classifier."""

from app.challenges.captum_deletion import reference
from app.core.evaluator import EvaluationOutput, Evaluator
from app.core.submission_adapter import SubmissionError


class DeletionAucEvaluator(Evaluator):
    id = "deletion_auc"
    label = "Deletion area under the confidence curve (lower is better)"

    def evaluate(self, predictions, targets, config):
        images = reference.datasets()["images"]
        try:
            result = reference.deletion_curves(images, predictions["attributions"])
        except (ValueError, KeyError) as exc:
            raise SubmissionError(str(exc)) from None
        fractions, curves, areas, _ = result
        predictions.info["deletion_result"] = (
            result  # reuse for display, avoiding duplicate inference
        )
        score = float(areas.mean())
        return EvaluationOutput(
            score,
            {
                "deletion_auc": round(score, 6),
                "confidence_at_25pct": round(float(curves[:, 7].mean()), 6),
                "n_explanations": len(images),
            },
        )
