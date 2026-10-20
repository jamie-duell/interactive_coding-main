from app.challenges.generative_models import reference
from app.core.evaluator import EvaluationOutput, Evaluator


class GenerativeMmdEvaluator(Evaluator):
    id = "generative_mmd"
    label = "Domain-feature RBF MMD² (lower is better)"

    def evaluate(self, predictions, targets, config):
        score, diversity = reference.score(predictions["samples"])
        return EvaluationOutput(
            score,
            {
                "mmd2": round(score, 6),
                "feature_diversity": round(diversity, 6),
                "n_samples": reference.N_SAMPLES,
                "model_family": predictions.info["model_family"],
                "description": predictions.info["description"],
            },
        )
