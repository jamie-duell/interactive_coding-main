"""Binary classification with two probability columns and imbalance-aware metrics."""

from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from app.core.evaluator import EvaluationOutput
from app.core.submission_adapter import PredictionSet, SubmissionError
from app.evaluators.multiclass_classification import MulticlassClassificationEvaluator, _check


class BinaryClassProbabilitiesEvaluator(MulticlassClassificationEvaluator):
    id = "binary_class_probabilities"
    label = "Binary classification (two-class probabilities / AUROC / balanced accuracy)"

    def evaluate(
        self, predictions: PredictionSet, targets: dict[str, np.ndarray], config: dict[str, Any]
    ) -> EvaluationOutput:
        p, y = _check(predictions, targets, config.get("dataset", "test"))
        if p.shape[1] != 2 or not np.all(np.isin(y, [0, 1])):
            raise SubmissionError("Binary classification requires two columns and labels 0 or 1.")
        if len(np.unique(y)) != 2:
            raise SubmissionError("AUROC evaluation requires both classes in the test targets.")
        out = super().evaluate(predictions, targets, config)
        out.metrics["auc"] = round(float(roc_auc_score(y, p[:, 1])), 6)
        out.metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(y, p.argmax(1))), 6)
        return out
