from __future__ import annotations

from app.core.evaluator import Evaluator
from app.core.registry import Registry
from app.evaluators.binary_class_probabilities import BinaryClassProbabilitiesEvaluator
from app.evaluators.binary_classification import BinaryClassificationEvaluator
from app.evaluators.deletion_auc import DeletionAucEvaluator
from app.evaluators.generative_mmd import GenerativeMmdEvaluator
from app.evaluators.multiclass_classification import MulticlassClassificationEvaluator


def register_all(registry: Registry[Evaluator]) -> None:
    registry.register(GenerativeMmdEvaluator(), replace=True)
    registry.register(DeletionAucEvaluator(), replace=True)
    registry.register(BinaryClassProbabilitiesEvaluator(), replace=True)
    registry.register(BinaryClassificationEvaluator(), replace=True)
    registry.register(MulticlassClassificationEvaluator(), replace=True)
