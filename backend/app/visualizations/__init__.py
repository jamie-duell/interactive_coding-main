from __future__ import annotations

from app.core.registry import Registry
from app.core.visualization import VisualizationType
from app.visualizations.accuracy_leaderboard import AccuracyLeaderboard
from app.visualizations.decision_boundary_arena import DecisionBoundaryArena
from app.visualizations.deletion_arena import DeletionArena
from app.visualizations.generalization_arena import GeneralizationArena
from app.visualizations.generative_arena import GenerativeArena


def register_all(registry: Registry[VisualizationType]) -> None:
    registry.register(GenerativeArena(), replace=True)
    registry.register(DeletionArena(), replace=True)
    registry.register(AccuracyLeaderboard(), replace=True)
    registry.register(DecisionBoundaryArena(), replace=True)
    registry.register(GeneralizationArena(), replace=True)
