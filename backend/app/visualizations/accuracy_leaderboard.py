"""Scores-only display for architecture competitions."""

from app.core.visualization import VisualizationType


class AccuracyLeaderboard(VisualizationType):
    id = "accuracy_leaderboard"
    label = "Test accuracy leaderboard"
