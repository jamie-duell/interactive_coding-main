"""Mean deletion curve and representative image/attribution pairs."""

from app.challenges.captum_deletion import reference
from app.core.visualization import VisualizationType


class DeletionArena(VisualizationType):
    id = "deletion_arena"
    label = "Deletion curves and attribution heatmaps"

    def submission_display(self, predictions, challenge):
        images = reference.datasets()["images"]
        maps = predictions["attributions"]
        result = predictions.info.get("deletion_result")
        if result is None:
            result = reference.deletion_curves(images, maps)
        fractions, curves, areas, targets = result
        return {
            "method": predictions.info.get("method", "Captum attribution"),
            "fractions": fractions.tolist(),
            "mean_curve": curves.mean(axis=0).tolist(),
            "examples": [
                {
                    "id": int(reference.datasets()["ids"][i]),
                    "target": int(targets[i]),
                    "pixels": images[i].tolist(),
                    "importance": maps[i].tolist(),
                    "curve": curves[i].tolist(),
                    "deletion_auc": float(areas[i]),
                }
                for i in range(min(4, len(images)))
            ],
        }
