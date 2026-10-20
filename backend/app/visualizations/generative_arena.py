import numpy as np

from app.core.visualization import VisualizationType


class GenerativeArena(VisualizationType):
    id = "generative_arena"
    label = "VAE and diffusion leaderboards with generated-image galleries"

    def submission_display(self, predictions, challenge):
        return {
            "model_family": predictions.info["model_family"],
            "description": predictions.info["description"],
            "samples": np.rint(predictions["samples"][:16] * 255)
            .astype(int)
            .reshape(16, 784)
            .tolist(),
        }
