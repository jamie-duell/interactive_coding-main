import zipfile
import zlib

import numpy as np

from app.challenges.generative_models import reference
from app.core.submission_adapter import PredictionSet, SubmissionAdapter, SubmissionError
from app.submissions.attribution_npz import _read_array


class GeneratedSamplesAdapter(SubmissionAdapter):
    id = "generated_samples"
    label = "Generated VAE/diffusion samples (NPZ)"
    accepted_extensions = (".npz",)
    description = "256 generated grayscale images, model family and benchmark digest."

    def run(self, artifact, ctx):
        try:
            with zipfile.ZipFile(artifact) as archive:
                names = archive.namelist()
                required = {
                    "samples.npy",
                    "model_family.npy",
                    "benchmark_sha256.npy",
                    "description.npy",
                }
                if len(names) != len(required) or set(names) != required:
                    raise ValueError(
                        "Expected samples, model_family, benchmark_sha256 and description only."
                    )
                samples = _read_array(archive, "samples.npy", (256, 28, 28), "f", 4)
                family = _read_array(archive, "model_family.npy", (), "U", 128).item()
                digest = _read_array(archive, "benchmark_sha256.npy", (), "U", 256).item()
                description = _read_array(archive, "description.npy", (), "U", 512).item()
            if family not in reference.FAMILIES:
                raise ValueError("model_family must be vae or diffusion.")
            if not np.isfinite(samples).all() or samples.min() < 0 or samples.max() > 1:
                raise ValueError("All generated pixels must be finite values in [0, 1].")
            if digest != reference.benchmark_sha256():
                raise ValueError("Benchmark digest differs; use the current notebook resources.")
            if not description.strip() or len(description) > 128:
                raise ValueError("Describe your experiment in 1 to 128 characters.")
        except (
            ValueError,
            OSError,
            KeyError,
            EOFError,
            zipfile.BadZipFile,
            RuntimeError,
            zlib.error,
        ) as exc:
            raise SubmissionError("Invalid generated-samples NPZ.", [str(exc)]) from None
        return PredictionSet(
            {"samples": samples.astype(np.float32)},
            info={"model_family": family, "description": description},
        )
