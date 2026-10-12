"""Bounded NPZ maps: read data only, reject object arrays and fabricated array headers."""

import io
import zipfile
import zlib

import numpy as np

from app.challenges.captum_deletion import reference
from app.core.submission_adapter import PredictionSet, SubmissionAdapter, SubmissionError

MAX_ARRAY_BYTES = 1024 * 1024
FIELDS = {"ids.npy", "attributions.npy", "method.npy", "model_sha256.npy"}


def _read_array(archive, name, shape, kinds, max_itemsize):
    with archive.open(name) as stream:
        content = stream.read(MAX_ARRAY_BYTES + 1)
    if len(content) > MAX_ARRAY_BYTES:
        raise ValueError(f"{name} exceeds the array size limit.")
    stream = io.BytesIO(content)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        actual_shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
    elif version == (2, 0):
        actual_shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=1024)
    else:
        raise ValueError("Use NumPy's standard save/savez format (NPY version 1 or 2).")
    if actual_shape != shape or dtype.kind not in kinds or dtype.itemsize > max_itemsize:
        raise ValueError(f"{name} has an invalid shape or dtype; expected {shape}.")
    return np.load(io.BytesIO(content), allow_pickle=False, max_header_size=1024)


class AttributionNpzAdapter(SubmissionAdapter):
    id = "attribution_npz"
    label = "Captum attribution maps (NPZ)"
    accepted_extensions = (".npz",)
    description = "Pixel attribution maps for the fixed model; the server computes deletion score."

    def run(self, artifact, ctx):
        ids = ctx.inputs["attributions"].ids
        n = len(ids)
        try:
            with zipfile.ZipFile(artifact) as archive:
                names = archive.namelist()
                if len(names) != len(FIELDS) or set(names) != FIELDS:
                    raise ValueError("Expected ids, attributions, method and model_sha256 only.")
                submitted_ids = _read_array(archive, "ids.npy", (n,), "iu", 8)
                maps = _read_array(archive, "attributions.npy", (n, 784), "fiu", 8)
                method = _read_array(archive, "method.npy", (), "U", 512).item()
                digest = _read_array(archive, "model_sha256.npy", (), "U", 256).item()
            if len(np.unique(submitted_ids)) != n or set(submitted_ids.tolist()) != set(
                ids.tolist()
            ):
                raise ValueError(
                    "Include every scoring image id exactly once, with no unknown ids."
                )
            if not np.all(np.isfinite(maps)):
                raise ValueError("Attributions contain NaN or Inf.")
            if digest != reference.model_sha256():
                raise ValueError(
                    "Model digest differs from the fixed model. Download the current resources."
                )
            if not method.strip() or len(method) > 128:
                raise ValueError("Provide a method/settings description of 1 to 128 characters.")
            lookup = {int(value): i for i, value in enumerate(submitted_ids)}
            aligned = maps[[lookup[int(value)] for value in ids]].astype(np.float64)
        except (
            ValueError,
            OSError,
            KeyError,
            EOFError,
            zipfile.BadZipFile,
            RuntimeError,
            zlib.error,
        ) as exc:
            raise SubmissionError("Invalid attribution NPZ.", [str(exc)]) from None
        return PredictionSet({"attributions": aligned}, info={"method": method, "source": self.id})
