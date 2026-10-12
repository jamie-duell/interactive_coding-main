"""Deletion definition, hostile NPZs, frozen model resources and lower-is-better ranking."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.challenges.captum_deletion import reference
from app.core import (
    AdapterContext,
    SubmissionError,
    adapters,
    challenges,
)
from app.worker import run_once


@pytest.fixture
def xai_ctx(settings, tmp_path):
    challenge = challenges.get("captum_deletion")
    return AdapterContext(challenge, challenge.inputs(), tmp_path, settings)


def archive_bytes(maps=None, **overrides):
    data = {
        "ids": reference.datasets()["ids"],
        "attributions": np.zeros((64, 784)) if maps is None else maps,
        "method": np.array("IntegratedGradients, 32 steps"),
        "model_sha256": np.array(reference.model_sha256()),
    }
    data.update(overrides)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **data)
    return buffer.getvalue()


def test_deletion_uses_original_target_black_baseline_endpoints_and_area():
    # Only pixel 0 matters: deleting it first must have lower AUC than deleting it last.
    images = np.zeros((1, 784), dtype=np.float32)
    images[0, 0] = 255

    def classifier(x):
        positive = np.where(x[:, 0] > 0, 0.9, 0.1)
        return np.column_stack([1 - positive, positive])

    first, last = np.zeros((1, 784)), np.zeros((1, 784))
    first[0, 0], last[0, 0] = 1, -1
    fractions, curve, area, targets = reference.deletion_curves(images, first, classifier)
    assert targets.tolist() == [1]
    assert fractions[0] == 0 and fractions[-1] == 1 and curve.shape == (1, 29)
    assert curve[0, 0] == 0.9 and np.all(curve[0, 1:] == 0.1)
    assert area[0] == pytest.approx((0.5 * (0.9 + 0.1) + 27 * 0.1) / 28)
    late_area = reference.deletion_curves(images, last, classifier)[2][0]
    assert late_area > area[0]
    # If the target were changed to the new prediction after deletion, this would be 0.9.
    assert curve[0, -1] == 0.1


def test_fixed_resources_and_public_api(client, xai_ctx):
    assert reference.datasets()["images"].shape == (64, 784)
    assert reference.datasets()["practice_images"].shape == (32, 784)
    assert reference.predict(reference.datasets()["images"]).shape == (64, 2)
    detail = client.get("/api/challenges/captum_deletion").json()
    assert detail["primary_metric"]["higher_is_better"] is False
    assert [t["id"] for t in detail["submission_types"]] == ["attribution_npz"]
    assert {r["name"] for r in detail["resources"]} == {
        "reference_model.npz",
        "explanation_data.npz",
    }
    assert (
        client.get("/api/challenges/captum_deletion/resources/reference_model.npz").content
        == reference.MODEL_PATH.read_bytes()
    )


def test_npz_rows_align_by_id(xai_ctx):
    maps = np.arange(64 * 784).reshape(64, 784)
    path = xai_ctx.workdir / "maps.npz"
    path.write_bytes(archive_bytes(maps[::-1], ids=reference.datasets()["ids"][::-1]))
    result = adapters.get("attribution_npz").run(path, xai_ctx)
    np.testing.assert_array_equal(result["attributions"], maps)
    assert result.info["method"].startswith("IntegratedGradients")


@pytest.mark.parametrize(
    "overrides",
    [
        {"attributions": np.zeros((63, 784))},
        {"attributions": np.full((64, 784), np.nan)},
        {"attributions": np.zeros((64, 784), dtype=object)},
        {"ids": np.zeros(64, dtype=int)},
        {"ids": np.arange(1, 65)},
        {"model_sha256": np.array("0" * 64)},
        {"method": np.array("")},
        {"score": np.array(0.0)},
    ],
)
def test_invalid_npz_is_a_student_facing_failure(xai_ctx, overrides):
    path = xai_ctx.workdir / "bad.npz"
    path.write_bytes(archive_bytes(**overrides))
    with pytest.raises(SubmissionError, match="Invalid attribution"):
        adapters.get("attribution_npz").run(path, xai_ctx)


def test_notebook_and_model_versions_agree():
    root = Path(__file__).resolve().parents[2]
    path = root / "examples/captum_deletion_colab.ipynb"
    if not path.exists():
        pytest.skip("notebook is outside backend-only Docker context")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert reference.model_sha256() in "\n".join(sources)
    for source in sources:
        if not source.startswith("%pip"):
            compile(source, str(path), "exec")


def test_fake_huge_npy_header_is_rejected_before_allocation(xai_ctx):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f8", "fortran_order": False, "shape": (10**12, 784)}
    )
    source = archive_bytes()
    path = xai_ctx.workdir / "forged.npz"
    with zipfile.ZipFile(io.BytesIO(source)) as original, zipfile.ZipFile(path, "w") as target:
        for name in original.namelist():
            target.writestr(
                name, header.getvalue() if name == "attributions.npy" else original.read(name)
            )
    with pytest.raises(SubmissionError):
        adapters.get("attribution_npz").run(path, xai_ctx)


def test_server_computes_score_and_lower_is_better(client, make_session, join, xai_ctx):
    session = make_session(challenge_id="captum_deletion")
    alice, bob = join(session, "Alice"), join(session, "Bob")
    rng = np.random.default_rng(1)
    maps = [rng.random((64, 784)), rng.random((64, 784))]
    scores = [
        float(reference.deletion_curves(reference.datasets()["images"], m)[2].mean()) for m in maps
    ]
    low, high = np.argsort(scores)

    def submit(student, index):
        response = client.post(
            f"/api/sessions/{session['id']}/submissions",
            headers={"X-Participant-Token": student["token"]},
            files={
                "file": ("attributions.npz", archive_bytes(maps[index]), "application/octet-stream")
            },
        )
        assert response.status_code == 202, response.text
        assert run_once("xai-test")
        result = client.get(f"/api/submissions/{response.json()['id']}").json()
        assert result["status"] == "succeeded", result
        assert result["primary_score"] == pytest.approx(scores[index])
        display = client.get(f"/api/submissions/{result['id']}/display").json()
        assert len(display["data"]["mean_curve"]) == 29
        assert len(display["data"]["examples"][0]["importance"]) == 784
        return result

    best = submit(alice, low)
    submit(bob, high)
    submit(alice, high)
    people = {
        p["display_name"]: p
        for p in client.get(f"/api/sessions/{session['id']}/live").json()["participants"]
    }
    assert people["Alice"]["rank"] == 1 and people["Bob"]["rank"] == 2
    assert people["Alice"]["best_submission_id"] == best["id"]
