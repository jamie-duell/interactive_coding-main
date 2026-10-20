"""Generative sample validation, benchmark math and independent live standings."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.challenges.generative_models import reference
from app.core import AdapterContext, SubmissionError, adapters, challenges
from app.worker import run_once

TEACHER = {"X-Teacher-Secret": "test-secret"}


@pytest.fixture
def generation_ctx(settings, tmp_path):
    challenge = challenges.get("generative_models")
    return AdapterContext(challenge, challenge.inputs(), tmp_path, settings)


def archive_bytes(**overrides):
    data = {
        "samples": np.random.default_rng(42).random((256, 28, 28), dtype=np.float32),
        "model_family": np.array("vae"),
        "benchmark_sha256": np.array(reference.benchmark_sha256()),
        "description": np.array("VAE baseline"),
    }
    data.update(overrides)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **data)
    return buffer.getvalue()


@pytest.mark.parametrize("family", ["vae", "diffusion"])
def test_adapter_accepts_each_family(generation_ctx, family):
    path = generation_ctx.workdir / "samples.npz"
    path.write_bytes(archive_bytes(model_family=np.array(family)))
    predictions = adapters.get("generated_samples").run(path, generation_ctx)
    assert predictions["samples"].shape == (256, 28, 28)
    assert predictions.info["model_family"] == family


@pytest.mark.parametrize(
    "overrides",
    [
        {"samples": np.zeros((255, 28, 28), dtype=np.float32)},
        {"samples": np.zeros((256, 28, 28), dtype=np.float64)},
        {"samples": np.zeros((256, 28, 28), dtype=object)},
        {"samples": np.full((256, 28, 28), np.nan, dtype=np.float32)},
        {"samples": np.full((256, 28, 28), -0.1, dtype=np.float32)},
        {"samples": np.full((256, 28, 28), 1.1, dtype=np.float32)},
        {"model_family": np.array("gan")},
        {"model_family": np.array(["vae"])},
        {"benchmark_sha256": np.array("0" * 64)},
        {"description": np.array("")},
        {"description": np.array("x" * 129)},
        {"score": np.array(0.0)},
    ],
)
def test_invalid_samples_fail_with_student_message(generation_ctx, overrides):
    path = generation_ctx.workdir / "bad.npz"
    path.write_bytes(archive_bytes(**overrides))
    with pytest.raises(SubmissionError, match="Invalid generated-samples"):
        adapters.get("generated_samples").run(path, generation_ctx)


def test_huge_forged_shape_rejected_before_allocation(generation_ctx):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f4", "fortran_order": False, "shape": (10**12, 28, 28)}
    )
    path = generation_ctx.workdir / "forged.npz"
    with zipfile.ZipFile(io.BytesIO(archive_bytes())) as original:
        with zipfile.ZipFile(path, "w") as target:
            for name in original.namelist():
                target.writestr(
                    name, header.getvalue() if name == "samples.npy" else original.read(name)
                )
    with pytest.raises(SubmissionError):
        adapters.get("generated_samples").run(path, generation_ctx)


def test_mmd_matches_hand_calculation_and_identical_distributions(monkeypatch):
    real = np.array([[0.0], [1.0]])
    generated = np.array([[2.0], [3.0]])
    monkeypatch.setattr(reference, "encode", lambda images: images)
    monkeypatch.setattr(
        reference,
        "distribution",
        lambda: {
            "reference_features": real,
            "feature_mean": np.zeros(1),
            "feature_scale": np.ones(1),
            "bandwidth_squared": np.array(1.0),
        },
    )
    assert reference.score(real)[0] == pytest.approx(0, abs=1e-12)
    # Kxx and Kyy each have two diagonal ones and two exp(-1/2) entries.
    expected = 1 + np.exp(-0.5) - 0.5 * (2 * np.exp(-2) + np.exp(-4.5) + np.exp(-0.5))
    score, diversity = reference.score(generated)
    assert score == pytest.approx(expected)
    assert diversity == pytest.approx(0.5)


def test_notebooks_and_public_benchmark(client):
    detail = client.get("/api/challenges/generative_models").json()
    assert set(detail["leaderboard_tracks"]) == {"vae", "diffusion"}
    assert detail["primary_metric"]["higher_is_better"] is False
    assert [t["id"] for t in detail["submission_types"]] == ["generated_samples"]
    for resource in detail["resources"]:
        response = client.get(f"/api/challenges/generative_models/resources/{resource['name']}")
        assert response.status_code == 200
        assert response.content == (reference.HERE / resource["name"]).read_bytes()
    root = Path(__file__).resolve().parents[2]
    if not (root / "examples").exists():
        pytest.skip("notebooks outside backend-only Docker context")
    assert len(detail["starter_notebooks"]) == 2
    for starter in detail["starter_notebooks"]:
        path = root / "examples" / starter["filename"]
        notebook = json.loads(path.read_text())
        sources = [
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        assert reference.benchmark_sha256() in "\n".join(sources)
        for source in sources:
            if not source.startswith("%pip"):
                compile(source, str(path), "exec")


def test_both_boards_keep_independent_bests_ranks_events_and_hidden_scores(
    client,
    make_session,
    join,
):
    session = make_session(challenge_id="generative_models")
    sid = session["id"]
    alice, bob = join(session, "Alice"), join(session, "Bob")
    sets = [
        np.random.default_rng(7).random((256, 28, 28), dtype=np.float32),
        np.zeros((256, 28, 28), dtype=np.float32),
    ]
    scores = [reference.score(images)[0] for images in sets]
    low, high = np.argsort(scores)
    assert scores[low] < scores[high]

    def submit(student, family, index):
        response = client.post(
            f"/api/sessions/{sid}/submissions",
            headers={"X-Participant-Token": student["token"]},
            files={
                "file": (
                    "samples.npz",
                    archive_bytes(samples=sets[index], model_family=np.array(family)),
                    "application/octet-stream",
                )
            },
        )
        assert response.status_code == 202, response.text
        assert run_once("generation-test")
        result = client.get(f"/api/submissions/{response.json()['id']}").json()
        assert result["status"] == "succeeded", result
        assert result["primary_score"] == pytest.approx(scores[index])
        display = client.get(f"/api/submissions/{result['id']}/display").json()
        assert display["data"]["model_family"] == family
        assert len(display["data"]["samples"]) == 16
        assert len(display["data"]["samples"][0]) == 784
        return result["id"]

    alice_vae = submit(alice, "vae", low)
    bob_vae = submit(bob, "vae", high)
    alice_diffusion = submit(alice, "diffusion", high)
    bob_diffusion = submit(bob, "diffusion", low)
    submit(alice, "vae", high)  # Worse VAE cannot replace her VAE best or her diffusion entry.
    submit(alice, "diffusion", high)  # Tied score keeps the earlier submission.
    people = {
        p["display_name"]: p for p in client.get(f"/api/sessions/{sid}/live").json()["participants"]
    }
    for name, vae_id, diffusion_id, vae_rank, diffusion_rank in [
        ("Alice", alice_vae, alice_diffusion, 1, 2),
        ("Bob", bob_vae, bob_diffusion, 2, 1),
    ]:
        tracks = people[name]["tracks"]
        assert tracks["vae"]["best_submission_id"] == vae_id
        assert tracks["diffusion"]["best_submission_id"] == diffusion_id
        assert tracks["vae"]["rank"] == vae_rank
        assert tracks["diffusion"]["rank"] == diffusion_rank
    assert people["Alice"]["tracks"]["vae"]["n_submissions"] == 2
    assert people["Alice"]["tracks"]["diffusion"]["n_submissions"] == 2

    event_text = client.get(f"/api/sessions/{sid}/events?after=0&once=true").text
    updates = [
        json.loads(line[6:]) for line in event_text.splitlines() if line.startswith("data: ")
    ]
    succeeded = {
        e["submission"]["id"]: e
        for e in updates
        if "submission" in e and e["submission"]["status"] == "succeeded"
    }
    # First diffusion entry improves that board even when worse than the participant's VAE.
    assert succeeded[alice_diffusion]["participant"]["improved"] is True
    assert (
        succeeded[alice_diffusion]["participant"]["tracks"]["vae"]["best_submission_id"]
        == alice_vae
    )

    token = {"X-Participant-Token": alice["token"]}
    client.patch(f"/api/teacher/sessions/{sid}", json={"scores_hidden": True}, headers=TEACHER)
    mine = client.get("/api/sessions/me", headers=token).json()
    for track in mine["participant"]["tracks"].values():
        assert track["best_score"] is None and track["rank"] is None
        assert track["best_metrics"] is None and track["latest_score"] is None
    teacher = client.get("/api/sessions/me", headers={**token, **TEACHER}).json()
    assert teacher["participant"]["tracks"]["vae"]["best_score"] == pytest.approx(scores[low])
    client.patch(f"/api/teacher/sessions/{sid}", json={"scores_hidden": False}, headers=TEACHER)
    mine = client.get("/api/sessions/me", headers=token).json()
    assert mine["participant"]["tracks"]["diffusion"]["rank"] == 2
    assert client.post(f"/api/teacher/sessions/{sid}/reset", headers=TEACHER).status_code == 200
    assert client.get(f"/api/sessions/{sid}/live").json()["participants"] == []
