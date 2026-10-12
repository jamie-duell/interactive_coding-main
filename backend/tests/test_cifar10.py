"""CIFAR-10 data layout, notebook export contract, and server-ranked accuracy."""

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.challenges.cifar10 import data
from app.config import get_settings
from app.core import challenges
from app.runners.base import RunnerResult
from app.scripts import build_cifar10
from app.worker import run_once


@pytest.fixture
def cifar(monkeypatch, tmp_path):
    # Synthetic images avoid a network download in unit tests. Keep the official shape.
    X = np.zeros((10_000, 3072), dtype=np.uint8)
    X[:, :1024] = 255  # red channel, so channel ordering is observable
    y = np.tile(np.arange(10, dtype=np.uint8), 1000)
    archive = tmp_path / "test.npz"
    np.savez_compressed(archive, X_test=X, y_test=y)
    monkeypatch.setattr(get_settings(), "cifar10_archive", archive)
    data.hidden_test_set.cache_clear()
    yield challenges.get("cifar10")
    data.hidden_test_set.cache_clear()


def test_binary_builder_preserves_official_order(monkeypatch, tmp_path):
    rows = np.zeros((10_000, 3073), dtype=np.uint8)
    rows[:, 0] = np.tile(np.arange(10), 1000)
    rows[:, 1:1025] = 255
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("cifar-10-batches-bin/test_batch.bin")
        member.size = rows.nbytes
        tar.addfile(member, io.BytesIO(rows.tobytes()))
    with pytest.raises(ValueError, match="checksum"):
        build_cifar10.build(archive, tmp_path / "invalid.npz")
    monkeypatch.setattr(build_cifar10, "MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    out = tmp_path / "prepared.npz"
    build_cifar10.build(archive, out)
    with np.load(out) as prepared:
        np.testing.assert_array_equal(prepared["y_test"], rows[:, 0])
        np.testing.assert_array_equal(prepared["X_test"], rows[:, 1:])


def test_inputs_and_resources(cifar, client):
    inputs = cifar.inputs()
    assert set(inputs) == {"test"}
    assert inputs["test"].X.shape == (10_000, 3072)
    assert inputs["test"].X.dtype == np.uint8
    np.testing.assert_array_equal(inputs["test"].ids, np.arange(10_000))
    image = inputs["test"].X[0].reshape(3, 32, 32)
    assert np.all(image[0] == 255) and np.all(image[1:] == 0)
    detail = client.get("/api/challenges/cifar10").json()
    assert detail["primary_metric"]["label"] == "test accuracy"
    assert detail["primary_metric"]["format"] == "%"
    assert [t["id"] for t in detail["submission_types"]] == ["python_bundle"]
    assert {r["name"] for r in detail["resources"]} == {"submission_template.py"}
    assert client.get("/api/challenges/cifar10/resources/cifar10_test.npz").status_code == 404
    display = client.get("/api/challenges/cifar10/display").json()
    assert display["visualization"] == "accuracy_leaderboard"
    assert display["data"]["class_labels"] == data.CLASS_LABELS
    assert display["data"]["n_test"] == 10_000


def test_server_scoring_and_best_submission_ranking(cifar, client, make_session, join, monkeypatch):
    class Runner:
        correct = 0

        def run(self, *, bundle_dir, inputs_dir, workdir):
            # The adapter only materialises pixels and a manifest, never test targets.
            assert {p.name for p in inputs_dir.iterdir()} == {"test.npy", "manifest.json"}
            X = np.load(inputs_dir / "test.npy")
            assert X.shape == (10_000, 3072)
            y = cifar.hidden_targets()["test"]
            predicted = (y + 1) % 10
            predicted[: self.correct] = y[: self.correct]
            probabilities = np.full((10_000, 10), 0.01)
            probabilities[np.arange(10_000), predicted] = 0.91
            return RunnerResult(predictions={"test": probabilities.tolist()}, runtime_seconds=1.0)

    runner = Runner()
    monkeypatch.setattr("app.submissions.python_bundle.get_runner", lambda _: runner)
    session = make_session(challenge_id="cifar10")
    alice, bob = join(session, "Alice"), join(session, "Bob")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("submission.py", "def predict(X): pass\n")

    def submit(student, correct):
        runner.correct = correct
        response = client.post(
            f"/api/sessions/{session['id']}/submissions",
            headers={"X-Participant-Token": student["token"]},
            files={"file": ("submission.zip", buffer.getvalue(), "application/zip")},
        )
        assert response.status_code == 202, response.text
        assert run_once("cifar-test")
        result = client.get(f"/api/submissions/{response.json()['id']}").json()
        assert result["status"] == "succeeded", result
        assert result["primary_score"] == correct / 10_000
        return result

    best = submit(alice, 8000)
    submit(bob, 9000)
    submit(alice, 7000)
    live = client.get(f"/api/sessions/{session['id']}/live").json()
    participants = {p["display_name"]: p for p in live["participants"]}
    assert participants["Bob"]["rank"] == 1 and participants["Bob"]["best_score"] == 0.9
    assert participants["Alice"]["rank"] == 2
    assert participants["Alice"]["best_score"] == 0.8
    assert participants["Alice"]["best_submission_id"] == best["id"]


def test_notebook_export_matches_server_template():
    root = Path(__file__).resolve().parents[2]
    notebook_path = root / "examples/cifar10_colab.ipynb"
    if not notebook_path.exists():
        pytest.skip("notebook is outside backend-only Docker context")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    template = (root / "backend/app/challenges/cifar10/submission_template.py").read_text()
    assert (
        next(s for s in sources if s.startswith("%%writefile submission.py"))
        .split("\n", 1)[1]
        .strip()
        == template.strip()
    )
    for source in sources:
        if source.startswith("%%writefile"):
            source = source.split("\n", 1)[1]
        compile(source, str(notebook_path), "exec")
    combined = "\n".join(sources)
    assert "train=False" not in combined
    assert "bundle.write(filename)" in combined
    assert '"model.py", "weights.pt"' in combined
