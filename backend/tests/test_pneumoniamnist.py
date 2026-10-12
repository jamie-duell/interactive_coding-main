"""MedMNIST preparation, binary metric semantics, notebook export, and leaderboard flow."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.challenges.pneumoniamnist import data
from app.config import get_settings
from app.core import PredictionSet, challenges, evaluators
from app.core.submission_adapter import SubmissionError
from app.runners.base import RunnerResult
from app.scripts import build_pneumoniamnist
from app.worker import run_once


@pytest.fixture
def pneumonia(monkeypatch, tmp_path):
    pixels = np.tile(np.arange(784, dtype=np.uint16) % 256, (624, 1)).astype(np.uint8)
    labels = np.array([0] * 234 + [1] * 390, dtype=np.uint8)
    archive = tmp_path / "test.npz"
    np.savez_compressed(archive, X_test=pixels, y_test=labels)
    monkeypatch.setattr(get_settings(), "pneumoniamnist_archive", archive)
    data.hidden_test_set.cache_clear()
    yield challenges.get("pneumoniamnist")
    data.hidden_test_set.cache_clear()


def test_preparation_keeps_only_test_split(monkeypatch, tmp_path):
    images = np.arange(624 * 784, dtype=np.uint32).reshape(624, 28, 28).astype(np.uint8)
    labels = np.array([0] * 234 + [1] * 390, dtype=np.uint8).reshape(-1, 1)
    source, prepared = tmp_path / "source.npz", tmp_path / "prepared.npz"
    np.savez_compressed(source, test_images=images, test_labels=labels, train_labels=[99])
    with pytest.raises(ValueError, match="checksum"):
        build_pneumoniamnist.build(source, prepared)
    assert not prepared.exists()
    monkeypatch.setattr(build_pneumoniamnist, "MD5", hashlib.md5(source.read_bytes()).hexdigest())
    build_pneumoniamnist.build(source, prepared)
    with np.load(prepared, allow_pickle=False) as z:
        assert set(z.files) == {"X_test", "y_test"}
        np.testing.assert_array_equal(z["X_test"], images.reshape(624, 784))
        np.testing.assert_array_equal(z["y_test"], labels[:, 0])


@pytest.mark.parametrize(
    "bad_images", [np.zeros((624, 28, 28, 3), dtype=np.uint8), np.zeros((624, 28, 28))]
)
def test_preparation_rejects_wrong_resolution_or_dtype(monkeypatch, tmp_path, bad_images):
    source = tmp_path / "invalid.npz"
    np.savez(source, test_images=bad_images, test_labels=np.zeros((624, 1), dtype=np.uint8))
    monkeypatch.setattr(build_pneumoniamnist, "MD5", hashlib.md5(source.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="grayscale"):
        build_pneumoniamnist.build(source, tmp_path / "out.npz")


def test_challenge_resources_and_official_input_contract(pneumonia, client):
    inputs, targets = pneumonia.inputs(), pneumonia.hidden_targets()
    assert set(inputs) == set(targets) == {"test"}
    assert inputs["test"].X.shape == (624, 784)
    assert inputs["test"].X.dtype == np.uint8
    np.testing.assert_array_equal(inputs["test"].ids, np.arange(624))
    assert np.bincount(targets["test"]).tolist() == [234, 390]
    detail = client.get("/api/challenges/pneumoniamnist").json()
    assert detail["primary_metric"]["format"] == "%"
    assert detail["evaluator"] == "binary_class_probabilities"
    assert [t["id"] for t in detail["submission_types"]] == ["python_bundle"]
    assert {r["name"] for r in detail["resources"]} == {"submission_template.py"}
    assert (
        client.get("/api/challenges/pneumoniamnist/resources/pneumoniamnist_test.npz").status_code
        == 404
    )
    display = client.get("/api/challenges/pneumoniamnist/display").json()
    assert display["visualization"] == "accuracy_leaderboard"
    assert display["data"] == {"class_labels": ["normal", "pneumonia"], "n_test": 624}


def test_accuracy_auc_and_imbalance_metrics(pneumonia):
    evaluator = evaluators.get(pneumonia.evaluator)
    labels = pneumonia.hidden_targets()["test"]
    perfect = np.full((624, 2), 0.01)
    perfect[np.arange(624), labels] = 0.99
    result = evaluator.evaluate(PredictionSet({"test": perfect}), {"test": labels}, {})
    assert (
        result.primary_score == result.metrics["auc"] == result.metrics["balanced_accuracy"] == 1.0
    )
    # Predicting the majority class is 62.5% accurate, but only 50% balanced accuracy/AUROC.
    constant = np.tile([0.1, 0.9], (624, 1))
    result = evaluator.evaluate(PredictionSet({"test": constant}), {"test": labels}, {})
    assert result.primary_score == 0.625
    assert result.metrics["auc"] == result.metrics["balanced_accuracy"] == 0.5
    # AUROC uses column 1 probabilities, and is not computed from thresholded labels.
    p1 = np.array([0.1, 0.6, 0.4, 0.9])
    probabilities = np.column_stack([1 - p1, p1])
    result = evaluator.evaluate(
        PredictionSet({"test": probabilities}), {"test": np.array([0, 0, 1, 1])}, {}
    )
    assert result.primary_score == 0.5 and result.metrics["auc"] == 0.75
    with pytest.raises(SubmissionError, match="two columns"):
        evaluator.evaluate(
            PredictionSet({"test": np.full((4, 3), 1 / 3)}), {"test": np.array([0, 0, 1, 1])}, {}
        )
    with pytest.raises(SubmissionError, match="both classes"):
        evaluator.evaluate(
            PredictionSet({"test": probabilities}), {"test": np.zeros(4, dtype=int)}, {}
        )


def test_uploaded_models_are_scored_and_ranked(pneumonia, client, make_session, join, monkeypatch):
    class Runner:
        correct = 0

        def run(self, *, bundle_dir, inputs_dir, workdir):
            assert {p.name for p in inputs_dir.iterdir()} == {"test.npy", "manifest.json"}
            assert np.load(inputs_dir / "test.npy").shape == (624, 784)
            y = pneumonia.hidden_targets()["test"]
            predicted = 1 - y
            predicted[: self.correct] = y[: self.correct]
            p = np.full((624, 2), 0.01)
            p[np.arange(624), predicted] = 0.99
            return RunnerResult(predictions={"test": p.tolist()}, runtime_seconds=1.0)

    runner = Runner()
    monkeypatch.setattr("app.submissions.python_bundle.get_runner", lambda _: runner)
    session = make_session(challenge_id="pneumoniamnist")
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
        assert run_once("pneumonia-test")
        result = client.get(f"/api/submissions/{response.json()['id']}").json()
        assert result["status"] == "succeeded", result
        assert result["primary_score"] == correct / 624
        assert {"auc", "balanced_accuracy", "log_loss"} <= set(result["metrics"])
        return result

    best = submit(alice, 500)
    submit(bob, 550)
    submit(alice, 450)
    live = client.get(f"/api/sessions/{session['id']}/live").json()
    people = {p["display_name"]: p for p in live["participants"]}
    assert people["Bob"]["rank"] == 1 and people["Bob"]["best_score"] == 550 / 624
    assert people["Alice"]["rank"] == 2 and people["Alice"]["best_submission_id"] == best["id"]


def test_notebook_architecture_and_export_contract():
    root = Path(__file__).resolve().parents[2]
    notebook_path = root / "examples/pneumoniamnist_colab.ipynb"
    if not notebook_path.exists():
        pytest.skip("notebook is outside backend-only Docker context")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    template = (root / "backend/app/challenges/pneumoniamnist/submission_template.py").read_text()
    assert (
        next(s for s in sources if s.startswith("%%writefile submission.py"))
        .split("\n", 1)[1]
        .strip()
        == template.strip()
    )
    for source in sources:
        if source.startswith("%pip"):
            continue
        if source.startswith("%%writefile"):
            source = source.split("\n", 1)[1]
        compile(source, str(notebook_path), "exec")
    combined = "\n".join(sources)
    assert 'PneumoniaMNIST(split="train"' in combined and 'PneumoniaMNIST(split="val"' in combined
    assert 'split="test"' not in combined
    assert '"model.py", "weights.pt"' in combined
