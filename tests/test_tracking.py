"""Tests for experiment tracking. Each test gets its own temporary MLflow database."""

import pandas as pd
import pytest

from spotcast import tracking, versioning
from spotcast.model import classification_scores
from spotcast.tracking import mlflow


@pytest.fixture
def temp_tracking(tmp_path, monkeypatch):
    """Point MLflow at a throwaway database, never at the project's mlflow.db."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(tracking, "ARTIFACTS_DIR", tmp_path / "artifacts")


def test_run_records_params_metrics_and_code_version(temp_tracking):
    with tracking.run("test-experiment", "first-run", profile="windsurf_intermediate"):
        mlflow.log_params({"max_depth": 5})
        mlflow.log_metric("accuracy", 0.9)

    runs = mlflow.search_runs(experiment_names=["test-experiment"])
    row = runs.iloc[0]
    assert row["tags.mlflow.runName"] == "first-run"
    assert row["params.max_depth"] == "5"  # MLflow stores params as text
    assert row["metrics.accuracy"] == pytest.approx(0.9)
    assert row["tags.profile"] == "windsurf_intermediate"
    assert row["tags.git_commit"]  # which code produced this result
    assert row["tags.git_dirty"] in {"true", "false", "unknown"}


def test_epoch_metrics_keep_their_steps(temp_tracking):
    with tracking.run("test-experiment", "lstm"):
        for epoch, mae in enumerate([2.3, 2.2, 2.4], start=1):
            mlflow.log_metric("val_mae", mae, step=epoch)
        run_id = mlflow.active_run().info.run_id

    history = mlflow.MlflowClient().get_metric_history(run_id, "val_mae")
    assert [(m.step, m.value) for m in history] == [(1, 2.3), (2, 2.2), (3, 2.4)]


def test_code_version_without_git(monkeypatch):
    monkeypatch.setattr(versioning, "_git", lambda *args: None)
    assert versioning.code_version() == {
        "git_commit": "unknown",
        "git_dirty": "unknown",
    }


@pytest.mark.parametrize(
    "text, key",
    [
        ("Kuşadası Pamucak", "kusadasi_pamucak"),
        ("Alaçatı", "alacati"),
        ("Akyaka", "akyaka"),
    ],
)
def test_metric_key(text, key):
    assert tracking.metric_key(text) == key


def test_classification_scores_per_class():
    actual = pd.Series(["bad", "bad", "good", "marginal"])
    predicted = ["bad", "good", "good", "bad"]

    scores = classification_scores(actual, predicted)

    assert scores["accuracy"] == pytest.approx(0.5)
    assert scores["recall_good"] == pytest.approx(1.0)  # the one good hour was found
    assert scores["precision_good"] == pytest.approx(0.5)  # but one "good" was wrong
    assert scores["recall_marginal"] == pytest.approx(0.0)
