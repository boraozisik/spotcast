"""Experiment tracking with MLflow.

Every training run is saved with its settings (params), its scores (metrics), the code
version (git commit) and, for the model we serve, the model itself. Results that used to
live only in the terminal and in our notes can now be compared side by side:

    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

Everything stays on this computer: mlflow.db (runs and the model registry) and
mlartifacts/ (saved files). Both are git-ignored.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

# MLflow prints a hint about a feature we don't use every time it is imported
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402  (must come after the line above)

from spotcast.config import PROJECT_ROOT  # noqa: E402
from spotcast.versioning import code_version  # noqa: E402

DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"


def tracking_uri() -> str:
    """Local SQLite by default; MLFLOW_TRACKING_URI can point to a shared server instead."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def use_experiment(name: str) -> None:
    """Select an experiment (a group of comparable runs), creating it the first time."""
    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(
            name, artifact_location=(ARTIFACTS_DIR / name).as_uri()
        )
    mlflow.set_experiment(name)


@contextmanager
def run(experiment: str, run_name: str, **tags: str) -> Iterator[mlflow.ActiveRun]:
    """One tracked run. Everything logged inside the `with` block belongs to it."""
    use_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags={**code_version(), **tags}) as active:
        yield active


def metric_key(text: str) -> str:
    """'Kuşadası Pamucak' → 'kusadasi_pamucak': safe in metric names and URLs."""
    plain = text.lower().translate(str.maketrans("çğıöşüâ", "cgiosua"))
    return "".join(c if c.isalnum() else "_" for c in plain)
