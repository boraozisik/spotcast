"""Training, evaluating, saving and using the suitability classifier."""

from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

from spotcast.config import MODELS_DIR
from spotcast.labels import is_unsafe
from spotcast.profiles import ActivityProfile

FEATURES = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wave_height",
    "wave_period",
    "is_day",
]
TARGET = "label"
CLASSES = ["bad", "marginal", "good"]  # fixed order for reports and matrices


def split_by_year(
    df: pd.DataFrame, test_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train on rows before test_year, test on rows in test_year."""
    years = df["time"].dt.year
    train = df[years < test_year]
    test = df[years == test_year]
    return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET]


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
    """A "model" that always predicts the most common label. Anything useful must beat it."""
    return DummyClassifier(strategy="most_frequent").fit(X_train, y_train)


def train_tree(
    X_train: pd.DataFrame, y_train: pd.Series, max_depth: int = 5
) -> DecisionTreeClassifier:
    """A decision tree: learns a chain of if/else questions from the data."""
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    return tree.fit(X_train, y_train)


def predict_with_guardrail(
    model: ClassifierMixin, X: pd.DataFrame, profile: ActivityProfile
) -> list[str]:
    """Model predictions, but unsafe conditions for this profile are always "bad"."""
    predictions = model.predict(X)
    return [
        "bad" if is_unsafe(profile, wind, wave, day) else prediction
        for prediction, wind, wave, day in zip(
            predictions, X["wind_speed_10m"], X["wave_height"], X["is_day"], strict=True
        )
    ]


def classification_scores(y_true: pd.Series, predictions: Any) -> dict[str, float]:
    """The numbers worth tracking: accuracy, macro F1, and precision/recall/F1 per class."""
    report = classification_report(
        y_true, predictions, labels=CLASSES, output_dict=True, zero_division=0
    )
    scores = {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": report["macro avg"]["f1-score"],
    }
    for label in CLASSES:
        scores[f"precision_{label}"] = report[label]["precision"]
        scores[f"recall_{label}"] = report[label]["recall"]
        scores[f"f1_{label}"] = report[label]["f1-score"]
    return scores


def evaluate(
    name: str, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Print accuracy, per-class precision/recall and the confusion matrix; return the scores."""
    predictions = model.predict(X_test)

    print(f"=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(
        classification_report(
            y_test, predictions, labels=CLASSES, digits=3, zero_division=0
        )
    )

    matrix = confusion_matrix(y_test, predictions, labels=CLASSES)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=CLASSES, columns=CLASSES))
    print()
    return classification_scores(y_test, predictions)


def show_tree(tree: DecisionTreeClassifier) -> str:
    """Print the learned rules and how much each feature was used; return the rules."""
    rules = export_text(tree, feature_names=FEATURES, decimals=2)
    print("=== Rules learned by the tree ===")
    print(rules)

    importances = pd.Series(tree.feature_importances_, index=FEATURES)
    print("Feature importance (share of the tree's decisions):")
    print(importances.sort_values(ascending=False).round(3))
    return rules


def model_path(profile: ActivityProfile, models_dir: Path = MODELS_DIR) -> Path:
    """Where the model for this profile is stored, e.g. models/windsurf_intermediate.joblib."""
    return models_dir / f"{profile.sport}_{profile.level}.joblib"


def feature_ranges(X: pd.DataFrame) -> dict[str, list[float]]:
    """The [min, max] of every numeric feature in the training data. Saved with the model,
    so live data outside what the model has seen can be noticed (it once called 40 knots
    "good", having never seen more than 27.7)."""
    numeric = X.select_dtypes("number")
    return {c: [float(numeric[c].min()), float(numeric[c].max())] for c in numeric}


def save_model(
    model: ClassifierMixin,
    profile: ActivityProfile,
    train_years: list[int],
    models_dir: Path = MODELS_DIR,
    ranges: dict[str, list[float]] | None = None,
    train_end: date | None = None,
) -> Path:
    """Save the model together with what is needed to use it correctly later."""
    bundle = {
        "model": model,
        "features": FEATURES,  # the exact columns, in the exact order, it was trained on
        "profile": profile,
        "train_years": train_years,
        # The last day of training data: retraining tests a new model on data after this
        "train_end": train_end.isoformat() if train_end else None,
        "feature_ranges": ranges or {},  # what the model has seen (for monitoring)
        "sklearn_version": sklearn.__version__,
    }
    path = model_path(profile, models_dir)
    models_dir.mkdir(exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(
    profile: ActivityProfile, models_dir: Path = MODELS_DIR
) -> dict[str, Any]:
    """Load a saved model bundle. Only load files you created yourself (see joblib docs)."""
    path = model_path(profile, models_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run `uv run spotcast` first."
        )

    bundle = joblib.load(path)
    if bundle["sklearn_version"] != sklearn.__version__:
        print(
            f"Warning: model was saved with scikit-learn {bundle['sklearn_version']}, "
            f"running {sklearn.__version__}. Retrain if results look wrong."
        )
    return bundle


def predict_conditions(
    bundle: dict[str, Any], conditions: pd.DataFrame
) -> pd.DataFrame:
    """Add a "prediction" column to a conditions table (e.g. the 7-day forecast)."""
    X = conditions[bundle["features"]]
    result = conditions.copy()
    result["prediction"] = predict_with_guardrail(bundle["model"], X, bundle["profile"])
    return result
