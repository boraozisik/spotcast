"""Training, evaluating, saving and using the suitability classifier.

ONE model serves every activity profile (Stage 11). It does not see raw wind and waves but
their distance to the profile's limits ("wind is 3 kn above this profile's maximum"). A
decision tree asks about one column at a time, so it can only compare wind with a limit if
that comparison is a column. With these relative features it learns the idea once and works
for every profile, even one added later without retraining (measured: macro F1 0.998 on a
profile left out of training; raw wind + limit columns: 0.65).
"""

from dataclasses import asdict
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
from spotcast.labels import add_labels, is_unsafe
from spotcast.profiles import ActivityProfile
from spotcast.shore import unsafe_offshore_flags

# What the model sees: conditions measured against one profile's limits
FEATURES = [
    "wind_over_min",  # wind - min_wind_kn (negative: not enough wind)
    "wind_over_good_min",  # wind - good_wind_min_kn
    "wind_under_good_max",  # good_wind_max_kn - wind
    "wind_over_max",  # wind - max_wind_kn (positive: unsafe)
    "wave_over_good_max",  # wave - good_wave_max_m
    "wave_over_max",  # wave - max_wave_m (zero or positive: unsafe)
    "offshore_over_max",  # offshore wind - max_offshore_wind_kn (positive: unsafe)
    "is_day",
]
# Raw inputs whose training range is saved with the model, to notice data drift
MONITORED_INPUTS = ["wind_speed_10m", "wave_height", "wave_period"]
MODEL_FILE = "suitability.joblib"
TARGET = "label"
CLASSES = ["bad", "marginal", "good"]  # fixed order for reports and matrices


def make_features(conditions: pd.DataFrame, profile: ActivityProfile) -> pd.DataFrame:
    """The model's input for one profile: each condition minus the profile's limit."""
    wind, wave = conditions["wind_speed_10m"], conditions["wave_height"]
    # Wind that blows from the land (outside a sheltered bay) counts here; else 0 kn
    offshore_wind = wind.where(unsafe_offshore_flags(conditions), 0.0)
    offshore_limit = (
        profile.max_offshore_wind_kn
        if profile.max_offshore_wind_kn is not None
        else profile.max_wind_kn
    )
    return pd.DataFrame(
        {
            "wind_over_min": wind - profile.min_wind_kn,
            "wind_over_good_min": wind - profile.good_wind_min_kn,
            "wind_under_good_max": profile.good_wind_max_kn - wind,
            "wind_over_max": wind - profile.max_wind_kn,
            "wave_over_good_max": wave - profile.good_wave_max_m,
            "wave_over_max": wave - profile.max_wave_m,
            "offshore_over_max": offshore_wind - offshore_limit,
            "is_day": conditions["is_day"],
        },
        index=conditions.index,
    )[FEATURES]


def training_table(
    history: pd.DataFrame, profiles: list[ActivityProfile]
) -> pd.DataFrame:
    """Every hour of history once per profile, with its features and rule-based label.

    3 spots x 42k hours x 12 profiles: ~1.5 million rows, a few seconds for a tree.
    """
    frames = []
    for profile in profiles:
        labeled = add_labels(history, profile)
        frames.append(
            make_features(labeled, profile).assign(
                label=labeled["label"],
                time=labeled["time"],
                spot=labeled["spot"],
                profile=profile.name,
            )
        )
    return pd.concat(frames, ignore_index=True)


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
    X_train: pd.DataFrame, y_train: pd.Series, max_depth: int = 7
) -> DecisionTreeClassifier:
    """A decision tree: learns a chain of if/else questions from the data.

    Depth 7 (measured in Stage 11, with the offshore rule): depth 5 reached macro F1 0.933
    and had no room left for the good/marginal edge of SUP and kayak; 7 reaches 1.000 with
    only 10 leaves, and deeper trees don't grow any further.
    """
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    return tree.fit(X_train, y_train)


def predict_with_guardrail(
    model: ClassifierMixin, conditions: pd.DataFrame, profile: ActivityProfile
) -> list[str]:
    """Model predictions for this profile, but unsafe conditions are always "bad"."""
    predictions = model.predict(make_features(conditions, profile))
    return [
        "bad" if is_unsafe(profile, wind, wave, day, offshore) else prediction
        for prediction, wind, wave, day, offshore in zip(
            predictions,
            conditions["wind_speed_10m"],
            conditions["wave_height"],
            conditions["is_day"],
            unsafe_offshore_flags(conditions),
            strict=True,
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


def model_path(models_dir: Path = MODELS_DIR) -> Path:
    """Where the one model for all profiles is stored: models/suitability.joblib."""
    return models_dir / MODEL_FILE


def feature_ranges(X: pd.DataFrame) -> dict[str, list[float]]:
    """The [min, max] of every numeric column (we pass MONITORED_INPUTS of the training
    history). Saved with the model,
    so live data outside what the model has seen can be noticed (it once called 40 knots
    "good", having never seen more than 27.7)."""
    numeric = X.select_dtypes("number")
    return {c: [float(numeric[c].min()), float(numeric[c].max())] for c in numeric}


def save_model(
    model: ClassifierMixin,
    profiles: list[ActivityProfile],
    train_years: list[int],
    models_dir: Path = MODELS_DIR,
    ranges: dict[str, list[float]] | None = None,
    train_end: date | None = None,
) -> Path:
    """Save the model together with what is needed to use it correctly later."""
    bundle = {
        "model": model,
        "features": FEATURES,  # the exact columns, in the exact order, it was trained on
        # The limits of the profiles it was trained on, for the record. Predictions use the
        # current profiles.toml: the features are relative, so a changed limit needs no
        # retraining.
        "profiles": {p.name: asdict(p) for p in profiles},
        "train_years": train_years,
        # The last day of training data: retraining tests a new model on data after this
        "train_end": train_end.isoformat() if train_end else None,
        "feature_ranges": ranges or {},  # what the model has seen (for monitoring)
        "sklearn_version": sklearn.__version__,
    }
    path = model_path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Load a saved model bundle. Only load files you created yourself (see joblib docs)."""
    path = model_path(models_dir)
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
    bundle: dict[str, Any], conditions: pd.DataFrame, profile: ActivityProfile
) -> pd.DataFrame:
    """Add a "prediction" column for this profile to a conditions table (e.g. the forecast)."""
    if bundle["features"] != FEATURES:
        raise ValueError(
            "The saved model was trained with different features. Retrain it: "
            "`uv run spotcast` (or delete models/ and let the job train a new one)."
        )
    result = conditions.copy()
    result["prediction"] = predict_with_guardrail(bundle["model"], conditions, profile)
    return result
