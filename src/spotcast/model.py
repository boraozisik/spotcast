"""Training and evaluating the suitability classifier."""

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

from spotcast.labels import is_unsafe

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


def predict_with_guardrail(model: ClassifierMixin, X: pd.DataFrame) -> list[str]:
    """Model predictions, but unsafe conditions are always forced to "bad"."""
    predictions = model.predict(X)
    return [
        "bad" if is_unsafe(wind, wave, day) else prediction
        for prediction, wind, wave, day in zip(
            predictions, X["wind_speed_10m"], X["wave_height"], X["is_day"]
        )
    ]


def evaluate(
    name: str, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """Print accuracy, per-class precision/recall and the confusion matrix."""
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


def show_tree(tree: DecisionTreeClassifier) -> None:
    """Print the learned rules and how much each feature was used."""
    print("=== Rules learned by the tree ===")
    print(export_text(tree, feature_names=FEATURES, decimals=2))

    importances = pd.Series(tree.feature_importances_, index=FEATURES)
    print("Feature importance (share of the tree's decisions):")
    print(importances.sort_values(ascending=False).round(3))
