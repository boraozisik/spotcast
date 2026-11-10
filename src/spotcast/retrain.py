"""Scheduled retraining with a champion/challenger check.

    uv run spotcast-retrain --out candidate      (GitHub Actions runs it every Monday)

- champion:   the model in use today (from models/, i.e. the approved `models` branch)
- challenger: a new model trained on everything up to the most recent days
- holdout:    the most recent days, which NEITHER model was trained on

Both are scored on the holdout, the way they would really be used (with the guardrail).
The challenger replaces the champion only if it is not worse and passes the safety check.
A promoted model then waits for a person to approve it (GitHub environment) before the
website uses it. What was evaluated is exactly what is shipped.

Labels here come from rules, so both models usually score the same. The point is the
process: in a project with real labels (e.g. user feedback), this is what catches a model
that got worse.
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from spotcast.config import HISTORY_END, HISTORY_START, MODELS_DIR
from spotcast.data import fetch_all_spots
from spotcast.labels import add_labels, is_unsafe
from spotcast.locations import TRAINING_LOCATIONS
from spotcast.model import (
    FEATURES,
    MONITORED_INPUTS,
    classification_scores,
    feature_ranges,
    load_model,
    predict_with_guardrail,
    save_model,
    train_tree,
    training_table,
)
from spotcast.profiles import PROFILES, ActivityProfile
from spotcast.shore import unsafe_offshore_flags
from spotcast.versioning import code_version

SETTLE_DAYS = 7  # ERA5 history is final only after ~5 days; wait a week to be safe
HOLDOUT_DAYS = 7  # the most recent days, kept out of training to judge both models
TOLERANCE = 0.01  # a challenger may be this much worse (noise) and still be promoted
CARD = "model_card.json"


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str


def holdout_window(champion_end: date, today: date) -> tuple[date, date] | None:
    """The days to judge both models on, or None if not enough new data has arrived.

    The champion must not have seen the holdout, and the challenger must get at least one
    day of training data the champion didn't have (otherwise it would learn nothing new).
    """
    data_end = today - timedelta(days=SETTLE_DAYS)
    start = data_end - timedelta(days=HOLDOUT_DAYS - 1)
    if start - timedelta(days=1) <= champion_end:
        return None
    return start, data_end


def score(
    model: Any, holdout: pd.DataFrame, profiles: list[ActivityProfile]
) -> dict[str, Any]:
    """How the model does on the holdout hours for every profile, as it is really used:
    with the guardrail. unsafe_called_good counts dangerous hours called "good" (any
    profile) and must be 0. per_profile shows a model that is bad for one sport only."""
    labels, predictions, per_profile, unsafe_good = [], [], {}, 0
    for profile in profiles:
        truth = add_labels(holdout, profile)["label"].tolist()
        predicted = predict_with_guardrail(model, holdout, profile)
        unsafe = [
            is_unsafe(profile, w, h, d, off)
            for w, h, d, off in zip(
                holdout["wind_speed_10m"],
                holdout["wave_height"],
                holdout["is_day"],
                unsafe_offshore_flags(holdout),
                strict=True,
            )
        ]
        unsafe_good += sum(
            u and p == "good" for u, p in zip(unsafe, predicted, strict=True)
        )
        per_profile[profile.name] = round(
            classification_scores(pd.Series(truth), predicted)["macro_f1"], 3
        )
        labels += truth
        predictions += predicted
    scores: dict[str, Any] = classification_scores(pd.Series(labels), predictions)
    scores["unsafe_called_good"] = unsafe_good
    scores["per_profile_macro_f1"] = per_profile
    return scores


def decide(champion: dict[str, float] | None, challenger: dict[str, float]) -> Decision:
    """The promotion rule. Safety first, then quality; newer data wins a tie."""
    if challenger["unsafe_called_good"] > 0:
        return Decision(False, "Challenger called dangerous hours good.")
    if champion is None:
        return Decision(True, "No champion yet: the challenger becomes the first one.")
    if challenger["macro_f1"] < champion["macro_f1"] - TOLERANCE:
        return Decision(
            False,
            f"Challenger is worse: macro F1 {challenger['macro_f1']:.3f} "
            f"vs champion {champion['macro_f1']:.3f}.",
        )
    return Decision(
        True,
        f"Challenger is not worse (macro F1 {challenger['macro_f1']:.3f} vs "
        f"{champion['macro_f1']:.3f}) and was trained on newer data.",
    )


def _champion(models_dir: Path) -> tuple[dict | None, date, int]:
    """The current model, the last day it was trained on, and its version number.

    A model in an older format (one per profile, before Stage 11) cannot be compared, so
    it counts as no champion; its version number still continues.
    """
    card = models_dir / CARD
    version = json.loads(card.read_text())["version"] if card.exists() else 0
    try:
        bundle = load_model(models_dir)
    except FileNotFoundError:
        bundle = None
    if bundle is None or bundle["features"] != FEATURES:
        return None, HISTORY_START - timedelta(days=1), version
    end = bundle.get("train_end")  # older models didn't record it
    return bundle, date.fromisoformat(end) if end else HISTORY_END, max(version, 1)


def retrain(
    out: Path,
    today: date,
    models_dir: Path = MODELS_DIR,
    fetch=fetch_all_spots,
) -> dict[str, Any]:
    """Run one champion/challenger round. Writes the report, and the new model if promoted."""
    profiles = list(PROFILES.values())
    champion, champion_end, version = _champion(models_dir)
    report: dict[str, Any] = {
        "checked_at": f"{datetime.now():%Y-%m-%dT%H:%M}",
        "champion_version": version or None,
        "champion_train_end": champion_end.isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)

    window = holdout_window(champion_end, today)
    if window is None:
        report |= {
            "promote": False,
            "reason": "Not enough new data since the champion was trained.",
        }
        (out / "report.json").write_text(json.dumps(report, indent=2))
        return report
    start, end = window

    history = fetch(TRAINING_LOCATIONS, HISTORY_START, end)
    day = history["time"].dt.date
    train, holdout = history[day < start], history[day >= start]
    table = training_table(train, profiles)

    challenger = train_tree(table[FEATURES], table["label"])
    challenger_scores = score(challenger, holdout, profiles)
    champion_scores = score(champion["model"], holdout, profiles) if champion else None
    decision = decide(champion_scores, challenger_scores)
    report |= {
        "holdout": [start.isoformat(), end.isoformat()],
        "holdout_rows": len(holdout),
        "champion": champion_scores,
        "challenger": challenger_scores,
        "promote": decision.promote,
        "reason": decision.reason,
    }

    if decision.promote:
        train_end = start - timedelta(days=1)
        save_model(
            challenger,
            profiles,
            sorted(train["time"].dt.year.unique().tolist()),
            models_dir=out,
            ranges=feature_ranges(train[MONITORED_INPUTS]),
            train_end=train_end,
        )
        card = {
            "version": version + 1,
            "profiles": [p.name for p in profiles],
            "train_start": HISTORY_START.isoformat(),
            "train_end": train_end.isoformat(),
            "train_rows": len(table),
            "holdout": report["holdout"],
            "holdout_scores": challenger_scores,
            "replaced_version": version or None,
            "code_version": code_version(),
        }
        (out / CARD).write_text(json.dumps(card, indent=2))
        report["new_version"] = card["version"]
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def summary_markdown(report: dict[str, Any]) -> str:
    """A short report for the GitHub Actions run page."""
    status = (
        "🆕 New champion (waiting for approval)" if report["promote"] else "No change"
    )
    lines = [f"### Model retraining: {status}", "", f"- {report['reason']}"]
    if "holdout" in report:
        lines.append(
            f"- Holdout: {report['holdout'][0]} to {report['holdout'][1]} ({report['holdout_rows']} hours)"
        )
        for name in ("champion", "challenger"):
            s = report.get(name)
            if s:
                lines.append(
                    f"- {name}: macro F1 {s['macro_f1']:.3f}, accuracy {s['accuracy']:.3f}, "
                    f"dangerous hours called good: {s['unsafe_called_good']}"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain and compare with the current model."
    )
    parser.add_argument("--out", type=Path, default=Path("candidate"))
    args = parser.parse_args()

    report = retrain(args.out, date.today())
    text = summary_markdown(report)
    print(text)
    # On GitHub Actions: show the report on the run page and tell the next job what happened
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as file:
            file.write(text)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as file:
            file.write(f"promote={str(report['promote']).lower()}\n")
