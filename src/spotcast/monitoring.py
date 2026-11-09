"""Monitoring for the batch job behind the website: is this run's data fit to publish?

There is no server to watch, so what can go wrong is the data itself: the weather API fails
for many places, returns impossible values, or starts sending conditions the model has never
seen. Each run builds a quality report and:

- STOPS (raises DataQualityError) on problems that make the data wrong. The workflow turns
  red, nothing is pushed, and the site keeps showing the last good data. Loud, not silent.
- RECORDS warnings that need attention but not a stop, e.g. values outside the model's
  training range (data drift). The guardrail already protects users from those.

The report goes into meta.json and into the GitHub Actions run summary.
"""

import os
from pathlib import Path
from typing import Any

import pandas as pd

# Values outside these are not weather, they are broken data
PLAUSIBLE = {
    "wind_speed_10m": (0.0, 100.0),
    "wave_height": (0.0, 20.0),
    "wave_period": (0.0, 30.0),
}
MAX_SKIPPED_SHARE = (
    0.2  # more places than this unreachable: something is wrong upstream
)


class DataQualityError(Exception):
    """This run's data must not be published."""


def _count_outside(values: pd.Series, low: float, high: float) -> int:
    known = values.dropna()
    return int(((known < low) | (known > high)).sum())


def quality_report(
    forecasts: pd.DataFrame, bundle: dict, places_total: int, skipped: list[str]
) -> dict[str, Any]:
    """Facts about one run's input data. `forecasts`: every hourly row that was fetched."""
    ranges = bundle.get("feature_ranges", {})
    return {
        "rows": len(forecasts),
        "places_total": places_total,
        "places_skipped": skipped,
        "missing_wave_share": round(float(forecasts["wave_height"].isna().mean()), 3)
        if len(forecasts)
        else 1.0,
        "implausible_values": {
            column: _count_outside(forecasts[column], low, high)
            for column, (low, high) in PLAUSIBLE.items()
            if column in forecasts
        },
        # Data drift: hours the model meets values it never saw in training
        "outside_training_range": {
            column: _count_outside(forecasts[column], low, high)
            for column, (low, high) in ranges.items()
            if column in forecasts
        },
    }


def problems(report: dict[str, Any]) -> list[str]:
    """Reasons not to publish. Empty list: the data is fit to publish."""
    found = []
    if report["rows"] == 0:
        found.append("No forecast rows were fetched.")
    total, skipped = report["places_total"], len(report["places_skipped"])
    if total and skipped / total > MAX_SKIPPED_SHARE:
        found.append(f"{skipped} of {total} places could not be fetched.")
    for column, count in report["implausible_values"].items():
        if count:
            found.append(
                f"{count} implausible {column} values (outside {PLAUSIBLE[column]})."
            )
    return found


def notices(report: dict[str, Any]) -> list[str]:
    """Worth a look, but not a reason to stop."""
    found = [
        f"{count} hours with {column} outside the model's training range."
        for column, count in report["outside_training_range"].items()
        if count
    ]
    if report["places_skipped"]:
        found.append(f"Skipped places: {', '.join(report['places_skipped'])}.")
    return found


def summary_markdown(report: dict[str, Any]) -> str:
    """A short report for the GitHub Actions run page."""
    stops, notes = problems(report), notices(report)
    status = "❌ Not published" if stops else "✅ Published"
    lines = [
        f"### Site data: {status}",
        "",
        f"- Forecast rows checked: {report['rows']}",
        f"- Places fetched: {report['places_total'] - len(report['places_skipped'])}"
        f" / {report['places_total']}",
        f"- Hours without wave data: {report['missing_wave_share']:.1%}",
    ]
    lines += [f"- ❌ {p}" for p in stops] + [f"- ⚠️ {w}" for w in notes]
    return "\n".join(lines) + "\n"


def write_run_summary(report: dict[str, Any]) -> None:
    """On GitHub Actions, add the report to the run's summary page (a no-op elsewhere)."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with Path(target).open("a", encoding="utf-8") as file:
            file.write(summary_markdown(report))
