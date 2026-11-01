"""Tests for picking the best spot and time window."""

from datetime import date

import pandas as pd

from spotcast.profiles import WINDSURF_INTERMEDIATE
from spotcast.recommend import Preferences, find_windows, hour_score, recommend


def make_predicted(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Rows of (spot, time, prediction, wind_kn)."""
    return pd.DataFrame(
        {
            "spot": [r[0] for r in rows],
            "time": pd.to_datetime([r[1] for r in rows]),
            "prediction": [r[2] for r in rows],
            "wind_speed_10m": [r[3] for r in rows],
            "wave_height": 0.5,
        }
    )


def test_good_hour_always_beats_marginal_hour():
    worst_good = min(hour_score("good", w, WINDSURF_INTERMEDIATE) for w in range(40))
    best_marginal = max(
        hour_score("marginal", w, WINDSURF_INTERMEDIATE) for w in range(40)
    )
    assert worst_good > best_marginal


def test_window_breaks_on_bad_hour_and_respects_min_hours():
    predicted = make_predicted(
        [
            ("A", "2026-11-07 10:00", "good", 18),
            ("A", "2026-11-07 11:00", "good", 18),
            ("A", "2026-11-07 12:00", "bad", 5),  # breaks the window
            ("A", "2026-11-07 13:00", "good", 18),  # alone: shorter than min_hours
        ]
    )

    windows = find_windows(predicted, WINDSURF_INTERMEDIATE, Preferences(min_hours=2))

    assert len(windows) == 1
    assert windows.iloc[0]["hours"] == 2
    assert windows.iloc[0]["end"] == pd.Timestamp("2026-11-07 11:00")


def test_evening_hours_are_excluded_by_preferences():
    predicted = make_predicted(
        [
            ("A", "2026-11-07 18:00", "good", 18),
            ("A", "2026-11-07 19:00", "good", 18),
            ("A", "2026-11-07 20:00", "good", 18),
        ]
    )

    windows = find_windows(
        predicted, WINDSURF_INTERMEDIATE, Preferences(latest_hour=18)
    )

    assert windows.empty  # only 18:00 is allowed, and one hour is not enough


def test_weekend_filter_and_spot_filter():
    predicted = make_predicted(
        [
            ("A", "2026-11-06 10:00", "good", 18),  # Friday
            ("A", "2026-11-06 11:00", "good", 18),
            ("A", "2026-11-07 10:00", "marginal", 12),  # Saturday
            ("A", "2026-11-07 11:00", "marginal", 12),
            ("B", "2026-11-07 10:00", "good", 18),  # Saturday, but spot B
            ("B", "2026-11-07 11:00", "good", 18),
        ]
    )

    prefs = Preferences.weekend(spots=frozenset({"A"}))
    best = recommend(predicted, WINDSURF_INTERMEDIATE, prefs)

    assert list(best["spot"]) == ["A"]
    assert best.iloc[0]["start"].date() == date(2026, 11, 7)


def test_windows_do_not_cross_midnight_or_spots():
    predicted = make_predicted(
        [
            ("A", "2026-11-07 09:00", "good", 18),
            ("B", "2026-11-07 10:00", "good", 18),  # different spot right after
        ]
    )
    prefs = Preferences(earliest_hour=0, latest_hour=23, min_hours=1)

    windows = find_windows(predicted, WINDSURF_INTERMEDIATE, prefs)

    assert len(windows) == 2


def test_longer_good_window_ranks_first():
    predicted = make_predicted(
        [
            ("A", "2026-11-07 10:00", "good", 20),
            ("A", "2026-11-07 11:00", "good", 20),
            ("B", "2026-11-07 10:00", "good", 20),
            ("B", "2026-11-07 11:00", "good", 20),
            ("B", "2026-11-07 12:00", "good", 20),
        ]
    )

    best = recommend(predicted, WINDSURF_INTERMEDIATE, Preferences())

    assert list(best["spot"]) == ["B", "A"]


def test_hours_in_the_past_are_skipped():
    predicted = make_predicted(
        [
            ("A", "2026-11-07 10:00", "good", 18),
            ("A", "2026-11-07 11:00", "good", 18),
            ("A", "2026-11-07 12:00", "good", 18),
        ]
    )
    prefs = Preferences(not_before=pd.Timestamp("2026-11-07 11:00"))

    windows = find_windows(predicted, WINDSURF_INTERMEDIATE, prefs)

    assert windows.iloc[0]["start"] == pd.Timestamp("2026-11-07 11:00")
    assert windows.iloc[0]["hours"] == 2
