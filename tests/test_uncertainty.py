"""Tests for the uncertainty warnings."""

from datetime import date, datetime

import pandas as pd
import pytest

from spotcast.uncertainty import (
    NO_WAVE_NOTE,
    UNCERTAIN_FROM_DAYS,
    days_ahead,
    lead_time_note,
    notes_for_results,
)

NOW = datetime(2026, 11, 5, 22, 0)  # late Thursday evening


def test_days_ahead_counts_calendar_days_not_hours():
    assert days_ahead(date(2026, 11, 5), NOW) == 0
    assert (
        days_ahead(pd.Timestamp("2026-11-06 07:00"), NOW) == 1
    )  # 9 hours, but tomorrow


@pytest.mark.parametrize(
    "days, warned", [(0, False), (UNCERTAIN_FROM_DAYS - 1, False), (5, True), (6, True)]
)
def test_lead_time_note_starts_at_five_days(days, warned):
    note = lead_time_note(days)
    assert (note is not None) == warned
    if warned:
        assert f"{days} gün sonrası" in note


def test_notes_use_the_farthest_day_and_spot_missing_waves():
    results = [
        {"days_ahead": 1, "label_counts": {"unknown": 3}},
        {"windows_best_first": [{"days_ahead": 2}, {"days_ahead": 6}]},
        {"error": "outside the forecast", "days_ahead": 20},  # errors are ignored
    ]
    assert notes_for_results(results) == [NO_WAVE_NOTE, lead_time_note(6)]


def test_no_notes_for_near_days_with_waves():
    assert notes_for_results([{"days_ahead": 2, "label_counts": {"good": 4}}]) == []
