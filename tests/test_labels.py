"""Tests for the rule-based labels and safety limits."""

import pandas as pd
import pytest

from spotcast.labels import add_labels, is_unsafe, label_conditions
from spotcast.profiles import WINDSURF_INTERMEDIATE, get_profile


@pytest.mark.parametrize(
    ("wind_kn", "wave_m", "is_day", "expected"),
    [
        # typical cases
        (18, 0.5, True, "good"),
        (12, 0.5, True, "marginal"),
        (27, 0.8, True, "marginal"),
        (8, 0.2, True, "bad"),  # not enough wind
        # danger always wins
        (18, 0.5, False, "bad"),  # night
        (18, 2.5, True, "bad"),  # big waves, even with ideal wind
        (40, 0.5, True, "bad"),  # storm
        # exact boundaries
        (10, 0.5, True, "marginal"),  # min_wind_kn itself is allowed
        (15, 1.0, True, "good"),  # lower edge of the good window
        (25, 1.0, True, "good"),  # upper edge of the good window
        (28, 0.5, True, "marginal"),  # max_wind_kn itself is allowed
        (28.1, 0.5, True, "bad"),
        (20, 2.0, True, "bad"),  # max_wave_m itself is dangerous
    ],
)
def test_windsurf_intermediate_labels(wind_kn, wave_m, is_day, expected):
    assert label_conditions(WINDSURF_INTERMEDIATE, wind_kn, wave_m, is_day) == expected


def test_low_wind_is_bad_but_not_unsafe():
    # "Nothing to do" is not the same as "dangerous"
    assert label_conditions(WINDSURF_INTERMEDIATE, 5, 0.1, True) == "bad"
    assert is_unsafe(WINDSURF_INTERMEDIATE, 5, 0.1, True) is False


def test_add_labels_adds_column_without_changing_input():
    df = pd.DataFrame(
        {
            "wind_speed_10m": [18.0, 40.0],
            "wave_height": [0.5, 0.5],
            "is_day": [True, True],
        }
    )

    labeled = add_labels(df, WINDSURF_INTERMEDIATE)

    assert list(labeled["label"]) == ["good", "bad"]
    assert "label" not in df.columns  # the original table is untouched


def test_get_profile_finds_known_profile():
    assert get_profile("windsurf", "intermediate") is WINDSURF_INTERMEDIATE


def test_get_profile_rejects_unknown_profile():
    with pytest.raises(ValueError, match="No profile for sup/beginner"):
        get_profile("sup", "beginner")
