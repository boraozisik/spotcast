"""Tests for the activity profiles: the real file, the checks, and calm-water sports."""

import pickle

import pytest

from spotcast.labels import is_unsafe, label_conditions
from spotcast.profiles import (
    LEVELS,
    PROFILES,
    SPORTS,
    ActivityProfile,
    ProfileError,
    get_profile,
    parse_profiles,
)
from spotcast.recommend import hour_score

ONE = """
[[profile]]
sport = "sup"
level = "beginner"
max_wind_kn = 10
max_wave_m = 0.5
min_wind_kn = 0
good_wind_min_kn = 0
good_wind_max_kn = 6
good_wave_max_m = 0.2
"""


# --- The real file ---


def test_every_sport_has_every_level():
    assert set(SPORTS) == {"windsurf", "kitesurf", "sup", "kayak"}
    assert set(PROFILES) == {(s, lvl) for s in SPORTS for lvl in LEVELS}


@pytest.mark.parametrize("sport", SPORTS)
def test_safety_limits_grow_with_skill(sport):
    """A beginner must never be allowed more than someone more experienced."""
    profiles = [get_profile(sport, level) for level in LEVELS]
    for easier, harder in zip(profiles, profiles[1:], strict=False):
        assert easier.max_wind_kn <= harder.max_wind_kn
        assert easier.max_wave_m <= harder.max_wave_m


# --- Checks on load ---


def test_valid_profile():
    [profile] = parse_profiles(ONE)
    assert profile.name == "sup/beginner"
    assert profile.ideal_wind == 3  # no ideal given: middle of 0-6


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            ("good_wind_max_kn = 6", "good_wind_max_kn = 12"),
            "good_wind_max <= max_wind",
        ),
        (
            ("good_wave_max_m = 0.2", "good_wave_max_m = 0.7"),
            "good_wave_max < max_wave",
        ),
        (('level = "beginner"', 'level = "pro"'), "level must be one of"),
        (("max_wind_kn = 10", 'max_wind_kn = "10"'), "must be a number"),
        (("max_wind_kn", "max_wnd_kn"), "unexpected keyword argument 'max_wnd_kn'"),
        (("min_wind_kn = 0\n", ""), "min_wind_kn"),  # missing field
    ],
)
def test_invalid_profile_is_rejected_with_its_name(change, message):
    with pytest.raises(ProfileError) as error:
        parse_profiles(ONE.replace(*change))
    assert "sup/" in str(error.value)
    assert message in str(error.value)


def test_ideal_wind_must_be_inside_the_good_range():
    with pytest.raises(ProfileError, match="ideal_wind"):
        parse_profiles(ONE + "ideal_wind_kn = 9\n")


def test_duplicate_profiles_are_rejected():
    with pytest.raises(ProfileError, match="duplicate profiles: sup/beginner"):
        parse_profiles(ONE + ONE)


def test_old_saved_profile_without_ideal_wind_still_works():
    """Model files from before ideal_wind_kn existed must still load and behave."""
    profile = get_profile("windsurf", "intermediate")
    old = pickle.loads(pickle.dumps(profile))
    del old.__dict__["ideal_wind_kn"]  # what an old pickle looks like
    assert old.ideal_wind_kn is None  # falls back to the class default
    assert old.ideal_wind == 20


# --- Calm-water sports: the logic turns around ---


@pytest.mark.parametrize(
    ("wind_kn", "wave_m", "expected"),
    [
        (0, 0.1, "good"),  # flat calm is perfect for SUP
        (7, 0.3, "good"),
        (11, 0.3, "marginal"),  # paddling back gets hard
        (14, 0.3, "marginal"),  # max_wind itself is allowed
        (15, 0.3, "bad"),  # unsafe for an intermediate SUP
        (3, 0.8, "bad"),  # waves at max_wave_m are unsafe
    ],
)
def test_sup_intermediate_labels(wind_kn, wave_m, expected):
    profile = get_profile("sup", "intermediate")
    assert label_conditions(profile, wind_kn, wave_m, True) == expected


def test_sup_unsafe_wind_is_fine_for_windsurf():
    # The same 20 kn: dangerous on a paddle board, ideal on a sail
    assert is_unsafe(get_profile("sup", "intermediate"), 20, 0.5, True)
    assert (
        label_conditions(get_profile("windsurf", "intermediate"), 20, 0.5, True)
        == "good"
    )


def test_recommender_prefers_calm_hours_for_sup():
    sup = get_profile("sup", "intermediate")
    assert hour_score("good", 1, sup) > hour_score("good", 6, sup)


def test_profile_object_is_still_a_frozen_dataclass():
    profile = get_profile("kayak", "beginner")
    assert isinstance(profile, ActivityProfile)
    with pytest.raises(AttributeError):
        profile.max_wind_kn = 99  # read-only
