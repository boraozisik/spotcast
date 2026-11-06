"""Tests for knots ↔ Beaufort conversions."""

import pytest

from spotcast.beaufort import add_beaufort_to_text, beaufort_force, knots_range


@pytest.mark.parametrize(
    ("knots", "force"),
    [
        (0, 0),
        (0.5, 0),
        (3, 1),
        (10, 3),
        (10.5, 3),
        (11, 4),
        (16, 4),
        (25, 6),
        (27.9, 6),
        (64, 12),
        (90, 12),
    ],
)
def test_beaufort_force(knots, force):
    assert beaufort_force(knots) == force


def test_knots_range():
    assert knots_range(4) == "11-16 knot"
    assert knots_range(12) == "64+ knot"


def test_add_beaufort_to_text():
    assert add_beaufort_to_text("25 knot rüzgâr") == "25 knot (6 Beaufort) rüzgâr"
    assert add_beaufort_to_text("10,5 kn esiyor") == "10,5 kn (3 Beaufort) esiyor"
    assert add_beaufort_to_text("rüzgâr yok") == "rüzgâr yok"
