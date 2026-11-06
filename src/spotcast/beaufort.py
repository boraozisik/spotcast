"""The Beaufort scale as code: converting between knots and Beaufort force.

Numbers are a job for code. Embeddings can't tell that 25 knots falls in 22-27 (force 6),
and LLMs are unreliable with such lookups, so every conversion goes through here.
"""

import re

# Wind speed range for each Beaufort force, in knots (WMO). None = no upper limit.
BEAUFORT_KNOTS = [
    (0, 1), (1, 3), (4, 6), (7, 10), (11, 16), (17, 21), (22, 27),
    (28, 33), (34, 40), (41, 47), (48, 55), (56, 63), (64, None),
]  # fmt: skip

KNOTS_IN_TEXT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:knot|kn\b|kt\b)", re.IGNORECASE)


def beaufort_force(knots: float) -> int:
    """The force whose range contains this speed. 10.5 kn falls between 7-10 and 11-16,
    so ranges are read as 'from this force's lower limit up to the next one's'."""
    force = 0
    for number, (low, _) in enumerate(BEAUFORT_KNOTS):
        if knots >= low:
            force = number
    return force


def knots_range(force: int) -> str:
    low, high = BEAUFORT_KNOTS[force]
    return f"{low}+ knot" if high is None else f"{low}-{high} knot"


def add_beaufort_to_text(text: str) -> str:
    """'25 knot rüzgâr' → '25 knot (6 Beaufort) rüzgâr', so search can match the scale."""

    def annotate(match: re.Match) -> str:
        knots = float(match.group(1).replace(",", "."))
        return f"{match.group(0)} ({beaufort_force(knots)} Beaufort)"

    return KNOTS_IN_TEXT.sub(annotate, text)
