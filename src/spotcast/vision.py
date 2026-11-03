"""Estimating wind from a photo of the sea with a local vision model (multimodal LLM).

Sailors have judged the wind by looking at the sea for 200 years: that is what the Beaufort
scale was made for. Here a vision model does the looking.

Run `uv run spotcast-vision` to test it on photos whose Beaufort force is known.
"""

import argparse
import base64
import csv
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from spotcast.config import PROJECT_ROOT
from spotcast.llm import VISION_MODEL, LLMOutputError, chat

SAMPLES_DIR = PROJECT_ROOT / "samples" / "photos" / "beaufort"

# Wind speed for each Beaufort force, in knots (WMO). Code converts; the model never does.
BEAUFORT_KNOTS = [
    (0, 1), (1, 3), (4, 6), (7, 10), (11, 16), (17, 21), (22, 27),
    (28, 33), (34, 40), (41, 47), (48, 55), (56, 63), (64, None),
]  # fmt: skip

# What the sea looks like at each force (shortened from the WMO sea criteria). Giving the
# model this reference is like giving a person the chart instead of asking from memory.
SEA_CRITERIA = """Beaufort sea criteria:
0: sea like a mirror.
1: ripples like scales, no foam crests.
2: small wavelets, crests look glassy and do not break.
3: large wavelets, crests begin to break, scattered white horses.
4: small waves becoming longer, fairly frequent white horses.
5: moderate waves, clearly longer; many white horses, some spray.
6: large waves begin to form; white foam crests everywhere, some spray.
7: sea heaps up; white foam from breaking waves blown in streaks.
8: moderately high waves; crests break into spindrift; well-marked foam streaks.
9: high waves; dense foam streaks; crests topple and roll over; spray reduces visibility.
10: very high waves with long overhanging crests; sea surface looks white with foam.
11: exceptionally high waves; sea completely covered with long white patches of foam.
12: air filled with foam and spray; sea completely white."""

SYSTEM_PROMPT = f"""You judge wind strength from photos of the sea, like a sailor would.
Look only at the water: the size of the waves, whether crests break, and how much white foam
there is. Ignore ships, people and the sky.

First describe what you see, then choose the Beaufort force that matches it best.

{SEA_CRITERIA}"""


class SeaAssessment(BaseModel):
    """The answer shape. The order matters: the model writes observations BEFORE the number,
    so the estimate is based on what it has just described."""

    whitecaps: Literal["none", "few", "many", "everywhere"] = Field(
        description="How much white foam is on the wave crests"
    )
    wave_size: Literal["flat", "ripples", "small", "moderate", "large", "very_large"]
    foam_streaks: bool = Field(description="Foam blown in streaks along the wind")
    description: str = Field(
        max_length=200, description="One short sentence about the sea surface"
    )
    beaufort: int = Field(ge=0, le=12, description="Best matching Beaufort force, 0-12")
    confidence: Literal["low", "medium", "high"]


@dataclass(frozen=True)
class PhotoResult:
    file: str
    actual: int
    assessment: SeaAssessment | None  # None: the model gave no usable answer

    @property
    def error(self) -> int | None:
        if self.assessment is None:
            return None
        return self.assessment.beaufort - self.actual


def knots_range(beaufort: int) -> str:
    low, high = BEAUFORT_KNOTS[beaufort]
    return f"{low}+ knot" if high is None else f"{low}-{high} knot"


def encode_image(path: Path) -> str:
    """Ollama takes images as base64 text inside the JSON request."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_messages(image_b64: str, with_criteria: bool = True) -> list[dict]:
    system = (
        SYSTEM_PROMPT
        if with_criteria
        else SYSTEM_PROMPT.split("\n\n" + SEA_CRITERIA)[0]
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "Assess the sea in this photo.",
            "images": [image_b64],
        },
    ]


def assess_photo(
    path: Path,
    with_criteria: bool = True,
    ask: Callable[..., SeaAssessment] = chat,
    model: str = VISION_MODEL,
) -> SeaAssessment:
    messages = build_messages(encode_image(path), with_criteria)
    return ask(
        messages,
        output_model=SeaAssessment,
        model=model,
        temperature=0,
        max_tokens=300,  # the whole answer is short; more means the model is looping
    )


def read_labels(directory: Path = SAMPLES_DIR) -> dict[str, int]:
    """The answer key: file name → true Beaufort force. The model never sees it."""
    with open(directory / "labels.csv", encoding="utf-8") as f:
        return {row["file"]: int(row["beaufort"]) for row in csv.DictReader(f)}


def evaluate(
    directory: Path = SAMPLES_DIR,
    with_criteria: bool = True,
    ask: Callable[..., SeaAssessment] = chat,
    model: str = VISION_MODEL,
) -> list[PhotoResult]:
    results = []
    for file, actual in sorted(read_labels(directory).items()):
        try:
            assessment = assess_photo(directory / file, with_criteria, ask, model)
        except LLMOutputError:
            assessment = None  # one bad answer must not stop the whole evaluation
        results.append(PhotoResult(file, actual, assessment))
    return results


def summarize(results: list[PhotoResult]) -> dict[str, float]:
    """A missing answer counts as wrong for exact/within_1; MAE and bias use real answers."""
    errors = [r.error for r in results if r.error is not None]
    n = len(results)
    return {
        "exact": sum(e == 0 for e in errors) / n,
        "within_1": sum(abs(e) <= 1 for e in errors) / n,
        "mae": sum(abs(e) for e in errors) / len(errors) if errors else float("nan"),
        # + means the model sees more wind than there is
        "bias": sum(errors) / len(errors) if errors else float("nan"),
        "failed": (n - len(errors)) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test vision models on sea photos.")
    parser.add_argument(
        "models", nargs="*", default=[VISION_MODEL], help="Ollama model names"
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the summary")
    args = parser.parse_args()

    if not SAMPLES_DIR.exists():
        print(f"No photos in {SAMPLES_DIR}. See CLAUDE.md, stage 6.")
        return

    summary = {}
    for model in args.models:
        for with_criteria in (False, True):
            name = f"{model} {'+ criteria' if with_criteria else ''}".strip()
            started = time.monotonic()
            results = evaluate(with_criteria=with_criteria, model=model)
            seconds = time.monotonic() - started
            if not args.quiet:
                print(f"\n=== {name} ===")
                for r in results:
                    a = r.assessment
                    if a is None:
                        print(f"! {r.file}: actual {r.actual:2d}  no usable answer")
                        continue
                    mark = "✓" if r.error == 0 else ("~" if abs(r.error) == 1 else "✗")
                    print(
                        f"{mark} {r.file}: actual {r.actual:2d}  predicted {a.beaufort:2d} "
                        f"({knots_range(a.beaufort)}, {a.confidence})  "
                        f"whitecaps={a.whitecaps}, waves={a.wave_size}"
                    )
            summary[name] = summarize(results) | {"seconds": seconds}

    print("\n=== Summary ===")
    for name, s in summary.items():
        print(
            f"{name:28s} exact {s['exact']:4.0%}  within ±1 {s['within_1']:4.0%}  "
            f"MAE {s['mae']:.2f}  bias {s['bias']:+.2f}  failed {s['failed']:3.0%}  "
            f"({s['seconds']:.0f} s)"
        )
