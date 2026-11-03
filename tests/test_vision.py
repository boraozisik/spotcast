"""Tests for the sea photo assessment. The real vision model is never called here."""

import pytest
from PIL import Image
from pydantic import ValidationError

from spotcast.llm import LLMOutputError
from spotcast.vision import (
    SEA_CRITERIA,
    PhotoResult,
    SeaAssessment,
    build_messages,
    evaluate,
    knots_range,
    summarize,
)


def assessment(beaufort: int) -> SeaAssessment:
    return SeaAssessment(
        whitecaps="few",
        wave_size="small",
        foam_streaks=False,
        description="test",
        beaufort=beaufort,
        confidence="medium",
    )


def test_knots_range_comes_from_code():
    assert knots_range(0) == "0-1 knot"
    assert knots_range(4) == "11-16 knot"
    assert knots_range(12) == "64+ knot"


def test_beaufort_outside_0_to_12_is_rejected():
    with pytest.raises(ValidationError):
        assessment(13)


def test_image_goes_into_the_user_message_and_criteria_can_be_switched_off():
    with_table = build_messages("BASE64", with_criteria=True)
    without = build_messages("BASE64", with_criteria=False)

    assert with_table[1]["images"] == ["BASE64"]
    assert SEA_CRITERIA in with_table[0]["content"]
    assert SEA_CRITERIA not in without[0]["content"]


def test_observations_come_before_the_number():
    # Structured output is written in field order: describe first, then decide
    fields = list(SeaAssessment.model_fields)
    assert fields.index("description") < fields.index("beaufort")


def test_summarize():
    results = [
        PhotoResult("a.jpg", 4, assessment(4)),  # exact
        PhotoResult("b.jpg", 4, assessment(5)),  # +1
        PhotoResult("c.jpg", 10, assessment(6)),  # -4: a storm seen as much calmer
    ]
    s = summarize(results)
    assert s["exact"] == pytest.approx(1 / 3)
    assert s["within_1"] == pytest.approx(2 / 3)
    assert s["mae"] == pytest.approx(5 / 3)
    assert s["bias"] == pytest.approx(-1.0)
    assert s["failed"] == 0


def test_a_broken_answer_is_counted_as_failed_not_a_crash(tmp_path):
    Image.new("RGB", (8, 8), "blue").save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 8), "blue").save(tmp_path / "b.jpg")
    (tmp_path / "labels.csv").write_text("file,beaufort\na.jpg,4\nb.jpg,4\n")
    answers = iter([LLMOutputError("cut off"), assessment(4)])

    def flaky_vision(*args, **kwargs):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    s = summarize(evaluate(tmp_path, ask=flaky_vision))

    assert s["failed"] == 0.5
    assert s["exact"] == 0.5  # the failed photo counts as not correct
    assert s["mae"] == 0  # but the error average only uses real answers


def test_evaluate_uses_the_answer_key(tmp_path):
    for name in ("calm.jpg", "storm.jpg"):
        Image.new("RGB", (8, 8), "blue").save(tmp_path / name)
    (tmp_path / "labels.csv").write_text("file,beaufort\ncalm.jpg,1\nstorm.jpg,9\n")

    def fake_vision(messages, output_model, model, temperature, max_tokens):
        assert messages[1]["images"]  # the image really is sent
        return assessment(3)

    results = evaluate(tmp_path, ask=fake_vision)

    assert [(r.file, r.actual, r.error) for r in results] == [
        ("calm.jpg", 1, 2),
        ("storm.jpg", 9, -6),
    ]
