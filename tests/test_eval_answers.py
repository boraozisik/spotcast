"""Tests for the answer evaluation: the rule checks must be right before we trust scores."""

from datetime import datetime

import pytest

from spotcast.agent import AgentResult, Step
from spotcast.evals import answers
from spotcast.evals.answers import (
    AnswerCase,
    Faithfulness,
    JudgeCase,
    check_answer,
    frozen_tools,
    invented_numbers,
    numbers_with_units,
    says_safe,
    suitable_claims,
    summarize_judge,
)
from spotcast.tools import TOOLS_BY_NAME


@pytest.mark.parametrize(
    "text, found",
    [
        ("Koşullar güvenli görünüyor.", ["güvenli"]),
        ("Bu saatler güvenlidir.", ["güvenlidir"]),
        ("Güvenle sörf yapabilirsin.", ["Güvenle"]),
        ("Güvenlik kurallarına uy.", []),  # "safety" is fine, "safe" is not
        ("Rüzgâr sörfü güvenlik notları", []),
    ],
)
def test_says_safe(text, found):
    assert says_safe(text) == found


def test_suitable_claims_ignore_negations():
    text = (
        "Cumartesi Mersin uygun değil. Uygun saat yok. Cumartesi uygun mu? "
        "Yarın uygun saat bulunamadı. Saat 12-15 arası uygun."
    )
    assert suitable_claims(text) == ["Saat 12-15 arası uygun."]


def test_numbers_with_units():
    text = (
        "Rüzgâr 1,8 ile 6 knot, en yüksek dalga 0.9 metre; öğlen 16-19 kn. Saat 12:00."
    )
    assert numbers_with_units(text) == [
        (1.8, "knot"),
        (6.0, "knot"),
        (0.9, "metre"),
        (16.0, "knot"),
        (19.0, "knot"),
    ]


def test_invented_numbers_allow_rounding_but_not_new_values():
    results = [{"daylight_hours": [{"wind_knots": 12.7, "wave_metres": 0.84}]}]
    text = "Rüzgâr 13 knot, dalga 0.8 metre. Öğleden sonra 25 knot olacak."
    assert invented_numbers(text, results, question="") == ["25 knot"]


def test_numbers_in_knowledge_text_and_question_count_as_known():
    results = [{"excerpts": [{"text": "4 Beaufort: 11-16 knot"}]}]
    text = "4 Beaufort 11-16 knot demektir, sorduğun 30 knot ise çok fazla."
    assert invented_numbers(text, results, question="30 knot olur mu?") == []


def test_frozen_tools_return_fixtures_and_restore_the_real_ones():
    real = TOOLS_BY_NAME["find_location"]
    with frozen_tools({"find_location": {"places": ["X"]}}):
        assert TOOLS_BY_NAME["find_location"].call({"name": "Mersin"}) == {
            "places": ["X"]
        }
        other = TOOLS_BY_NAME["recommend_spots"].call({})
        assert "error" in other  # tools without a fixture can't reach the internet
        bad = TOOLS_BY_NAME["find_location"].call({"nam": "Mersin"})
        assert "Invalid arguments" in bad["error"]  # argument checks are still real
    assert TOOLS_BY_NAME["find_location"] is real


def conditions_step(good=0, day="2026-11-07"):
    return Step(
        "check_conditions",
        {"latitude": 1, "longitude": 2, "day": day},
        {"label_counts": {"good": good, "marginal": 3}, "daylight_hours": []},
        0.1,
    )


def by_name(checks):
    return {c.name: c.passed for c in checks}


def test_check_answer_catches_uygun_without_good_hours():
    case = AnswerCase(
        "x", "Cumartesi?", tools={}, must_call=["check_conditions"],
        arguments={"check_conditions": {"day": "2026-11-07"}},
        must_mention=["sınırda"],
    )  # fmt: skip
    result = AgentResult("", [conditions_step(good=0)], llm_text="Cumartesi uygun.")

    checks = by_name(check_answer(case, result, ask_judge=None))

    assert checks["tools"] and checks["check_conditions arguments"]
    assert not checks["no 'uygun' without good hours"]
    assert not checks["mentions"]  # "sınırda" is missing


def test_safe_word_rule_is_for_condition_answers_only():
    text = "Suya girmemek en güvenli davranıştır."
    advice = AnswerCase("a", "q", tools={"search_knowledge": {}})
    conditions = AnswerCase("c", "q", tools={"check_conditions": {}})
    result = AgentResult("", [], llm_text=text)

    assert "no 'güvenli'" not in by_name(check_answer(advice, result, ask_judge=None))
    assert not by_name(check_answer(conditions, result, ask_judge=None))["no 'güvenli'"]


def test_check_answer_wrong_day_fails():
    case = AnswerCase(
        "x", "q", tools={}, arguments={"check_conditions": {"day": "2026-11-07"}}
    )
    result = AgentResult("", [conditions_step(day="2026-11-08")], llm_text="Sınırda.")
    assert not by_name(check_answer(case, result, ask_judge=None))[
        "check_conditions arguments"
    ]


def test_judge_sees_the_excerpts_the_agent_got():
    seen = {}

    def fake_judge(source, answer):
        seen.update(source=source, answer=answer)
        return Faithfulness(problems=["ters"], contradicts=True, adds=False)

    step = Step("search_knowledge", {}, {"excerpts": [{"text": "Önce X, sonra Y."}]}, 0)
    case = AnswerCase("x", "q", tools={}, judge=True)
    result = AgentResult("", [step], llm_text="X yapma, Y yap.")

    checks = by_name(check_answer(case, result, ask_judge=fake_judge))

    assert seen == {"source": "Önce X, sonra Y.", "answer": "X yapma, Y yap."}
    assert not checks["judge: no contradiction"]
    assert checks["judge: nothing added"]


def test_run_case_uses_frozen_tools(monkeypatch):
    def fake_agent(question, now):
        result = TOOLS_BY_NAME["check_conditions"].call(
            {"latitude": 1, "longitude": 2, "day": "2026-11-07"}
        )
        step = Step("check_conditions", {"day": "2026-11-07"}, result, 0)
        return AgentResult("", [step], llm_text="Sınırda, uygun değil.")

    fixture = {"label_counts": {"marginal": 2}}
    case = AnswerCase("x", "q", tools={"check_conditions": fixture})
    r = answers.run_case(case, datetime(2026, 11, 5), agent=fake_agent, ask_judge=None)

    assert r.result.steps[0].result == fixture
    assert r.passed


def test_summarize_judge_counts_missed_contradictions():
    def pair(human, judged):
        case = JudgeCase("n", "s", "a", contradicts=human, adds=False)
        return case, Faithfulness(problems=[], contradicts=judged, adds=False)

    s = summarize_judge([pair(True, False), pair(True, True), pair(False, True)])
    assert s["contradicts"] == {
        "agreement": pytest.approx(1 / 3),
        "missed": 1,
        "false_alarms": 1,
    }


def test_answer_cases_file_is_valid():
    now, cases = answers.load_answer_cases()
    assert now.weekday() == 3  # the date table and expected days assume a Thursday
    names = set(TOOLS_BY_NAME)
    for case in cases:
        assert set(case.tools) <= names, case.id
        assert set(case.must_call) <= set(case.tools), case.id
        assert not case.judge or "search_knowledge" in case.tools, case.id
    assert len({c.id for c in cases}) == len(cases)
