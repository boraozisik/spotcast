"""Tests for the agent loop. The LLM is replaced by a script of pre-written replies."""

import json
from datetime import datetime

import pytest

from spotcast import agent
from spotcast.agent import (
    UserProfile,
    decide_profile,
    read_profile,
    run_agent,
    schema_for_llm,
)
from spotcast.tools import TOOLS_BY_NAME, ProfileArgs

NOW = datetime(2026, 11, 5, 8, 0)  # Thursday


def reads(sport="windsurf", level="intermediate"):
    """A fake profile reader: pretends the user said this sport and level."""
    return lambda question: UserProfile(sport=sport, level=level)


def scripted_llm(*replies):
    """A fake LLM that answers with the given replies in order and records what it saw."""
    seen = []

    def llm(messages, tools):
        seen.append({"messages": [dict(m) for m in messages], "tools": tools})
        return replies[len(seen) - 1]

    llm.seen = seen
    return llm


class FakeTool:
    """Stands in for a real tool and remembers the arguments it got."""

    def __init__(self, result, args=ProfileArgs):
        self.result, self.args, self.received = result, args, None

    def call(self, arguments):
        self.received = arguments
        return self.result


def use_fake(monkeypatch, name, result):
    fake = FakeTool(result)
    monkeypatch.setitem(TOOLS_BY_NAME, name, fake)
    return fake


def tool_call(tool_name, **arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": tool_name, "arguments": arguments}}],
    }


def answer(text):
    return {"role": "assistant", "content": text}


def test_date_table_is_written_by_code():
    table = agent.date_table(NOW)
    assert "5 Kasım Perşembe = 2026-11-05 today" in table
    assert "7 Kasım Cumartesi = 2026-11-07" in table


def test_loop_runs_the_tool_and_gives_the_result_back(monkeypatch):
    use_fake(monkeypatch, "find_location", {"places": ["Mersin"]})
    llm = scripted_llm(tool_call("find_location", name="Mersin"), answer("Bulundu."))

    result = run_agent("Mersin nerede?", now=NOW, llm=llm, profile_reader=reads())

    assert result.answer == "Bulundu."
    assert [s.tool for s in result.steps] == ["find_location"]
    tool_message = llm.seen[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"]) == {"places": ["Mersin"]}


def test_unknown_tool_becomes_an_error_message():
    llm = scripted_llm(tool_call("delete_database"), answer("Yapamıyorum."))
    result = run_agent("?", now=NOW, llm=llm, profile_reader=reads())
    assert "There is no tool called delete_database" in result.steps[0].result["error"]


def test_the_loop_stops_after_max_steps():
    llm = scripted_llm(*[tool_call("delete_database")] * 3)
    result = run_agent("?", now=NOW, llm=llm, profile_reader=reads(), max_steps=3)
    assert result.stopped_early
    assert len(result.steps) == 3


def test_unsupported_activity_is_answered_by_code_without_the_agent():
    llm = scripted_llm()  # would fail if it were called

    result = run_agent(
        "Yarın dalışa gideceğim", now=NOW, llm=llm, profile_reader=reads(sport="other")
    )

    assert "henüz öneri veremiyorum" in result.answer
    assert "kano" in result.answer  # says what IS supported
    assert llm.seen == []  # the agent never ran


@pytest.mark.parametrize(
    ("sport", "level", "expected"),
    [
        ("unknown", "beginner", "Sporunu belirtmediğin için"),
        ("sup", "unknown", "Seviyeni belirtmediğin için"),
        ("unknown", "unknown", "Sporunu ve seviyeni belirtmediğin için"),
    ],
)
def test_code_says_which_assumption_it_made(sport, level, expected):
    who = decide_profile(UserProfile(sport=sport, level=level))
    assert who.refusal is None
    assert expected in who.level_note


def test_beginner_is_supported_now(monkeypatch):
    fake = use_fake(monkeypatch, "recommend_spots", {"windows_best_first": []})
    llm = scripted_llm(tool_call("recommend_spots"), answer("Tamam."))

    run_agent(
        "Yeni başlıyorum", now=NOW, llm=llm, profile_reader=reads(level="beginner")
    )

    assert fake.received["level"] == "beginner"


def test_llm_cannot_change_the_level(monkeypatch):
    fake = use_fake(monkeypatch, "recommend_spots", {"windows_best_first": []})
    llm = scripted_llm(tool_call("recommend_spots", level="advanced"), answer("Tamam."))

    run_agent("Orta seviyedeyim", now=NOW, llm=llm, profile_reader=reads())

    assert fake.received["level"] == "intermediate"  # code overwrote the LLM's value


def test_llm_does_not_even_see_sport_and_level():
    real = TOOLS_BY_NAME["check_conditions"]
    properties = schema_for_llm(real)["function"]["parameters"]["properties"]
    assert "level" not in properties and "sport" not in properties
    assert "level" in real.schema()["function"]["parameters"]["properties"]


def test_unknown_level_adds_a_note_and_the_caution(monkeypatch):
    use_fake(monkeypatch, "recommend_spots", {"windows_best_first": []})
    llm = scripted_llm(tool_call("recommend_spots"), answer("Cumartesi Alaçatı."))

    result = run_agent(
        "Nereye gideyim?", now=NOW, llm=llm, profile_reader=reads(level="unknown")
    )

    assert "orta seviye rüzgâr sörfü içindir" in result.answer
    assert "kendin kontrol et" in result.answer


def test_uncertainty_notes_are_added_by_code_even_if_the_llm_skips_them(monkeypatch):
    use_fake(
        monkeypatch,
        "check_conditions",
        {"days_ahead": 6, "label_counts": {"unknown": 5}, "warnings": ["..."]},
    )
    llm = scripted_llm(tool_call("check_conditions"), answer("Rüzgâr 12 knot."))

    result = run_agent("Gölde olur mu?", now=NOW, llm=llm, profile_reader=reads())

    assert result.llm_text == "Rüzgâr 12 knot."  # the LLM said nothing about either
    assert "dalga verisi yok" in result.answer
    assert "6 gün sonrası için tahmin" in result.answer
    # the order: LLM text, then uncertainty, then the general caution
    assert result.answer.index("6 gün") < result.answer.index("kendin kontrol et")


def test_knowledge_answer_quotes_the_source_word_for_word(monkeypatch):
    excerpt = {
        "title": "Dip Akıntıları",
        "section": "Ne yapmalı",
        "text": "Akıntıyla\nmücadele etme.",
    }
    use_fake(monkeypatch, "search_knowledge", {"excerpts": [excerpt]})
    # Even if the LLM gets it wrong, the original text is shown under the answer
    llm = scripted_llm(
        tool_call("search_knowledge", question="dip"), answer("Mücadele et!")
    )

    result = run_agent("Dip akıntısı?", now=NOW, llm=llm, profile_reader=reads())

    assert result.answer.startswith("Mücadele et!")
    assert "«Akıntıyla mücadele etme.»" in result.answer
    assert "kendin kontrol et" not in result.answer  # no conditions were given


@pytest.mark.parametrize(
    ("question", "sport", "level"),
    [
        ("Cumartesi Mersin'de rüzgâr sörfü yapacağım", "windsurf", None),
        ("Rüzgar sörfü, orta seviyedeyim", "windsurf", "intermediate"),
        ("İlk kez deneyeceğim, windsurf için Akyaka nasıl?", "windsurf", "beginner"),
        ("Kite için ileri seviye biri olarak soruyorum", "kitesurf", "advanced"),
        ("Kano ile yeni başlayan biriyim", "kayak", "beginner"),
        ("Ayakta kürek sörfü, orta seviye", "sup", "intermediate"),
        ("Kayak tatili için orta seviyeyim", None, "intermediate"),  # kayak = skiing
    ],
)
def test_known_words_are_read_by_code(question, sport, level):
    calls = []

    def fake_llm(messages, output_model, **kwargs):
        calls.append(messages)
        return UserProfile(sport="unknown", level="unknown")

    profile = read_profile(question, ask=fake_llm)

    assert profile.sport == (sport or "unknown")
    assert profile.level == (level or "unknown")
    # The LLM is only asked when code can't read something
    assert bool(calls) == (sport is None or level is None)


def test_nearby_answers_get_distances_and_guide_notes_from_code(monkeypatch):
    windows = [
        {"spot": "Seferihisar Sığacık", "distance_km": 39, "local_guide": False},
        {"spot": "Alaçatı", "distance_km": 69, "local_guide": True},
    ]
    use_fake(monkeypatch, "recommend_near", {"windows_best_first": windows})
    llm = scripted_llm(
        tool_call("recommend_near", latitude=38.4, longitude=27.1),
        answer("Her iki yerde de yerel rehber var."),  # the mistake an LLM once made
    )

    result = run_agent(
        "İzmir'deyim, nereye gideyim?", now=NOW, llm=llm, profile_reader=reads()
    )

    assert "Kuş uçuşu mesafe: Seferihisar Sığacık 39 km, Alaçatı 69 km" in result.answer
    assert "Seferihisar Sığacık için yerel rehberimiz yok" in result.answer
    assert "Alaçatı için" not in result.answer  # it has a guide
