"""Tests for the LLM explanation. The real LLM is never called here."""

import json
from datetime import datetime

import pandas as pd

from spotcast.explain import (
    MAX_SOURCE_DISTANCE,
    ExplainResult,
    Explanation,
    Source,
    Tip,
    build_messages,
    explain_recommendation,
    gather_sources,
    keep_cited_tips,
)
from spotcast.profiles import WINDSURF_INTERMEDIATE
from spotcast.rag import Chunk, SearchResult


def make_windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "spot": ["Alaçatı"],
            "start": [pd.Timestamp("2026-11-05 09:00")],
            "end": [pd.Timestamp("2026-11-05 18:00")],
            "hours": [10],
            "good_hours": [8],
            "avg_wind_kn": [16.58],
            "max_wave_m": [1.02],
            "score": [24.5],
        }
    )


def test_code_writes_the_dates_not_the_llm():
    messages = build_messages(
        make_windows(), WINDSURF_INTERMEDIATE, datetime(2026, 11, 2, 18)
    )
    facts = json.loads(messages[-1]["content"])  # the real question comes last

    window = facts["windows_best_first"][0]
    assert facts["today_is"] == "2 Kasım Pazartesi"
    assert window["day"] == "5 Kasım Perşembe"
    assert window["hours"] == "09:00-18:00"
    assert window["marginal_hours"] == 2
    assert window["max_wave_metres"] == 1.02  # not rounded down to an "ideal" 1.0
    assert window["wind_vs_ideal"] == "inside"  # 16.6 kn is within 15-25
    assert window["waves_vs_comfortable"] == "above"  # 1.02 m > 1.0 m


def test_system_prompt_forbids_inventing_and_calling_it_safe():
    system = build_messages(
        make_windows(), WINDSURF_INTERMEDIATE, datetime(2026, 11, 2)
    )[0]
    assert system["role"] == "system"
    assert "ONLY the facts" in system["content"]
    assert "Never describe conditions as safe" in system["content"]


def test_llm_receives_messages_and_answer_shape():
    seen = {}

    def fake_llm(messages, output_model):
        seen["messages"], seen["output_model"] = messages, output_model
        return Explanation(headline="h", reason="r", alternatives=[])

    result = explain_recommendation(
        make_windows(), WINDSURF_INTERMEDIATE, datetime(2026, 11, 2), ask=fake_llm
    )

    assert result.explanation.headline == "h"
    assert seen["output_model"] is Explanation
    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user", "assistant", "user"]  # rules, example, question


def test_no_windows_means_no_llm_call():
    def exploding_llm(*args, **kwargs):
        raise AssertionError(
            "the LLM must not be called when there is nothing to explain"
        )

    empty = make_windows().iloc[0:0]
    result = explain_recommendation(
        empty, WINDSURF_INTERMEDIATE, datetime(2026, 11, 2), ask=exploding_llm
    )

    assert "görünmüyor" in result.explanation.headline
    assert result.sources == []


def result_for(
    section: str, distance: float, spot: str | None = "Alaçatı"
) -> SearchResult:
    chunk = Chunk(
        file="spots/alacati.md" if spot else "beaufort.md",
        title="rehber",
        section=section,
        spot=spot,
        source="test",
        content=f"{section} metni",
    )
    return SearchResult(chunk=chunk, distance=distance)


def test_sources_are_searched_with_context_deduplicated_and_filtered():
    queries = []

    def fake_retrieve(question, spots):
        queries.append((question, spots))
        return [
            result_for("Alaçatı: tehlikeler", 0.30),
            result_for("Alaçatı: rüzgâr", 0.35),
            result_for("çok alakasız", MAX_SOURCE_DISTANCE + 0.1),
        ]

    sources = gather_sources(make_windows(), WINDSURF_INTERMEDIATE, fake_retrieve)

    # Queries name the spot and the sport, and search only that spot (plus general docs)
    assert all("Alaçatı" in q or "windsurf" in q for q, _ in queries)
    assert all(spots == ["Alaçatı"] for _, spots in queries)
    # Same chunk found twice is listed once; the unrelated one is dropped; numbered from 1
    assert [s.section for s in sources] == ["Alaçatı: tehlikeler", "Alaçatı: rüzgâr"]
    assert [s.id for s in sources] == [1, 2]


def test_sources_reach_the_prompt():
    sources = [Source(1, "rehber", "Alaçatı: tehlikeler", "test", "Kalabalık olur.")]
    messages = build_messages(
        make_windows(), WINDSURF_INTERMEDIATE, datetime(2026, 11, 2), sources
    )
    facts = json.loads(messages[-1]["content"])
    assert facts["sources"] == [
        {
            "id": 1,
            "title": "rehber",
            "section": "Alaçatı: tehlikeler",
            "text": "Kalabalık olur.",
        }
    ]


def test_tips_with_missing_or_invented_citations_are_dropped():
    sources = [Source(1, "a", "b", "c", "d"), Source(2, "a", "b", "c", "d")]
    answer = Explanation(
        headline="h",
        reason="r",
        alternatives=[],
        tips=[
            Tip(text="iyi", source_ids=[2, 1, 2]),
            Tip(text="kaynaksız", source_ids=[]),
            Tip(text="uydurma kaynak", source_ids=[7]),
        ],
    )

    kept = keep_cited_tips(answer, sources).tips

    assert [t.text for t in kept] == ["iyi"]
    assert kept[0].source_ids == [1, 2]


def test_only_cited_sources_are_shown():
    sources = [Source(1, "a", "b", "c", "d"), Source(2, "a", "b", "c", "d")]
    answer = Explanation(
        headline="h", reason="r", alternatives=[], tips=[Tip(text="t", source_ids=[2])]
    )
    assert [s.id for s in ExplainResult(answer, sources).cited_sources] == [2]
