"""Tests for the LLM explanation. The real LLM is never called here."""

import json
from datetime import datetime

import pandas as pd

from spotcast.explain import Explanation, build_messages, explain_recommendation
from spotcast.profiles import WINDSURF_INTERMEDIATE


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

    assert result.headline == "h"
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

    assert "görünmüyor" in result.headline
