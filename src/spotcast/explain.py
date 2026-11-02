"""Turning the recommendation table into a short Turkish explanation with the LLM.

Rule of thumb: let code do what code is good at (dates, numbers, sorting, safety) and let
the LLM do only what it is good at (turning facts into friendly language).
"""

import json
from collections.abc import Callable
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from spotcast.llm import chat
from spotcast.profiles import ActivityProfile

TURKISH_DAYS = [
    "Pazartesi",
    "Salı",
    "Çarşamba",
    "Perşembe",
    "Cuma",
    "Cumartesi",
    "Pazar",
]
TURKISH_MONTHS = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]  # fmt: skip

SYSTEM_PROMPT = """You are SpotCast, a careful assistant for water sports in Turkey.
Write in Turkish. Be short, friendly and concrete.

You get facts as JSON. "windows_best_first" lists time windows, the best one first.

Fill the answer like this:
- headline: one sentence recommending the FIRST window: spot, day and hours.
- reason: two sentences about the first window: average wind in knot, highest wave in metre,
  how many hours are "uygun" and "sınırda", and why some hours are only "sınırda"
  (use wind_vs_ideal and waves_vs_comfortable).
- alternatives: one sentence for each OTHER window (at most two): spot, day, hours,
  average wind, and its own good_hours and marginal_hours counts. Empty list if there is
  only one window.

Rules:
- Use ONLY the facts given. Never invent spots, days, hours or numbers.
- Copy spot names, day names, dates, hours and hour counts exactly as given for THAT
  window. The example answer only shows the style; never reuse its numbers or phrases.
- Always write units: knot, metre.
- Say "uygun" for good hours and "sınırda" for marginal hours.
- Never describe conditions as safe ("güvenli")."""

# One worked example (few-shot prompting): showing the model an answer works better than
# describing it. The numbers are made up and never shown to users.
EXAMPLE_FACTS = {
    "sport": "windsurf",
    "level": "intermediate",
    "windows_best_first": [
        {
            "spot": "Akyaka",
            "day": "12 Haziran Cuma",
            "hours": "11:00-17:00",
            "good_hours": 5,
            "marginal_hours": 2,
            "avg_wind_knots": 18.2,
            "max_wave_metres": 0.6,
            "wind_vs_ideal": "inside",
            "waves_vs_comfortable": "within",
        },
        {
            "spot": "Alaçatı",
            "day": "13 Haziran Cumartesi",
            "hours": "10:00-14:00",
            "good_hours": 2,
            "marginal_hours": 3,
            "avg_wind_knots": 13.1,
            "max_wave_metres": 0.4,
            "wind_vs_ideal": "below",
            "waves_vs_comfortable": "within",
        },
    ],
}
EXAMPLE_ANSWER = {
    "headline": "En iyi seçenek 12 Haziran Cuma, 11:00-17:00 arası Akyaka.",
    "reason": (
        "Ortalama rüzgâr 18,2 knot, dalga en fazla 0,6 metre; 5 saat uygun, 2 saat "
        "sınırda. Sınırda saatlerde rüzgâr ideal aralığın kenarında kalıyor."
    ),
    "alternatives": [
        (
            "13 Haziran Cumartesi 10:00-14:00 Alaçatı: ortalama 13,1 knot, 2 saat uygun, "
            "3 saat sınırda; rüzgâr çoğu saatte ideal aralığın altında."
        )
    ],
}

# Safety text is fixed, written by us, never generated: an LLM could word it wrongly.
CAUTION = (
    "Tahminler değişebilir. Suya girmeden önce sahilde rüzgârı ve dalgayı kendin "
    "kontrol et; emin olmadığında girme."
)


class Explanation(BaseModel):
    """The shape we force the LLM's answer into."""

    headline: str = Field(
        description="One sentence: the first window's spot, day, hours"
    )
    reason: str = Field(
        description="Two sentences about the first window, with numbers"
    )
    alternatives: list[str] = Field(
        description="One sentence per other window, max two"
    )


def turkish_date(when: datetime) -> str:
    return f"{when.day} {TURKISH_MONTHS[when.month - 1]} {TURKISH_DAYS[when.weekday()]}"


def describe_window(row: pd.Series, profile: ActivityProfile) -> dict:
    """Facts for one window. Code does dates and comparisons; the LLM only words them."""
    start, end = row["start"], row["end"]
    wind, wave = float(row["avg_wind_kn"]), float(row["max_wave_m"])
    if wind < profile.good_wind_min_kn:
        wind_vs_ideal = "below"
    elif wind > profile.good_wind_max_kn:
        wind_vs_ideal = "above"
    else:
        wind_vs_ideal = "inside"
    return {
        "spot": row["spot"],
        "day": turkish_date(start),
        "hours": f"{start:%H:%M}-{end:%H:%M}",
        "good_hours": int(row["good_hours"]),
        "marginal_hours": int(row["hours"] - row["good_hours"]),
        "avg_wind_knots": round(wind, 1),
        "max_wave_metres": round(wave, 2),  # 1.02 must not become an "ideal" 1.0
        "wind_vs_ideal": wind_vs_ideal,
        "waves_vs_comfortable": "above" if wave > profile.good_wave_max_m else "within",
    }


def build_messages(
    windows: pd.DataFrame, profile: ActivityProfile, now: datetime
) -> list[dict[str, str]]:
    facts = {
        "sport": profile.sport,
        "level": profile.level,
        "today_is": turkish_date(now),
        "ideal_wind_knots": f"{profile.good_wind_min_kn}-{profile.good_wind_max_kn}",
        "max_comfortable_wave_metres": profile.good_wave_max_m,
        "windows_best_first": [
            describe_window(row, profile) for _, row in windows.iterrows()
        ],
    }

    def as_json(data: dict) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": as_json(EXAMPLE_FACTS)},
        {"role": "assistant", "content": as_json(EXAMPLE_ANSWER)},
        {"role": "user", "content": as_json(facts)},
    ]


def explain_recommendation(
    windows: pd.DataFrame,
    profile: ActivityProfile,
    now: datetime,
    ask: Callable[..., Explanation] = chat,
) -> Explanation:
    """Ask the LLM to explain the windows. With no windows, answer without the LLM."""
    if windows.empty:
        return Explanation(
            headline="Önümüzdeki günlerde tercihlerine uyan bir zaman aralığı görünmüyor.",
            reason="Tahminlerde rüzgâr ya çok zayıf ya da seviyen için fazla güçlü.",
            alternatives=[],
        )
    return ask(build_messages(windows, profile, now), output_model=Explanation)


def render(explanation: Explanation) -> str:
    """Plain text version for the terminal."""
    lines = [explanation.headline, "", explanation.reason]
    if explanation.alternatives:
        lines += [
            "",
            "Diğer seçenekler:",
            *[f"- {a}" for a in explanation.alternatives],
        ]
    lines += ["", f"⚠️  {CAUTION}"]
    return "\n".join(lines)
