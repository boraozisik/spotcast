"""Turning the recommendation table into a short Turkish explanation with the LLM.

Rule of thumb: let code do what code is good at (dates, numbers, sorting, safety, checking
citations) and let the LLM do only what it is good at (turning facts into friendly language).

With a `retrieve` function (RAG), relevant parts of the knowledge documents are added to the
prompt as numbered sources, and the LLM must cite them in its tips.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from spotcast.llm import chat
from spotcast.profiles import ActivityProfile
from spotcast.rag import SearchResult

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

MAX_SOURCE_DISTANCE = 0.5  # farther than this is too unrelated to show the LLM
MAX_SOURCES = (
    5  # keep the prompt short: a small local model loses focus with too much text
)

SYSTEM_PROMPT = """You are SpotCast, a careful assistant for water sports in Turkey.
Write in Turkish. Be short, friendly and concrete.

You get facts as JSON. "windows_best_first" lists time windows, the best one first.
"sources" are numbered excerpts from SpotCast's guides.

Fill the answer like this:
- headline: one sentence recommending the FIRST window: spot, day and hours.
- reason: two sentences about the first window: average wind in knot, highest wave in metre,
  how many hours are "uygun" and "sınırda", and why some hours are only "sınırda"
  (use wind_vs_ideal and waves_vs_comfortable).
- alternatives: one sentence for each OTHER window (at most two): spot, day, hours,
  average wind, and its own good_hours and marginal_hours counts. Empty list if there is
  only one window.
- tips: one to three short, practical tips for the first window, each based ONLY on the
  sources. Put the ids of the sources you used in source_ids. If no source is useful,
  return an empty list.

Rules:
- Use ONLY the facts given. Never invent spots, days, hours, numbers or advice.
- Copy spot names, day names, dates, hours and hour counts exactly as given for THAT
  window. The example answer only shows the style; never reuse its content.
- Always write units: knot, metre.
- Say "uygun" for good hours and "sınırda" for marginal hours.
- Never describe conditions as safe ("güvenli")."""

# One worked example (few-shot prompting). Spots and sources are fictional on purpose, so
# the model can't copy real-looking facts from the example into a real answer.
EXAMPLE_FACTS = {
    "sport": "windsurf",
    "level": "intermediate",
    "windows_best_first": [
        {
            "spot": "Örnek Koyu",
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
            "spot": "Deneme Plajı",
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
    "sources": [
        {
            "id": 1,
            "title": "Örnek Koyu spot rehberi",
            "section": "Örnek Koyu: tehlikeler",
            "text": "Örnek Koyu'nun batı ucunda sığ kayalar vardır; oraya yaklaşma.",
        },
        {
            "id": 2,
            "title": "Örnek güvenlik notları",
            "section": "Mendirek",
            "text": "Mendirek yakınında rüzgâr gölgesi oluşur, yelken aniden boşalır.",
        },
    ],
}
EXAMPLE_ANSWER = {
    "headline": "En iyi seçenek 12 Haziran Cuma, 11:00-17:00 arası Örnek Koyu.",
    "reason": (
        "Ortalama rüzgâr 18,2 knot, dalga en fazla 0,6 metre; 5 saat uygun, 2 saat "
        "sınırda. Sınırda saatlerde rüzgâr ideal aralığın kenarında kalıyor."
    ),
    "alternatives": [
        (
            "13 Haziran Cumartesi 10:00-14:00 Deneme Plajı: ortalama 13,1 knot, 2 saat "
            "uygun, 3 saat sınırda; rüzgâr çoğu saatte ideal aralığın altında."
        )
    ],
    "tips": [
        {"text": "Koyun batı ucundaki sığ kayalardan uzak dur.", "source_ids": [1]},
        {
            "text": "Mendireğe yaklaşınca yelkenin aniden boşalabileceğini unutma.",
            "source_ids": [2],
        },
    ],
}

# Safety text is fixed, written by us, never generated: an LLM could word it wrongly.
CAUTION = (
    "Tahminler değişebilir. Suya girmeden önce sahilde rüzgârı ve dalgayı kendin "
    "kontrol et; emin olmadığında girme."
)

Retriever = Callable[[str, list[str] | None], list[SearchResult]]


class Tip(BaseModel):
    text: str = Field(description="One short practical tip in Turkish")
    source_ids: list[int] = Field(description="Ids of the sources this tip is based on")


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
    tips: list[Tip] = Field(
        default=[], description="Up to three tips, each citing sources"
    )


@dataclass(frozen=True)
class Source:
    """A numbered knowledge excerpt shown to the LLM and to the user."""

    id: int
    title: str
    section: str
    source: str
    text: str


@dataclass(frozen=True)
class ExplainResult:
    explanation: Explanation
    sources: list[Source]  # everything shown to the LLM

    @property
    def cited_sources(self) -> list[Source]:
        """Only the sources a tip actually uses: those are the ones worth showing a user."""
        cited = {i for tip in self.explanation.tips for i in tip.source_ids}
        return [s for s in self.sources if s.id in cited]


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


def gather_sources(
    windows: pd.DataFrame, profile: ActivityProfile, retrieve: Retriever
) -> list[Source]:
    """Search the knowledge base with queries built by code, then number the results.

    Vague questions find the wrong text, so each query names the spot and the sport.
    """
    best = windows.iloc[0]
    queries = [
        (
            f"{best['spot']} {profile.sport}: rüzgâr, dalga ve tehlikeler",
            [best["spot"]],
        ),
        (
            (
                f"{best['avg_wind_kn']:.0f} knot rüzgâr kaç Beaufort, "
                f"{profile.level} {profile.sport} için nasıl"
            ),
            [best["spot"]],
        ),
    ]
    found: dict[tuple[str, str], SearchResult] = {}
    for query, spots in queries:
        for result in retrieve(query, spots):
            key = (result.chunk.file, result.chunk.section)
            if result.distance <= MAX_SOURCE_DISTANCE and key not in found:
                found[key] = result

    closest = sorted(found.values(), key=lambda r: r.distance)[:MAX_SOURCES]
    return [
        Source(
            id=i,
            title=r.chunk.title,
            section=r.chunk.section,
            source=r.chunk.source,
            text=r.chunk.content,
        )
        for i, r in enumerate(closest, start=1)
    ]


def build_messages(
    windows: pd.DataFrame,
    profile: ActivityProfile,
    now: datetime,
    sources: list[Source] | None = None,
) -> list[dict[str, str]]:
    sources = sources or []
    facts = {
        "sport": profile.sport,
        "level": profile.level,
        "today_is": turkish_date(now),
        "ideal_wind_knots": f"{profile.good_wind_min_kn}-{profile.good_wind_max_kn}",
        "max_comfortable_wave_metres": profile.good_wave_max_m,
        "windows_best_first": [
            describe_window(row, profile) for _, row in windows.iterrows()
        ],
        "sources": [
            {"id": s.id, "title": s.title, "section": s.section, "text": s.text}
            for s in sources
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


def keep_cited_tips(explanation: Explanation, sources: list[Source]) -> Explanation:
    """Drop tips that cite nothing or cite a source we never gave. Never trust citations blindly."""
    valid_ids = {s.id for s in sources}
    tips = [
        Tip(text=tip.text, source_ids=sorted(set(tip.source_ids)))
        for tip in explanation.tips
        if tip.source_ids and set(tip.source_ids) <= valid_ids
    ]
    return explanation.model_copy(update={"tips": tips})


def explain_recommendation(
    windows: pd.DataFrame,
    profile: ActivityProfile,
    now: datetime,
    ask: Callable[..., Explanation] = chat,
    retrieve: Retriever | None = None,
) -> ExplainResult:
    """Ask the LLM to explain the windows. With no windows, answer without the LLM."""
    if windows.empty:
        return ExplainResult(
            Explanation(
                headline="Önümüzdeki günlerde tercihlerine uyan bir zaman aralığı görünmüyor.",
                reason="Tahminlerde rüzgâr ya çok zayıf ya da seviyen için fazla güçlü.",
                alternatives=[],
            ),
            sources=[],
        )

    sources = gather_sources(windows, profile, retrieve) if retrieve else []
    answer = ask(
        build_messages(windows, profile, now, sources), output_model=Explanation
    )
    return ExplainResult(keep_cited_tips(answer, sources), sources)


def render(result: ExplainResult) -> str:
    """Plain text version for the terminal."""
    e = result.explanation
    lines = [e.headline, "", e.reason]
    if e.alternatives:
        lines += ["", "Diğer seçenekler:", *[f"- {a}" for a in e.alternatives]]
    if e.tips:
        lines += ["", "İpuçları:"]
        lines += [
            f"- {t.text} {''.join(f'[{i}]' for i in t.source_ids)}" for t in e.tips
        ]
    lines += ["", f"⚠️  {CAUTION}"]
    if result.cited_sources:
        lines += ["", "Kaynaklar:"]
        lines += [f"[{s.id}] {s.title} → {s.section}" for s in result.cited_sources]
    return "\n".join(lines)
