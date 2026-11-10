"""A tool-calling agent, written by hand so every step is visible.

    question ──▶ LLM ──"call find_location(Mersin)"──▶ our code runs it ──▶ result ──▶ LLM
                  ▲                                                                   │
                  └────────────────────── repeat until the LLM answers ◀─────────────┘

The LLM decides WHICH tools to call and in what order. Our code decides WHAT tools exist,
checks every argument, and stops the loop if it runs too long.

Run: uv run spotcast-agent "Cumartesi Mersin sahilinde sörf yapacağım, uygun mu?"
"""

import argparse
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from spotcast.explain import CAUTION, turkish_date
from spotcast.llm import chat, chat_with_tools
from spotcast.locations import LOCATIONS, distance_km
from spotcast.profiles import LEVELS, PROFILES, SPORTS
from spotcast.tools import CONDITION_TOOLS, TOOLS, TOOLS_BY_NAME, ProfileArgs, Tool
from spotcast.uncertainty import notes_for_results

MAX_STEPS = 6  # each step is one LLM call; a normal question needs 2-4

SYSTEM_PROMPT = """You are SpotCast, an assistant for water sports in Turkey. Answer in Turkish.

Use the tools to get facts. Never guess wind, waves, coordinates or dates yourself.
- Place + day ("Saturday in Mersin") → find_location, then check_conditions.
- Where / when to go → recommend_spots.
- The user says where they live or start from ("I live in Izmir, where should I go?")
  → find_location, then recommend_near with those coordinates. Distances are
  straight-line (kuş uçuşu): say so.
- Advice or safety questions → search_knowledge.
If find_location returns several places, pick the coastal one that matches the question;
prefer a known_spot. A town centre is on land: for check_conditions use its
coastal_place_nearby coordinates when there is one; for recommend_near start from the
town itself.
Use the date table below to turn day names into dates; do not calculate dates yourself.

Answer rules:
- Use ONLY facts from tool results. Give numbers with units (knot, metre).
- Say "uygun" for good hours, "sınırda" for marginal, "uygun değil" for bad.
- Pass on every warning from the tools (no wave data, no local guide).
- For safety advice from search_knowledge, keep the guide's own words. Do not rephrase
  instructions like "do X, then Y": rewording them can flip their meaning.
- If all hours are "sınırda" (marginal), do not call the conditions "uygun".
- For a window, give the good and marginal hours separately ("4 saat uygun, 6 saat
  sınırda"); do not call the whole window "uygun" when part of it is marginal.
- For recommend_near: name the best place with its distance in km (kuş uçuşu) and
  mention the next option too.
- Never add names, neighbourhoods or details that are not in the tool results.
- If a tool returns an error, read it: fix your arguments and call again if you can;
  otherwise tell the user what is not possible.
- Never call conditions safe ("güvenli"). Keep the answer short: 3-6 sentences.

{date_table}"""


DEFAULT_SPORT, DEFAULT_LEVEL = "windsurf", "intermediate"
TURKISH_LEVELS = {
    "beginner": "yeni başlayan",
    "intermediate": "orta",
    "advanced": "ileri",
}
TURKISH_SPORTS = {
    "windsurf": "rüzgâr sörfü",
    "kitesurf": "uçurtma sörfü",
    "sup": "SUP",
    "kayak": "kano",
}

PROFILE_PROMPT = """Read the user's message. Which water sport and which skill level does the
user say? Use "unknown" when they don't say it. Do not guess."""


class UserProfile(BaseModel):
    """What the user said about themselves. The LLM only READS this from the question."""

    sport: Literal["windsurf", "kitesurf", "sup", "kayak", "other", "unknown"] = Field(
        description="windsurf = rüzgâr sörfü, kitesurf = uçurtma sörfü, "
        "sup = SUP / ayakta kürek sörfü, kayak = kano, other = any other activity"
    )
    level: Literal["beginner", "intermediate", "advanced", "unknown"] = Field(
        description="beginner = yeni başlayan, intermediate = orta, advanced = ileri"
    )


@dataclass
class Step:
    tool: str
    arguments: dict
    result: dict
    seconds: float


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_early: bool = False
    llm_text: str = (
        ""  # only what the LLM wrote, before code adds notes (for evaluation)
    )


def date_table(now: datetime) -> str:
    """The next 7 days written out by code, because LLMs often get weekdays wrong."""
    lines = ["Dates (today first):"]
    for offset in range(7):
        day = now + timedelta(days=offset)
        # Turkish words too: the user writes "yarın", and the model missed "tomorrow" for it
        label = {0: "today (bugün)", 1: "tomorrow (yarın)"}.get(offset, "")
        lines.append(f"- {turkish_date(day)} = {day:%Y-%m-%d} {label}".rstrip())
    return "\n".join(lines)


# Words code can recognise reliably. A small LLM read "rüzgâr sörfü" as SUP, so known
# words are matched by code first, and the LLM is only asked when none of them appear.
SPORT_WORDS = {
    "windsurf": r"rüzg[aâ]r ?sörf|windsurf",
    "kitesurf": r"kite|uçurtma",
    "sup": r"\bsup\b|kürek ?sörf|ayakta kürek",
    # Not the Turkish word "kayak": it means skiing
    "kayak": r"\bkano|kayaking|sea kayak",
}
LEVEL_WORDS = {
    # "yeni başl": başlayan, başlıyorum, başladım ("yeni başla" missed "başlıyorum")
    "beginner": r"yeni başl|acemi|başlangıç|ilk kez|ilk defa|hiç yapmadım|beginner",
    "intermediate": r"\borta\b|intermediate",
    "advanced": r"ileri seviye|\bileri\b|deneyimli|uzman|advanced",
}


# Activities SpotCast has no profile for. A small LLM once read "dalış" (diving) as SUP and
# answered with SUP limits, so code recognises these and the request is refused.
OTHER_ACTIVITIES = r"\bdal(ış|ma|mak|acağ|ıyor)|scuba|ş?norkel|jet ?ski|balık (avı|tut)"


def _match(words: dict[str, str], text: str) -> str | None:
    found = [name for name, pattern in words.items() if re.search(pattern, text)]
    return found[0] if len(found) == 1 else None  # none or several: not sure


def read_profile(question: str, ask: Callable[..., UserProfile] = chat) -> UserProfile:
    # Python lowercases "İ" to "i" plus a combining dot, so map Turkish capitals first
    text = question.replace("İ", "i").replace("I", "ı").lower()
    sport, level = _match(SPORT_WORDS, text), _match(LEVEL_WORDS, text)
    if sport is None and re.search(OTHER_ACTIVITIES, text):
        sport = "other"
    if sport is None or level is None:
        messages = [
            {"role": "system", "content": PROFILE_PROMPT},
            {"role": "user", "content": question},
        ]
        guess = ask(messages, output_model=UserProfile, temperature=0, max_tokens=60)
        sport, level = sport or guess.sport, level or guess.level
    return UserProfile(sport=sport, level=level)


def quoted_source(steps: list[Step]) -> str:
    """The best knowledge excerpt, copied word for word by code.

    The LLM once turned "get out of the current, then swim to shore" into its opposite.
    Showing the original lets the user see what the guide really says.
    """
    for step in reversed(steps):
        excerpts = (
            step.result.get("excerpts") if step.tool == "search_knowledge" else None
        )
        if excerpts:
            top = excerpts[0]
            text = " ".join(top["text"].split())
            return f"\n\n📖 Kaynak ({top['title']} → {top['section']}): «{text}»"
    return ""


def unsupported_answer(sport: str, level: str) -> str:
    """Fixed text, written by code: an LLM once 'helped' a beginner by using intermediate."""
    sports = ", ".join(TURKISH_SPORTS.get(s, s) for s in SPORTS)
    levels = ", ".join(
        TURKISH_LEVELS.get(lvl, lvl) for lvl in LEVELS if (sport, lvl) in PROFILES
    )
    if sport in SPORTS:  # a known sport at a level we have no profile for
        asked = f"{TURKISH_LEVELS.get(level, level)} seviye {TURKISH_SPORTS.get(sport, sport)}"
        hint = f"{TURKISH_SPORTS.get(sport, sport).capitalize()} için seviyeler: {levels}. "
    else:
        asked, hint = "Bu aktivite", ""
    return (
        f"SpotCast şu an sadece şu sporlar için değerlendirme yapabiliyor: {sports}. "
        f"{hint}{asked} için henüz öneri veremiyorum. Başka bir spor ya da seviyenin "
        f"sonuçları sana uygun olmayabilir, o yüzden onları kullanmanı önermiyorum."
    )


def schema_for_llm(tool: Tool) -> dict:
    """The tool schema WITHOUT sport and level: code sets those, the LLM can't change them."""
    schema = json.loads(json.dumps(tool.schema()))  # a deep copy
    params = schema["function"]["parameters"]
    for key in ("sport", "level"):
        params.get("properties", {}).pop(key, None)
        if key in params.get("required", []):
            params["required"].remove(key)
    return schema


def run_tool(name: str, arguments: Any, sport: str, level: str) -> dict:
    """Run one tool the LLM asked for. Unknown names and bad arguments become errors it can read."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return {
            "error": f"There is no tool called {name}. Tools: {', '.join(TOOLS_BY_NAME)}"
        }
    if isinstance(arguments, str):  # some models send arguments as a JSON string
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"error": "Arguments must be a JSON object."}
    if issubclass(tool.args, ProfileArgs):
        arguments = {
            **arguments,
            "sport": sport,
            "level": level,
        }  # whatever the LLM wrote
    return tool.call(arguments)


GAVE_UP = "Üzgünüm, bu soruyu cevaplayamadım. Lütfen daha açık bir şekilde sor."

# ---------- Rules on tool calls that code checks (the prompt asks; code makes sure) ----------

COORDINATE_TOOLS = {"check_conditions", "recommend_near"}
COORDINATES_FROM_KM = (
    2  # coordinates this close to a known point count as "not guessed"
)
# Day words in a question. "pazar" is not "pazartesi", "cuma" is not "cumartesi".
WEEKDAY_WORDS = [
    r"\bpazartesi",
    r"\bsalı",
    r"\bçarşamba",
    r"\bperşembe",
    r"\bcuma(?!rtesi)",
    r"\bcumartesi",
    r"\bpazar(?!tesi)",
]


def days_mentioned(question: str, now: datetime) -> set[date]:
    """The days the user named, as dates in the next 7 days. Empty: no day was named."""
    text = question.replace("İ", "i").replace("I", "ı").lower()
    today = now.date()
    days = set()
    if re.search(r"\bbugün", text):
        days.add(today)
    if re.search(r"\byarın", text):
        days.add(today + timedelta(days=1))
    weekdays = {i for i, word in enumerate(WEEKDAY_WORDS) if re.search(word, text)}
    if re.search(r"hafta ?sonu", text):
        weekdays |= {5, 6}
    days |= {today + timedelta(days=(wd - today.weekday()) % 7) for wd in weekdays}
    return days


def known_points(results: list[Any]) -> list[tuple[float, float]]:
    """Every latitude/longitude pair inside earlier tool results (find_location etc.)."""
    points = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            lat, lon = value.get("latitude"), value.get("longitude")
            if isinstance(lat, int | float) and isinstance(lon, int | float):
                points.append((float(lat), float(lon)))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(results)
    return points


def check_call(
    name: str, arguments: Any, question: str, now: datetime, earlier: list[Any]
) -> dict | None:
    """An error for the LLM if a tool call breaks a rule code can check, else None.

    - Coordinates must come from find_location or our own places: the LLM once skipped
      find_location and made up coordinates for Akyaka (they were near Bodrum).
    - The day must be the one the user named: "yarın" was sent as a Saturday.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None  # run_tool reports it
    if not isinstance(arguments, dict):
        return None

    lat, lon = arguments.get("latitude"), arguments.get("longitude")
    if (
        name in COORDINATE_TOOLS
        and isinstance(lat, int | float)
        and isinstance(lon, int | float)
    ):
        points = known_points(earlier) + [
            (loc.latitude, loc.longitude) for loc in LOCATIONS
        ]
        if not any(distance_km(lat, lon, *p) <= COORDINATES_FROM_KM for p in points):
            return {
                "error": "These coordinates did not come from find_location. Call "
                "find_location with the place name first; never guess coordinates."
            }

    expected = days_mentioned(question, now)
    if expected and arguments.get("day"):
        try:
            day = date.fromisoformat(str(arguments["day"]))
        except ValueError:
            return None  # the tool's own validation reports it
        if day not in expected:
            wanted = ", ".join(f"{d:%Y-%m-%d}" for d in sorted(expected))
            return {
                "error": f"The user asked about {wanted}, but you sent {day:%Y-%m-%d}. "
                "Use the date table and call again."
            }
    return None


@dataclass(frozen=True)
class Who:
    """The sport and level code decided on, or a refusal if SpotCast can't help yet."""

    sport: str
    level: str
    level_note: str = ""
    refusal: str | None = None


def decide_profile(said: UserProfile) -> Who:
    sport = DEFAULT_SPORT if said.sport == "unknown" else said.sport
    level = DEFAULT_LEVEL if said.level == "unknown" else said.level
    if (sport, level) not in PROFILES:
        return Who(sport, level, refusal=unsupported_answer(sport, level))
    # Code says which assumption it made: the limits differ a lot between sports
    missing = [
        what
        for what, unknown in (
            ("sporunu", said.sport == "unknown"),
            ("seviyeni", said.level == "unknown"),
        )
        if unknown
    ]
    note = (
        f"\n\nℹ️ {' ve '.join(missing).capitalize()} belirtmediğin için bu değerlendirme "
        f"{TURKISH_LEVELS[level]} seviye {TURKISH_SPORTS[sport]} içindir; senin durumun "
        f"farklıysa sonuçlar sana uymayabilir."
        if missing
        else ""
    )
    return Who(sport, level, level_note=note)


def nearby_notes(steps: list[Step]) -> list[str]:
    """Facts about recommend_near results that must not depend on the LLM: an LLM once
    said "both places have a local guide" when the best one had none."""
    windows = [
        w
        for s in steps
        if s.tool == "recommend_near" and "error" not in s.result
        for w in s.result.get("windows_best_first", [])
    ]
    if not windows:
        return []
    distances = dict.fromkeys(f"{w['spot']} {w['distance_km']} km" for w in windows)
    notes = [
        f"📏 Kuş uçuşu mesafe: {', '.join(distances)}. Yol mesafesi daha uzun olabilir."
    ]
    unguided = list(dict.fromkeys(w["spot"] for w in windows if not w["local_guide"]))
    if unguided:
        notes.append(
            f"ℹ️ {', '.join(unguided)} için yerel rehberimiz yok: akıntı, kayalık ve "
            f"yasak alanlar gibi yerel tehlikeleri orada sor."
        )
    return notes


def offshore_notes(conditions: list[dict]) -> list[str]:
    """What offshore means, written by code: an LLM once described it the wrong way round
    ("wind blowing from the sea inland")."""
    unsafe = sum(r.get("offshore_hours", 0) for r in conditions)
    sheltered = sum(r.get("sheltered_offshore_hours", 0) for r in conditions)
    notes = []
    if unsafe:
        notes.append(
            f"🌬️ Gündüz {unsafe} saatte rüzgâr karadan denize esiyor (offshore): seni "
            f"kıyıdan uzaklaştırır. Seviyenin sınırını aşan saatler uygun değil sayıldı."
        )
    if sheltered:
        notes.append(
            f"⚓ Gündüz {sheltered} saatte rüzgâr karadan esiyor ama burası korunaklı bir koy: "
            f"sürüklenirsen açık denize değil koyun karşı kıyısına varırsın. Yine de "
            f"rüzgâr açıkta daha sert ve hamlelidir, koyun içinde kal."
        )
    return notes


def finish_answer(content: str, steps: list[Step], level_note: str) -> str:
    """Add the texts that code writes, never the LLM: level note, caution, source quote."""
    answer = content.strip()
    conditions = [
        s.result for s in steps if s.tool in CONDITION_TOOLS and "error" not in s.result
    ]
    if conditions:
        answer += level_note
        # Uncertainty (far-ahead forecast, no wave data): the LLM is asked to pass tool
        # warnings on, but asking is not a guarantee, so code adds them too.
        for note in notes_for_results(conditions):
            answer += f"\n\n⏳ {note}"
        for note in nearby_notes(steps) + offshore_notes(conditions):
            answer += f"\n\n{note}"
        answer += f"\n\n⚠️ {CAUTION}"
    return answer + quoted_source(steps)


def run_agent(
    question: str,
    now: datetime | None = None,
    llm: Callable[..., dict] = chat_with_tools,
    profile_reader: Callable[[str], UserProfile] = read_profile,
    tools: list[Tool] = TOOLS,
    max_steps: int = MAX_STEPS,
    on_step: Callable[[Step], None] | None = None,
) -> AgentResult:
    now = now or datetime.now()

    # 1) Who is asking? Read it once, then code decides. The agent can never change it.
    who = decide_profile(profile_reader(question))
    if who.refusal:
        return AgentResult(who.refusal)
    sport, level, level_note = who.sport, who.level, who.level_note

    # 2) The agent loop
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(date_table=date_table(now))},
        {"role": "user", "content": question},
    ]
    schemas = [schema_for_llm(tool) for tool in tools]
    steps: list[Step] = []

    for _ in range(max_steps):
        reply = llm(messages, schemas)
        messages.append(reply)  # the conversation must include what the LLM asked for

        calls = reply.get("tool_calls") or []
        if not calls:  # no tool requested: this is the final answer
            content = reply.get("content") or ""
            answer = finish_answer(content, steps, level_note)
            return AgentResult(answer, steps, llm_text=content.strip())

        for call in calls:
            name = call["function"]["name"]
            arguments = call["function"].get("arguments", {})
            started = time.monotonic()
            earlier = [s.result for s in steps]
            result = check_call(name, arguments, question, now, earlier) or run_tool(
                name, arguments, sport, level
            )
            step = Step(name, arguments, result, time.monotonic() - started)
            steps.append(step)
            if on_step:
                on_step(step)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

    # Too many steps: the model is going in circles. Stop instead of looping forever.
    return AgentResult(GAVE_UP, steps, stopped_early=True)


EXAMPLE_QUESTIONS = [
    "Cumartesi Mersin sahilinde rüzgâr sörfü yapmak istiyorum, koşullar uygun mu?",
    "Bu hafta sonu rüzgâr sörfü için nereye gideyim? Orta seviyedeyim.",
    "Yeni başlıyorum, hafta sonu Alaçatı'ya gitsem olur mu?",
    "Dip akıntısına yakalanırsam ne yapmalıyım?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask SpotCast's agent a question.")
    parser.add_argument("question", nargs="*", help="Your question (in Turkish)")
    args = parser.parse_args()
    questions = [" ".join(args.question)] if args.question else EXAMPLE_QUESTIONS

    def show(step: Step) -> None:
        short = json.dumps(step.result, ensure_ascii=False, default=str)
        print(
            f"   🔧 {step.tool}({step.arguments}) → {short[:160]}... ({step.seconds:.1f} s)"
        )

    for question in questions:
        print(f"\n❓ {question}")
        started = time.monotonic()
        result = run_agent(question, on_step=show)
        print(f"\n💬 {result.answer}")
        print(
            f"   ({len(result.steps)} tool calls, {time.monotonic() - started:.0f} s)"
        )
