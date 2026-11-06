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
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from spotcast.explain import CAUTION, turkish_date
from spotcast.llm import chat, chat_with_tools
from spotcast.profiles import PROFILES
from spotcast.tools import TOOLS, TOOLS_BY_NAME, ProfileArgs, Tool

MAX_STEPS = 6  # each step is one LLM call; a normal question needs 2-4

SYSTEM_PROMPT = """You are SpotCast, an assistant for water sports in Turkey. Answer in Turkish.

Use the tools to get facts. Never guess wind, waves, coordinates or dates yourself.
- Place + day ("Saturday in Mersin") → find_location, then check_conditions.
- Where / when to go → recommend_spots.
- Advice or safety questions → search_knowledge.
If find_location returns several places, pick the coastal one that matches the question.
Use the date table below to turn day names into dates; do not calculate dates yourself.

Answer rules:
- Use ONLY facts from tool results. Give numbers with units (knot, metre).
- Say "uygun" for good hours, "sınırda" for marginal, "uygun değil" for bad.
- Pass on every warning from the tools (no wave data, no local guide).
- For safety advice from search_knowledge, keep the guide's own words. Do not rephrase
  instructions like "do X, then Y": rewording them can flip their meaning.
- If all hours are "sınırda" (marginal), do not call the conditions "uygun".
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
TURKISH_SPORTS = {"windsurf": "rüzgâr sörfü", "kitesurf": "uçurtma sörfü", "sup": "SUP"}

PROFILE_PROMPT = """Read the user's message. Which water sport and which skill level does the
user say? Use "unknown" when they don't say it. Do not guess."""


class UserProfile(BaseModel):
    """What the user said about themselves. The LLM only READS this from the question."""

    sport: Literal["windsurf", "kitesurf", "sup", "other", "unknown"] = Field(
        description="windsurf = rüzgâr sörfü, kitesurf = uçurtma sörfü, sup = SUP"
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
    llm_text: str = ""  # only what the LLM wrote, before code adds notes (for evaluation)


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
    "sup": r"\bsup\b|kürek ?sörf",
}
LEVEL_WORDS = {
    "beginner": r"yeni başla|acemi|başlangıç|ilk kez|ilk defa|hiç yapmadım|beginner",
    "intermediate": r"\borta\b|intermediate",
    "advanced": r"ileri seviye|\bileri\b|deneyimli|uzman|advanced",
}


def _match(words: dict[str, str], text: str) -> str | None:
    found = [name for name, pattern in words.items() if re.search(pattern, text)]
    return found[0] if len(found) == 1 else None  # none or several: not sure


def read_profile(question: str, ask: Callable[..., UserProfile] = chat) -> UserProfile:
    # Python lowercases "İ" to "i" plus a combining dot, so map Turkish capitals first
    text = question.replace("İ", "i").replace("I", "ı").lower()
    sport, level = _match(SPORT_WORDS, text), _match(LEVEL_WORDS, text)
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
    supported = ", ".join(
        f"{TURKISH_LEVELS.get(lvl, lvl)} seviye {TURKISH_SPORTS.get(spt, spt)}"
        for spt, lvl in sorted(PROFILES)
    )
    asked = (
        f"{TURKISH_LEVELS.get(level, level)} seviye {TURKISH_SPORTS.get(sport, sport)}"
    )
    return (
        f"SpotCast şu an sadece {supported} için değerlendirme yapabiliyor. "
        f"{asked.capitalize()} için henüz öneri veremiyorum. Başka bir seviyenin sonuçları "
        f"sana uygun olmayabilir, o yüzden onları kullanmanı önermiyorum."
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
    note = (
        f"\n\nℹ️ Seviyeni belirtmediğin için bu değerlendirme {TURKISH_LEVELS[level]} seviye "
        f"içindir; seviyen farklıysa sonuçlar sana uymayabilir."
        if said.level == "unknown"
        else ""
    )
    return Who(sport, level, level_note=note)


def finish_answer(content: str, steps: list[Step], level_note: str) -> str:
    """Add the texts that code writes, never the LLM: level note, caution, source quote."""
    answer = content.strip()
    gave_conditions = any(
        s.tool in ("check_conditions", "recommend_spots") and "error" not in s.result
        for s in steps
    )
    if gave_conditions:
        answer += level_note + f"\n\n⚠️ {CAUTION}"
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
            result = run_tool(name, arguments, sport, level)
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
