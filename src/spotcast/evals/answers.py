"""Are the agent's answers right? Measured on evals/answers.json.

The weather changes every day, so the tools are FROZEN: each test case brings the results
its tools must return. The real LLM runs; only the world around it is fixed. Then:

1. Rule checks (code, cheap, always the same result):
   - the right tools were called, with the right day,
   - required facts and warnings are mentioned,
   - never "güvenli" (safe),
   - no "uygun" (suitable) when no hour is good,
   - every knot/metre number in the answer comes from a tool result (no invented numbers).
2. LLM-as-judge (for advice from the knowledge base): a second LLM call compares the answer
   with the source text. Rules can't see that "do X, then Y" became "don't do X".
   A judge is an LLM too, so it is tested first on evals/judge.json (see `evaluate_judge`).
"""

import copy
import json
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from spotcast.agent import AgentResult, run_agent
from spotcast.config import PROJECT_ROOT
from spotcast.llm import chat
from spotcast.tools import TOOLS_BY_NAME

ANSWERS_FILE = PROJECT_ROOT / "evals" / "answers.json"
JUDGE_FILE = PROJECT_ROOT / "evals" / "judge.json"


# ---------- Test cases ----------


@dataclass(frozen=True)
class AnswerCase:
    id: str
    question: str
    tools: dict[str, dict]  # tool name → the result it returns in this test
    must_call: list[str] = field(default_factory=list)
    arguments: dict[str, dict] = field(default_factory=dict)  # tool → expected values
    must_mention: list[str] = field(default_factory=list)  # regexes
    judge: bool = False  # compare the answer with the knowledge excerpts


def load_answer_cases(path: Path = ANSWERS_FILE) -> tuple[datetime, list[AnswerCase]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return datetime.fromisoformat(data["now"]), [
        AnswerCase(**case) for case in data["cases"]
    ]


@contextmanager
def frozen_tools(results: dict[str, dict]) -> Iterator[None]:
    """Make every tool return a fixed result. Argument checks stay real, so the LLM's
    mistakes (wrong date format, missing field) still show up as errors."""
    original = dict(TOOLS_BY_NAME)
    try:
        for name, tool in original.items():
            fixed = results.get(
                name, {"error": "This tool is not available in this test."}
            )
            TOOLS_BY_NAME[name] = replace(
                tool, run=lambda args, fixed=fixed: copy.deepcopy(fixed)
            )
        yield
    finally:
        TOOLS_BY_NAME.clear()
        TOOLS_BY_NAME.update(original)


# ---------- Rule checks: pure functions on text ----------

# "güvenli", "güvenlidir", "güvenle" but not "güvenlik" (safety, as in "safety rules")
SAFE_WORD = re.compile(r"\bgüvenl(?:i(?!k)|e)\w*", re.IGNORECASE)
SUITABLE = re.compile(r"\buygun\b(?!\s+mu)", re.IGNORECASE)
NEGATION = re.compile(
    r"değil|olmayan|olmadığ|olmayacak|olmaz|yok|bulunm|bulunama|sayılmaz", re.IGNORECASE
)
NUMBER = r"(\d+(?:[.,]\d+)?)"
UNITS = {"knot": r"knot|kn\b|kt\b|deniz mili", "metre": r"metre|m\b"}
TOLERANCE = {"knot": 0.5, "metre": 0.1}  # rounding 12.7 to "13 knot" is fine
NUMBER_WITH_UNIT = re.compile(
    rf"{NUMBER}(?:\s*(?:-|–|ile|ila|ve)\s*{NUMBER})?\s*(?P<unit>{UNITS['knot']}|{UNITS['metre']})",
    re.IGNORECASE,
)


def says_safe(text: str) -> list[str]:
    return SAFE_WORD.findall(text)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def suitable_claims(text: str) -> list[str]:
    """Sentences that call something 'uygun' without a negation ('uygun değil', 'yok')."""
    return [s for s in sentences(text) if SUITABLE.search(s) and not NEGATION.search(s)]


def numbers_with_units(text: str) -> list[tuple[float, str]]:
    """'1,8 ile 6 knot' → [(1.8, 'knot'), (6.0, 'knot')]."""
    found = []
    for match in NUMBER_WITH_UNIT.finditer(text):
        unit = "knot" if re.fullmatch(UNITS["knot"], match["unit"], re.I) else "metre"
        for number in match.groups()[:2]:
            if number:
                found.append((float(number.replace(",", ".")), unit))
    return found


def known_numbers(results: list[Any], question: str) -> dict[str, list[float]]:
    """Every knot and metre value the tools returned (or the user wrote)."""
    known: dict[str, list[float]] = {"knot": [], "metre": []}

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, k)
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            for unit in known:
                if unit in key:  # wind_knots, avg_wind_knots, wave_metres, ...
                    known[unit].append(float(value))
        elif isinstance(value, str):  # numbers inside knowledge excerpts
            for number, unit in numbers_with_units(value):
                known[unit].append(number)

    walk(results)
    walk(question)
    return known


def invented_numbers(text: str, results: list[Any], question: str) -> list[str]:
    known = known_numbers(results, question)
    return [
        f"{number:g} {unit}"
        for number, unit in numbers_with_units(text)
        if not any(abs(number - k) <= TOLERANCE[unit] for k in known[unit])
    ]


CONDITION_TOOLS = {"check_conditions", "recommend_spots"}


def good_hours(result: AgentResult) -> int | None:
    """How many 'good' hours the tools reported. None if no conditions tool ran."""
    counts = []
    for step in result.steps:
        if "error" in step.result:
            continue
        if step.tool == "check_conditions":
            counts.append(step.result.get("label_counts", {}).get("good", 0))
        elif step.tool == "recommend_spots":
            counts += [w["good_hours"] for w in step.result["windows_best_first"]]
    return sum(counts) if counts else None


# ---------- LLM-as-judge ----------

JUDGE_PROMPT = """You check an answer against its SOURCE text. The source is the only truth.

contradicts = true if the answer:
- reverses an instruction or the order of steps ("first X, then Y" → "don't X, just Y"),
- says something is safe or easy that the source calls dangerous,
- weakens a prohibition ("never" → "only for experts"),
- gives a different number than the source.
adds = true if the answer gives advice or facts that are NOT in the source.
Leaving things out, shortening or rewording with the same meaning is fine: not a problem.

First write each problem in `problems` (quote the answer's words). Then decide."""


class Faithfulness(BaseModel):
    problems: list[str] = Field(
        description="Each problem, quoting the answer. Empty if there are none."
    )
    contradicts: bool
    adds: bool


def judge_whole(
    source: str, answer: str, ask: Callable[..., Any] = chat
) -> Faithfulness:
    """Version 1: one verdict for the whole answer."""
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"SOURCE:\n{source}\n\nANSWER:\n{answer}"},
    ]
    return ask(messages, output_model=Faithfulness, temperature=0, max_tokens=400)


CLAIMS_PROMPT = """Split the ANSWER into its separate claims (one instruction or fact each).
For each claim, find what the SOURCE says about the same thing and compare:
- "same": the source says this too (other words with the same meaning are fine),
- "opposite": the source says something different: a reversed instruction or step order,
  dangerous called safe, "never" weakened, a different number,
- "not_in_source": the source says nothing about it.
Only judge the ANSWER's claims. Things the source says but the answer leaves out are fine."""


class Claim(BaseModel):
    answer_says: str = Field(description="One claim, in the answer's words")
    source_says: str = Field(
        description="The source's words about the same thing, or 'nothing'"
    )
    relation: Literal["same", "opposite", "not_in_source"]


class ClaimCheck(BaseModel):
    claims: list[Claim]


def judge_claims(
    source: str, answer: str, ask: Callable[..., Any] = chat
) -> Faithfulness:
    """Version 2: the LLM compares one claim at a time, code makes the verdict.

    Small tasks are easier for a small model than one big judgement, and the quoted
    source words show a person exactly why a claim was marked.
    """
    messages = [
        {"role": "system", "content": CLAIMS_PROMPT},
        {"role": "user", "content": f"SOURCE:\n{source}\n\nANSWER:\n{answer}"},
    ]
    check = ask(messages, output_model=ClaimCheck, temperature=0, max_tokens=800)
    marked = [c for c in check.claims if c.relation != "same"]
    return Faithfulness(
        problems=[
            f"{c.relation}: «{c.answer_says}» ↔ «{c.source_says}»" for c in marked
        ],
        contradicts=any(c.relation == "opposite" for c in marked),
        adds=any(c.relation == "not_in_source" for c in marked),
    )


JUDGES = {"whole": judge_whole, "claims": judge_claims}
judge = judge_claims  # the one answer checks use; chosen by `spotcast-eval judge`


def knowledge_given(result: AgentResult) -> str:
    """The knowledge excerpts the agent actually received."""
    return "\n\n".join(
        excerpt["text"]
        for step in result.steps
        if step.tool == "search_knowledge"
        for excerpt in step.result.get("excerpts", [])
    )


# ---------- Running and checking ----------


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class CaseResult:
    case: AnswerCase
    result: AgentResult
    checks: list[Check]
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def check_answer(
    case: AnswerCase,
    result: AgentResult,
    ask_judge: Callable[[str, str], Faithfulness] | None = judge,
) -> list[Check]:
    text = result.llm_text
    called = [s.tool for s in result.steps]
    checks = [Check("finished", bool(text) and not result.stopped_early)]

    missing = [t for t in case.must_call if t not in called]
    checks.append(Check("tools", not missing, f"missing {missing}, called {called}"))

    for tool, expected in case.arguments.items():
        sent = [s.arguments for s in result.steps if s.tool == tool]
        ok = any(
            all(str(args.get(k)) == str(v) for k, v in expected.items())
            for args in sent
        )
        checks.append(
            Check(f"{tool} arguments", ok, f"expected {expected}, sent {sent}")
        )

    missing_words = [p for p in case.must_mention if not re.search(p, text, re.I)]
    checks.append(Check("mentions", not missing_words, f"missing {missing_words}"))

    # The rule is "never call CONDITIONS safe". In advice answers "the safest thing is not
    # to go in" is right, so there the judge compares the wording with the source instead.
    if CONDITION_TOOLS & set(case.tools):
        safe = says_safe(text)
        checks.append(Check("no 'güvenli'", not safe, f"found {safe}"))

    if good_hours(result) == 0:
        claims = suitable_claims(text)
        checks.append(Check("no 'uygun' without good hours", not claims, f"{claims}"))

    invented = invented_numbers(text, [s.result for s in result.steps], case.question)
    checks.append(
        Check("numbers from tools", not invented, f"not in tools: {invented}")
    )

    source = knowledge_given(result)
    if case.judge and ask_judge and source and text:
        verdict = ask_judge(source, text)
        problems = "; ".join(verdict.problems)
        checks.append(
            Check("judge: no contradiction", not verdict.contradicts, problems)
        )
        checks.append(Check("judge: nothing added", not verdict.adds, problems))
    return checks


def run_case(
    case: AnswerCase,
    now: datetime,
    agent: Callable[..., AgentResult] = run_agent,
    ask_judge: Callable[[str, str], Faithfulness] | None = judge,
) -> CaseResult:
    started = time.monotonic()
    with frozen_tools(case.tools):
        result = agent(case.question, now=now)
    seconds = time.monotonic() - started
    return CaseResult(case, result, check_answer(case, result, ask_judge), seconds)


def summarize_answers(results: list[CaseResult]) -> dict[str, float]:
    """Share of passed runs per check, plus 'all' = the whole case passed."""
    by_check: dict[str, list[bool]] = {}
    for r in results:
        for c in r.checks:
            by_check.setdefault(c.name, []).append(c.passed)
    summary = {"all": sum(r.passed for r in results) / len(results)}
    summary.update({name: sum(v) / len(v) for name, v in by_check.items()})
    return summary


def report_case(r: CaseResult, verbose: bool = False) -> str:
    mark = "✓" if r.passed else "✗"
    lines = [f"{mark} {r.case.id} ({r.seconds:.0f} s)"]
    for c in r.checks:
        if not c.passed:
            lines.append(f"     ✗ {c.name}: {c.detail}")
    if verbose or not r.passed:
        answer = " ".join(r.result.llm_text.split())
        lines.append(f"     💬 {answer[:400]}")
    return "\n".join(lines)


# ---------- Testing the judge itself ----------


@dataclass(frozen=True)
class JudgeCase:
    note: str
    source: str
    answer: str
    contradicts: bool
    adds: bool


def load_judge_cases(path: Path = JUDGE_FILE) -> list[JudgeCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [JudgeCase(**case) for case in data["cases"]]


def evaluate_judge(
    cases: list[JudgeCase], ask_judge: Callable[[str, str], Faithfulness] = judge
) -> list[tuple[JudgeCase, Faithfulness]]:
    return [(case, ask_judge(case.source, case.answer)) for case in cases]


def summarize_judge(results: list[tuple[JudgeCase, Faithfulness]]) -> dict[str, Any]:
    """Agreement with the human labels. A missed contradiction is the dangerous mistake."""
    summary: dict[str, Any] = {}
    for label in ("contradicts", "adds"):
        pairs = [(getattr(c, label), getattr(v, label)) for c, v in results]
        summary[label] = {
            "agreement": sum(h == j for h, j in pairs) / len(pairs),
            "missed": sum(h and not j for h, j in pairs),  # human: yes, judge: no
            "false_alarms": sum(j and not h for h, j in pairs),
        }
    return summary
