"""Does knowledge search find the right section? Measured on evals/retrieval.json.

Metrics, for each question with its expected sections:
- hit@1: the best result is one of the expected sections
- hit@k: an expected section is somewhere in the top k
- MRR (mean reciprocal rank): 1 if the right section is 1st, 1/2 if 2nd, 1/3 if 3rd, 0 if missing,
  averaged over all questions. One number that rewards finding it early.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from spotcast.config import PROJECT_ROOT
from spotcast.rag import SearchResult

CASES_FILE = PROJECT_ROOT / "evals" / "retrieval.json"

Search = Callable[[str, int, list[str] | None], list[SearchResult]]


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    expected: list[str]  # "file → section"; finding any of them counts
    spots: list[str] | None = None


@dataclass(frozen=True)
class RetrievalResult:
    case: RetrievalCase
    found: list[str]  # what search returned, best first
    rank: int | None  # 1-based position of the first expected section, None if missing


def label(result: SearchResult) -> str:
    return f"{result.chunk.file} → {result.chunk.section}"


def load_cases(path: Path = CASES_FILE) -> list[RetrievalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [RetrievalCase(**case) for case in data["cases"]]


def evaluate(
    cases: list[RetrievalCase], search: Search, k: int = 3
) -> list[RetrievalResult]:
    results = []
    for case in cases:
        found = [label(r) for r in search(case.question, k, case.spots)]
        rank = next(
            (i for i, f in enumerate(found, start=1) if f in case.expected), None
        )
        results.append(RetrievalResult(case, found, rank))
    return results


def summarize(results: list[RetrievalResult]) -> dict[str, float]:
    n = len(results)
    return {
        "hit@1": sum(r.rank == 1 for r in results) / n,
        "hit@k": sum(r.rank is not None for r in results) / n,
        "mrr": sum(1 / r.rank for r in results if r.rank) / n,
    }


def report(results: list[RetrievalResult], k: int) -> str:
    lines = []
    for r in results:
        mark = "✓" if r.rank == 1 else ("~" if r.rank else "✗")
        where = f"#{r.rank}" if r.rank else "missing"
        lines.append(f"{mark} [{where:>7}] {r.case.question}")
        if r.rank != 1:  # show what came first instead
            lines.append(f"      expected: {r.case.expected[0]}")
            lines.append(f"      got #1:   {r.found[0] if r.found else '-'}")
    s = summarize(results)
    lines.append(
        f"\nhit@1 {s['hit@1']:.0%}   hit@{k} {s['hit@k']:.0%}   MRR {s['mrr']:.2f}"
        f"   ({len(results)} questions)"
    )
    return "\n".join(lines)


def real_search(mode: str = "hybrid") -> Search:
    from spotcast.db import get_engine
    from spotcast.rag import search

    engine = get_engine()
    return lambda question, k, spots: search(
        engine, question, k=k, spots=spots, mode=mode
    )
