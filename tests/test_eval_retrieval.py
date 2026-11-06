"""Tests for the retrieval evaluation. A fake search engine returns fixed results."""

import pytest

from spotcast.evals.retrieval import (
    RetrievalCase,
    evaluate,
    load_cases,
    summarize,
)
from spotcast.rag import Chunk, SearchResult, load_knowledge


def result(file, section):
    chunk = Chunk(
        file=file, title="t", section=section, spot=None, source="s", content="c"
    )
    return SearchResult(chunk=chunk, distance=0.3)


def fake_search(answers):
    """Return prepared results per question."""
    return lambda question, k, spots: answers[question][:k]


def test_rank_metrics():
    cases = [
        RetrievalCase("first", ["a.md → A"]),
        RetrievalCase("second", ["a.md → A"]),
        RetrievalCase("missing", ["a.md → A"]),
        RetrievalCase("any of two", ["a.md → A", "b.md → B"]),
    ]
    search = fake_search(
        {
            "first": [result("a.md", "A"), result("x.md", "X")],
            "second": [result("x.md", "X"), result("a.md", "A")],
            "missing": [result("x.md", "X"), result("y.md", "Y")],
            "any of two": [result("b.md", "B")],
        }
    )

    results = evaluate(cases, search, k=3)
    s = summarize(results)

    assert [r.rank for r in results] == [1, 2, None, 1]
    assert s["hit@1"] == pytest.approx(2 / 4)
    assert s["hit@k"] == pytest.approx(3 / 4)
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0 + 1) / 4)


def test_spot_filter_is_passed_to_search():
    seen = []

    def search(question, k, spots):
        seen.append(spots)
        return []

    evaluate([RetrievalCase("q", ["a.md → A"], spots=["Alaçatı"])], search)
    assert seen == [["Alaçatı"]]


def test_test_set_only_expects_sections_that_exist():
    """If a heading is renamed, the test set must be updated too, or scores silently drop."""
    existing = {f"{c.file} → {c.section}" for c in load_knowledge()}
    for case in load_cases():
        for expected in case.expected:
            assert expected in existing, f"'{expected}' is not a knowledge section"
