"""Run the evaluations and print a score report: `uv run spotcast-eval`."""

import argparse

from spotcast.evals import retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SpotCast's evaluations.")
    parser.add_argument("-k", type=int, default=3, help="How many search results count")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show every question"
    )
    args = parser.parse_args()

    cases = retrieval.load_cases()
    print(f"=== Retrieval (evals/retrieval.json, top {args.k}) ===")
    for mode in ("vector", "hybrid"):
        results = retrieval.evaluate(cases, retrieval.real_search(mode), k=args.k)
        if args.verbose:
            print(f"\n--- {mode} ---")
            print(retrieval.report(results, args.k))
        s = retrieval.summarize(results)
        print(
            f"{mode:7s} hit@1 {s['hit@1']:4.0%}   hit@{args.k} {s['hit@k']:4.0%}   "
            f"MRR {s['mrr']:.2f}"
        )
