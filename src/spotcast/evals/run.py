"""Run the evaluations and print a score report.

    uv run spotcast-eval                 # retrieval (fast, no chat LLM)
    uv run spotcast-eval judge           # test the LLM judge on hand-labelled pairs
    uv run spotcast-eval answers         # the agent's answers (slow: ~15 s per question)
    uv run spotcast-eval answers --repeat 3 --only rip_current
"""

import argparse
import time

from spotcast.evals import answers, retrieval


def run_retrieval(args: argparse.Namespace) -> None:
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


def run_judge(args: argparse.Namespace) -> None:
    cases = answers.load_judge_cases()
    print(f"=== LLM judge (evals/judge.json, {len(cases)} labelled pairs) ===")
    for name, ask_judge in answers.JUDGES.items():
        print(f"\n--- {name} ---")
        started = time.monotonic()
        results = answers.evaluate_judge(cases, ask_judge)
        for case, verdict in results:
            ok = (case.contradicts, case.adds) == (verdict.contradicts, verdict.adds)
            print(
                f"{'✓' if ok else '✗'} human c={case.contradicts:d} a={case.adds:d}  "
                f"judge c={verdict.contradicts:d} a={verdict.adds:d}  {case.note}"
            )
            if args.verbose or not ok:
                for problem in verdict.problems:
                    print(f"      · {problem}")
        for label, s in answers.summarize_judge(results).items():
            print(
                f"{label:12s} agreement {s['agreement']:4.0%}   missed {s['missed']}   "
                f"false alarms {s['false_alarms']}"
            )
        print(f"({time.monotonic() - started:.0f} s)")


def run_answers(args: argparse.Namespace) -> None:
    now, cases = answers.load_answer_cases()
    if args.only:
        cases = [c for c in cases if c.id in args.only]
    ask_judge = None if args.no_judge else answers.judge
    print(
        f"=== Agent answers (evals/answers.json, {len(cases)} cases × {args.repeat}) ==="
    )
    results = []
    for case in cases:
        for _ in range(args.repeat):
            r = answers.run_case(case, now, ask_judge=ask_judge)
            results.append(r)
            print(answers.report_case(r, args.verbose), flush=True)

    print()
    for name, share in answers.summarize_answers(results).items():
        print(f"{share:5.0%}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SpotCast's evaluations.")
    parser.add_argument(
        "suite", nargs="?", default="retrieval", choices=["retrieval", "judge", "answers"]
    )
    parser.add_argument("-k", type=int, default=3, help="How many search results count")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show every question"
    )
    parser.add_argument("--repeat", type=int, default=1, help="Runs per answer case")
    parser.add_argument("--only", nargs="*", help="Only these answer case ids")
    parser.add_argument("--no-judge", action="store_true", help="Rule checks only")
    args = parser.parse_args()

    {"retrieval": run_retrieval, "judge": run_judge, "answers": run_answers}[
        args.suite
    ](args)
