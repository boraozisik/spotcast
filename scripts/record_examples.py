"""Record real agent conversations for the website's "Örnek konuşmalar" page.

    uv run python scripts/record_examples.py

The public site has no LLM (it can't run one for free), so it shows answers produced here,
on this computer, by the same agent code and the local model. They are labelled as such.
Needs Ollama and the database (docker compose up -d db).
"""

import json
import time
from datetime import datetime
from pathlib import Path

from spotcast.agent import Step, run_agent
from spotcast.llm import CHAT_MODEL

OUT = Path(__file__).resolve().parents[1] / "web" / "src" / "data" / "examples.json"

QUESTIONS = [
    (
        "Bilinen spotlar dışındaki bir yer",
        "Cumartesi Mersin sahilinde rüzgâr sörfü yapmak istiyorum, orta seviyedeyim. Uygun mu?",
    ),
    ("Öneri", "Bu hafta sonu rüzgâr sörfü için nereye gideyim? Orta seviyedeyim."),
    (
        "Desteklenmeyen seviye (kod cevaplar)",
        "Yeni başlıyorum, hafta sonu Alaçatı'ya gitsem olur mu?",
    ),
    ("Güvenlik bilgisi (RAG)", "Dip akıntısına yakalanırsam ne yapmalıyım?"),
    (
        "Güvenlik bilgisi (RAG)",
        "Rüzgâr karadan denize doğru eserse rüzgâr sörfü yapmak riskli mi?",
    ),
]


def summarize(step: Step) -> str:
    """A short, readable version of a tool result for the page."""
    r = step.result
    if "error" in r:
        return f"Hata: {r['error']}"
    if step.tool == "find_location":
        return "; ".join(f"{p['name']} ({p['region']})" for p in r["places"])
    if step.tool == "check_conditions":
        counts = ", ".join(f"{k}: {v}" for k, v in r["label_counts"].items())
        best = r["best_window"]["hours"] if r["best_window"] else "yok"
        return f"{r['day']} · saatler: {counts} · en iyi aralık: {best}"
    if step.tool == "recommend_spots":
        return " | ".join(
            f"{w['spot']} {w['day']} {w['hours']} (uygun {w['good_hours']}, sınırda {w['marginal_hours']})"
            for w in r["windows_best_first"]
        )
    if step.tool == "search_knowledge":
        return " | ".join(f"{e['title']} → {e['section']}" for e in r["excerpts"])
    return json.dumps(r, ensure_ascii=False)[:200]


def main() -> None:
    examples = []
    for topic, question in QUESTIONS:
        print(f"❓ {question}")
        started = time.monotonic()
        result = run_agent(question)
        seconds = time.monotonic() - started
        examples.append(
            {
                "topic": topic,
                "question": question,
                "steps": [
                    {
                        "tool": s.tool,
                        "arguments": s.arguments,
                        "summary": summarize(s),
                    }
                    for s in result.steps
                ],
                "llm_text": result.llm_text,
                # The full answer: the LLM's text plus what code adds (level note,
                # uncertainty notes, caution, source quote)
                "answer": result.answer,
                "seconds": round(seconds, 1),
            }
        )
        print(f"   {len(result.steps)} tool calls, {seconds:.0f} s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "recorded_at": f"{datetime.now():%Y-%m-%d %H:%M}",
                "model": CHAT_MODEL,
                "examples": examples,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
