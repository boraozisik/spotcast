"""Reading text from images (OCR) and turning it into knowledge documents for RAG.

    image ──OCR (vision model)──▶ lines of text
                                  └──LLM: organise + translate to Turkish──▶ knowledge/inbox/*.md
                                                                             (status: draft)
A person must review every draft and set `status: reviewed` before it is used by RAG:
OCR and translation can both be wrong, and a wrong safety rule is worse than none.

Run `uv run spotcast-ocr` to test OCR on samples/ocr/ and create drafts.
"""

import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from spotcast.config import PROJECT_ROOT
from spotcast.llm import CHAT_MODEL, VISION_MODEL, chat
from spotcast.rag import KNOWLEDGE_DIR

SAMPLES_DIR = PROJECT_ROOT / "samples" / "ocr"
INBOX_DIR = KNOWLEDGE_DIR / "inbox"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

OCR_PROMPT = """Transcribe all readable text in this image, line by line, top to bottom.
Copy it exactly: same language, same letters (keep Turkish letters like ç ğ ı İ ö ş ü),
same capitalisation. Do not translate, correct, summarise or add anything.
Skip tiny text inside logos."""

STRUCTURE_PROMPT = """You turn text read from a sign or brochure into a short knowledge
document for a water sports assistant in Turkey. Write in Turkish.

Rules:
- Use ONLY what the text says. Do not add advice, numbers or facts that are not in it.
- If the text is not Turkish, translate it faithfully.
- Group it into a few sections. Each heading must make sense alone, so name the topic in it
  (e.g. "Dip akıntısı: yakalanırsan ne yapmalı"), never just "Kurallar".
- Leave out web addresses, logos and decoration."""


class OcrText(BaseModel):
    lines: list[str] = Field(
        description="Every line of text, exactly as written, top to bottom"
    )


class DraftSection(BaseModel):
    heading: str = Field(description="Turkish heading that names the topic")
    text: str = Field(description="Turkish text of this section, only from the source")


class DraftDocument(BaseModel):
    title: str = Field(description="Turkish title of the document")
    sections: list[DraftSection]


# ---------- OCR ----------


def read_text(
    path: Path, ask: Callable[..., OcrText] = chat, model: str = VISION_MODEL
) -> list[str]:
    from spotcast.vision import encode_image  # same base64 helper as the sea photos

    messages = [
        {"role": "system", "content": OCR_PROMPT},
        {"role": "user", "content": "Read the text.", "images": [encode_image(path)]},
    ]
    answer = ask(
        messages, output_model=OcrText, model=model, temperature=0, max_tokens=1500
    )
    return [line.strip() for line in answer.lines if line.strip()]


# ---------- Measuring OCR quality against the answer key ----------


def normalize(text: str) -> str:
    """Make texts comparable: one case, plain quotes, no bullets, single spaces.

    Turkish İ/ı are mapped first, because Python's lower() turns "İ" into two characters.
    (So "I" and "ı" count as the same letter here; a small price for a fair comparison.)
    """
    text = text.replace("İ", "i").replace("I", "i").replace("ı", "i").lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[•◆▪●·]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: the fewest single-letter edits that turn a into b."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # delete
                    current[j - 1] + 1,  # insert
                    previous[j - 1] + (char_a != char_b),  # replace (free if same)
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(expected: str, actual: str) -> float:
    """Share of characters that are wrong, missing or extra. 0 = perfect. Order matters."""
    e, a = normalize(expected), normalize(actual)
    return edit_distance(e, a) / max(len(e), 1)


def word_recall(expected: str, actual: str) -> float:
    """Share of the expected words that were read at all, in any order."""
    words = lambda text: Counter(re.findall(r"[\w'.]+", normalize(text)))
    e, a = words(expected), words(actual)
    found = sum(min(count, a[word]) for word, count in e.items())
    return found / max(sum(e.values()), 1)


# ---------- From OCR text to a knowledge draft ----------


def structure_text(
    lines: list[str], ask: Callable[..., DraftDocument] = chat, model: str = CHAT_MODEL
) -> DraftDocument:
    messages = [
        {"role": "system", "content": STRUCTURE_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
    return ask(messages, output_model=DraftDocument, model=model, temperature=0.1)


def draft_markdown(doc: DraftDocument, source: str) -> str:
    """A knowledge document in the same format as the others, marked as a draft."""
    parts = [
        "---",
        f"title: {doc.title}",
        f"source: {source}",
        "status: draft",
        "---",
        "",
        f"# {doc.title}",
        "",
        "<!-- Created automatically from an image (OCR + LLM). Check every sentence against",
        "     the original, fix mistakes, then change status to reviewed and run",
        "     `uv run spotcast-index`. Drafts are never used by RAG. -->",
    ]
    for section in doc.sections:
        parts += ["", f"## {section.heading}", "", section.text]
    return "\n".join(parts) + "\n"


def write_draft(
    doc: DraftDocument, source: str, name: str, inbox: Path = INBOX_DIR
) -> Path:
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / f"{name}.md"
    path.write_text(draft_markdown(doc, source), encoding="utf-8")
    return path


def main() -> None:
    images = sorted(
        p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No images in {SAMPLES_DIR}.")
        return

    for image in images:
        print(f"\n=== {image.name} ===")
        lines = read_text(image)
        print("\n".join(f"  | {line}" for line in lines))

        answer_key = image.with_suffix(".txt")
        if answer_key.exists():
            expected, actual = answer_key.read_text(encoding="utf-8"), "\n".join(lines)
            print(
                f"  character error rate {character_error_rate(expected, actual):.1%}  "
                f"word recall {word_recall(expected, actual):.0%}"
            )

        doc = structure_text(lines)
        path = write_draft(
            doc, source=f"OCR: {image.name} (taslak, kontrol edilmedi)", name=image.stem
        )
        print(f"  draft → {path.relative_to(PROJECT_ROOT)}")
