"""Tests for OCR helpers and drafts. The real vision model and LLM are never called here."""

import pytest
from PIL import Image

from spotcast.ocr import (
    DraftDocument,
    DraftSection,
    OcrText,
    character_error_rate,
    draft_markdown,
    edit_distance,
    normalize,
    read_text,
    word_recall,
    write_draft,
)
from spotcast.rag import split_document


def test_normalize_handles_turkish_capitals_quotes_and_bullets():
    assert normalize("• DİKKAT: Don’t   GO") == "dikkat: don't go"
    assert normalize("IŞIK") == normalize("ışık")  # same letters, different case


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("same", "same") == 0


def test_character_error_rate_and_word_recall():
    expected = "Never swim alone"
    assert character_error_rate(expected, "NEVER SWIM ALONE") == 0
    assert character_error_rate(expected, "Never swin alone") == pytest.approx(1 / 16)
    # Word recall ignores order and extra words, CER does not
    assert word_recall(expected, "alone swim never NOAA") == 1
    assert word_recall(expected, "Never alone") == pytest.approx(2 / 3)


def test_read_text_sends_the_image_and_drops_empty_lines(tmp_path):
    image = tmp_path / "sign.png"
    Image.new("RGB", (8, 8), "white").save(image)
    seen = {}

    def fake_vision(messages, output_model, **kwargs):
        seen["images"] = messages[1]["images"]
        return OcrText(lines=["  DİKKAT ", "", "Suya girmeyin."])

    assert read_text(image, ask=fake_vision) == ["DİKKAT", "Suya girmeyin."]
    assert seen["images"]


def make_draft() -> DraftDocument:
    return DraftDocument(
        title="Tabela",
        sections=[
            DraftSection(
                heading="Dip akıntısı: ne yapmalı", text="Akıntıya karşı yüzme."
            )
        ],
    )


def test_drafts_are_marked_and_never_reach_rag(tmp_path):
    path = write_draft(
        make_draft(), source="OCR: test.png", name="test", inbox=tmp_path
    )
    text = path.read_text(encoding="utf-8")

    assert "status: draft" in text
    assert "## Dip akıntısı: ne yapmalı" in text
    assert split_document(text, "inbox/test.md") == []  # human review comes first


def test_reviewed_draft_is_used_by_rag():
    reviewed = draft_markdown(make_draft(), "OCR: test.png").replace(
        "status: draft", "status: reviewed"
    )
    chunks = split_document(reviewed, "inbox/test.md")
    assert [c.section for c in chunks] == ["Dip akıntısı: ne yapmalı"]
    assert chunks[0].content == "Akıntıya karşı yüzme."
