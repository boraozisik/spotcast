"""Tests for splitting knowledge documents. No database or Ollama needed."""

from spotcast.locations import SPOTS
from spotcast.rag import load_knowledge, parse_front_matter, split_document

DOC = """---
title: Test rehberi
spot: Alaçatı
source: test notları
---

# Test rehberi

<!-- a note for writers, must not reach the LLM -->

## Alaçatı: rüzgâr

Kuzeyden eser.
<!-- another note -->

## Alaçatı: boş bölüm

<!-- only a comment, so this section has no content -->

## Alaçatı: dalga

Küçüktür.
"""


def test_front_matter_is_read():
    meta, body = parse_front_matter(DOC)
    assert meta == {
        "title": "Test rehberi",
        "spot": "Alaçatı",
        "source": "test notları",
    }
    assert body.lstrip().startswith("# Test rehberi")


def test_one_chunk_per_section_without_comments_or_empty_sections():
    chunks = split_document(DOC, "spots/test.md")

    assert [c.section for c in chunks] == ["Alaçatı: rüzgâr", "Alaçatı: dalga"]
    assert chunks[0].content == "Kuzeyden eser."  # the HTML comment is gone
    assert chunks[0].spot == "Alaçatı"
    assert chunks[0].source == "test notları"
    assert "Alaçatı: rüzgâr" in chunks[0].text_for_embedding()  # heading gives context


def test_document_without_spot_is_general():
    general = DOC.replace("spot: Alaçatı\n", "")
    assert split_document(general, "general.md")[0].spot is None


def test_real_knowledge_folder():
    """Checks the documents themselves, not just the code."""
    chunks = load_knowledge()
    spot_names = {spot.name for spot in SPOTS}

    assert len(chunks) > 0
    assert all(c.content and "<!--" not in c.content for c in chunks)
    for chunk in chunks:
        if chunk.spot is not None:
            assert chunk.spot in spot_names, f"unknown spot in {chunk.file}"
            # A section must make sense on its own when search returns it
            assert chunk.spot in chunk.section, (
                f"{chunk.file}: add the spot name to '{chunk.section}'"
            )


def test_every_reviewed_document_reaches_rag():
    """A reviewed document with no `## ` heading gives no chunks and is silently ignored."""
    from spotcast.rag import KNOWLEDGE_DIR, parse_front_matter

    used = {c.file for c in load_knowledge()}
    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        file = path.relative_to(KNOWLEDGE_DIR).as_posix()
        meta, _ = parse_front_matter(path.read_text(encoding="utf-8"))
        if meta.get("status") != "draft":
            assert file in used, f"{file} has no '## ' section, so RAG never uses it"


def test_keyword_query_keeps_only_safe_words():
    from spotcast.rag import keyword_query

    assert (
        keyword_query("Fırtınada sörf yapılır mı?") == "Fırtınada | sörf | yapılır | mı"
    )
    assert (
        keyword_query("'; DROP TABLE x; --") == "DROP | TABLE"
    )  # no SQL or query syntax
    assert keyword_query("?!") is None


def test_reciprocal_rank_fusion_rewards_agreement():
    from spotcast.rag import reciprocal_rank_fusion

    by_meaning = [1, 2, 3]
    by_words = [3, 4]
    # 3 is in both lists: 1/63 + 1/61 beats even 1, which is first in only one list
    assert reciprocal_rank_fusion(by_meaning, by_words)[:2] == [3, 1]
    assert reciprocal_rank_fusion([7, 8]) == [7, 8]
