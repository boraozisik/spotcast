"""Retrieval for RAG: split the knowledge documents, store their vectors, find the relevant ones.

    knowledge/*.md ──split──▶ chunks ──embed──▶ vectors ──store──▶ PostgreSQL (pgvector)
    question ──embed──▶ vector ──nearest chunks──▶ facts for the LLM prompt, with sources

Run `uv run spotcast-index` after changing anything in knowledge/.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    or_,
    select,
    text,
)

from spotcast.config import PROJECT_ROOT
from spotcast.llm import EMBED_DIMENSIONS, embed

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

metadata = MetaData()
chunks_table = Table(
    "knowledge_chunks",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("file", String, nullable=False),  # e.g. spots/alacati.md
    Column("title", String, nullable=False),  # document title
    Column("section", String, nullable=False),  # the "## ..." heading
    Column("spot", String),  # NULL for general documents (Beaufort, safety)
    Column("source", String, nullable=False),  # where the information comes from
    Column("content", Text, nullable=False),
    Column("embedding", Vector(EMBED_DIMENSIONS), nullable=False),
)


@dataclass(frozen=True)
class Chunk:
    file: str
    title: str
    section: str
    spot: str | None
    source: str
    content: str

    def text_for_embedding(self) -> str:
        """The heading carries context (spot name, topic), so it is embedded with the body."""
        return f"{self.title}\n{self.section}\n\n{self.content}"


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    distance: float  # cosine distance: 0 = same meaning, bigger = less related


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Read the `key: value` lines between the two `---` lines at the top of a file."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, flags=re.DOTALL)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, match.group(2)


def split_document(raw: str, file: str) -> list[Chunk]:
    """One chunk per `## ` section. HTML comments are notes for writers, so they are removed.

    Documents with `status: draft` give no chunks: a person must review them first.
    """
    meta, body = parse_front_matter(raw)
    if meta.get("status") == "draft":
        return []
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)

    chunks = []
    # Splitting on lines that start with "## " gives [text before first section, section, ...]
    for part in re.split(r"^## ", body, flags=re.MULTILINE)[1:]:
        heading, _, content = part.partition("\n")
        content = content.strip()
        if content:  # an empty section has nothing to say
            chunks.append(
                Chunk(
                    file=file,
                    title=meta.get("title", file),
                    section=heading.strip(),
                    spot=meta.get("spot") or None,
                    source=meta.get("source", file),
                    content=content,
                )
            )
    return chunks


def load_knowledge(directory: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(directory.rglob("*.md")):
        file = path.relative_to(directory).as_posix()
        chunks += split_document(path.read_text(encoding="utf-8"), file)
    return chunks


def rebuild_index(
    engine: Engine,
    chunks: list[Chunk],
    embed_texts: Callable[[list[str]], list[list[float]]] = embed,
) -> int:
    """Replace everything in the table with these chunks. Simple and always consistent."""
    vectors = embed_texts([c.text_for_embedding() for c in chunks])
    rows = [
        {
            "file": c.file,
            "title": c.title,
            "section": c.section,
            "spot": c.spot,
            "source": c.source,
            "content": c.content,
            "embedding": v,
        }
        for c, v in zip(chunks, vectors, strict=True)
    ]
    with (
        engine.begin() as conn
    ):  # one transaction: readers never see a half-built table
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        metadata.create_all(conn)
        conn.execute(chunks_table.delete())
        if rows:
            conn.execute(chunks_table.insert(), rows)
    return len(rows)


def search(
    engine: Engine,
    question: str,
    k: int = 4,
    spots: list[str] | None = None,
    embed_texts: Callable[[list[str]], list[list[float]]] = embed,
) -> list[SearchResult]:
    """The k chunks closest in meaning to the question.

    With `spots`, only those spots' guides and the general documents are searched, so a
    question about Alaçatı never gets an answer from the Akyaka guide.
    """
    query_vector = embed_texts([question])[0]
    distance = chunks_table.c.embedding.cosine_distance(query_vector).label("distance")
    query = select(chunks_table, distance).order_by(distance).limit(k)
    if spots:
        query = query.where(
            or_(chunks_table.c.spot.in_(spots), chunks_table.c.spot.is_(None))
        )

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [
        SearchResult(
            chunk=Chunk(
                file=row["file"],
                title=row["title"],
                section=row["section"],
                spot=row["spot"],
                source=row["source"],
                content=row["content"],
            ),
            distance=float(row["distance"]),
        )
        for row in rows
    ]


def main() -> None:
    """Rebuild the index from knowledge/ and try a few questions."""
    from spotcast.db import get_engine

    engine = get_engine()
    chunks = load_knowledge()
    count = rebuild_index(engine, chunks)
    print(f"Indexed {count} chunks from {KNOWLEDGE_DIR.name}/\n")

    examples = [
        ("Alaçatı'da rüzgâr hangi yönden eser?", None),
        ("Rüzgâr karadan denize doğru eserse ne olur?", None),
        ("25 knot rüzgâr kaç Beaufort, orta seviye için nasıl?", None),
        ("Hangi ayda gitmeliyim?", ["Akyaka"]),
    ]
    for question, spots in examples:
        where = f" (spots: {', '.join(spots)})" if spots else ""
        print(f"? {question}{where}")
        for result in search(engine, question, k=3, spots=spots):
            print(
                f"   {result.distance:.3f}  {result.chunk.file} → {result.chunk.section}"
            )
        print()
