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
    func,
    literal_column,
    or_,
    select,
    text,
)

from spotcast.beaufort import add_beaufort_to_text
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


FUSION_K = 60  # the usual constant in Reciprocal Rank Fusion; dampens rank differences
CANDIDATES = 10  # how many results each search method contributes to the fusion


def keyword_query(question: str) -> str | None:
    """Words joined with OR for PostgreSQL full-text search: 'fırtınada | sörf | yapılır'.

    Only letters and digits are kept, so the user's text can't break the query syntax.
    """
    words = [w for w in re.findall(r"\w+", question) if len(w) > 1]
    return " | ".join(words) if words else None


def reciprocal_rank_fusion(*rankings: list[int]) -> list[int]:
    """Merge ranked id lists: each list gives 1 / (FUSION_K + rank) to each id it contains.

    Uses only positions, so scores on different scales (distance vs. text rank) never mix.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (FUSION_K + rank)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


TURKISH = literal_column("'turkish'::regconfig")
# The searchable text of a chunk, with Turkish stemming ("fırtınada" → "fırtına")
DOCUMENT = func.to_tsvector(
    TURKISH,
    chunks_table.c.title + " " + chunks_table.c.section + " " + chunks_table.c.content,
)
MAX_TERM_SHARE = 0.3  # a word in more than 30% of chunks says little about any of them


def _rare_terms(conn, words: str) -> str | None:
    """Keep only the question's words that are rare in the knowledge base.

    PostgreSQL's ts_rank ignores how common a word is, so "rüzgâr" (in 28 of 34 chunks)
    counted as much as "akyaka". Dropping common words is a simple stand-in for IDF.
    """
    lexemes = (
        conn.execute(
            select(
                func.unnest(func.tsvector_to_array(func.to_tsvector(TURKISH, words)))
            )
        )
        .scalars()
        .all()
    )
    total = conn.execute(select(func.count()).select_from(chunks_table)).scalar_one()
    counts = dict(
        conn.execute(
            text(
                "SELECT word, ndoc FROM ts_stat($$SELECT to_tsvector('turkish', "
                "title || ' ' || section || ' ' || content) FROM knowledge_chunks$$)"
            )
        ).all()
    )
    rare = [w for w in lexemes if counts.get(w, 0) <= MAX_TERM_SHARE * total]
    # Lexemes are already stemmed, so they are queried with the 'simple' config as-is
    return " | ".join("'" + w.replace("'", "''") + "'" for w in rare) or None


def search(
    engine: Engine,
    question: str,
    k: int = 4,
    spots: list[str] | None = None,
    embed_texts: Callable[[list[str]], list[list[float]]] = embed,
    mode: str = "hybrid",
) -> list[SearchResult]:
    """The k chunks that best answer the question.

    mode="vector": by meaning only (embeddings).
    mode="hybrid": meaning AND exact words (PostgreSQL full-text search, Turkish stemming),
    merged with Reciprocal Rank Fusion. Words like "112" or "hamle" are found reliably.

    With `spots`, only those spots' guides and the general documents are searched, so a
    question about Alaçatı never gets an answer from the Akyaka guide.
    """
    # Query expansion: "25 knot" → "25 knot (6 Beaufort)". Code does the number lookup that
    # neither embeddings nor word search can do.
    question = add_beaufort_to_text(question)
    query_vector = embed_texts([question])[0]
    distance = chunks_table.c.embedding.cosine_distance(query_vector).label("distance")

    def only_allowed_spots(query):
        if not spots:
            return query
        return query.where(
            or_(chunks_table.c.spot.in_(spots), chunks_table.c.spot.is_(None))
        )

    with engine.connect() as conn:
        by_meaning = (
            conn.execute(
                only_allowed_spots(select(chunks_table.c.id).order_by(distance)).limit(
                    CANDIDATES
                )
            )
            .scalars()
            .all()
        )
        rankings = [list(by_meaning)]

        words = keyword_query(question) if mode == "hybrid" else None
        rare = _rare_terms(conn, words) if words else None
        if rare:
            terms = func.to_tsquery(literal_column("'simple'::regconfig"), rare)
            text_rank = func.ts_rank_cd(DOCUMENT, terms)
            by_words = (
                conn.execute(
                    only_allowed_spots(
                        select(chunks_table.c.id)
                        .where(DOCUMENT.op("@@")(terms))
                        .order_by(text_rank.desc())
                    ).limit(CANDIDATES)
                )
                .scalars()
                .all()
            )
            rankings.append(list(by_words))

        best_ids = reciprocal_rank_fusion(*rankings)[:k]
        rows = (
            conn.execute(
                select(chunks_table, distance).where(chunks_table.c.id.in_(best_ids))
            )
            .mappings()
            .all()
        )

    rows = sorted(rows, key=lambda row: best_ids.index(row["id"]))
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
