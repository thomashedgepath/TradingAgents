"""Write path: upsert vendor payloads and (re)build their search index.

The scheduler re-runs the same fetches on a loop, so most cycles return text
that is byte-identical to what is already stored. Ingestion detects that by
content hash and touches only ``fetched_at`` — no re-chunking, no re-embedding.
That is what makes a tight schedule affordable: the cost of a cycle tracks how
much genuinely *changed*, not how many jobs ran.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta

from .chunking import chunk_text
from .db import connection, ensure_schema, has_vector
from .embedding import get_embedder, to_pgvector
from .models import Document, utcnow

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Per-cycle counters, surfaced in worker logs and ``--status``."""

    stored: int = 0
    unchanged: int = 0
    chunks: int = 0
    embedded: int = 0

    def __iadd__(self, other: IngestResult) -> IngestResult:
        self.stored += other.stored
        self.unchanged += other.unchanged
        self.chunks += other.chunks
        self.embedded += other.embedded
        return self


def _upsert_document(cur, doc: Document) -> tuple[int, bool]:
    """Insert or update one document. Returns ``(doc_id, content_changed)``.

    The previous hash is read first rather than inferred from the upsert, so
    "changed" means the text actually differs — not merely that a row existed.
    """
    now = utcnow()
    cur.execute("SELECT id, content_hash FROM corpus_doc WHERE doc_key = %s", (doc.doc_key,))
    row = cur.fetchone()

    if row is not None and row[1] == doc.content_hash:
        cur.execute("UPDATE corpus_doc SET fetched_at = %s WHERE id = %s", (now, row[0]))
        return row[0], False

    cur.execute(
        """
        INSERT INTO corpus_doc
            (doc_key, ticker, category, source, title, body, url,
             content_hash, published_at, fetched_at, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (doc_key) DO UPDATE SET
            ticker       = EXCLUDED.ticker,
            category     = EXCLUDED.category,
            source       = EXCLUDED.source,
            title        = EXCLUDED.title,
            body         = EXCLUDED.body,
            url          = EXCLUDED.url,
            content_hash = EXCLUDED.content_hash,
            published_at = EXCLUDED.published_at,
            fetched_at   = EXCLUDED.fetched_at,
            metadata     = EXCLUDED.metadata
        RETURNING id
        """,
        (
            doc.doc_key, doc.ticker, doc.category, doc.source, doc.title, doc.body,
            doc.url, doc.content_hash, doc.published_at, now, json.dumps(doc.metadata),
        ),
    )
    return cur.fetchone()[0], True


def _index_document(cur, doc_id: int, doc: Document, embedder, vector_ok: bool) -> tuple[int, int]:
    """Replace a document's chunks. Returns ``(chunks_written, embedded)``.

    Delete-then-insert rather than a diff: chunk boundaries shift when the text
    changes, so ordinals are not stable across versions and a partial update
    would leave orphaned slices of the previous payload in the index.
    """
    cur.execute("DELETE FROM corpus_chunk WHERE doc_id = %s", (doc_id,))

    # The title carries the ticker and window, which the body often does not
    # repeat; prefixing it makes chunk 0 findable by name as well as content.
    chunks = chunk_text(f"{doc.title}\n\n{doc.body}" if doc.title else doc.body)
    if not chunks:
        return 0, 0

    vectors: list[list[float]] = []
    if vector_ok and embedder is not None:
        vectors = embedder.encode(chunks)
        if vectors and len(vectors) != len(chunks):
            logger.warning(
                "embedder returned %d vectors for %d chunks; storing without vectors",
                len(vectors), len(chunks),
            )
            vectors = []

    if vectors:
        cur.executemany(
            "INSERT INTO corpus_chunk (doc_id, ordinal, text, embedding) "
            "VALUES (%s, %s, %s, %s::vector)",
            [
                (doc_id, i, text, to_pgvector(vec))
                for i, (text, vec) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
        return len(chunks), len(chunks)

    cur.executemany(
        "INSERT INTO corpus_chunk (doc_id, ordinal, text) VALUES (%s, %s, %s)",
        [(doc_id, i, text) for i, text in enumerate(chunks)],
    )
    return len(chunks), 0


def ingest_documents(
    documents: list[Document],
    *,
    conn=None,
    config: dict | None = None,
    embedder=None,
) -> IngestResult:
    """Store ``documents`` and refresh the index for those whose text changed.

    Pass ``conn``/``embedder`` to reuse the worker's connection and loaded
    model across jobs; omit them for a one-off call.
    """
    result = IngestResult()
    if not documents:
        return result

    if conn is None:
        with connection(config) as own_conn:
            ensure_schema(own_conn, embed_dim=_embed_dim(config))
            return ingest_documents(
                documents, conn=own_conn, config=config, embedder=embedder
            )

    if embedder is None:
        embedder = get_embedder(config)
    vector_ok = has_vector(conn)

    with conn.cursor() as cur:
        for doc in documents:
            doc_id, changed = _upsert_document(cur, doc)
            if not changed:
                result.unchanged += 1
                continue
            written, embedded = _index_document(cur, doc_id, doc, embedder, vector_ok)
            result.stored += 1
            result.chunks += written
            result.embedded += embedded
    conn.commit()
    return result


def prune(conn, *, retention_days: int | None) -> int:
    """Delete documents last seen longer than ``retention_days`` ago.

    Keyed on ``fetched_at``, not ``published_at``: a document the schedule still
    refreshes is live regardless of how old the underlying article is, and
    dropping it would leave the read path with a hole the vendor already covers.
    Chunks follow via ON DELETE CASCADE.
    """
    if not retention_days:
        return 0
    cutoff = utcnow() - timedelta(days=int(retention_days))
    with conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_doc WHERE fetched_at < %s", (cutoff,))
        removed = cur.rowcount or 0
    conn.commit()
    if removed:
        logger.info("pruned %d documents older than %d days", removed, retention_days)
    return removed


def _embed_dim(config: dict | None) -> int:
    if config is None:
        from tradingagents.dataflows.config import get_config

        config = get_config()
    return int(config.get("knowledge_embed_dim", 384))
