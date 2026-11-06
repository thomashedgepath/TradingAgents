"""Postgres connection handling and idempotent schema management.

``psycopg`` is imported lazily and never at module import time: the core install
does not depend on it, and both the vendor read path and the search tool must be
importable (and degrade) on a machine that has neither the driver nor a database.
Every entry point here raises :class:`KnowledgeUnavailable` rather than letting a
driver ``ImportError`` or a connection refusal escape.
"""

from __future__ import annotations

import atexit
import logging
import re
import threading

logger = logging.getLogger(__name__)

# One pool per DSN, created on first use. Keyed by DSN so a test or a caller
# that switches databases gets its own pool instead of silently reusing another.
_POOLS: dict[str, object] = {}
_POOL_LOCK = threading.Lock()

# Per-DSN record of what the deployed schema actually supports, so the read and
# write paths stop probing after the first connection.
_SCHEMA_STATE: dict[str, dict] = {}


class KnowledgeUnavailable(RuntimeError):
    """The knowledge corpus cannot be reached or is not configured.

    Callers treat this as "no corpus" and fall back to their live path; it is
    never a reason to abort an analysis.
    """


def _require_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise KnowledgeUnavailable(
            "psycopg is not installed. Install the knowledge extra: "
            'pip install ".[knowledge]"'
        ) from exc
    return psycopg, dict_row


def get_dsn(config: dict | None = None) -> str:
    """Return the configured DSN, or raise if the corpus is switched off."""
    if config is None:
        from tradingagents.dataflows.config import get_config

        config = get_config()
    dsn = config.get("knowledge_db_url")
    if not dsn:
        raise KnowledgeUnavailable(
            "knowledge_db_url is not set; the background knowledge base is "
            "disabled. Set TRADINGAGENTS_KNOWLEDGE_DB_URL to enable it."
        )
    return dsn


def get_pool(config: dict | None = None):
    """Get or create the connection pool for the configured DSN.

    Falls back to unpooled connections when ``psycopg_pool`` is absent, so the
    base ``psycopg`` install is enough to read the corpus.
    """
    dsn = get_dsn(config)
    with _POOL_LOCK:
        if dsn in _POOLS:
            return _POOLS[dsn]
        _require_psycopg()
        try:
            from psycopg_pool import ConnectionPool
        except ImportError:
            _POOLS[dsn] = None  # signals "connect directly"
            return None
        try:
            pool = ConnectionPool(dsn, min_size=0, max_size=4, open=True, timeout=10)
        except Exception as exc:
            raise KnowledgeUnavailable(f"could not open a connection pool: {exc}") from exc
        # A pool runs background threads. Without this, a CLI process that
        # merely searched the corpus once exits with psycopg warnings about
        # threads it could not stop.
        atexit.register(close_pools)
        _POOLS[dsn] = pool
        return pool


def close_pools() -> None:
    """Close every open pool. Used by the worker on shutdown and by tests."""
    with _POOL_LOCK:
        for pool in _POOLS.values():
            if pool is not None:
                try:
                    pool.close()
                except Exception:  # pragma: no cover - best-effort shutdown
                    logger.debug("pool close failed", exc_info=True)
        _POOLS.clear()
        _SCHEMA_STATE.clear()


class _Connection:
    """Context manager yielding a connection from the pool, or a direct one."""

    def __init__(self, config: dict | None):
        self._config = config
        self._pool = None
        self._ctx = None
        self._conn = None

    def __enter__(self):
        dsn = get_dsn(self._config)
        self._pool = get_pool(self._config)
        try:
            if self._pool is not None:
                self._ctx = self._pool.connection()
                self._conn = self._ctx.__enter__()
            else:
                psycopg, _ = _require_psycopg()
                self._conn = psycopg.connect(dsn, connect_timeout=10)
        except KnowledgeUnavailable:
            raise
        except Exception as exc:
            raise KnowledgeUnavailable(f"could not connect to the corpus: {exc}") from exc
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        if self._conn is not None:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
            self._conn.close()
        return False


def connection(config: dict | None = None) -> _Connection:
    """Open a corpus connection. Raises ``KnowledgeUnavailable`` if it can't."""
    return _Connection(config)


def dict_cursor(conn):
    """Cursor yielding dict rows, so callers index by column name."""
    _, dict_row = _require_psycopg()
    return conn.cursor(row_factory=dict_row)


# --- schema -----------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS corpus_doc (
    id            BIGSERIAL PRIMARY KEY,
    doc_key       TEXT        NOT NULL UNIQUE,
    ticker        TEXT        NOT NULL,
    category      TEXT        NOT NULL,
    source        TEXT        NOT NULL,
    title         TEXT        NOT NULL DEFAULT '',
    body          TEXT        NOT NULL,
    url           TEXT,
    content_hash  TEXT        NOT NULL,
    published_at  TIMESTAMPTZ,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS corpus_doc_lookup_idx
    ON corpus_doc (ticker, category, fetched_at DESC);
CREATE INDEX IF NOT EXISTS corpus_doc_fetched_idx ON corpus_doc (fetched_at);

CREATE TABLE IF NOT EXISTS corpus_chunk (
    id       BIGSERIAL PRIMARY KEY,
    doc_id   BIGINT NOT NULL REFERENCES corpus_doc(id) ON DELETE CASCADE,
    ordinal  INT    NOT NULL,
    text     TEXT   NOT NULL,
    tsv      tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (doc_id, ordinal)
);

CREATE INDEX IF NOT EXISTS corpus_chunk_tsv_idx ON corpus_chunk USING GIN (tsv);
CREATE INDEX IF NOT EXISTS corpus_chunk_doc_idx ON corpus_chunk (doc_id);

CREATE TABLE IF NOT EXISTS ingest_state (
    job                  TEXT NOT NULL,
    ticker               TEXT NOT NULL,
    last_attempt_at      TIMESTAMPTZ,
    last_success_at      TIMESTAMPTZ,
    last_status          TEXT,
    last_error           TEXT,
    consecutive_failures INT  NOT NULL DEFAULT 0,
    documents            INT  NOT NULL DEFAULT 0,
    PRIMARY KEY (job, ticker)
);
"""


def _vector_dim(conn) -> int | None:
    """Deployed width of ``corpus_chunk.embedding``, or None if absent."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.atttypmod
              FROM pg_attribute a
              JOIN pg_class c ON c.oid = a.attrelid
             WHERE c.relname = 'corpus_chunk'
               AND a.attname = 'embedding'
               AND NOT a.attisdropped
            """
        )
        row = cur.fetchone()
    # pgvector stores the declared dimension directly in atttypmod.
    return int(row[0]) if row and row[0] and row[0] > 0 else None


def ensure_schema(conn, *, embed_dim: int = 384) -> dict:
    """Create or update the corpus schema. Idempotent and safe to call often.

    Returns the schema state, notably ``{"vector": bool}``. The vector column is
    added opportunistically: on a stock ``postgres`` image without the pgvector
    extension the corpus still works, full-text only.
    """
    if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim <= 0:
        raise ValueError(f"knowledge_embed_dim must be a positive int, got {embed_dim!r}")

    with conn.cursor() as cur:
        cur.execute(_DDL)
    # Committed before the extension is attempted: a failed CREATE EXTENSION
    # aborts the transaction, and rolling that back would take the base tables
    # with it — leaving a full-text-only deployment with no schema at all.
    conn.commit()

    state = {"vector": False, "dim": None}

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.warning(
            "pgvector unavailable (%s); the corpus will run full-text only. "
            "Use the pgvector/pgvector image to enable semantic search.", exc,
        )
        _SCHEMA_STATE[_dsn_key(conn)] = state
        return state

    existing = _vector_dim(conn)
    if existing is not None and existing != embed_dim:
        # Silently keeping the old width would make every insert fail at
        # runtime with an opaque dimension error, so say exactly what to do.
        raise ValueError(
            f"corpus_chunk.embedding is vector({existing}) but "
            f"knowledge_embed_dim is {embed_dim}. Either set the config back to "
            f"{existing}, or re-index: DROP the embedding column and re-run the "
            f"worker with --backfill."
        )

    with conn.cursor() as cur:
        # embed_dim is validated as a positive int above, so interpolating it
        # is safe; ADD COLUMN takes no bind parameters.
        cur.execute(
            f"ALTER TABLE corpus_chunk ADD COLUMN IF NOT EXISTS embedding vector({embed_dim})"
        )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS corpus_chunk_embedding_idx "
                "ON corpus_chunk USING hnsw (embedding vector_cosine_ops)"
            )
    except Exception as exc:
        # HNSW needs pgvector >= 0.5. Without the index search still works,
        # just with an exact scan, so this is a warning and not a failure.
        conn.rollback()
        logger.warning("could not create the HNSW index (%s); falling back to exact scan", exc)

    conn.commit()
    state = {"vector": True, "dim": embed_dim}
    _SCHEMA_STATE[_dsn_key(conn)] = state
    return state


def _dsn_key(conn) -> str:
    """Best-effort stable key for a connection's database, for schema caching."""
    info = getattr(conn, "info", None)
    dsn = getattr(info, "dsn", None)
    if isinstance(dsn, str) and dsn:
        # Never key on (or log) a DSN containing credentials.
        return re.sub(r"password=[^\s]*", "password=***", dsn)
    return "default"


def schema_state(conn) -> dict:
    """Cached schema state for ``conn``, probing once if not yet known."""
    key = _dsn_key(conn)
    if key not in _SCHEMA_STATE:
        dim = _vector_dim(conn)
        _SCHEMA_STATE[key] = {"vector": dim is not None, "dim": dim}
    return _SCHEMA_STATE[key]


def has_vector(conn) -> bool:
    """Whether semantic search is available on this database."""
    try:
        return bool(schema_state(conn)["vector"])
    except Exception:
        return False
