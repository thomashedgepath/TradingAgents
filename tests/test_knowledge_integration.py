"""Real-Postgres tests for the corpus. Skipped unless a database is provided.

The rest of the suite mocks the driver, which proves the code's decisions but
not that the SQL runs. These do, against a live database:

    docker run -d --rm --name ta-kb-test -p 127.0.0.1:55433:5432 \\
      -e POSTGRES_USER=tradingagents -e POSTGRES_PASSWORD=tradingagents \\
      -e POSTGRES_DB=tradingagents pgvector/pgvector:pg17

    TRADINGAGENTS_TEST_KNOWLEDGE_DSN=postgresql://tradingagents:tradingagents@127.0.0.1:55433/tradingagents \\
      .venv/bin/python -m pytest tests/test_knowledge_integration.py -q

Plain ``postgres:17`` works too and exercises the full-text-only path.
"""

import copy
import os

import pytest

DSN = os.environ.get("TRADINGAGENTS_TEST_KNOWLEDGE_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DSN, reason="TRADINGAGENTS_TEST_KNOWLEDGE_DSN not set; no corpus database",
    ),
]

EMBED_DIM = 384


class StubEmbedder:
    """Deterministic 384-dim vectors: exercises the real SQL without torch."""

    dim = EMBED_DIM
    name = "stub"

    def encode(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * EMBED_DIM
            for i, ch in enumerate(text.lower()[:EMBED_DIM]):
                vec[i % EMBED_DIM] += (ord(ch) % 13) / 13.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out

    def encode_one(self, text):
        return self.encode([text])[0]


@pytest.fixture
def config():
    from tradingagents.default_config import DEFAULT_CONFIG

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["knowledge_db_url"] = DSN
    return cfg


@pytest.fixture
def conn(config):
    from tradingagents.knowledge.db import connection, ensure_schema

    with connection(config) as c:
        ensure_schema(c, embed_dim=EMBED_DIM)
        with c.cursor() as cur:
            cur.execute("DELETE FROM corpus_doc")
            cur.execute("DELETE FROM ingest_state")
        c.commit()
        yield c


def _docs():
    from tradingagents.knowledge.models import GLOBAL_TICKER, Document

    return [
        Document(
            doc_key="news:NVDA", ticker="NVDA", category="news", source="yfinance",
            title="NVDA news",
            body="Nvidia reported record data center revenue with gross margins expanding.",
        ),
        Document(
            doc_key="global_news", ticker=GLOBAL_TICKER, category="global_news",
            source="yfinance", title="Global macro news",
            body="The Federal Reserve held rates steady amid sticky services inflation.",
        ),
    ]


def test_schema_is_idempotent(conn, config):
    from tradingagents.knowledge.db import ensure_schema

    first = ensure_schema(conn, embed_dim=EMBED_DIM)
    second = ensure_schema(conn, embed_dim=EMBED_DIM)
    assert first == second


def test_dimension_drift_fails_loudly(conn):
    from tradingagents.knowledge.db import ensure_schema, has_vector

    if not has_vector(conn):
        pytest.skip("no pgvector on this database")
    with pytest.raises(ValueError, match="re-index"):
        ensure_schema(conn, embed_dim=EMBED_DIM * 2)


def test_unchanged_payload_is_not_reindexed(conn, config):
    from tradingagents.knowledge.ingest import ingest_documents

    docs = _docs()
    first = ingest_documents(docs, conn=conn, config=config, embedder=StubEmbedder())
    assert first.stored == len(docs)

    second = ingest_documents(docs, conn=conn, config=config, embedder=StubEmbedder())
    assert (second.stored, second.unchanged, second.chunks) == (0, len(docs), 0)


def test_changed_payload_is_reindexed(conn, config):
    from tradingagents.knowledge.ingest import ingest_documents
    from tradingagents.knowledge.models import Document

    docs = _docs()
    ingest_documents(docs, conn=conn, config=config, embedder=StubEmbedder())

    updated = Document(**{**docs[0].__dict__, "body": docs[0].body + " Updated on tariffs."})
    result = ingest_documents([updated], conn=conn, config=config, embedder=StubEmbedder())
    assert result.stored == 1 and result.chunks >= 1


def test_full_text_search_finds_a_document(conn, config):
    from tradingagents.knowledge.ingest import ingest_documents
    from tradingagents.knowledge.search import search

    ingest_documents(_docs(), conn=conn, config=config, embedder=StubEmbedder())
    hits = search("gross margins", ticker="NVDA", conn=conn, config=config,
                  embedder=StubEmbedder())
    assert hits
    assert hits[0].ticker == "NVDA"


def test_ticker_query_also_returns_corpus_wide_documents(conn, config):
    from tradingagents.knowledge.ingest import ingest_documents
    from tradingagents.knowledge.models import GLOBAL_TICKER
    from tradingagents.knowledge.search import search

    ingest_documents(_docs(), conn=conn, config=config, embedder=StubEmbedder())
    hits = search("Federal Reserve", ticker="NVDA", conn=conn, config=config,
                  embedder=StubEmbedder())
    assert any(h.ticker == GLOBAL_TICKER for h in hits)


def test_hybrid_fusion_marks_dual_matches(conn, config):
    from tradingagents.knowledge.db import has_vector
    from tradingagents.knowledge.ingest import ingest_documents
    from tradingagents.knowledge.search import search

    if not has_vector(conn):
        pytest.skip("no pgvector on this database")

    result = ingest_documents(_docs(), conn=conn, config=config, embedder=StubEmbedder())
    assert result.embedded == result.chunks > 0

    hits = search("record revenue margins", conn=conn, config=config,
                  embedder=StubEmbedder(), limit=5)
    assert hits
    # The fused query must attribute each hit to the arm(s) that found it.
    assert {h.matched_by for h in hits} <= {"fts", "vector", "both"}
    assert any(h.matched_by in ("both", "vector") for h in hits)


def test_search_without_vectors_still_works(conn, config):
    """Full-text arm alone, as on a database with no pgvector."""
    from tradingagents.knowledge.ingest import ingest_documents
    from tradingagents.knowledge.search import search

    ingest_documents(_docs(), conn=conn, config=config, embedder=None)
    hits = search("inflation", conn=conn, config=config, embedder=None)
    assert hits and all(h.matched_by == "fts" for h in hits)


def test_freshness_gate_hides_a_stale_payload(conn, config):
    from tradingagents.knowledge.ingest import ingest_documents
    from tradingagents.knowledge.search import latest_payload

    ingest_documents(_docs(), conn=conn, config=config, embedder=StubEmbedder())

    assert latest_payload("NVDA", "news", doc_key="news:NVDA", max_age_days=3,
                          conn=conn, config=config)
    # 0 admits nothing: the vendor reports a miss and the router falls through.
    assert latest_payload("NVDA", "news", doc_key="news:NVDA", max_age_days=0,
                          conn=conn, config=config) is None


def test_job_state_tracks_failures_then_resets(conn):
    from tradingagents.knowledge.state import load_states, record_attempt

    record_attempt(conn, "news", "NVDA", status="error", error="vendor down")
    record_attempt(conn, "news", "NVDA", status="error", error="still down")
    assert load_states(conn)[("news", "NVDA")]["consecutive_failures"] == 2

    record_attempt(conn, "news", "NVDA", status="ok", documents=1)
    state = load_states(conn)[("news", "NVDA")]
    assert state["consecutive_failures"] == 0
    assert state["last_success_at"] is not None


def test_prune_removes_documents_and_their_chunks(conn, config):
    from tradingagents.knowledge.ingest import ingest_documents, prune

    ingest_documents(_docs(), conn=conn, config=config, embedder=StubEmbedder())
    # retention_days=1 with everything just fetched must keep it all.
    assert prune(conn, retention_days=1) == 0

    with conn.cursor() as cur:
        cur.execute("UPDATE corpus_doc SET fetched_at = now() - interval '400 days'")
    conn.commit()

    assert prune(conn, retention_days=180) == len(_docs())
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM corpus_chunk")
        assert cur.fetchone()[0] == 0  # cascade
