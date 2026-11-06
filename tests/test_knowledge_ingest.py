"""Ingestion must be cheap when nothing changed and correct when it did.

The scheduler re-runs the same fetches on a loop, so the content-hash
short-circuit is what keeps a tight cadence affordable: an unchanged payload
must not be re-chunked or re-embedded.
"""

import pytest

from tradingagents.knowledge import ingest as ingest_module
from tradingagents.knowledge.ingest import ingest_documents, prune
from tradingagents.knowledge.models import Document

from ._knowledge_fakes import FakeConn, FakeEmbedder


def _doc(body="Some news body that is long enough to index.", key="news:NVDA"):
    return Document(
        doc_key=key, ticker="NVDA", category="news", source="yfinance",
        title="NVDA news", body=body,
    )


def _conn_with_existing(hash_value=None, doc_id=1):
    """Fake where the SELECT finds an existing row with ``hash_value``."""
    return FakeConn(responses={
        "SELECT ID, CONTENT_HASH": (doc_id, hash_value) if hash_value else None,
        "INSERT INTO CORPUS_DOC": (doc_id,),
    })


@pytest.fixture(autouse=True)
def _no_vector_probe(monkeypatch):
    # has_vector() would query pg_attribute; the arm under test is chosen
    # per-test by overriding this.
    monkeypatch.setattr(ingest_module, "has_vector", lambda conn: True)


@pytest.mark.unit
def test_unchanged_document_is_not_reindexed():
    doc = _doc()
    conn = _conn_with_existing(hash_value=doc.content_hash)

    result = ingest_documents([doc], conn=conn, config={}, embedder=FakeEmbedder())

    assert (result.unchanged, result.stored, result.chunks) == (1, 0, 0)
    # Only the fetched_at touch — no delete, no chunk insert.
    assert conn.ran("UPDATE corpus_doc SET fetched_at")
    assert not conn.ran("DELETE FROM corpus_chunk")
    assert conn.many == []


@pytest.mark.unit
def test_changed_document_is_reindexed_and_embedded():
    doc = _doc()
    conn = _conn_with_existing(hash_value="a-different-hash")
    embedder = FakeEmbedder()

    result = ingest_documents([doc], conn=conn, config={}, embedder=embedder)

    assert result.stored == 1
    assert result.chunks >= 1
    assert result.embedded == result.chunks
    # Stale chunks are cleared before the new ones land: boundaries shift.
    assert conn.ran("DELETE FROM corpus_chunk")
    insert_sql, rows = conn.many[0]
    assert "embedding" in insert_sql and "::vector" in insert_sql
    assert len(rows) == result.chunks
    assert embedder.calls


@pytest.mark.unit
def test_new_document_is_inserted():
    conn = _conn_with_existing(hash_value=None)
    result = ingest_documents([_doc()], conn=conn, config={}, embedder=FakeEmbedder())

    assert (result.stored, result.unchanged) == (1, 0)
    assert conn.ran("INSERT INTO corpus_doc")


@pytest.mark.unit
def test_falls_back_to_text_only_when_embedder_fails():
    conn = _conn_with_existing(hash_value="old")
    result = ingest_documents(
        [_doc()], conn=conn, config={}, embedder=FakeEmbedder(fail=True),
    )

    assert result.stored == 1
    assert result.chunks >= 1
    assert result.embedded == 0
    insert_sql, _ = conn.many[0]
    assert "embedding" not in insert_sql


@pytest.mark.unit
def test_no_vector_column_means_no_vector_insert(monkeypatch):
    monkeypatch.setattr(ingest_module, "has_vector", lambda conn: False)
    conn = _conn_with_existing(hash_value="old")
    embedder = FakeEmbedder()

    result = ingest_documents([_doc()], conn=conn, config={}, embedder=embedder)

    assert result.embedded == 0
    assert "embedding" not in conn.many[0][0]
    # And the model is never invoked, so a full-text deployment pays nothing.
    assert embedder.calls == []


@pytest.mark.unit
def test_title_is_indexed_with_the_body():
    conn = _conn_with_existing(hash_value="old")
    ingest_documents([_doc()], conn=conn, config={}, embedder=FakeEmbedder(fail=True))
    chunk_texts = [row[2] for row in conn.many[0][1]]
    assert any("NVDA news" in t for t in chunk_texts)


@pytest.mark.unit
def test_empty_document_list_is_a_noop():
    conn = FakeConn()
    result = ingest_documents([], conn=conn, config={})
    assert (result.stored, result.unchanged) == (0, 0)
    assert conn.statements == []


@pytest.mark.unit
def test_prune_is_a_noop_without_retention():
    conn = FakeConn(rowcount=5)
    assert prune(conn, retention_days=None) == 0
    assert conn.statements == []


@pytest.mark.unit
def test_prune_deletes_by_fetched_at():
    conn = FakeConn(rowcount=3)
    assert prune(conn, retention_days=30) == 3
    assert conn.ran("DELETE FROM corpus_doc WHERE fetched_at <")
