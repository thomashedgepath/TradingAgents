"""Hybrid search must fuse both arms when it can and degrade to one when it can't.

The vector arm is conditional on pgvector *and* a loadable embedding model, so
the query is built at runtime; these tests pin the shape it takes in each case,
plus the filter semantics both arms must share.
"""

from datetime import datetime, timezone

import pytest

from tradingagents.knowledge import search as search_module
from tradingagents.knowledge.models import GLOBAL_TICKER, SearchHit
from tradingagents.knowledge.search import RRF_K, format_hits, search

from ._knowledge_fakes import FakeConn, FakeEmbedder


@pytest.fixture
def captured(monkeypatch):
    """Capture the SQL and params search() would execute, returning no rows."""
    calls = []

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def execute(self, sql, params=None):
            calls.append((" ".join(str(sql).split()), params))
        def fetchall(self): return []

    monkeypatch.setattr(search_module, "dict_cursor", lambda conn: Cur())
    return calls


@pytest.mark.unit
def test_blank_query_short_circuits(captured):
    assert search("   ", conn=FakeConn()) == []
    assert captured == []


@pytest.mark.unit
def test_full_text_only_when_no_vector_column(captured, monkeypatch):
    monkeypatch.setattr(search_module, "has_vector", lambda conn: False)

    search("margin pressure", conn=FakeConn(), config={})

    sql, params = captured[0]
    assert "fts AS" in sql
    assert "vec AS" not in sql
    assert "qvec" not in params
    assert "websearch_to_tsquery" in sql


@pytest.mark.unit
def test_both_arms_fused_when_vector_available(captured, monkeypatch):
    monkeypatch.setattr(search_module, "has_vector", lambda conn: True)

    search("margin pressure", conn=FakeConn(), config={}, embedder=FakeEmbedder())

    sql, params = captured[0]
    assert "fts AS" in sql and "vec AS" in sql
    # Reciprocal rank fusion, one term per arm, with the shared damping constant.
    assert sql.count("%(rrf_k)s") == 2
    assert params["rrf_k"] == RRF_K
    assert params["qvec"].startswith("[")


@pytest.mark.unit
def test_vector_arm_dropped_when_the_model_will_not_load(captured, monkeypatch):
    """pgvector present but no usable embedder: keyword arm still runs."""
    monkeypatch.setattr(search_module, "has_vector", lambda conn: True)

    search("x", conn=FakeConn(), config={}, embedder=FakeEmbedder(fail=True))

    sql, params = captured[0]
    assert "vec AS" not in sql
    assert "qvec" not in params


@pytest.mark.unit
def test_ticker_filter_includes_corpus_wide_documents(captured, monkeypatch):
    monkeypatch.setattr(search_module, "has_vector", lambda conn: False)

    search("fed", ticker="nvda", conn=FakeConn(), config={})

    _, params = captured[0]
    # Macro and global news are context for every symbol, so a ticker search
    # must not exclude them.
    assert params["tickers"] == ["NVDA", GLOBAL_TICKER]


@pytest.mark.unit
def test_filters_are_identical_across_arms(captured, monkeypatch):
    """Divergent filters would fuse rankings over different populations."""
    monkeypatch.setattr(search_module, "has_vector", lambda conn: True)

    search(
        "x", ticker="NVDA", category="news", lookback_days=30,
        conn=FakeConn(), config={}, embedder=FakeEmbedder(),
    )

    sql, params = captured[0]
    assert sql.count("d.ticker = ANY(%(tickers)s)") == 2
    assert sql.count("d.category = ANY(%(categories)s)") == 2
    assert sql.count("COALESCE(d.published_at, d.fetched_at) >= %(since)s") == 2
    assert params["categories"] == ["news"]


@pytest.mark.unit
def test_candidate_pool_is_wider_than_the_final_limit(captured, monkeypatch):
    monkeypatch.setattr(search_module, "has_vector", lambda conn: False)
    search("x", limit=5, conn=FakeConn(), config={})
    _, params = captured[0]
    assert params["pool"] > params["limit"]


def _hit(**kw):
    base = {
        "ticker": "NVDA", "category": "news", "source": "yfinance",
        "title": "NVDA news", "text": "Revenue grew.",
        "url": "https://example.invalid/a",
        "published_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
        "fetched_at": datetime(2026, 8, 5, tzinfo=timezone.utc),
        "score": 0.5, "matched_by": "both",
    }
    base.update(kw)
    return SearchHit(**base)


@pytest.mark.unit
def test_empty_results_tell_the_agent_not_to_infer_absence():
    out = format_hits([], "tariffs")
    assert out.startswith("NO_RESULTS")
    assert "do not treat this as evidence of absence" in out


@pytest.mark.unit
def test_rendered_hits_carry_provenance_and_dates():
    out = format_hits([_hit()], "revenue")
    # Without the collection date an agent would date stored text to today.
    assert "published: 2026-08-01" in out
    assert "indexed: 2026-08-05" in out
    assert "via yfinance" in out
    assert "Revenue grew." in out


@pytest.mark.unit
def test_global_documents_are_labelled_as_such():
    out = format_hits([_hit(ticker=GLOBAL_TICKER, title="Macro")], "cpi")
    assert "global" in out
    assert GLOBAL_TICKER not in out


@pytest.mark.unit
def test_undated_hits_are_rendered_without_crashing():
    out = format_hits([_hit(published_at=None, fetched_at=None, url=None)], "q")
    assert "undated" in out
