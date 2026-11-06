"""The corpus must behave like any other vendor in the routing chain.

The point of registering it in ``VENDOR_METHODS`` is that DB-first reads with
live fallback need no new call-site logic. These tests pin the two behaviours
that make it safe: an unreachable or stale corpus is a *miss* that falls
through, never an error that aborts an analysis.
"""

import copy

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface, knowledge_base
from tradingagents.dataflows.errors import NoMarketDataError, VendorNotConfiguredError
from tradingagents.knowledge.db import KnowledgeUnavailable


@pytest.fixture(autouse=True)
def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _set_vendors(chain):
    config_module._config["data_vendors"]["news_data"] = chain


@pytest.mark.unit
def test_registered_for_every_research_method():
    expected = {
        "get_news", "get_global_news", "get_insider_transactions",
        "get_fundamentals", "get_balance_sheet", "get_cashflow",
        "get_income_statement", "get_macro_indicators", "get_prediction_markets",
    }
    registered = {m for m, v in interface.VENDOR_METHODS.items() if "knowledge_base" in v}
    assert registered == expected
    assert "knowledge_base" in interface.VENDOR_LIST


@pytest.mark.unit
def test_prices_and_indicators_are_never_served_from_the_corpus():
    """Exact price claims must stay live — that is what the verified snapshot
    exists to guarantee, and a replayed OHLCV payload would undermine it."""
    assert "knowledge_base" not in interface.VENDOR_METHODS["get_stock_data"]
    assert "knowledge_base" not in interface.VENDOR_METHODS["get_indicators"]


@pytest.mark.unit
def test_unconfigured_corpus_falls_through_to_the_live_vendor(monkeypatch):
    _set_vendors("knowledge_base,yfinance")
    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_news"], "yfinance",
        lambda *a, **k: "LIVE NEWS",
    )
    # No knowledge_db_url configured, which is the default.
    assert interface.route_to_vendor("get_news", "NVDA", "2026-08-01", "2026-08-06") == "LIVE NEWS"


@pytest.mark.unit
def test_stale_corpus_entry_falls_through_to_the_live_vendor(monkeypatch):
    """A payload past the freshness gate is a miss, not a stale answer."""
    _set_vendors("knowledge_base,yfinance")
    config_module._config["knowledge_db_url"] = "postgresql://x/y"
    # latest_payload applies the max-age filter in SQL, so a stale row simply
    # is not returned.
    monkeypatch.setattr(
        "tradingagents.knowledge.search.latest_payload", lambda *a, **k: None,
    )
    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_news"], "yfinance",
        lambda *a, **k: "LIVE NEWS",
    )
    assert interface.route_to_vendor("get_news", "NVDA", "2026-08-01", "2026-08-06") == "LIVE NEWS"


@pytest.mark.unit
def test_database_error_degrades_rather_than_aborting(monkeypatch):
    config_module._config["knowledge_db_url"] = "postgresql://x/y"

    def boom(*a, **k):
        raise RuntimeError("connection reset mid-query")

    monkeypatch.setattr("tradingagents.knowledge.search.latest_payload", boom)
    with pytest.raises(VendorNotConfiguredError):
        knowledge_base.get_news("NVDA", "2026-08-01", "2026-08-06")


@pytest.mark.unit
def test_unavailable_corpus_raises_the_not_configured_error(monkeypatch):
    config_module._config["knowledge_db_url"] = "postgresql://x/y"

    def unavailable(*a, **k):
        raise KnowledgeUnavailable("no route to host")

    monkeypatch.setattr("tradingagents.knowledge.search.latest_payload", unavailable)
    with pytest.raises(VendorNotConfiguredError):
        knowledge_base.get_news("NVDA", "2026-08-01", "2026-08-06")


@pytest.mark.unit
def test_miss_raises_no_data_naming_the_watchlist(monkeypatch):
    config_module._config["knowledge_db_url"] = "postgresql://x/y"
    monkeypatch.setattr(
        "tradingagents.knowledge.search.latest_payload", lambda *a, **k: None,
    )
    with pytest.raises(NoMarketDataError) as exc:
        knowledge_base.get_news("NVDA", "2026-08-01", "2026-08-06")
    assert "watchlist" in str(exc.value)


@pytest.mark.unit
def test_hit_is_returned_with_a_provenance_header(monkeypatch):
    from datetime import datetime, timezone

    config_module._config["knowledge_db_url"] = "postgresql://x/y"
    monkeypatch.setattr(
        "tradingagents.knowledge.search.latest_payload",
        lambda *a, **k: {
            "body": "NVDA beat estimates.",
            "source": "yfinance",
            "fetched_at": datetime(2026, 8, 5, 9, 30, tzinfo=timezone.utc),
        },
    )
    out = knowledge_base.get_news("NVDA", "2026-08-01", "2026-08-06")

    # Without the header an agent cannot tell stored text from a live fetch and
    # would date the content to now.
    assert out.startswith("[Research corpus:")
    assert "2026-08-05 09:30 UTC" in out
    assert "NVDA beat estimates." in out


@pytest.mark.unit
def test_corpus_first_beats_the_live_vendor_when_it_has_data(monkeypatch):
    _set_vendors("knowledge_base,yfinance")
    config_module._config["knowledge_db_url"] = "postgresql://x/y"
    monkeypatch.setattr(
        "tradingagents.knowledge.search.latest_payload",
        lambda *a, **k: {"body": "STORED", "source": "yfinance", "fetched_at": None},
    )
    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_news"], "yfinance",
        lambda *a, **k: "LIVE NEWS",
    )
    out = interface.route_to_vendor("get_news", "NVDA", "2026-08-01", "2026-08-06")
    assert "STORED" in out


@pytest.mark.unit
def test_default_vendor_chain_still_prefers_live_sources():
    """Registering the corpus must not change behaviour for existing users."""
    for method in ("get_news", "get_fundamentals", "get_macro_indicators"):
        vendors = list(interface.VENDOR_METHODS[method].keys())
        assert vendors[-1] == "knowledge_base", method
