"""Wiring guarantees for the corpus subsystem.

Two classes of regression are covered here:

* the worker must never read its own output (the self-feeding loop), and
* the corpus must stay strictly optional — no configured database means the
  pipeline behaves exactly as it did before this feature existed.
"""

import copy

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.agents.utils.research_tools import CORPUS_CATEGORIES, search_research
from tradingagents.knowledge.jobs import JOB_SPECS, JOBS_BY_NAME
from tradingagents.knowledge.runtime import build_worker_config, live_vendor_config


@pytest.fixture(autouse=True)
def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


# --- the worker must not read its own output --------------------------------

@pytest.mark.unit
def test_worker_config_strips_the_corpus_from_every_chain():
    stripped = live_vendor_config({
        "data_vendors": {
            "news_data": "knowledge_base,yfinance",
            "fundamental_data": "knowledge_base",
            "macro_data": "fred",
        },
        "tool_vendors": {"get_news": "knowledge_base,alpha_vantage"},
    })
    assert stripped["data_vendors"]["news_data"] == "yfinance"
    assert stripped["tool_vendors"]["get_news"] == "alpha_vantage"
    # A chain that was *only* the corpus becomes "default" (any live vendor)
    # rather than empty, which route_to_vendor would reject.
    assert stripped["data_vendors"]["fundamental_data"] == "default"
    assert stripped["data_vendors"]["macro_data"] == "fred"


@pytest.mark.unit
def test_worker_config_does_not_mutate_its_input():
    original = {"data_vendors": {"news_data": "knowledge_base,yfinance"}}
    live_vendor_config(original)
    assert original["data_vendors"]["news_data"] == "knowledge_base,yfinance"


@pytest.mark.unit
def test_built_worker_config_never_routes_to_the_corpus():
    config = build_worker_config({
        "data_vendors": {"news_data": "knowledge_base,yfinance"},
    })
    for chain in config["data_vendors"].values():
        assert "knowledge_base" not in chain


# --- the job table ----------------------------------------------------------

@pytest.mark.unit
def test_job_names_are_unique():
    assert len(JOBS_BY_NAME) == len(JOB_SPECS)


@pytest.mark.unit
def test_every_job_has_a_sane_cadence_and_collector():
    for spec in JOB_SPECS:
        assert spec.cadence_minutes > 0, spec.name
        assert callable(spec.collect), spec.name


@pytest.mark.unit
def test_job_categories_match_the_searchable_categories():
    """A collected category the search tool cannot filter on is unreachable."""
    assert {spec.category for spec in JOB_SPECS} <= set(CORPUS_CATEGORIES)


@pytest.mark.unit
def test_no_job_collects_prices_or_indicators():
    methods = set()
    for spec in JOB_SPECS:
        methods.add(spec.name)
    assert "market" not in methods
    assert "indicators" not in methods


@pytest.mark.unit
@pytest.mark.parametrize("sentinel", [
    "NO_DATA_AVAILABLE: nothing for FOO",
    "DATA_UNAVAILABLE: optional macro_data could not be retrieved",
    "",
    "   ",
    None,
])
def test_vendor_non_answers_are_not_stored(sentinel):
    """Storing a vendor's apology would serve it back as if it were research."""
    from tradingagents.knowledge.jobs import _usable

    assert _usable(sentinel) is None


@pytest.mark.unit
def test_real_payloads_pass_through():
    from tradingagents.knowledge.jobs import _usable

    assert _usable("## NVDA news\n\nRevenue grew.") == "## NVDA news\n\nRevenue grew."


# --- the corpus stays optional ---------------------------------------------

@pytest.mark.unit
def test_corpus_is_disabled_by_default():
    assert default_config.DEFAULT_CONFIG["knowledge_db_url"] is None
    assert default_config.DEFAULT_CONFIG["knowledge_watchlist"] == []


@pytest.mark.unit
def test_search_tool_returns_a_sentinel_when_no_corpus_is_configured():
    out = search_research.func("anything")
    assert out.startswith("CORPUS_UNAVAILABLE")
    # The model must not read "I could not look" as "there is nothing".
    assert "do not cite the corpus or its absence as evidence" in out


@pytest.mark.unit
def test_search_tool_rejects_an_unknown_category():
    out = search_research.func("q", category="not_a_category")
    assert out.startswith("INVALID_CATEGORY")
    for category in CORPUS_CATEGORIES:
        assert category in out


@pytest.mark.unit
def test_search_tool_swallows_backend_errors(monkeypatch):
    config_module._config["knowledge_db_url"] = "postgresql://x/y"
    monkeypatch.setattr(
        "tradingagents.knowledge.search.search",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = search_research.func("q")
    assert out.startswith("CORPUS_UNAVAILABLE")


# --- analyst binding --------------------------------------------------------

@pytest.mark.unit
def test_research_tool_is_bound_to_the_reasoning_analysts():
    from unittest.mock import MagicMock

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    nodes = TradingAgentsGraph._create_tool_nodes(MagicMock())
    for key in ("news", "fundamentals"):
        names = {t.name for t in nodes[key].tools_by_name.values()}
        assert "search_research" in names, key


@pytest.mark.unit
def test_market_analyst_cannot_reach_the_corpus():
    """Its exact price/indicator claims must come from the live snapshot."""
    from unittest.mock import MagicMock

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    nodes = TradingAgentsGraph._create_tool_nodes(MagicMock())
    names = {t.name for t in nodes["market"].tools_by_name.values()}
    assert "search_research" not in names
    assert "get_verified_market_snapshot" in names
