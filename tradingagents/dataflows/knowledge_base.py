"""The background corpus as a dataflow vendor.

Registering the corpus in ``VENDOR_METHODS`` rather than special-casing it at
the call sites means it composes with the existing routing rules for free:

    data_vendors = {"news_data": "knowledge_base,yfinance"}

reads the pre-indexed corpus first and falls through to live yfinance on a miss,
using the same ordered-chain semantics as any other vendor pair (#988).

Every miss is signalled with a typed error the router already understands:

* :class:`~tradingagents.dataflows.errors.VendorNotConfiguredError` — no corpus
  configured, no driver installed, or the database is unreachable. The router
  logs it and moves to the next vendor.
* :class:`~tradingagents.dataflows.errors.NoMarketDataError` — the corpus is up
  but has nothing fresh enough for this request.

The freshness gate (``knowledge_max_age_days``) is what makes DB-first safe: a
payload past its window is treated as a miss, so the fallback vendor fetches
live rather than an analysis silently running on week-old news.

Prices and technical indicators are intentionally absent from this module. They
must be fetched live so the market analyst's verified snapshot stays the source
of truth for exact price claims (see ``knowledge/jobs.py``).
"""

from __future__ import annotations

import logging

from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError

logger = logging.getLogger(__name__)


def _fetch(ticker: str, category: str, *, doc_key: str, label: str) -> str:
    """Return a stored payload or raise the appropriate vendor error."""
    try:
        from tradingagents.knowledge.db import KnowledgeUnavailable
        from tradingagents.knowledge.search import latest_payload
    except ImportError as exc:  # pragma: no cover - defensive
        raise VendorNotConfiguredError(f"knowledge base unavailable: {exc}") from exc

    config = get_config()
    max_age = config.get("knowledge_max_age_days")

    try:
        row = latest_payload(
            ticker, category, doc_key=doc_key, max_age_days=max_age, config=config,
        )
    except KnowledgeUnavailable as exc:
        raise VendorNotConfiguredError(str(exc)) from exc
    except Exception as exc:
        # A malformed query or a mid-flight database restart must not abort the
        # analysis — degrade to the next vendor and make the reason visible.
        logger.warning("knowledge base query failed for %s/%s: %s", ticker, category, exc)
        raise VendorNotConfiguredError(f"knowledge base query failed: {exc}") from exc

    if not row:
        raise NoMarketDataError(
            ticker,
            detail=(
                f"no {label} in the research corpus within the last {max_age} day(s); "
                "the ticker may not be on the ingestion watchlist"
            ),
        )

    fetched = row.get("fetched_at")
    stamp = fetched.strftime("%Y-%m-%d %H:%M UTC") if fetched else "unknown time"
    # The provenance header is not decoration: without it an agent cannot tell a
    # pre-indexed payload from a live fetch, and would date the content to now.
    header = (
        f"[Research corpus: {label} for {ticker}, collected from "
        f"{row.get('source', 'unknown')} at {stamp}]"
    )
    return f"{header}\n\n{row['body']}"


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Stored ticker news.

    The date range is accepted for signature compatibility but not used to
    select a document: the corpus keeps one rolling window per ticker, refreshed
    on a schedule. The stored window is stated in the payload header, so an
    agent asking for a narrower range sees exactly what it got.
    """
    return _fetch(ticker, "news", doc_key=f"news:{ticker}", label="news")


def get_global_news(curr_date: str, look_back_days=None, limit=None) -> str:
    from tradingagents.knowledge.models import GLOBAL_TICKER

    return _fetch(
        GLOBAL_TICKER, "global_news", doc_key="global_news", label="global macro news",
    )


def get_insider_transactions(symbol: str) -> str:
    return _fetch(
        symbol, "insider", doc_key=f"insider:{symbol}", label="insider transactions",
    )


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    return _fetch(
        ticker, "fundamentals", doc_key=f"overview:{ticker}", label="fundamentals",
    )


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _fetch(
        ticker, "fundamentals", doc_key=f"balance_sheet:{ticker}", label="balance sheet",
    )


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _fetch(
        ticker, "fundamentals", doc_key=f"cashflow:{ticker}", label="cash flow statement",
    )


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _fetch(
        ticker, "fundamentals", doc_key=f"income_statement:{ticker}",
        label="income statement",
    )


def get_macro_indicators(indicator: str, curr_date: str = None, look_back_days=None) -> str:
    from tradingagents.knowledge.models import GLOBAL_TICKER

    return _fetch(
        GLOBAL_TICKER, "macro", doc_key=f"macro:{indicator}",
        label=f"macro series {indicator}",
    )


def get_prediction_markets(topic: str, limit=None) -> str:
    from tradingagents.knowledge.models import GLOBAL_TICKER

    return _fetch(
        GLOBAL_TICKER, "prediction_markets", doc_key=f"prediction:{topic}",
        label=f"prediction markets for {topic}",
    )
