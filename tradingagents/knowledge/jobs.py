"""The ingestion job table: what gets collected, how often, and from where.

``JOB_SPECS`` is the single source of truth for background collection, in the
same spirit as ``VENDOR_METHODS`` and ``_ENV_OVERRIDES``. To collect a new
source, add a row — the scheduler, the ``--status`` output and the worker all
read it. Nothing branches on job names anywhere else.

Every job routes through ``route_to_vendor``, so a job automatically honours the
configured vendor chain and inherits vendor fallback. The worker installs a
config with ``knowledge_base`` stripped from the vendor chain first, so
collection always hits live vendors and can never read its own output.

**Prices and technical indicators are deliberately not collected.** The market
analyst is required to ground exact price and indicator claims in
``get_verified_market_snapshot`` (see CLAUDE.md), and replaying a cached OHLCV
payload into that path would reintroduce precisely the fabricated-price-level
class of bug that snapshot exists to prevent. Prices stay live, always.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .models import GLOBAL_TICKER, Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobSpec:
    """One recurring collection task.

    ``cadence_minutes`` is the minimum gap between successful runs; the
    scheduler polls more often than any cadence and runs whatever came due.
    ``per_ticker`` jobs fan out across the watchlist, the rest run once per
    cycle against ``GLOBAL_TICKER``.
    """

    name: str
    category: str
    cadence_minutes: int
    collect: Callable[..., list[Document]]
    per_ticker: bool = True
    # Fixed argument sets for jobs that fan out over topics rather than tickers
    # (macro series, prediction-market topics).
    variants: tuple[str, ...] = field(default=())


def _route(method: str, *args):
    """Call a dataflow method through the normal vendor routing layer."""
    from tradingagents.dataflows.interface import route_to_vendor

    return route_to_vendor(method, *args)


def _vendor_for(method: str) -> str:
    """The vendor that will actually serve ``method``, for provenance labelling."""
    from tradingagents.dataflows.interface import get_category_for_method, get_vendor

    try:
        vendor = get_vendor(get_category_for_method(method), method)
    except Exception:
        return "unknown"
    first = [v.strip() for v in str(vendor).split(",") if v.strip()]
    return first[0] if first and first[0] != "default" else "auto"


def _usable(payload) -> str | None:
    """Normalize a vendor return value, rejecting its non-answer sentinels.

    Vendors signal "no data" in-band with a sentinel string rather than by
    raising (see ``route_to_vendor``). Storing one would make the corpus serve
    an apology where the read path expects data, so they are dropped here and
    the job records zero documents instead.
    """
    if payload is None:
        return None
    if isinstance(payload, dict):
        # Alpha Vantage helpers can return the raw JSON envelope.
        payload = payload.get("data") or payload.get("message") or str(payload)
    text = str(payload).strip()
    if not text:
        return None
    if text.startswith(("NO_DATA_AVAILABLE", "DATA_UNAVAILABLE", "NO_RESULTS")):
        return None
    return text


def _doc(
    *, key: str, ticker: str, category: str, method: str, title: str, body: str,
    published_at: datetime | None = None, metadata: dict | None = None,
) -> Document:
    return Document(
        doc_key=key,
        ticker=ticker,
        category=category,
        source=_vendor_for(method),
        title=title,
        body=body,
        published_at=published_at,
        metadata={"method": method, **(metadata or {})},
    )


# --- collectors -------------------------------------------------------------
# Each returns a list of Documents (possibly empty). Raising is fine: the
# scheduler records the failure against the job and moves on to the next one.


def collect_news(ticker: str, now: datetime, config: dict) -> list[Document]:
    lookback = int(config.get("knowledge_news_lookback_days", 7))
    start = (now - timedelta(days=lookback)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    body = _usable(_route("get_news", ticker, start, end))
    if not body:
        return []
    # The key omits the date window on purpose: one rolling row per ticker that
    # each cycle refreshes, rather than a new row per day that would make the
    # corpus grow without bound and split retrieval across near-duplicates.
    return [_doc(
        key=f"news:{ticker}", ticker=ticker, category="news", method="get_news",
        title=f"{ticker} news, {start} to {end}", body=body, published_at=now,
        metadata={"window_start": start, "window_end": end},
    )]


def collect_global_news(_ticker: str, now: datetime, config: dict) -> list[Document]:
    lookback = config.get("global_news_lookback_days")
    limit = config.get("global_news_article_limit")
    body = _usable(_route("get_global_news", now.strftime("%Y-%m-%d"), lookback, limit))
    if not body:
        return []
    return [_doc(
        key="global_news", ticker=GLOBAL_TICKER, category="global_news",
        method="get_global_news", title=f"Global macro news as of {now:%Y-%m-%d}",
        body=body, published_at=now,
    )]


def collect_insider(ticker: str, now: datetime, _config: dict) -> list[Document]:
    body = _usable(_route("get_insider_transactions", ticker))
    if not body:
        return []
    return [_doc(
        key=f"insider:{ticker}", ticker=ticker, category="insider",
        method="get_insider_transactions", title=f"{ticker} insider transactions",
        body=body, published_at=now,
    )]


def collect_fundamentals(ticker: str, now: datetime, _config: dict) -> list[Document]:
    """Overview plus the three statements, as four separately retrievable docs."""
    date_str = now.strftime("%Y-%m-%d")
    specs = [
        ("get_fundamentals", (ticker, date_str), "overview", f"{ticker} fundamentals"),
        ("get_balance_sheet", (ticker, "quarterly", date_str), "balance_sheet",
         f"{ticker} quarterly balance sheet"),
        ("get_cashflow", (ticker, "quarterly", date_str), "cashflow",
         f"{ticker} quarterly cash flow statement"),
        ("get_income_statement", (ticker, "quarterly", date_str), "income_statement",
         f"{ticker} quarterly income statement"),
    ]
    docs = []
    for method, args, kind, title in specs:
        try:
            body = _usable(_route(method, *args))
        except Exception as exc:
            # One missing statement must not cost the other three.
            logger.warning("%s failed for %s: %s", method, ticker, exc)
            continue
        if body:
            docs.append(_doc(
                key=f"{kind}:{ticker}", ticker=ticker, category="fundamentals",
                method=method, title=title, body=body, published_at=now,
                metadata={"statement": kind},
            ))
    return docs


def collect_macro(indicator: str, now: datetime, _config: dict) -> list[Document]:
    body = _usable(_route("get_macro_indicators", indicator, now.strftime("%Y-%m-%d"), None))
    if not body:
        return []
    return [_doc(
        key=f"macro:{indicator}", ticker=GLOBAL_TICKER, category="macro",
        method="get_macro_indicators", title=f"Macro series: {indicator}",
        body=body, published_at=now, metadata={"indicator": indicator},
    )]


def collect_prediction_markets(topic: str, now: datetime, _config: dict) -> list[Document]:
    body = _usable(_route("get_prediction_markets", topic, None))
    if not body:
        return []
    return [_doc(
        key=f"prediction:{topic}", ticker=GLOBAL_TICKER, category="prediction_markets",
        method="get_prediction_markets", title=f"Prediction markets: {topic}",
        body=body, published_at=now, metadata={"topic": topic},
    )]


# Macro series and prediction-market topics worth keeping warm corpus-wide.
# Kept modest by default: every variant is an API call each cycle.
DEFAULT_MACRO_SERIES = (
    "fed_funds_rate", "10y_treasury", "yield_curve", "cpi", "core_pce",
    "unemployment", "nonfarm_payrolls", "real_gdp", "vix",
)

DEFAULT_PREDICTION_TOPICS = (
    "Fed rate cut", "recession", "inflation", "US election",
)


# The job table. Cadences reflect how fast each source actually moves: news
# hourly, prediction markets a few times a day, fundamentals and insider filings
# daily, macro series twice a day (most are monthly releases).
JOB_SPECS: tuple[JobSpec, ...] = (
    JobSpec("news", "news", 60, collect_news, per_ticker=True),
    JobSpec("fundamentals", "fundamentals", 1440, collect_fundamentals, per_ticker=True),
    JobSpec("insider", "insider", 1440, collect_insider, per_ticker=True),
    JobSpec("global_news", "global_news", 60, collect_global_news, per_ticker=False),
    JobSpec("macro", "macro", 720, collect_macro, per_ticker=False,
            variants=DEFAULT_MACRO_SERIES),
    JobSpec("prediction_markets", "prediction_markets", 360, collect_prediction_markets,
            per_ticker=False, variants=DEFAULT_PREDICTION_TOPICS),
)

JOBS_BY_NAME = {spec.name: spec for spec in JOB_SPECS}


def job_units(spec: JobSpec, watchlist: list[str]) -> list[str]:
    """The individual work items for a job: tickers, variants, or a single run.

    The returned value is the ``ticker`` column of ``ingest_state``, so each
    unit is scheduled and retried independently — one failing macro series does
    not hold back the other eight.
    """
    if spec.per_ticker:
        return list(watchlist)
    if spec.variants:
        return list(spec.variants)
    return [GLOBAL_TICKER]
