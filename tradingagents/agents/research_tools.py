"""Retrieval tool over the background research corpus.

Unlike the other data tools, this one is not routed through ``VENDOR_METHODS``:
it is not a source of new data but a search across everything the background
worker has already collected, which is a different operation with no vendor
equivalent. The vendor-shaped read path is ``dataflows/knowledge_base.py``.

The tool never raises into the agent loop. An unreachable corpus returns an
explicit sentinel telling the model to use the live tools instead — the same
contract the routing layer uses for a missing vendor, and for the same reason:
an agent must never treat "I could not look" as "there is nothing".
"""

from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Categories the background worker collects. Named in the tool description so
# the model can filter instead of retrieving across the whole corpus.
CORPUS_CATEGORIES = (
    "news", "global_news", "fundamentals", "insider", "macro", "prediction_markets",
)


@tool
def search_research(
    query: Annotated[str, "Natural-language search query, e.g. 'margin pressure from tariffs'"],
    ticker: Annotated[
        str | None,
        "Restrict to one ticker; corpus-wide documents (macro, global news) are "
        "always included. Omit to search everything.",
    ] = None,
    category: Annotated[
        str | None,
        "Restrict to one category: news, global_news, fundamentals, insider, "
        "macro, prediction_markets. Omit to search all.",
    ] = None,
    lookback_days: Annotated[
        int | None, "Only return material from the last N days. Omit for no limit."
    ] = None,
    limit: Annotated[int | None, "Max passages to return; omit for the configured default"] = None,
) -> str:
    """
    Search the pre-indexed research corpus accumulated by the background worker.

    This searches accumulated research — news, fundamentals, insider activity,
    macro series and prediction markets collected on a schedule — using hybrid
    full-text and semantic (vector) retrieval. It is broader and deeper in time
    than the live tools, which only see a recent window, so it is the right tool
    for questions about what has been said about a topic over time, for finding
    material you cannot name a date for, and for cross-referencing a claim
    against earlier coverage.

    It returns previously collected passages, each stamped with its source and
    collection date. It does NOT return live data: for the current price, the
    latest indicator values, or breaking news, use the live tools.

    Args:
        query (str): Natural-language search query
        ticker (str): Optional ticker filter
        category (str): Optional category filter
        lookback_days (int): Optional recency filter
        limit (int): Maximum passages to return

    Returns:
        str: Ranked passages with source, publication and indexing dates
    """
    try:
        from tradingagents.dataflows.config import get_config
        from tradingagents.knowledge.db import KnowledgeUnavailable
        from tradingagents.knowledge.search import format_hits, search
    except ImportError as exc:  # pragma: no cover - defensive
        return _unavailable(f"the knowledge base module could not be imported ({exc})")

    config = get_config()
    if limit is None:
        limit = config.get("knowledge_search_limit", 8)

    if category and category not in CORPUS_CATEGORIES:
        return (
            f"INVALID_CATEGORY: {category!r} is not a corpus category. "
            f"Valid values: {', '.join(CORPUS_CATEGORIES)}. Retry without the "
            "filter to search everything."
        )

    # A model that means "recent" sometimes passes 0, which as a literal bound
    # matches nothing. Treat it as "no limit" here rather than returning a
    # confusing empty result; search() keeps the literal semantics.
    if lookback_days is not None and int(lookback_days) <= 0:
        lookback_days = None

    try:
        hits = search(
            query,
            ticker=ticker,
            category=category,
            lookback_days=lookback_days,
            limit=int(limit),
            config=config,
        )
    except KnowledgeUnavailable as exc:
        return _unavailable(str(exc))
    except Exception as exc:
        logger.warning("corpus search failed for %r: %s", query, exc)
        return _unavailable(f"the search failed ({exc})")

    return format_hits(hits, query)


def _unavailable(reason: str) -> str:
    return (
        f"CORPUS_UNAVAILABLE: the research corpus could not be searched — {reason}. "
        "This says nothing about whether relevant information exists. Use the live "
        "data tools instead, and do not cite the corpus or its absence as evidence."
    )
