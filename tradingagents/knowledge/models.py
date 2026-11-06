"""Corpus record types shared by the ingestion worker and the read path.

A *document* is one vendor payload: the exact text a dataflow vendor returned
for a (ticker, category, source) triple at a point in time. A *chunk* is a
searchable slice of that document. Keeping the whole payload lets the
``knowledge_base`` vendor replay a stored answer verbatim, while chunks give
the hybrid search useful retrieval granularity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Sentinel used in place of a ticker for corpus-wide documents (global news,
# macro series, prediction markets). A real symbol never contains '*', so this
# cannot collide, and it keeps `ticker` NOT NULL and directly indexable.
GLOBAL_TICKER = "*GLOBAL*"


def utcnow() -> datetime:
    """Timezone-aware UTC now. Every stored timestamp goes through this."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Document:
    """One vendor payload, addressed by ``doc_key`` and versioned by content.

    ``doc_key`` is the stable identity of a recurring fetch (e.g. "yfinance
    news for NVDA"), so each cycle updates one row instead of appending a new
    one. ``content_hash`` is what actually changed: when it matches the stored
    row, ingestion touches ``fetched_at`` and skips re-chunking and re-embedding
    entirely, which is what keeps an hourly schedule cheap.
    """

    doc_key: str
    ticker: str
    category: str
    source: str
    title: str
    body: str
    url: str | None = None
    published_at: datetime | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            f"{self.title}\n{self.body}".encode()
        ).hexdigest()

    @property
    def is_global(self) -> bool:
        return self.ticker == GLOBAL_TICKER


@dataclass(frozen=True)
class SearchHit:
    """One chunk returned by hybrid search, with its document's provenance."""

    ticker: str
    category: str
    source: str
    title: str
    text: str
    url: str | None
    published_at: datetime | None
    fetched_at: datetime | None
    score: float
    # Which retrieval arm(s) surfaced this chunk — "fts", "vector", or "both".
    # Surfaced in the rendered output so a reader can tell a keyword match from
    # a semantic one when judging relevance.
    matched_by: str = "fts"
