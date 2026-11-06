"""Background research corpus: scheduled ingestion + hybrid retrieval.

A companion Postgres service accumulates vendor payloads on a schedule so an
analysis reads an already-indexed corpus instead of fetching everything live.
Two consumers:

* the ``knowledge_base`` dataflow vendor (``tradingagents/dataflows/knowledge_base.py``),
  which replays a stored payload for a normal tool call, and
* the ``search_research`` agent tool, which runs hybrid full-text + vector
  search across everything ever collected.

The whole subsystem is optional and degrades: with no database configured, no
driver installed, or no pgvector extension, callers fall back to live vendors
and full-text-only search respectively. Nothing here can abort an analysis.

Names are exported lazily so importing this package costs nothing and never
requires ``psycopg`` to be installed.

The read-path functions are deliberately *not* re-exported here: ``search`` is
already the name of a submodule, so binding it as a function too would make
``tradingagents.knowledge.search`` mean different things before and after that
submodule is imported. Import them from their module instead::

    from tradingagents.knowledge.search import search, latest_payload
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "Document",
    "GLOBAL_TICKER",
    "KnowledgeUnavailable",
    "SearchHit",
    "chunk_text",
    "ingest_documents",
]

# Exported name -> submodule that defines it.
_LAZY = {
    "Document": "models",
    "GLOBAL_TICKER": "models",
    "SearchHit": "models",
    "KnowledgeUnavailable": "db",
    "chunk_text": "chunking",
    "ingest_documents": "ingest",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # import_module, not `from . import x`: the latter re-enters this hook via
    # the package's attribute lookup and recurses.
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
