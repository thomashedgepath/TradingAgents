"""Read path: hybrid chunk search and whole-payload replay.

Two shapes of read, for two callers:

* :func:`search` — reciprocal-rank fusion over a full-text arm and a vector arm,
  backing the ``search_research`` agent tool. RRF is used rather than a weighted
  score blend because ``ts_rank_cd`` and cosine distance are not commensurable;
  fusing *ranks* needs no per-corpus tuning and degrades cleanly to whichever
  arm is available.
* :func:`latest_payload` — the most recent stored payload for a (ticker,
  category), backing the ``knowledge_base`` dataflow vendor.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from .db import connection, dict_cursor, has_vector
from .embedding import get_embedder, to_pgvector
from .models import GLOBAL_TICKER, SearchHit, utcnow

logger = logging.getLogger(__name__)

# RRF damping. 60 is the value from the original Cormack et al. formulation and
# behaves well without tuning: it keeps rank-1 from dominating so a chunk found
# by both arms outranks one that any single arm ranked first.
RRF_K = 60

# How many candidates each arm contributes before fusion. Wider than the final
# limit so a chunk ranked poorly by one arm can still be lifted by the other.
POOL_MULTIPLIER = 5
MIN_POOL = 40


def _filters(ticker, category, since) -> tuple[str, dict]:
    """Build the WHERE fragment shared by both arms, plus its parameters.

    Both arms must filter identically or fusion compares different candidate
    populations, so the fragment is built once and interpolated into each.
    """
    clauses: list[str] = []
    params: dict = {}

    if ticker:
        # A ticker search also matches corpus-wide documents (global news,
        # macro, prediction markets) — those are context for every symbol.
        params["tickers"] = [ticker.upper(), GLOBAL_TICKER]
        clauses.append("d.ticker = ANY(%(tickers)s)")
    if category:
        categories = [category] if isinstance(category, str) else list(category)
        params["categories"] = categories
        clauses.append("d.category = ANY(%(categories)s)")
    if since is not None:
        params["since"] = since
        # published_at is frequently NULL (fundamentals, macro snapshots), so
        # fall back to fetched_at rather than dropping those documents.
        clauses.append("COALESCE(d.published_at, d.fetched_at) >= %(since)s")

    return (" AND " + " AND ".join(clauses) if clauses else ""), params


_FTS_ARM = """
    fts AS (
        SELECT c.id,
               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.id) AS rank
          FROM corpus_chunk c
          JOIN corpus_doc d ON d.id = c.doc_id,
               q
         WHERE c.tsv @@ q.tsq {filters}
         ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.id
         LIMIT %(pool)s
    )
"""

_VECTOR_ARM = """
    vec AS (
        SELECT c.id,
               ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(qvec)s::vector, c.id) AS rank
          FROM corpus_chunk c
          JOIN corpus_doc d ON d.id = c.doc_id
         WHERE c.embedding IS NOT NULL {filters}
         ORDER BY c.embedding <=> %(qvec)s::vector, c.id
         LIMIT %(pool)s
    )
"""


def _build_query(filters: str, *, use_vector: bool) -> str:
    arms = [_FTS_ARM.format(filters=filters)]
    if use_vector:
        arms.append(_VECTOR_ARM.format(filters=filters))

    vec_join = "LEFT JOIN vec ON vec.id = c.id" if use_vector else ""
    vec_score = "+ COALESCE(1.0 / (%(rrf_k)s + vec.rank), 0)" if use_vector else ""
    vec_flag = "(vec.id IS NOT NULL)" if use_vector else "FALSE"
    vec_where = " OR vec.id IS NOT NULL" if use_vector else ""

    return f"""
    WITH q AS (SELECT websearch_to_tsquery('english', %(query)s) AS tsq),
    {",".join(arms)}
    SELECT d.ticker, d.category, d.source, d.title, d.url,
           d.published_at, d.fetched_at, c.text,
           (COALESCE(1.0 / (%(rrf_k)s + fts.rank), 0) {vec_score}) AS score,
           (fts.id IS NOT NULL) AS by_fts,
           {vec_flag} AS by_vec
      FROM corpus_chunk c
      JOIN corpus_doc d ON d.id = c.doc_id
      LEFT JOIN fts ON fts.id = c.id
      {vec_join}
     WHERE (fts.id IS NOT NULL{vec_where})
     ORDER BY score DESC, d.fetched_at DESC
     LIMIT %(limit)s
    """


def search(
    query: str,
    *,
    ticker: str | None = None,
    category: str | list[str] | None = None,
    lookback_days: int | None = None,
    limit: int = 8,
    conn=None,
    config: dict | None = None,
    embedder=None,
) -> list[SearchHit]:
    """Hybrid search over the corpus, best match first.

    Raises ``KnowledgeUnavailable`` if the corpus is unreachable; returns an
    empty list when it is reachable but has nothing matching.
    """
    if not query or not query.strip():
        return []

    if conn is None:
        with connection(config) as own_conn:
            return search(
                query, ticker=ticker, category=category, lookback_days=lookback_days,
                limit=limit, conn=own_conn, config=config, embedder=embedder,
            )

    # `is not None`, not truthiness: 0 is a meaningful bound ("today only"),
    # and treating it as "no filter" would silently widen the search.
    since = (
        utcnow() - timedelta(days=int(lookback_days))
        if lookback_days is not None else None
    )
    filters, params = _filters(ticker, category, since)

    qvec = None
    if has_vector(conn):
        if embedder is None:
            embedder = get_embedder(config)
        if embedder is not None:
            vec = embedder.encode_one(query)
            qvec = to_pgvector(vec) if vec else None

    params.update({
        "query": query,
        "limit": int(limit),
        "pool": max(MIN_POOL, int(limit) * POOL_MULTIPLIER),
        "rrf_k": RRF_K,
    })
    if qvec is not None:
        params["qvec"] = qvec

    sql = _build_query(filters, use_vector=qvec is not None)
    with dict_cursor(conn) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    hits = []
    for row in rows:
        by_fts, by_vec = row["by_fts"], row["by_vec"]
        hits.append(SearchHit(
            ticker=row["ticker"],
            category=row["category"],
            source=row["source"],
            title=row["title"],
            text=row["text"],
            url=row["url"],
            published_at=row["published_at"],
            fetched_at=row["fetched_at"],
            score=float(row["score"]),
            matched_by="both" if by_fts and by_vec else ("vector" if by_vec else "fts"),
        ))
    return hits


def latest_payload(
    ticker: str,
    category: str,
    *,
    source: str | None = None,
    doc_key: str | None = None,
    max_age_days: int | None = None,
    conn=None,
    config: dict | None = None,
) -> dict | None:
    """Most recent stored payload for a (ticker, category), or None.

    ``max_age_days`` is a hard freshness gate: an older payload is reported as a
    miss so the caller falls through to a live vendor instead of quietly serving
    stale data into an analysis. ``None`` disables the gate; ``0`` admits
    nothing, which is a valid way to turn corpus reads off without unregistering
    the vendor.
    """
    if conn is None:
        with connection(config) as own_conn:
            return latest_payload(
                ticker, category, source=source, doc_key=doc_key,
                max_age_days=max_age_days, conn=own_conn, config=config,
            )

    clauses = ["ticker = %(ticker)s", "category = %(category)s"]
    params: dict = {"ticker": ticker.upper(), "category": category}
    if source:
        clauses.append("source = %(source)s")
        params["source"] = source
    if doc_key:
        clauses.append("doc_key = %(doc_key)s")
        params["doc_key"] = doc_key
    if max_age_days is not None:
        params["cutoff"] = utcnow() - timedelta(days=int(max_age_days))
        clauses.append("fetched_at >= %(cutoff)s")

    with dict_cursor(conn) as cur:
        cur.execute(
            "SELECT doc_key, ticker, category, source, title, body, url, "
            "published_at, fetched_at, metadata FROM corpus_doc "
            f"WHERE {' AND '.join(clauses)} ORDER BY fetched_at DESC LIMIT 1",
            params,
        )
        return cur.fetchone()


def format_hits(hits: list[SearchHit], query: str) -> str:
    """Render hits as markdown for an agent prompt.

    Provenance (source, dates, URL) leads each hit so the agent can attribute
    and date every claim it lifts, and so a stale hit is visibly stale.
    """
    if not hits:
        return (
            f"NO_RESULTS: the research corpus has nothing matching {query!r}. "
            "The topic may not be covered by the ingestion watchlist, or the "
            "corpus may not have been populated yet. Use the live data tools "
            "instead; do not treat this as evidence of absence."
        )

    lines = [f"## Research corpus results for {query!r}", ""]
    for i, hit in enumerate(hits, 1):
        published = hit.published_at.strftime("%Y-%m-%d") if hit.published_at else "undated"
        fetched = hit.fetched_at.strftime("%Y-%m-%d") if hit.fetched_at else "unknown"
        scope = "global" if hit.ticker == GLOBAL_TICKER else hit.ticker
        lines.append(
            f"### {i}. {hit.title or hit.category} "
            f"[{scope} · {hit.category} · via {hit.source}]"
        )
        lines.append(
            f"published: {published} | indexed: {fetched} | "
            f"match: {hit.matched_by} | score: {hit.score:.4f}"
        )
        if hit.url:
            lines.append(f"url: {hit.url}")
        lines.append("")
        lines.append(hit.text)
        lines.append("")
    return "\n".join(lines)
