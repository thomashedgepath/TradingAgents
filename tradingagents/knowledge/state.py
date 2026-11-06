"""Per-job-unit scheduling state, persisted so restarts don't refetch.

The worker is expected to be restarted freely (compose up/down, image rebuilds).
Keeping last-success timestamps in Postgres rather than in memory means a
restart resumes the schedule instead of stampeding every vendor at once.
"""

from __future__ import annotations

from .db import dict_cursor
from .models import utcnow


def load_states(conn) -> dict[tuple[str, str], dict]:
    """Load all scheduling state, keyed by ``(job, unit)``."""
    with dict_cursor(conn) as cur:
        cur.execute("SELECT * FROM ingest_state")
        return {(row["job"], row["ticker"]): row for row in cur.fetchall()}


def record_attempt(
    conn,
    job: str,
    unit: str,
    *,
    status: str,
    documents: int = 0,
    error: str | None = None,
) -> None:
    """Record the outcome of one job unit.

    ``consecutive_failures`` drives the scheduler's backoff, so it resets to 0
    on success and increments on failure. ``last_success_at`` is only advanced
    on success — a failing job keeps its old success time and therefore stays
    due, rather than being silently rescheduled a full cadence into the future.
    """
    now = utcnow()
    success = status == "ok"
    # Truncated: an error message is for the operator, and an enormous vendor
    # traceback in a status table helps nobody.
    error_text = (error or "")[:2000] or None

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingest_state
                (job, ticker, last_attempt_at, last_success_at, last_status,
                 last_error, consecutive_failures, documents)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (job, ticker) DO UPDATE SET
                last_attempt_at      = EXCLUDED.last_attempt_at,
                last_success_at      = COALESCE(EXCLUDED.last_success_at,
                                                ingest_state.last_success_at),
                last_status          = EXCLUDED.last_status,
                last_error           = EXCLUDED.last_error,
                consecutive_failures = CASE
                    WHEN EXCLUDED.last_status = 'ok' THEN 0
                    ELSE ingest_state.consecutive_failures + 1
                END,
                documents            = EXCLUDED.documents
            """,
            (
                job, unit, now, now if success else None, status,
                error_text, 0 if success else 1, documents,
            ),
        )
    conn.commit()


def summarize(conn) -> list[dict]:
    """All job state, ordered for human reading (worst first)."""
    with dict_cursor(conn) as cur:
        cur.execute(
            "SELECT * FROM ingest_state "
            "ORDER BY consecutive_failures DESC, last_success_at ASC NULLS FIRST, job, ticker"
        )
        return list(cur.fetchall())


def corpus_stats(conn) -> dict:
    """Document/chunk counts by category, for ``--status``.

    The embedded count is only selected when the vector column exists — on a
    full-text-only deployment referencing it would make the whole query fail.
    """
    from .db import has_vector

    embedded = (
        "COUNT(c.id) FILTER (WHERE c.embedding IS NOT NULL)"
        if has_vector(conn) else "0"
    )
    with dict_cursor(conn) as cur:
        cur.execute(
            f"""
            SELECT d.category,
                   COUNT(DISTINCT d.id) AS documents,
                   COUNT(c.id)          AS chunks,
                   {embedded}           AS embedded,
                   MAX(d.fetched_at)    AS newest
              FROM corpus_doc d
              LEFT JOIN corpus_chunk c ON c.doc_id = d.id
             GROUP BY d.category
             ORDER BY d.category
            """
        )
        return {row["category"]: row for row in cur.fetchall()}
