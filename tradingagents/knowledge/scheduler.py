"""The looping job runner.

Scheduling is deliberately *pull*-based: rather than holding timers, each cycle
asks the database which units are due right now. That makes the worker stateless
between cycles, so restarts, crashes and multiple short-lived runs all behave
identically, and ``--once`` from a cron job is the same code path as the loop.
"""

from __future__ import annotations

import logging
import random
import signal
import threading
from datetime import datetime, timedelta

from .db import connection, ensure_schema
from .embedding import get_embedder
from .ingest import ingest_documents, prune
from .jobs import JOB_SPECS, JobSpec, job_units
from .models import utcnow
from .state import load_states, record_attempt

logger = logging.getLogger(__name__)

# Failure backoff: a unit that keeps failing is retried after cadence * 2^n,
# capped so a transient vendor outage doesn't park a job for days.
MAX_BACKOFF_MULTIPLIER = 8


def _next_due(spec: JobSpec, state: dict | None) -> datetime | None:
    """When this unit next becomes eligible. ``None`` means "due now".

    A unit that has never succeeded is always due, so a fresh corpus fills on
    the first cycle rather than waiting out a cadence.
    """
    if state is None or state.get("last_success_at") is None:
        return None

    failures = int(state.get("consecutive_failures") or 0)
    multiplier = min(2 ** failures, MAX_BACKOFF_MULTIPLIER) if failures else 1
    gap = timedelta(minutes=spec.cadence_minutes * multiplier)

    if failures:
        # Back off from the last *attempt*, not the last success — otherwise a
        # unit whose last success is long past would retry every single cycle.
        base = state.get("last_attempt_at") or state["last_success_at"]
    else:
        base = state["last_success_at"]
    return base + gap


def due_units(
    states: dict[tuple[str, str], dict],
    watchlist: list[str],
    now: datetime,
    specs: tuple[JobSpec, ...] = JOB_SPECS,
) -> list[tuple[JobSpec, str]]:
    """Which ``(spec, unit)`` pairs are eligible at ``now``.

    Pure function of the persisted state — no clock or database access of its
    own — so scheduling behaviour is directly testable.
    """
    due = []
    for spec in specs:
        for unit in job_units(spec, watchlist):
            nxt = _next_due(spec, states.get((spec.name, unit)))
            if nxt is None or nxt <= now:
                due.append((spec, unit))
    return due


def run_unit(spec: JobSpec, unit: str, *, conn, config: dict, embedder=None) -> int:
    """Run one job unit and record its outcome. Returns documents stored.

    Never raises: a vendor failure is recorded against the unit and the cycle
    continues. One broken source must not stop the corpus from updating.
    """
    now = utcnow()
    try:
        documents = spec.collect(unit, now, config)
    except Exception as exc:
        logger.warning("job %s/%s failed: %s", spec.name, unit, exc)
        record_attempt(conn, spec.name, unit, status="error", error=str(exc))
        return 0

    if not documents:
        # Distinct from an error: the vendor answered, it just had nothing.
        # Recorded as success so the unit backs off to its normal cadence.
        record_attempt(conn, spec.name, unit, status="empty", documents=0)
        return 0

    try:
        result = ingest_documents(documents, conn=conn, config=config, embedder=embedder)
    except Exception as exc:
        logger.warning("ingest failed for %s/%s: %s", spec.name, unit, exc)
        conn.rollback()
        record_attempt(conn, spec.name, unit, status="ingest_error", error=str(exc))
        return 0

    record_attempt(conn, spec.name, unit, status="ok", documents=len(documents))
    logger.info(
        "%s/%s: %d changed, %d unchanged, %d chunks (%d embedded)",
        spec.name, unit, result.stored, result.unchanged, result.chunks, result.embedded,
    )
    return result.stored


def run_cycle(config: dict, *, conn, embedder=None, watchlist: list[str] | None = None) -> dict:
    """Run every due unit once. Returns a summary for logging."""
    if watchlist is None:
        watchlist = resolve_watchlist(config)

    states = load_states(conn)
    now = utcnow()
    pending = due_units(states, watchlist, now)

    if not pending:
        return {"ran": 0, "stored": 0}

    logger.info("cycle: %d unit(s) due", len(pending))
    stored = 0
    for spec, unit in pending:
        stored += run_unit(spec, unit, conn=conn, config=config, embedder=embedder)

    removed = prune(conn, retention_days=config.get("knowledge_retention_days"))
    return {"ran": len(pending), "stored": stored, "pruned": removed}


def resolve_watchlist(config: dict) -> list[str]:
    """Normalize the configured watchlist: upper-cased, de-duplicated, ordered.

    Order is preserved so the first cycle after a restart works through the list
    predictably rather than in whatever order a set happens to iterate.
    """
    raw = config.get("knowledge_watchlist") or []
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",")]

    seen, out = set(), []
    for item in raw:
        ticker = str(item).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


class Worker:
    """Owns the loop, the connection and the loaded embedding model."""

    def __init__(self, config: dict):
        self.config = config
        self.watchlist = resolve_watchlist(config)
        self.tick_seconds = max(10, int(config.get("knowledge_tick_seconds", 300)))
        self._stop = threading.Event()
        self._embedder = None

    def install_signal_handlers(self) -> None:
        """Stop cleanly on SIGTERM/SIGINT so ``docker compose down`` is quick."""
        def handle(signum, _frame):
            logger.info("received signal %s; finishing the current cycle", signum)
            self._stop.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handle)
            except ValueError:  # pragma: no cover - not on the main thread
                logger.debug("could not install handler for %s", sig)

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.config)
        return self._embedder

    def run_once(self) -> dict:
        with connection(self.config) as conn:
            ensure_schema(conn, embed_dim=int(self.config.get("knowledge_embed_dim", 384)))
            return run_cycle(
                self.config, conn=conn, embedder=self.embedder, watchlist=self.watchlist,
            )

    def run_forever(self) -> None:
        logger.info(
            "knowledge worker starting: %d ticker(s), %d job(s), %ds tick",
            len(self.watchlist), len(JOB_SPECS), self.tick_seconds,
        )
        if not self.watchlist:
            logger.warning(
                "watchlist is empty; only corpus-wide jobs (global news, macro, "
                "prediction markets) will run. Set TRADINGAGENTS_KNOWLEDGE_WATCHLIST."
            )

        while not self._stop.is_set():
            try:
                summary = self.run_cycle_guarded()
                if summary.get("ran"):
                    logger.info("cycle complete: %s", summary)
            except Exception:
                logger.exception("cycle failed; retrying after the next tick")

            # Jitter so several workers (or a restart storm) don't align their
            # cycles and hit every vendor simultaneously.
            self._stop.wait(self.tick_seconds * random.uniform(0.9, 1.1))

        logger.info("knowledge worker stopped")

    def run_cycle_guarded(self) -> dict:
        """One cycle, opening a fresh connection so a dropped socket self-heals."""
        return self.run_once()

    def stop(self) -> None:
        self._stop.set()
