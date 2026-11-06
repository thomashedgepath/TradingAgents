"""Knowledge worker entry point: ``python -m tradingagents.knowledge``.

    python -m tradingagents.knowledge              # loop forever (the compose service)
    python -m tradingagents.knowledge --once       # one cycle, then exit (cron-friendly)
    python -m tradingagents.knowledge --backfill   # ignore cadence; run every job now
    python -m tradingagents.knowledge --status     # what's in the corpus, what's failing
    python -m tradingagents.knowledge --search Q   # query the corpus from the shell
"""

from __future__ import annotations

import argparse
import logging
import sys

from .db import KnowledgeUnavailable, close_pools, connection, ensure_schema
from .jobs import JOB_SPECS, job_units
from .runtime import build_worker_config, install
from .scheduler import Worker, resolve_watchlist, run_unit
from .state import corpus_stats, summarize

logger = logging.getLogger(__name__)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.knowledge",
        description="Background research corpus worker for TradingAgents.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run one cycle and exit")
    mode.add_argument(
        "--backfill", action="store_true",
        help="run every job now regardless of cadence, then exit",
    )
    mode.add_argument("--status", action="store_true", help="print corpus and job status")
    mode.add_argument("--search", metavar="QUERY", help="run a hybrid search and print hits")
    parser.add_argument(
        "--watchlist", help="comma-separated tickers, overriding the configured watchlist",
    )
    parser.add_argument("--ticker", help="restrict --search to one ticker")
    parser.add_argument("--limit", type=int, default=8, help="--search result count")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _fmt(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value else "never"


def cmd_status(config) -> int:
    with connection(config) as conn:
        ensure_schema(conn, embed_dim=int(config.get("knowledge_embed_dim", 384)))
        stats = corpus_stats(conn)
        states = summarize(conn)

    print("Corpus")
    if not stats:
        print("  (empty — run --backfill to populate)")
    for category, row in stats.items():
        print(
            f"  {category:<20} {row['documents']:>5} docs  {row['chunks']:>6} chunks  "
            f"{row['embedded']:>6} embedded  newest {_fmt(row['newest'])}"
        )

    print("\nJobs")
    if not states:
        print("  (nothing scheduled yet)")
    for row in states:
        flag = "!" if row["consecutive_failures"] else " "
        print(
            f" {flag}{row['job']:<20} {row['ticker']:<18} {str(row['last_status'] or '-'):<12} "
            f"last ok {_fmt(row['last_success_at'])}"
            + (f"  fails={row['consecutive_failures']}" if row["consecutive_failures"] else "")
        )
        if row["last_error"]:
            print(f"      {row['last_error'][:160]}")
    return 0


def cmd_search(config, query: str, ticker: str | None, limit: int) -> int:
    from .search import format_hits, search

    with connection(config) as conn:
        hits = search(query, ticker=ticker, limit=limit, conn=conn, config=config)
    print(format_hits(hits, query))
    return 0


def cmd_backfill(config, watchlist) -> int:
    """Run every unit of every job, ignoring cadence."""
    worker = Worker(config)
    total = 0
    with connection(config) as conn:
        ensure_schema(conn, embed_dim=int(config.get("knowledge_embed_dim", 384)))
        for spec in JOB_SPECS:
            for unit in job_units(spec, watchlist):
                total += run_unit(
                    spec, unit, conn=conn, config=config, embedder=worker.embedder,
                )
    logger.info("backfill complete: %d document(s) stored or updated", total)
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    config = build_worker_config()
    if args.watchlist:
        config["knowledge_watchlist"] = [
            t.strip() for t in args.watchlist.split(",") if t.strip()
        ]
    install(config)
    watchlist = resolve_watchlist(config)

    try:
        if args.status:
            return cmd_status(config)
        if args.search:
            return cmd_search(config, args.search, args.ticker, args.limit)
        if args.backfill:
            return cmd_backfill(config, watchlist)

        worker = Worker(config)
        if args.once:
            summary = worker.run_once()
            logger.info("cycle complete: %s", summary)
            return 0

        worker.install_signal_handlers()
        worker.run_forever()
        return 0
    except KnowledgeUnavailable as exc:
        # The corpus being unreachable is a configuration problem, not a crash:
        # report it in one line rather than a traceback.
        logger.error("%s", exc)
        return 2
    finally:
        close_pools()


if __name__ == "__main__":
    raise SystemExit(main())
