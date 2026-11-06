"""Runs the agent graph on a worker thread and turns it into an event stream.

The graph is streamed in ``values`` mode, so each chunk is the whole state
rather than a delta. Node identity is therefore inferred from which state keys
have filled — the same technique the CLI uses to drive its live display.

Only one run is allowed at a time. ``dataflows.config.set_config`` installs the
active config into a module-level global that every vendor call reads, so two
concurrent runs with different settings would silently contend over it.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from cli.models import AnalystType
from cli.utils import detect_asset_type, filter_analysts_for_asset_type, normalize_ticker_symbol
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)

# Pipeline stages in execution order. ``key`` is the state field that proves the
# stage finished; analyst stages are dropped when that analyst is not selected.
PIPELINE: list[dict[str, str]] = [
    {"id": "market",       "label": "Market Analyst",       "key": "market_report",         "group": "Analysts"},
    {"id": "social",       "label": "Sentiment Analyst",    "key": "sentiment_report",      "group": "Analysts"},
    {"id": "news",         "label": "News Analyst",         "key": "news_report",           "group": "Analysts"},
    {"id": "fundamentals", "label": "Fundamentals Analyst", "key": "fundamentals_report",   "group": "Analysts"},
    {"id": "bull",         "label": "Bull Researcher",      "key": "_bull",                 "group": "Research"},
    {"id": "bear",         "label": "Bear Researcher",      "key": "_bear",                 "group": "Research"},
    {"id": "manager",      "label": "Research Manager",     "key": "investment_plan",       "group": "Research"},
    {"id": "trader",       "label": "Trader",               "key": "trader_investment_plan", "group": "Trading"},
    {"id": "aggressive",   "label": "Aggressive Analyst",   "key": "_aggressive",           "group": "Risk"},
    {"id": "conservative", "label": "Conservative Analyst", "key": "_conservative",         "group": "Risk"},
    {"id": "neutral",      "label": "Neutral Analyst",      "key": "_neutral",              "group": "Risk"},
    {"id": "portfolio",    "label": "Portfolio Manager",    "key": "final_trade_decision",  "group": "Portfolio"},
]

_ANALYST_STAGE_IDS = {"market", "social", "news", "fundamentals"}


def stages_for(selected_analysts: list[str]) -> list[dict[str, str]]:
    """The pipeline as it will actually execute for this analyst selection."""
    chosen = set(selected_analysts)
    return [
        stage for stage in PIPELINE
        if stage["id"] not in _ANALYST_STAGE_IDS or stage["id"] in chosen
    ]


def _debate_done(state: dict, side: str) -> bool:
    """Whether a named researcher/debater has spoken at least once.

    The debate sub-states accumulate per-speaker transcripts, so a non-empty
    history is the signal. These are the only stages with no dedicated
    top-level state key of their own.
    """
    if side in ("_bull", "_bear"):
        debate = state.get("investment_debate_state") or {}
        return bool(debate.get("bull_history" if side == "_bull" else "bear_history"))
    risk = state.get("risk_debate_state") or {}
    field = {
        "_aggressive": "aggressive_history",
        "_conservative": "conservative_history",
        "_neutral": "neutral_history",
    }[side]
    return bool(risk.get(field))


def _completed_stage_ids(state: dict, stages: list[dict[str, str]]) -> set[str]:
    done = set()
    for stage in stages:
        key = stage["key"]
        if key.startswith("_"):
            if _debate_done(state, key):
                done.add(stage["id"])
        elif state.get(key):
            done.add(stage["id"])
    return done


class Job:
    """One dashboard-initiated analysis, with a replayable event buffer.

    Events are buffered rather than only broadcast so that a browser that
    connects late — or reconnects after a refresh — replays the whole run
    instead of joining mid-stream and showing a half-empty pipeline.
    """

    _ids = itertools.count(1)

    def __init__(self, params: dict[str, Any]):
        self.id = f"job-{next(Job._ids)}-{datetime.now():%H%M%S}"
        self.params = params
        self.events: list[dict] = []
        self.done = threading.Event()
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []

    def emit(self, kind: str, **payload) -> None:
        event = {"kind": kind, "at": datetime.now().isoformat(timespec="seconds"), **payload}
        with self._lock:
            self.events.append(event)
            subscribers = list(self._subscribers)
        for sub in subscribers:
            sub.put(event)

    def subscribe(self) -> tuple[list[dict], queue.Queue]:
        """Return buffered history plus a queue receiving everything after it."""
        sub: queue.Queue = queue.Queue()
        with self._lock:
            backlog = list(self.events)
            self._subscribers.append(sub)
        return backlog, sub

    def unsubscribe(self, sub: queue.Queue) -> None:
        with self._lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)


class Runner:
    """Owns the single active job and the thread executing it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active: Job | None = None
        self._last: Job | None = None

    @property
    def active(self) -> Job | None:
        with self._lock:
            return self._active

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            for job in (self._active, self._last):
                if job is not None and job.id == job_id:
                    return job
        return None

    def start(self, params: dict[str, Any]) -> Job:
        with self._lock:
            if self._active is not None and not self._active.done.is_set():
                raise RuntimeError("An analysis is already running.")
            job = Job(params)
            self._active = job
            self._last = job

        thread = threading.Thread(target=self._execute, args=(job,), daemon=True)
        thread.start()
        return job

    def _execute(self, job: Job) -> None:
        try:
            self._run_graph(job)
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI, never raised into the server
            logger.exception("Dashboard run failed")
            job.emit("error", message=str(exc), detail=traceback.format_exc(limit=4))
        finally:
            job.done.set()
            job.emit("finished")
            with self._lock:
                self._active = None

    def _run_graph(self, job: Job) -> None:
        params = job.params
        ticker = params["ticker"]
        trade_date = params["date"]
        analysts = params["analysts"]

        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = params["debate_rounds"]
        config["max_risk_discuss_rounds"] = params["debate_rounds"]

        asset_type = detect_asset_type(ticker).value
        stages = stages_for(analysts)

        job.emit(
            "started",
            ticker=ticker,
            date=trade_date,
            asset_type=asset_type,
            analysts=analysts,
            stages=stages,
            provider=config["llm_provider"],
            deep_model=config["deep_think_llm"],
            quick_model=config["quick_think_llm"],
            debate_rounds=params["debate_rounds"],
        )

        job.emit("log", message=f"Initialising graph for {ticker} on {trade_date}")
        graph = TradingAgentsGraph(
            selected_analysts=tuple(analysts), debug=False, config=config
        )

        # Resolve outcomes for earlier decisions on this ticker before analysing.
        # This is what turns a pending history row into a scored one, so it is
        # worth announcing rather than doing invisibly.
        job.emit("log", message="Resolving outcomes for earlier decisions…")
        graph._resolve_pending_entries(ticker)

        past_context = graph.memory_log.get_past_context(ticker)
        instrument_context = graph.resolve_instrument_context(ticker, asset_type)
        job.emit("identity", context=instrument_context, has_memory=bool(past_context))

        init_state = graph.propagator.create_initial_state(
            ticker, trade_date, asset_type=asset_type,
            past_context=past_context, instrument_context=instrument_context,
        )
        args = graph.propagator.get_graph_args()

        started = datetime.now()
        seen: set[str] = set()
        final_state: dict = {}

        for chunk in graph.graph.stream(init_state, **args):
            final_state.update(chunk)

            messages = chunk.get("messages") or []
            if messages:
                calls = getattr(messages[-1], "tool_calls", None) or []
                for call in calls:
                    job.emit("tool", name=call.get("name", "?"), args=_short_args(call.get("args")))

            done = _completed_stage_ids(chunk, stages)
            for stage in stages:
                if stage["id"] in done and stage["id"] not in seen:
                    seen.add(stage["id"])
                    job.emit(
                        "stage",
                        id=stage["id"],
                        label=stage["label"],
                        elapsed=round((datetime.now() - started).total_seconds(), 1),
                        completed=len(seen),
                        total=len(stages),
                    )

        elapsed = (datetime.now() - started).total_seconds()

        decision = graph.process_signal(final_state["final_trade_decision"])
        graph._log_state(trade_date, final_state)
        graph.memory_log.store_decision(
            ticker=ticker, trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = (
            Path(config["results_dir"]) / "reports"
            / f"{safe_ticker_component(ticker)}_{stamp}"
        )
        graph.save_reports(final_state, ticker, save_path=save_path)

        job.emit(
            "complete",
            signal=decision,
            elapsed=round(elapsed, 1),
            run_id=save_path.name,
            decision=final_state.get("final_trade_decision", ""),
        )


def _short_args(args: Any, limit: int = 90) -> str:
    if not args:
        return ""
    try:
        text = ", ".join(f"{k}={v}" for k, v in args.items())
    except AttributeError:
        text = str(args)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def validate_params(payload: dict) -> dict:
    """Normalise and check a run request, mirroring the CLI's own handling.

    Raises ``ValueError`` with a message intended to be shown verbatim in the UI.
    """
    raw_ticker = (payload.get("ticker") or "").strip()
    if not raw_ticker:
        raise ValueError("Ticker is required.")
    ticker = normalize_ticker_symbol(raw_ticker)

    date = (payload.get("date") or "").strip()
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format.") from None

    valid = {a.value for a in AnalystType}
    analysts = [a for a in (payload.get("analysts") or []) if a in valid]
    if not analysts:
        raise ValueError("Select at least one analyst.")
    # Crypto has no company fundamentals to fetch; the CLI drops that analyst
    # rather than letting it run against an instrument it cannot analyse.
    asset_type = detect_asset_type(ticker)
    analysts = [
        a.value for a in filter_analysts_for_asset_type(
            [AnalystType(a) for a in analysts], asset_type
        )
    ]
    if not analysts:
        raise ValueError("No analysts remain for this asset type.")

    try:
        rounds = int(payload.get("debate_rounds", 1))
    except (TypeError, ValueError):
        raise ValueError("Research depth must be a number.") from None
    if not 1 <= rounds <= 5:
        raise ValueError("Research depth must be between 1 and 5.")

    return {
        "ticker": ticker,
        "date": date,
        "analysts": analysts,
        "debate_rounds": rounds,
    }
