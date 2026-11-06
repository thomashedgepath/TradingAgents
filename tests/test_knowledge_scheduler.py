"""Scheduling must be restart-safe, fair across units, and back off on failure.

``due_units`` is a pure function of persisted state, so these tests pin the
scheduling policy directly rather than through a running worker.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.knowledge import scheduler as sched
from tradingagents.knowledge.jobs import JobSpec, job_units
from tradingagents.knowledge.models import GLOBAL_TICKER
from tradingagents.knowledge.scheduler import due_units, resolve_watchlist, run_unit

from ._knowledge_fakes import FakeConn

NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


def _spec(name="news", cadence=60, per_ticker=True, variants=(), collect=None):
    return JobSpec(
        name=name, category=name, cadence_minutes=cadence,
        collect=collect or (lambda unit, now, cfg: []),
        per_ticker=per_ticker, variants=variants,
    )


def _state(success_delta_min=None, failures=0, attempt_delta_min=None):
    return {
        "last_success_at": NOW - timedelta(minutes=success_delta_min)
        if success_delta_min is not None else None,
        "last_attempt_at": NOW - timedelta(minutes=attempt_delta_min)
        if attempt_delta_min is not None else None,
        "consecutive_failures": failures,
    }


@pytest.mark.unit
def test_never_run_units_are_due_immediately():
    specs = (_spec(),)
    due = due_units({}, ["NVDA", "AAPL"], NOW, specs)
    assert [unit for _, unit in due] == ["NVDA", "AAPL"]


@pytest.mark.unit
def test_recent_success_is_not_due_until_the_cadence_elapses():
    specs = (_spec(cadence=60),)
    states = {("news", "NVDA"): _state(success_delta_min=30)}
    assert due_units(states, ["NVDA"], NOW, specs) == []

    states = {("news", "NVDA"): _state(success_delta_min=61)}
    assert len(due_units(states, ["NVDA"], NOW, specs)) == 1


@pytest.mark.unit
def test_failures_back_off_exponentially_from_the_last_attempt():
    specs = (_spec(cadence=60),)
    # 3 failures -> 8x cadence = 480 min. Attempted 100 minutes ago: not yet.
    states = {("news", "NVDA"): _state(
        success_delta_min=1000, failures=3, attempt_delta_min=100,
    )}
    assert due_units(states, ["NVDA"], NOW, specs) == []

    states = {("news", "NVDA"): _state(
        success_delta_min=1000, failures=3, attempt_delta_min=500,
    )}
    assert len(due_units(states, ["NVDA"], NOW, specs)) == 1


@pytest.mark.unit
def test_backoff_is_capped():
    specs = (_spec(cadence=60),)
    # 20 failures must not park the job for years — cap is 8x cadence.
    states = {("news", "NVDA"): _state(
        success_delta_min=99999, failures=20, attempt_delta_min=481,
    )}
    assert len(due_units(states, ["NVDA"], NOW, specs)) == 1


@pytest.mark.unit
def test_units_are_scheduled_independently():
    specs = (_spec(cadence=60),)
    states = {("news", "NVDA"): _state(success_delta_min=5)}
    due = due_units(states, ["NVDA", "AAPL"], NOW, specs)
    assert [unit for _, unit in due] == ["AAPL"]


@pytest.mark.unit
def test_job_units_cover_tickers_variants_and_singletons():
    assert job_units(_spec(per_ticker=True), ["A", "B"]) == ["A", "B"]
    assert job_units(
        _spec(per_ticker=False, variants=("cpi", "vix")), ["A"]
    ) == ["cpi", "vix"]
    assert job_units(_spec(per_ticker=False), ["A"]) == [GLOBAL_TICKER]


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        (["nvda", "AAPL"], ["NVDA", "AAPL"]),
        (["NVDA", "nvda", " NVDA "], ["NVDA"]),      # de-duplicated
        ("NVDA, AAPL ,", ["NVDA", "AAPL"]),          # accepts a raw string
        ([], []),
        (None, []),
    ],
)
def test_resolve_watchlist_normalizes(raw, expected):
    assert resolve_watchlist({"knowledge_watchlist": raw}) == expected


@pytest.mark.unit
def test_run_unit_records_error_without_raising(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        sched, "record_attempt",
        lambda conn, job, unit, **kw: recorded.update(kw),
    )

    def boom(unit, now, cfg):
        raise RuntimeError("vendor exploded")

    stored = run_unit(_spec(collect=boom), "NVDA", conn=FakeConn(), config={})

    assert stored == 0
    assert recorded["status"] == "error"
    assert "vendor exploded" in recorded["error"]


@pytest.mark.unit
def test_run_unit_records_empty_as_success(monkeypatch):
    """A vendor with nothing to say must back off normally, not retry hard."""
    recorded = {}
    monkeypatch.setattr(
        sched, "record_attempt",
        lambda conn, job, unit, **kw: recorded.update(kw),
    )
    run_unit(_spec(collect=lambda u, n, c: []), "NVDA", conn=FakeConn(), config={})
    assert recorded["status"] == "empty"


@pytest.mark.unit
def test_run_unit_rolls_back_when_ingest_fails(monkeypatch):
    from tradingagents.knowledge.models import Document

    recorded = {}
    monkeypatch.setattr(
        sched, "record_attempt", lambda conn, job, unit, **kw: recorded.update(kw),
    )

    def bad_ingest(*a, **k):
        raise RuntimeError("db write failed")

    monkeypatch.setattr(sched, "ingest_documents", bad_ingest)
    doc = Document(
        doc_key="k", ticker="NVDA", category="news", source="v", title="t", body="b",
    )
    conn = FakeConn()
    run_unit(_spec(collect=lambda u, n, c: [doc]), "NVDA", conn=conn, config={})

    assert recorded["status"] == "ingest_error"
    assert conn.rollbacks == 1
