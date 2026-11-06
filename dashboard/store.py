"""Read-only views over what a TradingAgents run leaves on disk.

Two independent stores back the dashboard:

* the **report tree** written by :func:`tradingagents.reporting.write_report_tree`
  — one directory per run, holding each agent's markdown output;
* the **memory log** parsed by :class:`~tradingagents.agents.utils.memory.TradingMemoryLog`
  — one entry per decision, later amended in place with the realised return,
  alpha, and a reflection.

The report tree is the detail view; the memory log is the only place that knows
how a decision actually turned out, so it is what the history view is built on.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import RATINGS_5_TIER, parse_rating

# Directory names written by write_report_tree, in pipeline order, mapped to the
# files each may contain. Keeping the order here means the UI never has to sort.
SECTION_LAYOUT: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("1_analysts", "Analyst Team", [
        ("market.md", "Market Analyst"),
        ("sentiment.md", "Sentiment Analyst"),
        ("news.md", "News Analyst"),
        ("fundamentals.md", "Fundamentals Analyst"),
    ]),
    ("2_research", "Research Team", [
        ("bull.md", "Bull Researcher"),
        ("bear.md", "Bear Researcher"),
        ("manager.md", "Research Manager"),
    ]),
    ("3_trading", "Trading Team", [
        ("trader.md", "Trader"),
    ]),
    ("4_risk", "Risk Management", [
        ("aggressive.md", "Aggressive Analyst"),
        ("conservative.md", "Conservative Analyst"),
        ("neutral.md", "Neutral Analyst"),
    ]),
    ("5_portfolio", "Portfolio Manager", [
        ("decision.md", "Final Decision"),
    ]),
]

# Report directories are named "<TICKER>_<YYYYmmdd>_<HHMMSS>". The ticker itself
# may contain dots, dashes, carets or equals (e.g. 0700.HK, BRK-B, GC=F), so the
# timestamp is anchored from the right rather than splitting on the first "_".
_RUN_DIR_RE = re.compile(r"^(?P<ticker>.+)_(?P<stamp>\d{8}_\d{6})$")


def _reports_root(config: dict) -> Path:
    return Path(config["results_dir"]) / "reports"


def _parse_run_dir(path: Path) -> dict | None:
    match = _RUN_DIR_RE.match(path.name)
    if not match:
        return None
    try:
        created = datetime.strptime(match.group("stamp"), "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return {
        "id": path.name,
        "ticker": match.group("ticker"),
        "created": created.isoformat(timespec="seconds"),
    }


def list_runs(config: dict) -> list[dict]:
    """Every saved report run, newest first.

    The rating is re-derived from the portfolio decision with the same parser
    the memory log uses, so a run shows the same rating in both views even
    though nothing on disk records it as a discrete field.
    """
    root = _reports_root(config)
    if not root.is_dir():
        return []

    runs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        meta = _parse_run_dir(child)
        if meta is None:
            continue

        decision_file = child / "5_portfolio" / "decision.md"
        decision = _read(decision_file)
        meta["rating"] = parse_rating(decision) if decision else None
        meta["complete"] = (child / "complete_report.md").is_file()
        meta["sections"] = _count_sections(child)
        runs.append(meta)

    runs.sort(key=lambda r: r["created"], reverse=True)
    return runs


def _count_sections(run_dir: Path) -> int:
    return sum(
        1
        for directory, _label, files in SECTION_LAYOUT
        for filename, _title in files
        if (run_dir / directory / filename).is_file()
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def get_run(config: dict, run_id: str) -> dict | None:
    """Load one run's full report tree, grouped by pipeline stage.

    ``run_id`` is validated against the parsed directory listing rather than
    joined onto the reports root directly — a caller-supplied id must never be
    able to walk out of the reports directory.
    """
    root = _reports_root(config)
    run_dir = root / run_id
    meta = _parse_run_dir(Path(run_id))
    if meta is None or not run_dir.is_dir() or run_dir.parent != root:
        return None

    groups = []
    for directory, label, files in SECTION_LAYOUT:
        entries = []
        for filename, title in files:
            body = _read(run_dir / directory / filename)
            if body.strip():
                entries.append({"title": title, "body": body})
        if entries:
            groups.append({"label": label, "entries": entries})

    decision = _read(run_dir / "5_portfolio" / "decision.md")
    meta["rating"] = parse_rating(decision) if decision else None
    meta["groups"] = groups
    return meta


# --- Memory log -------------------------------------------------------------


def _pct_to_float(value: str | None) -> float | None:
    """Turn a stored "+2.4%" tag field back into 0.024.

    Returns None for pending entries and for anything that does not parse, so a
    hand-edited log can never crash the history view.
    """
    if not value:
        return None
    try:
        return float(value.strip().rstrip("%")) / 100.0
    except ValueError:
        return None


def _holding_to_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value.strip().rstrip("d"))
    except ValueError:
        return None


def history(config: dict) -> dict:
    """Every logged decision plus aggregates, newest first.

    Aggregates are computed over resolved entries only — a pending entry has no
    outcome yet, and counting it as a zero would drag every average toward it.
    """
    entries = TradingMemoryLog(config).load_entries()

    rows = []
    for entry in entries:
        raw = _pct_to_float(entry.get("raw"))
        alpha = _pct_to_float(entry.get("alpha"))
        rows.append({
            "date": entry.get("date"),
            "ticker": entry.get("ticker"),
            "rating": entry.get("rating"),
            "pending": bool(entry.get("pending")),
            "raw_return": raw,
            "alpha_return": alpha,
            "holding_days": _holding_to_int(entry.get("holding")),
            "decision": entry.get("decision", ""),
            "reflection": entry.get("reflection", ""),
        })

    rows.sort(key=lambda r: (r["date"] or "", r["ticker"] or ""), reverse=True)
    return {"entries": rows, "summary": _summarize(rows)}


def _summarize(rows: list[dict]) -> dict:
    resolved = [r for r in rows if not r["pending"] and r["alpha_return"] is not None]
    alphas = [r["alpha_return"] for r in resolved]
    raws = [r["raw_return"] for r in resolved if r["raw_return"] is not None]

    by_rating = dict.fromkeys(RATINGS_5_TIER, 0)
    for row in rows:
        if row["rating"] in by_rating:
            by_rating[row["rating"]] += 1

    tickers = sorted({r["ticker"] for r in rows if r["ticker"]})

    return {
        "total": len(rows),
        "pending": sum(1 for r in rows if r["pending"]),
        "resolved": len(resolved),
        # Mean alpha, and how often alpha was positive. Both are descriptive
        # only: entries are not equally weighted, holding periods vary, and the
        # sample is whatever tickers happened to be run.
        "avg_alpha": (sum(alphas) / len(alphas)) if alphas else None,
        "avg_raw": (sum(raws) / len(raws)) if raws else None,
        "beat_benchmark": sum(1 for a in alphas if a > 0),
        "by_rating": by_rating,
        "tickers": tickers,
    }
