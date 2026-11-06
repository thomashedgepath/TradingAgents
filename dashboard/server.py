"""FastAPI app serving the dashboard UI and its JSON/SSE API.

Binds to localhost by default: the API can start analyses that spend real money
on hosted providers, and it carries no authentication of its own.
"""

from __future__ import annotations

import json
import queue
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from cli.models import AnalystType
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.api_key_env import get_api_key_env

from . import store
from .runner import PIPELINE, Runner, validate_params

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="TradingAgents Dashboard", docs_url=None, redoc_url=None)
runner = Runner()


def _config() -> dict:
    return DEFAULT_CONFIG


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    """Everything the UI needs to render its header and pre-fill the run form."""
    config = _config()
    provider = config["llm_provider"]
    key_env = get_api_key_env(provider)
    # Report only whether a key is present, never any part of its value.
    import os
    key_set = bool(os.environ.get(key_env)) if key_env else None

    active = runner.active
    return {
        "provider": provider,
        "deep_model": config["deep_think_llm"],
        "quick_model": config["quick_think_llm"],
        "backend_url": config.get("backend_url"),
        "api_key_env": key_env,
        "api_key_set": key_set,
        "results_dir": str(config["results_dir"]),
        "memory_log_path": str(config["memory_log_path"]),
        "checkpoint_enabled": bool(config.get("checkpoint_enabled")),
        "debate_rounds": config["max_debate_rounds"],
        "analysts": [
            {"value": a.value, "label": label}
            for a, label in [
                (AnalystType.MARKET, "Market"),
                (AnalystType.SOCIAL, "Sentiment"),
                (AnalystType.NEWS, "News"),
                (AnalystType.FUNDAMENTALS, "Fundamentals"),
            ]
        ],
        "pipeline": PIPELINE,
        "today": datetime.now().strftime("%Y-%m-%d"),
        "active_job": active.id if active else None,
    }


@app.get("/api/history")
def get_history() -> dict:
    return store.history(_config())


@app.get("/api/reports")
def get_reports() -> dict:
    return {"runs": store.list_runs(_config())}


@app.get("/api/reports/{run_id}")
def get_report(run_id: str) -> dict:
    run = store.get_run(_config(), run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return run


@app.post("/api/run")
def start_run(payload: dict) -> dict:
    try:
        params = validate_params(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    try:
        job = runner.start(params)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None

    return {"job_id": job.id, "params": params}


@app.get("/api/run/{job_id}/events")
def stream_events(job_id: str) -> StreamingResponse:
    """Server-sent events for one run, replaying anything already emitted.

    Declared with ``def`` (not ``async def``) on purpose: Starlette iterates a
    sync generator in a worker thread, so the blocking queue read below cannot
    stall the event loop.
    """
    job = runner.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")

    def generate():
        backlog, sub = job.subscribe()
        try:
            for event in backlog:
                yield _sse(event)
            while True:
                try:
                    event = sub.get(timeout=15)
                except queue.Empty:
                    if job.done.is_set():
                        break
                    yield ": keepalive\n\n"   # holds the connection through quiet stages
                    continue
                yield _sse(event)
                if event["kind"] == "finished":
                    break
        finally:
            job.unsubscribe(sub)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def main() -> None:
    """Console-script entry point (``tradingagents-dashboard``)."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="TradingAgents local web dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    parser.add_argument("--port", type=int, default=8008, help="port (default: 8008)")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"WARNING: binding to {args.host} exposes an unauthenticated API that can "
            "start billable analyses. Only do this on a trusted network."
        )

    print(f"TradingAgents dashboard → http://{args.host}:{args.port}")
    uvicorn.run(
        "dashboard.server:app" if args.reload else app,
        host=args.host, port=args.port, reload=args.reload, log_level="warning",
    )


if __name__ == "__main__":
    main()
