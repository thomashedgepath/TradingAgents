# AGENTS.md

Shared rules for every coding agent working in this repo (Claude Code, Codex, or anything else).
**This file is the source of truth.** `CLAUDE.md` is a thin `@AGENTS.md` import plus anything that
only makes sense inside Claude Code. Put a rule here if it binds any agent; never duplicate it there.

## What this is

Multi-agent LLM trading research framework built on LangGraph. Checkout of
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) at `main` (v0.3.1),
with local uncommitted work on top (dashboard, knowledge base). Research scaffold, **not trading
advice** — never present its output as investment guidance.

Stack: Python >=3.10 (local venv is 3.13), LangGraph/LangChain, Typer + Rich CLI, optional FastAPI
dashboard, optional Postgres/pgvector research corpus. `pyproject.toml` is authoritative;
`requirements.txt` contains only `.`.

## Layout

| Path | Role |
|---|---|
| `tradingagents/graph/` | LangGraph wiring, checkpointing, reflection, signal parsing |
| `tradingagents/agents/` | Prompt-bearing node factories (`create_*(llm)`), Pydantic schemas |
| `tradingagents/dataflows/` | Data vendors + routing table (`interface.py::VENDOR_METHODS`) |
| `tradingagents/llm_clients/` | Per-provider clients, capability table, model catalog |
| `tradingagents/knowledge/` | Background research corpus: scheduled ingestion + hybrid search (`[knowledge]` extra) |
| `cli/` | Interactive CLI (`tradingagents` entry point, `cli.main:app`) |
| `dashboard/` | Local FastAPI web UI (`[dashboard]` extra; `python -m dashboard`) |
| `tests/` | Offline, fully mocked pytest suite |
| `main.py` | Programmatic example; `test.py` is a stray network-hitting script, not a test |

## Environment and commands

System Python has no `pip`/`ensurepip`. `.venv/` already works — don't recreate it with a plain
`python3 -m venv`. If you must: `python3 -m venv --without-pip .venv`, bootstrap pip with
`get-pip.py`, then `.venv/bin/pip install -e ".[dev,dashboard]"`.

Always call `.venv/bin/python`, `.venv/bin/pip`, `.venv/bin/ruff`, `.venv/bin/tradingagents`.
Nothing is on `PATH` and no environment is activated.

**Run everything from the repo root.** `.env` is found with `find_dotenv(usecwd=True)`
(`tradingagents/__init__.py`); from elsewhere the API key and every `TRADINGAGENTS_*` override are
silently dropped.

```bash
.venv/bin/python -m pytest -q                      # full suite (~95s); CI runs it on py3.10-3.13
.venv/bin/python -m pytest tests/test_fred.py -q   # one file; -k "<substr>"; -m unit|integration|smoke
.venv/bin/ruff check .                             # lint gate, must be clean (ruff check . --fix)

.venv/bin/tradingagents                            # interactive CLI; --clear-checkpoints resets resume state
.venv/bin/python -m dashboard                      # web UI, http://127.0.0.1:8008
.venv/bin/python main.py                           # programmatic example

docker compose run --rm tradingagents              # CLI in Docker (reads .env; volume holds ~/.tradingagents)
docker compose --profile ollama run --rm tradingagents-ollama
```

Current baseline: 649 passed, 13 skipped. Expected skips: `test_bedrock_provider` (optional
`langchain_aws`), `test_deepseek_reasoning` (live call, needs a key), and the 11
`test_knowledge_integration` cases (need `TRADINGAGENTS_TEST_KNOWLEDGE_DSN`, see below).
`RuntimeWarning: Model 'X' is not in the known model list` is tests exercising that path. The suite is offline and mocked: it catches wiring regressions but proves
nothing about real model or vendor behaviour — run the pipeline for that.

CI (`.github/workflows/ci.yml`): pytest on 3.10–3.13, a clean `pip install .` import smoke test
(catches undeclared runtime deps), and `ruff check .` over the whole repo. All three must stay green.

## Architecture essentials

Linear pipeline, not a swarm (`graph/setup.py`):

```
analysts (sequential, each with a tool loop + "Msg Clear" node)
  → Bull ⇄ Bear → Research Manager → Trader
  → Aggressive ⇄ Conservative ⇄ Neutral → Portfolio Manager
```

Analysts use `quick_thinking_llm`; only Research Manager and Portfolio Manager get
`deep_thinking_llm`. `GraphSetup` takes exactly two LLMs — per-agent models would be a change there.
Programmatic entry: `TradingAgentsGraph(config=...).propagate(ticker, date)` → `(final_state,
signal)`, with `config` a dict copied from `DEFAULT_CONFIG`.

Cross-cutting mechanisms to know before touching agent code:

- **Instrument identity** is resolved once from yfinance and injected into every prompt
  (`agent_utils.resolve_instrument_identity`) so agents can't invent a different company (#814).
- **Grounding**: the market analyst must call `get_verified_market_snapshot`; it is the source of
  truth for exact price/indicator claims. It exists because earlier versions fabricated prices.
- **Structured output**: Research Manager, Trader, Portfolio Manager and Sentiment bind schemas from
  `agents/schemas.py` and render back to markdown via `render_*`. Providers without structured
  output fall back to free text.

## Conventions (observed in the code)

Single-source tables — extend the table, never add branches at the call site:

| Table | Purpose |
|---|---|
| `default_config.py::_ENV_OVERRIDES` | one row exposes a config key as `TRADINGAGENTS_*`; coercion follows the default's type |
| `llm_clients/api_key_env.py::PROVIDER_API_KEY_ENV` | provider → API key env var |
| `llm_clients/openai_client.py::OPENAI_COMPATIBLE_PROVIDERS` | one `ProviderSpec` per OpenAI-compatible provider; only genuinely different APIs (Anthropic, Google, Azure, Bedrock) get their own client |
| `llm_clients/capabilities.py` | per-model API quirks instead of model-name `if` ladders |
| `dataflows/interface.py::VENDOR_METHODS` | `{tool: {vendor: impl}}`; the configured chain is exact, no silent fallback (#988); `OPTIONAL_CATEGORIES` degrade to a sentinel string |
| `knowledge/jobs.py::JOB_SPECS` | one row per background collection job; scheduler, `--status`, worker all read it |

- Comments carrying an issue number (`#814`, `#1088`, …) record *why* non-obvious code exists.
  Preserve them when refactoring.
- Adding an analyst touches: `agents/analysts/<new>.py`, export in `agents/__init__.py`,
  `ANALYST_NODE_SPECS` (`graph/analyst_execution.py`), `should_continue_<key>`
  (`graph/conditional_logic.py`), `_create_tool_nodes` (`graph/trading_graph.py`), `AnalystType`
  (`cli/models.py`), plus a field on `AgentState`.
- New runtime deps go in an optional extra (`[bedrock]`, `[dashboard]`, `[knowledge]`,
  `[embeddings]`), not the core `dependencies` list.
- Ruff: `line-length = 100`, select `E W F I B UP C4 SIM`, E501 ignored, `F401` allowed in
  `__init__.py`. `ruff format` is deliberately not adopted repo-wide yet; don't mass-reformat.

## Gotchas

- **The CLI has no `analyze` subcommand.** `analyze` is the only `@app.command()`, so Typer promotes
  it to root: `tradingagents --checkpoint`. A second command would invert this.
- **Env overrides apply at import time.** Setting `os.environ` after importing `default_config`
  does nothing. `TRADINGAGENTS_TEMPERATURE` arrives as a `str` (default is `None`); call sites cast.
- **Runs are not independent.** Each run appends to `~/.tradingagents/memory/trading_memory.md`
  and the next same-ticker run feeds past decisions into the Portfolio Manager prompt. Account for
  this when A/B-testing prompts.
- `graph.stream()` uses `stream_mode="values"`: each chunk is the whole state; the first is the
  initial state echoed back.
- **One analysis per process.** `dataflows.config.set_config` installs a module-level global.
- `openai_reasoning_effort` applies only when `llm_provider == "openai"` — unreachable via
  OpenRouter or any OpenAI-compatible endpoint.

## Providers, data, secrets

- `.env` at the repo root selects everything; `.env.example` lists every variable. `.env` and
  `.env.enterprise` are gitignored — never commit them, never paste key values into chat or logs.
- OpenAI-compatible servers: `llm_provider=openai_compatible` + `TRADINGAGENTS_LLM_BACKEND_URL`.
  For hosted providers (`openrouter`, `groq`, …) leave `backend_url` unset — an explicit value
  overrides the provider endpoint and silently misroutes every request.
- Local LM Studio (`http://localhost:1234/v1`): the default 8192 context overflows on the Market
  Analyst's first tool result. Load with e.g. `lms load <model> -c 32768 --parallel 1 --gpu max -y`.
- Data defaults to keyless yfinance everywhere; Alpha Vantage, FRED (macro) and Polymarket
  (prediction markets) are alternatives, the latter two optional-degrading. Exchange-suffixed
  tickers (`0700.HK`, `BTC-USD`) work; the alpha benchmark resolves via `config["benchmark_map"]`.
- State lives under `~/.tradingagents/`: `logs/`, `cache/checkpoints/<TICKER>.db`,
  `memory/trading_memory.md`. Override with `TRADINGAGENTS_RESULTS_DIR`, `TRADINGAGENTS_CACHE_DIR`,
  `TRADINGAGENTS_MEMORY_LOG_PATH`.
- The dashboard binds to localhost by default on purpose: its API starts billable analyses and
  has no auth. Don't expose it.

## Knowledge base (optional corpus)

```bash
docker compose --profile knowledge up -d               # knowledge-db (pgvector) + knowledge-worker
docker compose run --rm knowledge-worker --status      # also: --backfill, --search "q" --ticker NVDA
.venv/bin/pip install -e ".[knowledge,embeddings]"     # to read it from the host
.venv/bin/python -m tradingagents.knowledge --once
```

Two consumers, deliberately different: `knowledge_base` is a **vendor** in `VENDOR_METHODS`
(`data_vendors={"news_data": "knowledge_base,yfinance"}` reads the corpus, falls through on a
miss); `search_research` is a **tool** on the news and fundamentals analysts (hybrid FTS ⊕ pgvector).
Sentiment pre-fetches a corpus block instead because it never tool-calls (#1130).

Invariants, each guarding a real failure — do not relax them:

- The worker never routes through `knowledge_base` (`runtime.live_vendor_config` strips it), or it
  would re-store its own output forever.
- Prices and indicators are never collected or served from the corpus.
- `knowledge_max_age_days` is a hard freshness gate; stale = miss → live fetch. `0` admits nothing,
  `None` disables.
- Ingestion is content-hashed; unchanged payloads only touch `fetched_at`.
- Everything degrades: no DB → `CORPUS_UNAVAILABLE`; no pgvector/sentence-transformers → FTS only.
- `ensure_schema` commits base DDL before `CREATE EXTENSION`, or a failed extension rolls the
  tables back too.

Unit tests mock the driver. Real SQL: start `pgvector/pgvector:pg17` (or `postgres:17`, FTS-only)
and run `tests/test_knowledge_integration.py` with `TRADINGAGENTS_TEST_KNOWLEDGE_DSN` set — the
DSN recipe is at the top of that file.

## Workflow rules

- **Do not commit or push unless asked.** The tree carries substantial uncommitted work
  (dashboard, knowledge base, README/compose changes). Stage by explicit path — never `git add -A`.
- Both gates (`pytest -q`, `ruff check .`) green before declaring anything done. If you did not run
  it, say so.
- When reporting on a run, check cited figures against actual tool output rather than trusting
  the generated report; the README's *Reproducibility* section says what is expected to move.
- Don't edit `CHANGELOG.md` or bump the version unprompted; this tracks upstream.
