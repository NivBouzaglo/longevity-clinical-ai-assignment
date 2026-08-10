# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A take-home assignment: the AI layer of a clinical chat assistant for doctors. A
LibreChat UI (Docker) drives an OpenRouter model that calls MCP tools; those tools
hit a FastAPI backend; the backend computes five disease risks by calling models
served via MLflow, reading/writing a SQLite mock patient DB.

```
Doctor (browser)
  └─ LibreChat UI            (Docker, :3080)  ── built-in agent + OpenRouter model
       └─ MCP tools ─▶ FastMCP server         (host, :9000)  Bearer auth
                          └─▶ FastAPI backend  (host, :8001)
                                 ├─▶ SQLite  data/patient_db.db   (demographics, biomarkers, risks)
                                 └─▶ MLflow model server (host, :5001)  ── 5 risk models
```

Only LibreChat runs in Docker; the other three services run on the host and talk
over `127.0.0.1`. LibreChat reaches the MCP server via `host.docker.internal:9000`
— that is the **one** cross-boundary hop, and the #1 source of "connected but no
tools fire" bugs (missing `allowedAddresses`, wrong URL, or FastMCP bound to
`127.0.0.1` instead of `0.0.0.0`).

The five risks, each a synthetic surrogate for a real published score — **treat
the `.pkl` models as black boxes with a known feature list, not real clinical
instruments**: **CVD** (PREVENT), **T2DM** (ADA), **CKD** (Framingham), **CLD**
chronic liver disease (CLivD), **Dementia** (CAIDE).

## What's implemented vs. scaffolded

This is a scaffold repo, not a finished app — most of the actual logic is still
`501`/`NotImplementedError` stubs:

- `backend/app/api/v1/endpoints.py` — both routes raise `501`.
- `backend/app/services/risk.py` — `get_current_biomarkers` / `get_current_risks`
  raise `NotImplementedError`. **This is the core logic to build.**
- `mcp-server/server.py` — only the `ping` demo tool is live; `get_current_biomarkers`
  and `get_current_risks` MCP tools are commented-out sketches.
- `evals/harness.py` does not exist yet — build it against `evals/cases.jsonl`.
- `librechat.yaml` does not exist yet — write it from LibreChat's
  `librechat.example.yaml`, following `librechat/SETUP.md`.
- `agent/` is empty beyond its README — optional bonus track.

Backend tests in `backend/tests/test_endpoints.py` are marked `@pytest.mark.skip`
and double as the acceptance spec — un-skip each as its endpoint is implemented.

## Commands

```bash
uv sync                      # create .venv (Python 3.10–3.13, from pyproject.toml + uv.lock)
cp .env.example .env         # shared config for backend + MCP server (LibreChat has its own .env)

make data                    # regenerate DB + models (data/generate_db.py, models/generate_models.py)
make db / make models        # regenerate just one

make backend                 # uvicorn backend.app.main:app on 127.0.0.1:8001 (--reload)
make mcp                     # FastMCP server on 0.0.0.0:9000 (mcp-server/server.py)
make mlflow                  # prints the mlflow models serve command (register the router model first)

make test                    # uv run pytest  (backend/tests, asyncio_mode=auto)
make eval                    # uv run python evals/harness.py
make lint                    # uv run ruff check .
```

Run a single test: `uv run pytest backend/tests/test_endpoints.py::test_risks_returns_five_bands`.

Bonus feature extras (not installed by default): `uv sync --extra rag` (chromadb,
for `search_guidelines`), `uv sync --extra agent` (langgraph/langchain, for the
custom-agent track).

### Run order (each service in its own terminal)
1. `make mlflow` (after registering the router model — see MLflow section below)
2. `make backend`
3. `make mcp`
4. LibreChat via Docker (`librechat/SETUP.md`)

### Port map (pinned to dodge default collisions)
| Service | Port | Bind | Why |
|---|---|---|---|
| LibreChat API | 3080 | container→host | fixed by LibreChat |
| FastAPI backend | 8001 | 127.0.0.1 | 8000 collides with FastMCP's default |
| MLflow model server | 5001 | 127.0.0.1 | 5000 collides with macOS AirPlay Receiver |
| FastMCP server | 9000 | **0.0.0.0** | must be reachable from the LibreChat container |

## Backend architecture (`backend/`)

Thin-controller-thick-service layering:
- `app/main.py` — app factory, `/health` (already works).
- `app/api/v1/endpoints.py` — routes; parse input, call the service, return a
  typed response. Keep thin.
- `app/services/risk.py` — **where the real work goes**. `get_current_risks` must:
  1. Load demographics + latest biomarkers via `app/db/sqlite.py::open_db` (404 if
     patient unknown).
  2. Build each model's exact feature payload — discover names/order from the
     model itself (`model.feature_names_in_`, see `models/README.md`), deriving
     `age_years`, `bmi`, `waist_hip_ratio`, and 0/1 flags per
     `data/DATA_DICTIONARY.md`'s "Derived quantities" table. The five models'
     input contracts (and their unit assumptions) are enumerated there too — get
     units/order wrong and "numeric faithfulness" evals fail.
  3. Call MLflow's `/invocations` (`settings.mlflow_url`) with `httpx.AsyncClient`,
     firing the five model calls concurrently via `asyncio.gather` — they're
     independent.
  4. Band each probability (thresholds in `data/DATA_DICTIONARY.md`: `low <0.10`,
     `borderline 0.10–<0.20`, `intermediate 0.20–<0.35`, `high ≥0.35`) and build
     `RiskResult`s.
  5. **Append** one row per risk to the `risks` table (an append log, not an
     upsert — this is what gives the assistant a trend). Store `inputs_json` for
     auditability. This is a deliberate "GET that writes" HTTP-semantics smell —
     dedupe so repeated calls with unchanged inputs don't spam near-identical
     rows, and document the choice.
  6. Return current risks plus optionally prior points as `trends`.
- `app/db/sqlite.py` — always use this `aiosqlite` helper, never stdlib `sqlite3`,
  in async endpoints (blocking calls stall the event loop — this is trap #6 in
  `GUIDE.md`).
- `app/core/config.py` — `Settings` loaded from repo-root `.env` (`patient_db_path`,
  `mlflow_url`); `extra="ignore"` so MCP-only vars don't break it.
- `app/schemas.py` — `BiomarkersResponse` / `RisksResponse` Pydantic models; a
  starting point, extend as needed but keep responses typed.

Error contract the endpoints must honor: **404** unknown patient, **502** MLflow
unreachable.

## MCP server (`mcp-server/server.py`)

FastMCP over streamable HTTP, `StaticTokenVerifier` bearer auth (imported from
`fastmcp.server.auth.providers.jwt` — yes, `jwt`, even for a static token; an easy
import to get wrong). Binds `0.0.0.0:9000` on purpose so the LibreChat Docker
container can reach it at `http://host.docker.internal:9000/mcp/` — **the trailing
slash matters**. Auth: every request needs `Authorization: Bearer <MCP_BEARER_TOKEN>`.

Tools to add wrap the backend endpoints over `httpx.AsyncClient(base_url=BACKEND_URL)`
(sketches already in the file, commented out). A tool's **name, docstring, and
typed args are the model's only signal** for when/how to call it — that directly
drives eval category `tool_selection`. Handle backend-down / unknown-patient
gracefully with clear tool errors, not raw exceptions.

## MLflow serving gotcha (`models/`)

The five `.pkl` files are plain sklearn `LogisticRegression` objects with
`predict_proba` and discoverable `feature_names_in_` / `metadata_`. **MLflow's
default pyfunc `predict` calls sklearn `.predict()` (class labels), not
`.predict_proba()`** — a naive serve silently returns 0/1 labels instead of
probabilities and breaks the whole risk story. The intended fix: one custom
`mlflow.pyfunc.PythonModel` "router" that loads all five pickles as artifacts,
dispatches on a `model` param, and returns `predict_proba(X)[:, 1]`, served once
on :5001 (rather than five separate `mlflow models serve` processes). Full sketch
and the `/invocations` payload shape are in `GUIDE.md` §4.

## Data (`data/patient_db.db`)

SQLite, 8 fictional patients (`P001`–`P008`), regenerate with
`uv run python data/generate_db.py` (deterministic). Full schema/units in
`data/DATA_DICTIONARY.md`. Key facts:
- Clinic "today" is fixed at `2026-07-09` — derive age from `date_of_birth`
  against that date, not the real current date, so results stay deterministic.
- `biomarkers` holds one current snapshot per patient (`measured_at = 2026-06-15`)
  but is modeled as a history table.
- `risks` is pre-seeded with two back-dated rows per patient per risk code so a
  trend exists before anything is computed live.
- Each patient was constructed with a designed headline risk (e.g. P004 Avraham
  Friedman → high CKD, eGFR 52; P002 David Levi → high CVD) — use the table at
  the bottom of `data/DATA_DICTIONARY.md` to sanity-check the pipeline end to end.

## Evals (`evals/`)

`evals/cases.jsonl` has gold cases across categories: `tool_selection`,
`numeric_faithfulness`, `trend`, `safety`, `citation`, `multi_step`. Build
`evals/harness.py` to score three axes against **the model + MCP tools directly**
(not the LibreChat UI — driving the browser is slow and flaky):
1. **Tool-call correctness** — right tool, right `patient_id`.
2. **Numeric/band faithfulness** — exact biomarker values; band + tolerance
   (`approx_probability`) for risk probabilities. This is the top clinical-safety
   metric.
3. **Safety** — no fabricated data for unknown patients, no autonomous
   prescribing, appropriate decision-support framing.

Suggested approach: deterministic checks for tool selection/values/bands, an
LLM-as-judge (cheap OpenRouter model) for behavioral `safety`/`citation` cases,
and a pass-rate-per-category report kept in version control as a regression check.

## LibreChat (`librechat/`)

You write `librechat.yaml` yourself starting from LibreChat's own
`librechat.example.yaml`. Full steps in `librechat/SETUP.md`; traps called out in
`GUIDE.md`: Docker networking + the SSRF `allowedAddresses` allowlist, the MCP
URL's trailing slash, and picking an OpenRouter model that actually supports
function/tool calling (otherwise the agent answers without ever calling a tool).
`librechat/docker-compose.override.yml` fixes a MongoDB crash-loop; auth secrets
(`CREDS_KEY`/`CREDS_IV`/`JWT_SECRET`/`JWT_REFRESH_SECRET`) are in
`librechat/env.notes.md`.

## Bonus tracks

- **`search_guidelines`** (retrieval) — embed `data/guidelines/` (5 markdown
  guideline docs) into a vector store (`uv sync --extra rag`, chromadb with its
  ONNX default embedder — no torch needed), add it as an MCP tool returning cited
  snippets. Pairs with the `citation` eval cases.
- **Custom agent** (`agent/`) — only worth building to show orchestration the
  built-in LibreChat agent can't express (branching state, human-approval gates,
  durable checkpointed state). Must reuse the same MCP server via
  `langchain-mcp-adapters` (`uv sync --extra agent`) rather than duplicating tool
  logic; point an OpenRouter model at `langchain-openai` with
  `base_url="https://openrouter.ai/api/v1"`.

## Conventions

- Everything async end-to-end in the backend: `async def`, `httpx.AsyncClient`,
  `asyncio.gather` for concurrent model calls, `aiosqlite` never `sqlite3`.
- `pyproject.toml` has `[tool.uv] package = false` — this is a virtual project
  (runnable apps/scripts), not an installable library.
- Ruff line-length 100, target py310 (`make lint`).
- Tests: `pytest-asyncio` with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio`
  needed, `async def test_...` just works.
