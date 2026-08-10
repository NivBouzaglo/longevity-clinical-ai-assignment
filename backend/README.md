# Backend (FastAPI)

A FastAPI service exposing two clinical endpoints over the mock patient database.
The scaffold boots and `/health` works; **the two endpoints return `501` until you
implement them.**

## Endpoints to implement
| Method | Path | Returns |
|---|---|---|
| GET | `/api/v1/get_current_biomarkers?patient_id=P001` | latest biomarker snapshot |
| GET | `/api/v1/get_current_risks?patient_id=P001` | five risks computed live + persisted |

`get_current_risks` is the meaty one: read the patient, build each model's feature
payload, call the MLflow model server, band the probabilities, **append** a row per
risk to the `risks` table (so the assistant can show a trend), and return them.

## Where things live
- `app/main.py` — app factory + `/health` (done).
- `app/api/v1/endpoints.py` — the two routes (501 stubs → implement).
- `app/services/risk.py` — **the core logic** you write (payload build, MLflow call, append).
- `app/db/sqlite.py` — async `aiosqlite` connection helper (use this, not stdlib `sqlite3`).
- `app/schemas.py` — response models (a starting point; extend as needed).
- `app/core/config.py` — settings (`PATIENT_DB_PATH`, `MLFLOW_URL`) from the repo-root `.env`.

## Run
```bash
uv sync                       # from repo root, once
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload
curl "http://127.0.0.1:8001/health"
```
Port **8001** is deliberate — 8000 collides with FastMCP's default. See the root
[`GUIDE.md`](../GUIDE.md) for the full port map and the MLflow serving steps.

## Test
```bash
uv run pytest            # test_health passes now; un-skip the rest as you go
```
The skipped tests in `tests/test_endpoints.py` are your acceptance spec.

## What we look for
Async end-to-end (`async def`, `httpx.AsyncClient`, `asyncio.gather` for the model
calls, non-blocking SQLite), a clean 404 for unknown patients and 502 when the model
server is down, correct feature payloads (units + order), and a sensible approach to
the "GET that writes" question.
