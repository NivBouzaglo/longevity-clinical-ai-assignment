# GUIDE — build & run the Longevity clinical assistant

This is the manual. It covers setup, the port map, the run order, each component,
and the handful of traps that eat the most time. Read it once end-to-end before
starting.

The system, and the one network boundary that matters:
```
Browser ──▶ LibreChat api            (Docker container, :3080)
LibreChat container ──▶ FastMCP      http://host.docker.internal:9000/mcp/   ← ONLY cross-boundary hop
FastMCP (host) ──▶ FastAPI           http://127.0.0.1:8001                   (both on host)
FastAPI (host) ──▶ MLflow            http://127.0.0.1:5001/invocations       (both on host)
FastAPI (host) ──▶ SQLite            data/patient_db.db
```
Only LibreChat runs in Docker. Your three services run on the host and talk to
each other over `127.0.0.1`. LibreChat reaches your MCP server via
`host.docker.internal`.

## Prerequisites
- **[uv](https://docs.astral.sh/uv/)** (`uv` will fetch 3.12 if you don't have one).
- **Docker** (for [LibreChat](https://github.com/danny-avila/librechat) UI).
- An **OpenRouter API key** (you will need to create a free [account](https://openrouter.ai/sign-up) and create an [API key](https://openrouter.ai/settings/keys)).

```bash
uv sync            # creates .venv from pyproject.toml + uv.lock
cp .env.example .env
```

## Port map (pinned — these dodge every default collision)
| Service | Port | Bind | Why this port |
|---|---|---|---|
| LibreChat api | 3080 | container→host | fixed by LibreChat |
| FastAPI backend | **8001** | 127.0.0.1 | 8000 collides with FastMCP's default |
| MLflow model server | **5001** | 127.0.0.1 | 5000 collides with macOS AirPlay Receiver |
| FastMCP server | **9000** | **0.0.0.0** | must be reachable from the LibreChat container |

## Run order (each in its own terminal)
1. `make mlflow` → serve the models on :5001 *(after you register them — §4)*
2. `make backend` → FastAPI on :8001
3. `make mcp` → FastMCP on :9000
4. LibreChat via Docker (`librechat/SETUP.md`)

---

## §1 Data (provided)
`data/patient_db.db` — 8 patients across three tables (`demographics`,
`biomarkers`, `risks`). The `risks` table ships with back-dated history so a trend
exists before you compute anything. Full column/unit reference:
[`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md). Regenerate any time with
`make db` (deterministic).

## §2 Backend — implement the two endpoints
Details in [`backend/README.md`](backend/README.md). In short:

- `GET /api/v1/get_current_biomarkers?patient_id=P001` → latest snapshot.
- `GET /api/v1/get_current_risks?patient_id=P001` → compute all five risks live,
  band them, **append** to the `risks` table, return them (optionally with trend).

Key expectations:
- **Async all the way.** Use `aiosqlite` (provided helper in `app/db/sqlite.py`),
  `httpx.AsyncClient` for MLflow, and `asyncio.gather` to call the five models
  concurrently.
- **Build each payload from the model's own feature list** (`model.feature_names_in_`);
  derive `age`, `bmi`, `waist_hip_ratio`, and the 0/1 flags. Units matter.
- **404** for an unknown patient, **502** if the model server is unreachable.
- The append makes a `GET` mutate state — a known HTTP-semantics smell. Handle it
  deliberately (e.g. dedupe so repeated calls don't spam near-identical rows) and
  say what you chose in your writeup.

Boot + test:
```bash
make backend        # /health works immediately; endpoints 501 until implemented
make test           # un-skip tests in backend/tests/test_endpoints.py as you go
```

## §3 MCP server — add the two tools
Details in [`mcp-server/README.md`](mcp-server/README.md). The skeleton already
boots with `StaticTokenVerifier` bearer auth over streamable HTTP and a `ping`
demo tool. Add `get_current_biomarkers` and `get_current_risks` as tools that call
your backend. Good names + docstrings + typed args are how the model knows when to
call them.
```bash
make mcp            # http://0.0.0.0:9000/mcp/  (trailing slash!)
```
Smoke-test it with the small client in the MCP README.

## §4 Serve the models (MLflow)
Goal: your backend POSTs a feature payload and gets a **probability** back.

**The gotcha:** MLflow's default pyfunc `predict` calls sklearn `.predict()`
(class *labels* — 0/1), **not** `.predict_proba()`. A naive serve returns labels
and silently breaks the whole risk story. You must surface probabilities.

**Recommended: one custom pyfunc "router" model, served once on :5001.** It loads
all five pickles and routes on a `model` param, returning the positive-class
probability. This beats running five `mlflow models serve` processes.

Sketch (you finish the routing + input handling):
```python
import mlflow, pandas as pd

class RiskRouter(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        self.models = {name: pickle.load(open(path, "rb"))
                       for name, path in context.artifacts.items()}
    def predict(self, context, model_input: pd.DataFrame, params=None):
        name = (params or {}).get("model")
        model = self.models[name]
        X = model_input[list(model.feature_names_in_)]      # order matters
        return model.predict_proba(X)[:, 1]                 # probability of the risk

# log once with all five pickles as artifacts, then serve:
#   mlflow.pyfunc.log_model(... python_model=RiskRouter(), artifacts={... 5 pkls ...},
#                           signature=... with a `model` param ...)
#   uv run mlflow models serve -m <model-uri> -p 5001 --env-manager local
```
`--env-manager local` reuses your venv (fast; no conda). Refs:
[custom pyfunc predict override](https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/creating-custom-pyfunc/notebooks/override-predict/) ·
[serving multiple models with pyfunc](https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial/).

**Call shape** (`/invocations`): send a dataframe row + the `model` param, get a
probability:
```bash
curl -s http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split": {"columns": ["age_years","diabetes","hypertension","proteinuria_trace_plus","egfr"],
                      "data": [[72, 1, 1, 1, 52]]},
  "params": {"model": "framingham_ckd"}
}'
# -> {"predictions": [0.50...]}
```
(That row is P004's CKD payload — expect a high probability.)

## §5 LibreChat — run it and wire the MCP server
You assemble `librechat.yaml` yourself, starting from LibreChat's own
`librechat.example.yaml` (see its **"Example MCP Servers Object Structure"** and
`mcpSettings` sections). Official docs:
[librechat.yaml config](https://www.librechat.ai/docs/configuration/librechat_yaml) ·
[MCP servers](https://www.librechat.ai/docs/mcp_servers). Full steps + traps in
[`librechat/SETUP.md`](librechat/SETUP.md). The three things people get wrong:
1. **Docker networking** — LibreChat runs in a container while your MCP server runs on
   the host, and LibreChat blocks host/private addresses by default. The reachable
   hostname and the allowlist are both explained in the `mcpSettings` comments of
   `librechat.example.yaml`.
2. **The MCP URL** — correct host, port, and path; streamable-HTTP is picky about the
   trailing slash.
3. **Tool-capable model** — pick an OpenRouter model that supports function calling,
   or the agent answers without ever calling your tools.

## §6 Evals
Build `evals/harness.py` over [`evals/cases.jsonl`](evals/cases.jsonl). Score three
axes — tool-call correctness, numeric/band faithfulness, safety — and point it at
the **model + MCP tools directly**, not the LibreChat UI. See
[`evals/README.md`](evals/README.md).

## §7 Bonus
- **Retrieval (`search_guidelines`)** — embed `data/guidelines/` into a vector store
  (`uv sync --extra rag`), add an MCP tool that returns cited snippets. Pairs with
  the `citation` eval cases.
- **Custom agent** — only if you want to show orchestration/state/gates the built-in
  agent can't. See [`agent/README.md`](agent/README.md) (`uv sync --extra agent`).

---

## Troubleshooting — the six traps
1. **MCP "connected" but no tools fire** → missing `allowedAddresses`, wrong URL
   (no `/mcp/`, or `localhost`), or FastMCP bound to `127.0.0.1`. Test reachability
   from inside the container (see `librechat/SETUP.md` §5).
2. **MongoDB crash-loops on start** → use `librechat/docker-compose.override.yml`.
3. **LibreChat login broken** → set `CREDS_KEY`/`CREDS_IV`/`JWT_SECRET`/
   `JWT_REFRESH_SECRET` (see `librechat/env.notes.md`).
4. **Agent never calls a tool** → the model doesn't support tool calling; pick another.
5. **Risks look wrong / labels not probabilities** → the MLflow `predict_proba`
   gotcha (§4), or wrong feature order/units in the payload.
6. **Event loop stalls / slow** → you used blocking `sqlite3`; use `aiosqlite`.

## End-to-end verification checklist
- [ ] `make data` regenerates the DB + models without error.
- [ ] `make backend` → `curl :8001/health` returns `ok`; both endpoints work (not 501).
- [ ] `make test` → all tests pass (skips removed).
- [ ] MLflow `/invocations` returns a probability for the P004 CKD payload above.
- [ ] MCP `ping` and both real tools callable with the bearer token.
- [ ] In LibreChat: *"What are Avraham Friedman's (P004) current risks, and how has
      his kidney risk trended?"* → grounded answer citing real values + a worsening
      CKD trend.
- [ ] `evals/harness.py` runs and reports pass rates.
