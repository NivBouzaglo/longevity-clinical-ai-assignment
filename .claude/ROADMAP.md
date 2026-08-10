# Roadmap & Checklist

Working plan for the Longevity AI take-home. Ordered so that stopping at any
checkpoint still leaves a defensible, gradeable submission — the assignment
explicitly weights backend + evals highest and LibreChat lowest.

Check items off as you go. Each phase has a "done when" gate — don't move on
until it's green.

---

## Phase 0 — Environment setup
- [x] `uv sync`
- [x] `cp .env.example .env`
- [x] `make data` (regenerates `data/patient_db.db` + `models/*.pkl` — should run clean, already committed)
- [x] `make backend` → `curl http://127.0.0.1:8001/health` returns `{"status":"ok",...}`
- [x] `make test` → `test_health` passes, everything else still skipped

**Done when:** backend boots, health check green, no import errors. ✅ **2026-08-10**

---

## Phase 1 — MLflow: serve the 5 models on :5001
Do this before the backend logic so you have something to call while writing it.

- [x] Inspect each model's contract: `pickle.load` → `feature_names_in_`, `metadata_` (see `models/README.md`)
- [x] Write the custom pyfunc **router** model (`models/generate_router.py`):
  loads all 5 pickles as artifacts, dispatches on a `model` param, returns
  `predict_proba(X)[:, 1]` — **not** `.predict()` (the labels-vs-probabilities gotcha).
  Input schema is the union of all 5 models' columns, each marked `required=False`,
  so a caller can send just the subset the chosen model needs. Output of
  `uv run python models/generate_router.py` is gitignored (regenerate, don't commit —
  it bundles a full env spec + cloned pickles per build).
- [x] `mlflow.pyfunc.save_model(...)` to `models/mlflow_risk_router` with all 5 artifacts + the param/column signature
- [x] `uv run mlflow models serve -m models/mlflow_risk_router -p 5001 --env-manager local`
- [x] Smoke test with the P004 CKD payload from `GUIDE.md` §4 — expect a **high** probability:
  ```bash
  curl -s http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d '{
    "dataframe_split": {"columns": ["age_years","diabetes","hypertension","proteinuria_trace_plus","egfr"],
                        "data": [[72, 1, 1, 1, 52]]},
    "params": {"model": "framingham_ckd"}
  }'
  ```

**Done when:** `/invocations` returns a float probability (not `0`/`1`) for at least one model.
✅ **2026-08-10** — verified for all 5 (CVD 0.427, T2DM 0.374, CKD 0.500, CLD 0.218,
DEMENTIA 0.436 on representative payloads); an unknown `model` param returns a clean
`400` with a readable message instead of crashing.

---

## Phase 2 — Backend: implement the two endpoints
This is the core logic. Work in `backend/app/services/risk.py`, wire through
`backend/app/api/v1/endpoints.py`.

### 2a. `get_current_biomarkers`
- [x] Query `demographics` + `biomarkers` for the patient via `app/db/sqlite.py::open_db`
- [x] 404 if patient not found
- [x] Map to `BiomarkersResponse` / `BiomarkerSnapshot`
- [x] Un-skip `test_biomarkers_known_patient` and `test_unknown_patient_returns_404` → green
  (+4 more tests added: second-patient regression guard, full response-shape check,
  422 on missing param, 404 on empty patient_id)

### 2b. `get_current_risks`
- [x] Load demographics + latest biomarkers (404 if unknown)
- [x] Build a payload-building helper per model: read `model.feature_names_in_`,
  derive `age_years` (vs. clinic-today `2026-07-09`), `bmi`, `waist_hip_ratio`,
  `sex_male`, `current_smoker`, `proteinuria_trace_plus`, `bp_treated` — see
  `data/DATA_DICTIONARY.md` "Derived quantities" table and `models/README.md`
  "Input contracts" for exact names/units per model
- [x] Call MLflow via `httpx.AsyncClient`, all 5 models concurrently with `asyncio.gather`
- [x] Band each probability (`low <0.10 / borderline <0.20 / intermediate <0.35 / high ≥0.35`)
- [x] Append one row per risk to `risks` (store `inputs_json`); dedupe strategy:
  compare this call's canonical `inputs_json` against the most recent stored row
  for that (patient, risk_code) — if identical, skip the INSERT and surface the
  existing row as "current" instead of writing a near-duplicate. Documented inline
  in `risk.py` as the deliberate handling of the "GET that writes" smell.
- [x] 502 if MLflow unreachable (connection errors + non-2xx + malformed response shape)
- [x] Return `RisksResponse` with current risks + `trends` (last 5 prior points per risk_code)
- [x] Un-skip `test_risks_returns_five_bands` and `test_risks_are_appended` → green
  (+5 more tests: response-shape/metadata, trend-excludes-current regression test,
  repeat-call-no-duplicate-rows, 404, 422, 502-when-MLflow-down via mocked httpx)

**Done when:** `make test` passes fully (no skips left), and manually:
`curl "http://127.0.0.1:8001/api/v1/get_current_risks?patient_id=P004"` shows CKD as `high`.
✅ **2026-08-10** — 15/15 tests passing, `ruff check .` clean. CKD for P004 = 0.50 (high).

**Built via a 3-agent workflow** (implementer → test-writer → reviewer, looped once
on a real finding): first review pass caught a genuine BLOCKER — `trends[risk_code]`
leaked the "current" value into its own trend list on the dedup no-op path (but
correctly excluded it on the fresh-insert path), verified empirically by the
reviewer via live curl calls, not just code reading. Sent back to implementer,
fixed, re-verified independently (including the reviewer reverting the fix to
confirm the new regression test actually catches it), re-reviewed: **PASS**, no
BLOCKER/MAJOR. Three MINOR items noted but not blocking, left as documented
trade-offs: (1) `asyncio.gather` without `return_exceptions=True` can produce
"exception never retrieved" log noise on partial 5-model failure — functionally
harmless, caller still gets a correct 502; (2) no transaction/locking around the
dedup read-then-write, so two concurrent calls for the same patient could both
insert — unlikely given the assistant's one-call-per-turn pattern; (3) fixed
(stale endpoints.py docstring, applied directly).

---

## Phase 3 — MCP server: add the two tools
Work in `mcp-server/server.py`, uncomment/finish the sketches.

- [ ] `get_current_biomarkers(patient_id)` — wraps the backend call
- [ ] `get_current_risks(patient_id)` — wraps the backend call
- [ ] Clear docstrings + typed args (this is the model's only signal for tool selection)
- [ ] Graceful errors when backend is down or patient unknown
- [ ] `make mcp` → boots on `0.0.0.0:9000/mcp/`
- [ ] Smoke test with the small `fastmcp.Client` script from `mcp-server/README.md`:
  list tools, call `ping`, call both real tools with a known `patient_id`

**Done when:** both tools are listed and callable with the bearer token, against
the live backend + MLflow.

---

## Phase 4 — Evals harness (before or in parallel with LibreChat)
The assignment explicitly says: point this at the model + MCP tools directly,
not the LibreChat UI. Doing this before LibreChat wiring de-risks the highest-
signal deliverable and gives you a way to sanity-check tool behavior without
fighting Docker.

- [ ] Write `evals/harness.py`: a loop that, for each case in `evals/cases.jsonl`,
  sends the `question` to an OpenAI-compatible chat model (OpenRouter) with the
  MCP tools attached (or a small client against the running MCP server), captures
  the tool-call trace + final answer
- [ ] Scorer 1 — **tool-call correctness**: called tool == `expected_tool`, arg
  `patient_id` matches (skip when `expected_tool` is `any`/`none`)
- [ ] Scorer 2 — **numeric/band faithfulness**: for `expected_facts` of kind
  `biomarker`, exact value match; for kind `risk`, band match + probability within
  `tolerance` of `approx_probability`
- [ ] Scorer 3 — **safety**: LLM-as-judge (cheap OpenRouter model + rubric) for
  `safety`/`citation`/behavioral facts — no fabrication, appropriate framing
- [ ] Emit a pass-rate-per-category report + list of failures
- [ ] `make eval` runs it end to end and is reproducible
- [ ] Add at least one case of your own that catches a real regression

**Done when:** `make eval` produces a report with per-category pass rates against
the live backend/MCP stack.

---

## Phase 5 — LibreChat wiring
Lowest signal, highest friction (Docker + SSRF). Time-box this.

- [ ] Copy `librechat.example.yaml` → `librechat.yaml`, fill in the `mcpServers`
  block pointing at `http://host.docker.internal:9000/mcp/` (trailing slash!)
  and the bearer token
- [ ] Set `allowedAddresses` so LibreChat's SSRF guard allows `host.docker.internal`
- [ ] `librechat/docker-compose.override.yml` to avoid the MongoDB crash-loop
- [ ] Set `CREDS_KEY`/`CREDS_IV`/`JWT_SECRET`/`JWT_REFRESH_SECRET` per `librechat/env.notes.md`
- [ ] Set `OPENROUTER_KEY`, pick a **tool-calling-capable** model
- [ ] Boot LibreChat via Docker, confirm MCP server shows "connected" **and tools
  actually fire** (not just connected)
- [ ] End-to-end check: ask *"What are Avraham Friedman's (P004) current risks,
  and how has his kidney risk trended?"* → grounded answer, real values, worsening
  CKD trend

**Done when:** the checklist question above gets a correct, grounded answer in
the LibreChat UI.

**Fallback if this fights you:** stub/script the agent→tool path instead and
make sure Phase 4's evals hit the model + MCP tools directly — that's where the
grading looks anyway.

---

## Phase 6 — Bonus (only once Phases 0–4 are solid)
- [ ] `search_guidelines(query, k)` MCP tool: `uv sync --extra rag`, embed
  `data/guidelines/*.md` into chromadb, return cited snippets
- [ ] Add/verify `citation` category eval cases pass against it
- [ ] (Optional) Custom agent in `agent/`: `uv sync --extra agent`, reuse the
  same MCP tools via `langchain-mcp-adapters` — only build if it demonstrates
  something the built-in agent can't (branching, approval gate, durable state)

---

## Phase 7 — Wrap-up
- [ ] `make lint` clean
- [ ] Full `make test` + `make eval` clean run from a fresh clone/venv if possible
- [ ] Write `SOLUTION.md`: how to run it (+ LibreChat version tag pinned), what
  you built vs. what's left, trade-offs (GET-that-writes, unit assumptions),
  where/how AI tools were used and what you changed or rejected from their output
- [ ] Final read-through: can you explain every decision in a follow-up interview?

---

## Suggested order of work (session-by-session)
1. Phase 0 + Phase 1 (env + MLflow) — unblocks everything else
2. Phase 2 (backend logic) — the core exercise
3. Phase 3 (MCP tools) — thin wrapper, fast once Phase 2 is solid
4. Phase 4 (evals) — highest remaining signal, don't skip for LibreChat
5. Phase 5 (LibreChat) — time-boxed, has a documented fallback
6. Phase 6 (bonus) — only with time to spare
7. Phase 7 (SOLUTION.md) — always do this, even if incomplete