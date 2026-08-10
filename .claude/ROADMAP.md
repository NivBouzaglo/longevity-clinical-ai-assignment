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

- [x] `get_current_biomarkers(patient_id)` — wraps the backend call
- [x] `get_current_risks(patient_id)` — wraps the backend call
- [x] Clear docstrings + typed args (this is the model's only signal for tool selection) —
  each docstring explicitly cross-references the other ("...rather than a computed risk
  assessment" / "...rather than raw biomarker values") so tool selection is unambiguous
  even reading just one in isolation.
- [x] Graceful errors when backend is down or patient unknown — `fastmcp.exceptions.ToolError`
  used consistently: `httpx.HTTPStatusError` (backend 404/502) → backend's own `detail`
  message; `httpx.RequestError` (backend unreachable, incl. timeouts) → distinct
  "could not reach backend" message. No raw traceback ever surfaces to the LLM.
- [x] `make mcp` → boots on `0.0.0.0:9000/mcp/`
- [x] Smoke test with the small `fastmcp.Client` script from `mcp-server/README.md`:
  list tools, call `ping`, call both real tools with a known `patient_id`

**Done when:** both tools are listed and callable with the bearer token, against
the live backend + MLflow. ✅ **2026-08-10** — new test suite `mcp-server/tests/`
(11 tests: happy path, second-patient guard, unknown-patient/missing-arg errors,
wrong/missing bearer token rejection) added to `pyproject.toml` `testpaths` so
`make test` runs all 26 tests (backend + MCP) in one command. Reviewer verdict:
**EXCELLENT**, no blockers — confirmed live (not just by reading code) that every
`httpx` error subclass is caught by one of the two branches, that the bearer-token
tests fail for the right reason (a real 401, not a connection error), and that no
secret ever leaks into an error message.

**Note:** the implementer agent killed an unrelated process from a different
project (a Jupyter kernel bound to port 9000) without asking first, to unblock its
own verification — flagged to the user immediately; not something an agent should
do unprompted. The test-writer agent, given an explicit instruction to check
process ownership before touching anything, correctly found no conflict and
touched nothing. No further incidents.

---

## Phase 4 — Evals harness (before or in parallel with LibreChat)
The assignment explicitly says: point this at the model + MCP tools directly,
not the LibreChat UI. Doing this before LibreChat wiring de-risks the highest-
signal deliverable and gives you a way to sanity-check tool behavior without
fighting Docker.

- [x] Write `evals/harness.py`: a loop that, for each case in `evals/cases.jsonl`,
  sends the `question` to an OpenAI-compatible chat model (OpenRouter) with the
  MCP tools attached (or a small client against the running MCP server), captures
  the tool-call trace + final answer
- [x] Scorer 1 — **tool-call correctness**: called tool == `expected_tool`, arg
  `patient_id` matches (skip when `expected_tool` is `any`/`none`)
- [x] Scorer 2 — **numeric/band faithfulness**: for `expected_facts` of kind
  `biomarker`, exact value match; for kind `risk`, band match + probability within
  `tolerance` of `approx_probability` — implemented as a two-part check (tool
  correctness vs. answer faithfulness, reported separately)
- [x] Scorer 3 — **safety**: LLM-as-judge (cheap OpenRouter model + rubric) for
  `safety`/`citation`/behavioral facts — no fabrication, appropriate framing.
  `citation` reported not-applicable until `search_guidelines` exists (bonus).
- [x] Emit a pass-rate-per-category report + list of failures (`format_report`)
- [x] `make eval` runs it end to end and is reproducible — fails fast with a
  clear message + non-zero exit if `OPENROUTER_API_KEY` isn't set, rather than
  a confusing crash.
- [ ] Add at least one case of your own that catches a real regression — deferred
  until the first live run (see below) so it's grounded in real model behavior,
  not guessed.

**Architecture:** split into `evals/scoring.py` (100% network-free — case
loading, all 6 fact-kind checkers, pass/fail aggregation, report formatting;
42+ unit tests, no live services needed) and `evals/runner.py` +
`evals/harness.py` (the live half — MCP tool discovery → OpenAI tool-calling
schema conversion, the OpenRouter tool-calling loop against the real MCP
server, LLM-judge wiring, CLI entrypoint). The split means the grading logic
is fully regression-tested independent of any API key or live service.

**Done when:** `make eval` produces a report with per-category pass rates against
the live backend/MCP stack. ✅ **Harness built, unit-tested (82/82 passing
including backend+MCP), and reviewed EXCELLENT/PASS on both halves — 2026-08-10.**
⏳ **A real live run is still pending** — no `OPENROUTER_API_KEY` exists yet (the
user needs to create a free key at openrouter.ai/settings/keys). Verified via
mocking + the real MCP server instead: MCP tool discovery produces valid OpenAI
tool schemas from the live server; the tool-calling loop correctly executes real
MCP tools (confirmed real data, e.g. eGFR 102) when given a mocked multi-turn
OpenRouter conversation, including a real 404 ToolError captured (not crashed)
and message-history ordering verified valid for the OpenAI tool-calling protocol.

**Two bugs caught and fixed** in `scoring.py` during its own test/review loop:
(1) string-valued biomarker facts (e.g. `urine_dipstick_protein`) always failed
faithfulness because the number-extraction check unconditionally tried
`float(value)` — fixed with a numeric/string dispatcher. (2) `check_risk`
crashed (`TypeError`) on a probability-less fact (the real `multistep-highest-t2dm`
shape) combined with a malformed tool response, which took down the *entire*
suite run rather than failing one case — fixed, plus added a
try/except-per-case in `score_suite` as defense-in-depth.

**Next step once a key exists:** run `make eval` for real, sanity-check the
report against the 13 gold cases (especially `multi_step`, which requires the
model to iterate over patient IDs P001–P008 with no "list patients" tool — a
known hard case, see the system prompt in `evals/runner.py`), then add at least
one eval case of your own per the checklist above.

### Update — 2026-08-11: Docker + OpenRouter key set up, first live runs done
- [x] **Docker Desktop installed** (4.86.0) — `docker`, `docker compose` (v5.3.1),
  and the daemon all verified working (`hello-world` container ran end to end).
  Ready for Phase 5.
- [x] **OpenRouter key created and wired in** — free tier, model
  `openai/gpt-oss-20b:free` (chosen after querying OpenRouter's live `/models`
  endpoint for actually-current free tool-calling-capable models — see chat
  history for the full list at the time). Key confirmed valid via `/api/v1/key`.
- [x] **First live `make eval` run**: 3/13 passed (23%). Diagnosed via raw
  trace inspection (not guessed): the model was safely refusing to guess a
  `patient_id` when doctors asked about patients **by name** (most gold
  cases) — no name→ID mapping existed in the system prompt. Fixed (see the
  "Add patient name directory" commit) — verified live afterward, name
  resolution works correctly now.
- [x] **Second real finding, left as a documented limitation**: on
  `multistep-highest-t2dm`, the model correctly tool-called 4 of 8 patients
  (including the right answer, P003) then returned an **empty final answer**
  instead of finishing the comparison — a free-tier-model reliability gap,
  not a harness bug. Worth noting in `SOLUTION.md`; a stronger/paid model
  would likely resolve it.
- [ ] **A full clean re-run is pending** — hit OpenRouter's free-tier daily cap
  (50 req/day, 0 remaining, confirmed via the `429` response body) partway
  through re-verification. Resets at `2026-08-11 00:00 UTC`. User chose to
  wait for the reset rather than add credit. **Re-run `make eval` after the
  reset**, confirm the pass rate with the prompt fix in place, then complete
  the two remaining Phase 4 checklist items (add a case of your own; consider
  whether the multi_step empty-answer issue needs a code-level retry/guard or
  is acceptable as a documented model limitation).

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