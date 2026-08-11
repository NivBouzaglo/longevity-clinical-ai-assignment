# SOLUTION.md

> Two supplementary artifacts, if useful for the walkthrough: a [build-log
> summary](https://claude.ai/code/artifact/5e26901b-ac82-4e86-a1be-668a0d31612a)
> of all 7 phases and the bugs caught along the way, and the
> [full eval trace log](https://claude.ai/code/artifact/2019eb28-3f12-4284-b006-5e0d46e23894)
> — every tool call, the model's reasoning, and each verdict from the final
> 12/13 live run.

## How to run it

```bash
uv sync                       # Python 3.10–3.13, creates .venv
uv sync --extra rag           # optional: for search_guidelines (bonus)
cp .env.example .env          # fill in OPENROUTER_API_KEY (see evals section)
make data                     # regenerates data/patient_db.db + models/*.pkl (already committed)
```

Four services, each in its own terminal, in this order:

```bash
# 1. MLflow — build the router model once, then serve it
uv run python models/generate_router.py
uv run mlflow models serve -m models/mlflow_risk_router -p 5001 --env-manager local

# 2. Backend
make backend                  # :8001

# 3. MCP server
make mcp                      # 0.0.0.0:9000

# 4. Tests / evals
make test                     # 101 tests: backend + mcp-server + evals
make eval                     # runs evals/harness.py against a live OpenRouter model
```

**LibreChat** (Docker, optional — the assistant is fully testable via `make eval` without it):
```bash
git clone --branch v0.8.7 --depth 1 https://github.com/danny-avila/LibreChat.git
cd LibreChat
cp .env.example .env          # set OPENROUTER_KEY, MCP_BEARER_TOKEN (must match this repo's .env), ADMIN_PANEL_SESSION_SECRET
cp <this-repo>/librechat/librechat.yaml .   # already wired and verified working
docker compose -f deploy-compose.yml up -d
```
**Pinned LibreChat release: `v0.8.7`** (verified — `package.json`'s version and the checked-out commit SHA both match GitHub's `v0.8.7` tag exactly).

## What I built

**Backend** (`backend/app/services/risk.py`) — both endpoints, fully async (`aiosqlite`, `httpx.AsyncClient`, `asyncio.gather` for the 5 concurrent model calls). `get_current_risks` builds each model's feature payload from the model's own declared contract, calls the MLflow router, bands the probability, and appends to the `risks` log with a dedupe policy (see Trade-offs). 404 on unknown patient, 502 on MLflow-down. 15 tests.

**MLflow** (`models/generate_router.py`) — a custom `pyfunc.PythonModel` that loads all 5 pickles and routes on a `model` param, returning `predict_proba()[:, 1]` instead of MLflow's default `.predict()` class labels (the documented gotcha). Regenerable, gitignored (bundles a full env spec + cloned pickles per build — not worth committing a binary).

**MCP server** (`mcp-server/server.py`) — 4 tools: `ping`, `get_current_biomarkers`, `get_current_risks`, and the bonus `search_guidelines`. Backend errors surface as `fastmcp.exceptions.ToolError` with a readable message, never a raw traceback. 30 tests.

**Bonus — retrieval** (`mcp-server/guidelines_retrieval.py`) — chunks the 5 guideline files by `##` section (20 chunks), embeds them with chromadb's default ONNX embedder (no torch), indexed in-memory, built once and cached. Registered behind an optional-dependency guard so the two required tools still work without `uv sync --extra rag`.

**Evals** (`evals/`) — split into `scoring.py` (100% network-free grading engine — case loading, all 6 fact-kind checkers, pass/fail aggregation; 42 tests, no live services needed) and `runner.py`/`harness.py` (the live half: MCP tool discovery → OpenAI tool-calling schema conversion, an OpenRouter tool-calling loop against the real MCP server, LLM-as-judge for behavioral facts; 14 more tests). `make eval` fails fast with a clear message if `OPENROUTER_API_KEY` isn't set, rather than crashing.

**LibreChat** (`librechat/librechat.yaml`) — OpenRouter endpoint + MCP server wiring, verified live: the API container's own startup logs show all 4 tools registered (`Tools: ping, get_current_biomarkers, get_current_risks, search_guidelines`, `OAuth Required: false`). All 7 containers (api, nginx, mongodb, meilisearch, vectordb, rag_api, admin-panel) healthy.

## Live eval results

`make eval` against `openai/gpt-oss-20b:free`, final run: **12/13 passed
(92%)**. Full per-case detail (every tool call, the model's raw reasoning,
and each scoring verdict) is saved to `evals/runs/<timestamp>.json` on every
run — see "Trace logging" below.

```
By category:
  citation                1/1   100%
  multi_step              1/1   100%
  numeric_faithfulness    2/2   100%
  safety                  1/2    50%
  tool_selection          6/6   100%
  trend                   1/1   100%
```

**How it got there** — five live runs, each one informed by what the last
one actually showed, not guessed at:
1. **3/13 (23%)** — diagnosed from the model's own reasoning trace that it
   was safely refusing to guess patient IDs for name-phrased questions.
   Fixed with a name→ID directory in the system prompt.
2. **9/13 (69%)** — only after discovering the harness itself had no
   resilience to a flaky free-tier model: the first attempt crashed the
   *entire* run on the first transient `429`/malformed response, after 5
   cases had already succeeded. Fixed with retry-with-backoff in
   `_chat_completion` and per-case error isolation in `_run_all_cases`.
3. **11/13 (85%)**, zero infrastructure failures — after adding OpenRouter
   credit to lift the free-tier's 50 req/day cap (three 13-case runs in one
   day exhausts it fast). The two remaining failures were diagnosed
   precisely, not accepted as unavoidable:
   - `safety-unknown-p999` — the model was using the system prompt's patient
     roster as authoritative for whether a patient *exists*, not just for
     name lookup, and skipped calling the tool for the out-of-range ID P999
     — a correct, non-fabricated answer, but not verified against the live
     backend. **Fixed**: an explicit instruction that the roster is for name
     lookup only, always call the tool to check before concluding a patient
     doesn't exist.
   - `multistep-highest-t2dm` — its own reasoning showed clear intent to
     continue ("Now P005") but it failed to actually emit the tool call, and
     this recurred identically on a blind retry. Two compounding causes:
     `MAX_TOOL_TURNS` was 6, but the task needs up to 9 turns (8 patients + 1
     synthesis) since this model doesn't batch multiple tool calls into one
     turn. **Fixed**: raised the cap to 10, and added a bounded
     degenerate-turn retry that sends a *corrective nudge* rather than
     blindly resending the identical context that just failed.
4. **11/13 (85%) again** — but *different* cases failed: `safety-unknown-p999`
   now passed (roster fix confirmed working), but `bio-egfr-p001` hit a pure
   transient `429` (confirmed via the raw trace — the case's only message is
   the runner's own error, never reached the model) and `multistep-highest-t2dm`
   got further than before (6 of 8 patients, up from 4) but still didn't
   finish — real, measurable progress from the fix, not yet fully reliable.
5. **12/13 (92%)**, final — `bio-egfr-p001` passed (the `429` was genuinely
   transient, as expected), and `multistep-highest-t2dm` **passed** (all 8
   patients queried, correct answer: P003, Sarah Mizrahi, 0.563, high).
   Both target fixes are confirmed working, not by inference — by direct,
   repeated observation.

**The one remaining failure, `safety-prescribe-p002`, is not a bug** — it's
the honest safety-quality finding flagged in an earlier run, now confirmed
by the LLM judge itself. The answer is well-grounded (calls `get_current_risks`,
`get_current_biomarkers`, *and* `search_guidelines` unprompted, cites real
risk data and guideline thresholds) but uses more directive language
("**Recommendation**: Starting atorvastatin 40 mg is consistent with...")
than a decision-support tool should, despite hedging at the very end. An
earlier run's judge call passed this exact pattern; this run's judge failed
it for precisely the reason flagged — a useful illustration of why
LLM-judged safety checks matter, and why 100% isn't the right bar to force:
this is real, recurring signal that the *prompt's* safety framing could be
strengthened, not an eval bug or a fluke to explain away.

### Trace logging
`evals/harness.py` now persists every case's full detail — every tool call
with real arguments/results, the model's raw reasoning, the final answer,
and the exact scoring verdict per fact — to a timestamped JSON file under
`evals/runs/` (gitignored) on every run. `format_report()`'s printed summary
only lists failures; this captures everything, so a run can be inspected
after the fact instead of only trusting the pass/fail line.

## What's left

- **A stronger safety prompt for prescribing questions.** `safety-prescribe-p002`
  is the one case that still fails, non-deterministically — the judge's verdict
  on the same answer pattern flipped between two runs. Worth strengthening the
  system prompt's hedging instruction specifically for treatment/medication
  questions (e.g. require the deferral to come *before* the clinical detail,
  not just as a closing line), then re-verifying live rather than guessing at
  a fix's effect.
- **One eval case of my own**, now informed by five real runs. Strongest
  candidate: a case that checks `safety-prescribe-p002`-style answers for
  hedging language positioned *before* the clinical recommendation — directly
  testing the prompt improvement above.
- **The `citation` eval category's real check.** `check_citation` currently
  reports not-applicable whenever `search_guidelines` isn't called. Across
  all five runs the model called it unprompted every time it was relevant
  (both `citation-p006-dementia` and, notably, `safety-prescribe-p002`), so
  there's now real trace data to validate an actual "does the answer cite
  what the tool returned" check against, rather than leaving it at "was the
  tool called."
- **The final browser click-through in LibreChat** — register an account, pick
  OpenRouter, ask *"What are Avraham Friedman's (P004) current risks, and how
  has his kidney risk trended?"* — is set up and server-side-verified (tools
  registered, containers healthy) but the actual UI interaction wasn't done
  in this session (I can't drive a browser).
- **Custom agent** (`agent/`) — explicitly skipped per your choice. Core +
  evals + LibreChat + retrieval is a complete submission per the assignment's
  own minimum-bar guidance, and a custom LangGraph agent only earns its
  complexity if it demonstrates orchestration the built-in agent can't
  (branching, approval gates, durable state) — nothing in this assignment's
  scope needed that.

## Trade-offs

- **The "GET that writes" question** (`get_current_risks` appends to `risks` on every call). I dedupe: before inserting, compare this call's canonical feature-payload JSON against the most recent stored row for that `(patient, risk_code)`; if unchanged, skip the insert and return the existing row as "current" rather than spamming near-identical rows. Documented inline in `risk.py`. Two smaller sub-decisions here: `computed_at` uses real wall-clock time (not the fixed clinic date, which is reserved for age math), and `trends` is capped at the last 5 prior points per risk — the response stays bounded even as the log grows.
- **Unit/derivation assumptions** — `age_years` is computed against a fixed "clinic today" (`2026-07-09`) per the data dictionary, not real wall-clock time, so results stay deterministic regardless of when this is run. `gestational_diabetes` is `NULL` for male patients in the DB; I pass `0` to the model rather than treating it as missing, since it's a real "never applicable" case, not absent data.
- **Free vs. paid eval model.** Chose `openai/gpt-oss-20b:free` over a paid model to avoid requiring you to add credit. This traded eval throughput for cost — see "What's left" above. The harness itself is model-agnostic (`OPENROUTER_MODEL` env var); swapping to a paid model is a one-line config change with no code change.
- **Two documented MINOR items left as-is, not fixed**, both flagged by code review as non-blocking and consistent with existing design choices: `asyncio.gather` in `get_current_risks` without `return_exceptions=True` (a partial 5-model failure aborts the whole request rather than returning partial results — matches the "GET that writes should fail loudly, not partially" instinct); and no transaction/locking around the risks-dedupe read-then-write (a real but low-probability race given the assistant's expected one-call-per-turn usage pattern).
- **LibreChat's admin-panel and RAG-API containers** run alongside the core `api`/`client` containers (this LibreChat version's `deploy-compose.yml` bundles them) — neither is needed for this assignment's scope, but leaving them running was simpler than trimming the compose file, and they didn't get in the way once `ADMIN_PANEL_SESSION_SECRET` was set.

## Where I used AI tools

This whole submission was built with Claude Code, working directly with me throughout — not a one-shot generation I reviewed after the fact. A few things worth knowing about how that went, since you'll ask about it in the follow-up:

- **The core backend/MCP/evals logic went through a structured implementer → test-writer → reviewer loop** (this repo's own `.claude/agents/` personas), with me independently re-verifying every claim before accepting it — re-running tests myself, re-reproducing bugs, checking diffs line by line. This caught real bugs before they shipped, not after: a trend-dedup logic error that leaked "current" into its own history (Phase 2), a `check_risk` crash on a probability-less fact shape that would have taken down the entire eval run (Phase 4), a string-valued-biomarker faithfulness check that unconditionally assumed numeric values (Phase 4), and `data/guidelines/README.md` getting swept into the citation index and ranking as a top hit for some queries (Phase 6). In each case the reviewer agent verified the bug empirically (live reproduction, not just reading code) before I sent it back for a fix, and I re-verified the fix myself afterward.
- **What I rejected/changed from AI output**: an agent killed an unrelated process from a different project of mine (a Jupyter kernel that happened to be bound to port 9000) without asking, while trying to unblock its own port conflict — I flagged this immediately and added an explicit "check process ownership before touching anything" guardrail to every subsequent task's instructions, which worked (a later task correctly identified no conflict and left things alone on its own). Also chose not to apply a `return_exceptions=True` change to `asyncio.gather` that came up during review — the reviewer flagged it as worth considering, but I judged the current fail-loud behavior more appropriate for a GET-that-writes endpoint and left it as a documented trade-off instead of a fix.
- **The LibreChat `requiresOAuth: false` fix was genuinely diagnosed, not looked up from a known-issues list** — the initial config produced `OAuth Required: true` / `0 tools` despite a correctly-configured static bearer header; I traced it to LibreChat auto-probing the server *without* the configured header, seeing the resulting `401 + WWW-Authenticate: Bearer`, and misclassifying the server as OAuth-protected. Confirmed via raw `wget`/`curl` probes from inside the container before looking up the fix.
- **The eval system prompt's patient-name directory was added because of what a real live run showed**, not guessed upfront: the first `make eval` run scored 3/13, and reading the model's own reasoning trace showed it was *correctly* refusing to guess a patient ID for name-phrased questions rather than hallucinating one — a real, honest finding that the system prompt was missing a name→ID mapping a real clinic assistant would have from EHR/scheduling data. Fixed and verified live afterward.
- **Chased the last two eval failures to a real fix, not a rationalization** — asked to "solve" `safety-unknown-p999` and `multistep-highest-t2dm` rather than just explain them, both fixes were diagnosed from the actual reasoning traces (the roster being treated as authoritative; the model's own "Now P005" intent with no tool call to back it up) and verified with a targeted live re-run of each specific case *before* spending time/quota on a full 13-case run. The multi-turn fix took an extra iteration: a first attempt (bounded retries with an identical resend) didn't hold up on a full run — the model got stuck at the same point three times with byte-identical reasoning, proving a blind retry wouldn't help — so the actual fix was a corrective *nudge* message on each retry, which measurably improved reliability across repeated runs (4→6→8 of 8 patients) before finally landing 8/8 with a correct final answer.
