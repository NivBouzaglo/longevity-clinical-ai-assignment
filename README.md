# Longevity AI — AI Engineer Take-Home

Build the AI layer of a **clinical chat assistant**. The users are doctors in a
single clinic (all doctors can see all patients). Through a chat UI they ask about
a patient's biomarkers and disease risks; behind the chat, tools pull that
patient's data and compute five disease risks in real time from ML models.

Your job is to assemble and wire up that system, then show you can measure its
quality. Most of the "hard to get right" infrastructure (mock data, the risk
models, a FastAPI skeleton, a booting MCP server) is provided — **your work is the
logic, the tools, and the evaluation.**

> 👉 **Start with [`GUIDE.md`](GUIDE.md)** — it has the full setup, the port map,
> the run order, and the traps to avoid. This page is the map; the GUIDE is the manual.

> 🎯 **This take-home is the backbone of your next interview.** We won't just score
> the repo — we'll sit down with you to walk through your code, your trade-offs, and
> make a small change or two together. So optimize for work you can explain and
> extend, not for a polished demo. See **How we evaluate** and **Submission** below.

## Architecture
```
Doctor (browser)
  └─ LibreChat UI            (Docker, :3080)  ── built-in agent + OpenRouter model
       └─ MCP tools ─▶ FastMCP server         (host, :9000)  Bearer auth
                          └─▶ FastAPI backend  (host, :8001)
                                 ├─▶ SQLite  data/patient_db.db   (demographics, biomarkers, risks)
                                 └─▶ MLflow model server (host, :5001)  ── 5 risk models
```
The five risks: **CVD** (PREVENT), **T2DM** (ADA), **CKD** (Framingham), **CLD —
chronic liver disease** (CLivD score), **Dementia** (CAIDE). `get_current_risks`
computes them live and **appends** each result to the `risks` table so the assistant
can show a **trend** over time.

> ℹ️ **The models are synthetic stand-ins.** The five `.pkl` models are mock models
> trained on generated data — they are *not* validated implementations of the real
> published equations, and nothing here is clinically valid. Don't spend time
> reproducing the real scores; treat them as black boxes with a known feature list.

## What's provided vs. what you build
| Provided (ready to use) | You build |
|---|---|
| Mock DB `data/patient_db.db` (+ generator, data dictionary) | The two backend endpoint bodies (`get_current_biomarkers`, `get_current_risks`) |
| 5 risk models `models/*.pkl` (+ generator, model card) | Serving those models with **MLflow** |
| FastAPI skeleton that boots (`/health` works, endpoints return 501) | The two **MCP tools** on the booting FastMCP skeleton |
| FastMCP skeleton with bearer auth + a demo tool | LibreChat setup (`.env`, `librechat.yaml`, run it, wire MCP) |
| LibreChat setup notes + the Docker/SSRF clues (you write `librechat.yaml`) | An **evaluation harness** over `evals/cases.jsonl` |
| Eval gold cases; guideline corpus (bonus RAG) | **Bonus:** retrieval tool + custom agent |

## Your tasks
1. **Backend** — implement the two endpoints (`backend/`). Build each model's
   payload by inspecting the models, call MLflow, band + append the risks.
2. **MLflow** — serve the five models on `:5001` (`GUIDE.md` §4).
3. **MCP** — add the two tools to `mcp-server/server.py` (bearer-authed).
4. **LibreChat** — run it and wire the MCP server (`librechat/SETUP.md`).
5. **Evals** — build `evals/harness.py`: tool-call correctness, numeric
   faithfulness, safety.
6. **Bonus** — `search_guidelines` retrieval tool; and/or a custom agent (`agent/`).

## Where to focus (and the minimum bar)
The signal we care about most is the **backend logic** and the **evaluation harness**.
The MLflow/MCP/LibreChat wiring is real, but it's glue — don't let it eat your time.

- **Minimum viable submission:** the two backend endpoints and the eval harness,
  working and tested. If you get there, you have a passing submission.
- LibreChat wiring is the highest-friction, lowest-signal step (Docker + SSRF; see the
  traps in `GUIDE.md`). If it fights you, stub or script the agent→tool path and point
  your evals at the **model + MCP tools directly** — that's where we look anyway.
- Do the **bonus** (retrieval + citations, custom agent) only once the core works. If
  you have limited time, a small `search_guidelines` + citation-faithfulness eval is
  more relevant to the role than a polished chat UI.

## Quickstart
```bash
uv sync            # Python 3.10–3.13; creates .venv
cp .env.example .env
make data          # (re)generate the DB + models — already committed, just to prove it runs
make backend       # boots on :8001; /health works, endpoints 501 until you implement them
```
Then follow [`GUIDE.md`](GUIDE.md) for MLflow, MCP, and LibreChat.

> 🔑 **OpenRouter:** you need a free account + API key. Pick a model that supports
> **function/tool calling**, or the agent will answer without ever calling your tools
> (see `GUIDE.md` trap #4). The mock patient data is synthetic — fine to send to a
> third-party model.

## How we evaluate
We weight these roughly in this order — **backend and evals carry the most signal**:

| Dimension | What we look for |
|---|---|
| **Python / async backend** | `async` end-to-end, `httpx.AsyncClient`, concurrent model calls, non-blocking SQLite, clean 404/502, typed responses |
| **Evaluation mindset** | Real, reproducible metrics — especially numeric faithfulness — not "looked right once" |
| **LLM app building** | Clear tool contracts (names, docstrings, typed args, errors); MCP auth + transport correct; the agent reliably calls the right tool |
| **Communication** | Clear writeup; documented trade-offs (e.g. the GET-that-writes question, unit assumptions); honest "what's left" |
| **Retrieval (bonus)** | Embeddings + retrieval + citation-faithfulness via `search_guidelines` |

Correctness matters here — these answers inform clinical conversations. Prefer
"I can't verify that" over a confident fabrication, in both the product and your evals.

## Using AI tools
**Use whatever AI tooling you like — Claude, Cursor, Copilot, etc. We do too, and this
is an AI Engineer role.** We don't score you on whether you used AI; we score the
result and, more importantly, your understanding of it. In the follow-up interview
we'll ask you to walk through and extend your own code, so make sure you can explain
every decision.

## Repo layout
```
data/        mock DB + generator + DATA_DICTIONARY.md + guidelines/ (bonus corpus)
models/      5 risk .pkl + generator + model card (README.md)
backend/     FastAPI app (implement the 2 endpoints) + tests
mcp-server/  FastMCP server (add the 2 tools)
evals/       gold cases + build the harness
librechat/   SETUP.md + setup clues (you write librechat.yaml yourself)
agent/       bonus custom-agent track
GUIDE.md     the manual — read this
```

## Submission
Share a repo (or archive) with your code plus a short `SOLUTION.md`:

- **How to run it** — the commands, plus the LibreChat release tag you pinned.
- **What you built** and what you didn't get to (an honest "what's left" is a plus).
- **Trade-offs** you made (e.g. the GET-that-writes question, unit assumptions).
- **Where you used AI tools** and what you changed or rejected from their output — we'll
  use this as a starting point for the walkthrough.
