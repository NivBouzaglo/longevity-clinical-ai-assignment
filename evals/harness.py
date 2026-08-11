"""evals/harness.py — CLI entrypoint for the evaluation harness.

Run:  uv run python evals/harness.py   (also what `make eval` runs)

Loads `.env`, connects to the live MCP server, drives every case in
`evals/cases.jsonl` through `evals/runner.py` against a real OpenRouter model,
scores the resulting traces with `evals/scoring.py`, and prints the report.

Requires, all already running on the host (not in Docker):
  - the FastAPI backend on :8001
  - the MLflow model server on :5001 (only needed for `get_current_risks` cases)
  - the FastMCP server on :9000 (`uv run python mcp-server/server.py`)
See mcp-server/README.md / repo GUIDE.md §6 for how to start them.

Design choices
--------------
* **Sequential, not concurrent, case execution.** This is a 13-case eval set,
  not a load test. Running cases one at a time keeps progress output linear
  (one case's tool calls don't interleave with another's in the logs) and
  avoids bursting concurrent requests at OpenRouter's rate limits or the
  single shared MLflow model server. `asyncio.gather` would trade that
  debuggability for a modest wall-clock win that doesn't matter at this scale.
* **Exit code policy: 0 whenever the run completes, non-zero only on
  infrastructure failure.** A low pass rate is exactly what this harness
  exists to surface in its report — it's informational, not a script failure,
  so it must not flip the exit code. Non-zero is reserved for things that mean
  the run itself didn't actually happen: a missing API key, an unreachable MCP
  server, or an unhandled exception while driving a case. This mirrors how
  most eval harnesses let CI gate on "did the harness run", not "did the model
  do well today".
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

REPO_ROOT = Path(__file__).resolve().parents[1]
# Allow `python evals/harness.py` (not `python -m evals.harness`) to still
# import the `evals` package as `evals.runner`/`evals.scoring`, matching how
# the Makefile's `eval` target invokes this file.
sys.path.insert(0, str(REPO_ROOT))

from evals.runner import discover_tool_schemas, make_judge_fn, run_case  # noqa: E402
from evals.scoring import (  # noqa: E402
    Case,
    SuiteResult,
    Trace,
    format_report,
    load_cases,
    score_suite,
)

load_dotenv(REPO_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = (
    os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free").strip() or "openai/gpt-oss-20b:free"
)
# Optional cheaper/separate judge model; falls back to the same model driving the cases.
OPENROUTER_JUDGE_MODEL = os.getenv("OPENROUTER_JUDGE_MODEL", "").strip() or OPENROUTER_MODEL
MCP_BEARER_TOKEN = os.getenv("MCP_BEARER_TOKEN", "dev-longevity-token-change-me")
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:9000/mcp/")

CASES_PATH = REPO_ROOT / "evals" / "cases.jsonl"


async def _run_all_cases(cases: list[Case]) -> dict[str, Trace]:
    """Run every case sequentially against one shared MCP connection + HTTP client.

    A single case failing (even after runner.py's own retries — e.g. the free
    model's upstream pool having a bad few seconds) must not lose every OTHER
    case's already-collected result. Each case is isolated in its own
    try/except; a failure is recorded as an empty Trace (no tool calls, no
    answer) so evals.scoring still scores it — as a fail, correctly — rather
    than silently vanishing from the report or crashing the whole run.
    """
    traces: dict[str, Trace] = {}
    async with Client(MCP_URL, auth=BearerAuth(MCP_BEARER_TOKEN)) as mcp_client:
        tool_schemas = await discover_tool_schemas(mcp_client)
        async with httpx.AsyncClient() as http_client:
            for case in cases:
                print(f"  [{case.id}] ({case.category}) running...", file=sys.stderr)
                try:
                    traces[case.id] = await run_case(
                        case,
                        mcp_client,
                        tool_schemas,
                        api_key=OPENROUTER_API_KEY,
                        model=OPENROUTER_MODEL,
                        http_client=http_client,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad case must not sink the run
                    print(f"  [{case.id}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                    traces[case.id] = Trace(
                        case_id=case.id,
                        final_answer="",
                        raw_messages=[{"role": "system", "content": f"runner error: {exc}"}],
                    )
    return traces


def main() -> int:
    if not OPENROUTER_API_KEY:
        print(
            "ERROR: OPENROUTER_API_KEY is not set in .env.\n"
            "Get a free key at https://openrouter.ai/settings/keys, set "
            "OPENROUTER_API_KEY in the repo-root .env, then re-run `make eval`.",
            file=sys.stderr,
        )
        return 1

    cases = load_cases(CASES_PATH)
    print(f"Loaded {len(cases)} cases from {CASES_PATH}", file=sys.stderr)
    print(f"Model: {OPENROUTER_MODEL}  Judge model: {OPENROUTER_JUDGE_MODEL}  MCP: {MCP_URL}", file=sys.stderr)

    try:
        traces = asyncio.run(_run_all_cases(cases))
    except KeyboardInterrupt:
        print("\nEval run interrupted by user.", file=sys.stderr)
        return 130
    except httpx.ConnectError as exc:
        print(
            f"ERROR: could not reach OpenRouter or the clinical backend: {exc}",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as exc:
        if "failed to connect" in str(exc).lower():
            print(
                f"ERROR: could not reach the MCP server at {MCP_URL} ({exc}).\n"
                "Start it first: `uv run python mcp-server/server.py` (and the backend/MLflow "
                "servers it depends on — see mcp-server/README.md / GUIDE.md §6).",
                file=sys.stderr,
            )
            return 1
        print(f"ERROR: eval run failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - any other infra failure should exit non-zero with a clear message
        print(f"ERROR: eval run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    judge_fn = make_judge_fn(OPENROUTER_JUDGE_MODEL, OPENROUTER_API_KEY)
    suite = score_suite(cases, traces, judge_fn=judge_fn)
    print(format_report(suite))

    run_path = _dump_run(traces, suite)
    print(f"\nFull run detail (every tool call, reasoning, and answer): {run_path}", file=sys.stderr)
    return 0


def _dump_run(traces: dict[str, Trace], suite: SuiteResult) -> Path:
    """Persist the full run — every case's trace (tool calls, reasoning,
    final answer) and scoring detail — to a timestamped JSON file.

    `format_report`'s printed summary only lists failures; this captures
    everything, including passing cases' tool calls and the raw model
    reasoning, so a run can be inspected after the fact rather than only
    trusting the pass/fail line. Gitignored (a run artifact, not source —
    same reasoning as `models/mlflow_risk_router/`).
    """
    runs_dir = REPO_ROOT / "evals" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = runs_dir / f"{stamp}.json"

    data = {
        "timestamp_utc": stamp,
        "model": OPENROUTER_MODEL,
        "judge_model": OPENROUTER_JUDGE_MODEL,
        "overall": {"passed": suite.passed, "total": suite.total, "pass_rate": suite.pass_rate},
        "by_category": {
            cat: {"passed": s.passed, "total": s.total} for cat, s in sorted(suite.by_category.items())
        },
        "cases": [
            {
                "id": result.case_id,
                "category": result.category,
                "question": result.question,
                "passed": result.passed,
                "trace": dataclasses.asdict(traces[result.case_id]) if result.case_id in traces else None,
                "tool_selection": dataclasses.asdict(result.tool_selection),
                "fact_results": [dataclasses.asdict(fr) for fr in result.fact_results],
            }
            for result in suite.case_results
        ],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


if __name__ == "__main__":
    sys.exit(main())
