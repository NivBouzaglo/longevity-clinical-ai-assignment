"""Tests for `evals/harness.py` — the CLI entrypoint.

Per this repo's convention (see `evals/tests/test_scoring.py`), plain `pytest`
functions. `harness.main()` is synchronous itself (it drives the async work via
`asyncio.run` internally), so no `pytest-asyncio` involvement is needed here —
these are ordinary sync tests.

`OPENROUTER_API_KEY` is read once at import time from `.env` into the module
global `evals.harness.OPENROUTER_API_KEY`. Since it's empty in this repo's
`.env`, the "missing key" path below is actually the real, unmodified
behavior; monkeypatching it is only needed for the "happy path" test, which
must fake a present key without ever making a real network call.

Neither test here makes any real HTTP/MCP call: the missing-key path returns
before touching the network at all, and the happy-path test replaces
`_run_all_cases` and `make_judge_fn` (the only two places `main()` touches the
network) with stand-ins.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import evals.harness as harness
from evals.scoring import Case, Trace, load_cases


def test_main_returns_1_and_prints_guidance_when_api_key_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(harness, "OPENROUTER_API_KEY", "")

    with patch.object(harness, "_run_all_cases", new=AsyncMock()) as mock_run:
        exit_code = harness.main()

    assert exit_code == 1
    mock_run.assert_not_called()  # confirms no network path was taken

    captured = capsys.readouterr()
    assert "OPENROUTER_API_KEY" in captured.err
    assert "openrouter.ai" in captured.err.lower()


def test_main_happy_path_returns_0_and_prints_report(monkeypatch, capsys) -> None:
    monkeypatch.setattr(harness, "OPENROUTER_API_KEY", "fake-key-for-test")

    cases = load_cases(harness.CASES_PATH)
    fake_traces = {c.id: Trace(case_id=c.id, tool_calls=[], final_answer="stub answer") for c in cases}

    def fake_judge_fn(prompt: str) -> str:
        return "PASS: stub verdict."

    with (
        patch.object(harness, "_run_all_cases", new=AsyncMock(return_value=fake_traces)),
        patch.object(harness, "make_judge_fn", return_value=fake_judge_fn) as mock_make_judge_fn,
    ):
        exit_code = harness.main()

    assert exit_code == 0
    mock_make_judge_fn.assert_called_once()

    captured = capsys.readouterr()
    assert "EVAL REPORT" in captured.out
    assert "Overall:" in captured.out


async def test_run_all_cases_isolates_a_failing_case() -> None:
    """One case's `run_case` raising (e.g. a flaky upstream exhausting
    runner.py's own retries) must not lose the other cases' already-collected
    traces or crash the whole run — see the `_run_all_cases` docstring."""
    cases = [
        Case(id="ok-1", category="x", question="q1", expected_tool="any", patient_id="P001"),
        Case(id="fails", category="x", question="q2", expected_tool="any", patient_id="P002"),
        Case(id="ok-2", category="x", question="q3", expected_tool="any", patient_id="P003"),
    ]

    async def fake_run_case(case, *_args, **_kwargs):
        if case.id == "fails":
            raise RuntimeError("simulated upstream failure after retries exhausted")
        return Trace(case_id=case.id, tool_calls=[], final_answer=f"answer for {case.id}")

    fake_client_cm = MagicMock()
    fake_client_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    fake_client_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(harness, "Client", return_value=fake_client_cm),
        patch.object(harness, "discover_tool_schemas", new=AsyncMock(return_value=[])),
        patch.object(harness, "run_case", new=fake_run_case),
    ):
        traces = await harness._run_all_cases(cases)

    assert set(traces) == {"ok-1", "fails", "ok-2"}
    assert traces["ok-1"].final_answer == "answer for ok-1"
    assert traces["ok-2"].final_answer == "answer for ok-2"
    # the failed case still gets a (failing) Trace, not a missing entry
    assert traces["fails"].final_answer == ""
    assert traces["fails"].tool_calls == []
    assert "runner error" in traces["fails"].raw_messages[0]["content"]
