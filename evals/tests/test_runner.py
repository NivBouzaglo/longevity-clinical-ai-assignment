"""Tests for `evals/runner.py` — the network-touching half of the eval harness.

Per this repo's convention (mirrors `mcp-server/tests/test_tools.py`): plain
`pytest` functions, `pytest-asyncio`'s `asyncio_mode = "auto"` (pyproject.toml)
means `async def test_...` needs no special decorator or marker.

Two networks are involved here, handled very differently:
  - The MCP tool-calling side (`tool_to_openai_schema`, `discover_tool_schemas`,
    and the tool-execution parts of `run_case`) is tested against the LIVE MCP
    server (mirrors `mcp-server/tests/test_tools.py`'s `client`/`MCP_URL`
    pattern) — free, deterministic, already covered by its own test suite, no
    reason to fake it.
  - The OpenRouter side (`_chat_completion`, called from inside `run_case`, and
    the plain `httpx.post` inside `make_judge_fn`) is ALWAYS mocked
    (`unittest.mock.patch`, matching
    `backend/tests/test_endpoints.py::test_risks_returns_502_when_mlflow_unreachable`'s
    pattern) — there is no `OPENROUTER_API_KEY` configured in this environment,
    so a real call is not an option.

Requires the backend (:8001), MLflow (:5001), and the MCP server (:9000) all
running — see mcp-server/README.md / repo GUIDE.md §6.

Only `get_current_biomarkers` and `ping` are exercised below (both read-only);
none of these tests call `get_current_risks`, so `data/patient_db.db` is never
touched and no `reset_db`-style fixture is needed here.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

from evals.harness import MCP_BEARER_TOKEN, MCP_URL
from evals.runner import (
    MAX_TOOL_TURNS,
    discover_tool_schemas,
    make_judge_fn,
    run_case,
    tool_to_openai_schema,
)
from evals.scoring import Case, _parse_judge_verdict


@pytest_asyncio.fixture
async def mcp_client() -> Client:
    """An authenticated fastmcp.Client connected to the live MCP server."""
    async with Client(MCP_URL, auth=BearerAuth(MCP_BEARER_TOKEN)) as c:
        yield c


@pytest_asyncio.fixture
async def http_client() -> httpx.AsyncClient:
    """A placeholder AsyncClient for `run_case`'s signature.

    Never actually reaches the network: `_chat_completion` is mocked in every
    `run_case` test below, so this object is passed through but not used.
    """
    async with httpx.AsyncClient() as c:
        yield c


@pytest_asyncio.fixture
async def tool_schemas(mcp_client: Client) -> list[dict]:
    return await discover_tool_schemas(mcp_client)


def _synthetic_case(case_id: str = "synthetic-run-case", question: str = "test question") -> Case:
    return Case(
        id=case_id,
        category="test",
        question=question,
        expected_tool="get_current_biomarkers",
        patient_id="P001",
    )


def _tool_call_message(calls: list[tuple[str, str, dict]]) -> dict:
    """Build an OpenRouter-shaped assistant message requesting one or more tool calls.

    ``calls`` is a list of ``(id, tool_name, arguments_dict)``; ``arguments_dict``
    is JSON-encoded here to match OpenRouter's real wire shape (`function.arguments`
    is always a JSON *string*, never a dict).
    """
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        for call_id, name, args in calls
                    ],
                }
            }
        ]
    }


def _final_answer_message(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


# ---------------------------------------------------------------------------
# tool_to_openai_schema
# ---------------------------------------------------------------------------


async def test_tool_to_openai_schema_matches_shape_for_real_tool(mcp_client: Client) -> None:
    tools = await mcp_client.list_tools()
    biomarkers_tool = next(t for t in tools if t.name == "get_current_biomarkers")
    schema = tool_to_openai_schema(biomarkers_tool)

    assert schema == {
        "type": "function",
        "function": {
            "name": "get_current_biomarkers",
            "description": biomarkers_tool.description,
            "parameters": biomarkers_tool.inputSchema,
        },
    }
    assert schema["function"]["parameters"]["required"] == ["patient_id"]
    assert schema["function"]["parameters"]["properties"]["patient_id"]["type"] == "string"


async def test_tool_to_openai_schema_handles_zero_parameter_tool(mcp_client: Client) -> None:
    """`ping` takes no arguments — the converted schema must not choke on an
    empty `properties`/no `required` list."""
    tools = await mcp_client.list_tools()
    ping_tool = next(t for t in tools if t.name == "ping")
    schema = tool_to_openai_schema(ping_tool)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "ping"
    assert schema["function"]["parameters"]["properties"] == {}


def test_tool_to_openai_schema_handles_missing_description() -> None:
    """A fake Tool-like object (no live server needed) with `description=None`
    must not crash — falls back to an empty string per the docstring."""

    class FakeTool:
        name = "fake_tool"
        description = None
        inputSchema = {"type": "object", "properties": {}, "required": []}

    schema = tool_to_openai_schema(FakeTool())
    assert schema["function"]["name"] == "fake_tool"
    assert schema["function"]["description"] == ""
    assert schema["function"]["parameters"] == {"type": "object", "properties": {}, "required": []}


# ---------------------------------------------------------------------------
# discover_tool_schemas
# ---------------------------------------------------------------------------


async def test_discover_tool_schemas_against_live_server(mcp_client: Client) -> None:
    schemas = await discover_tool_schemas(mcp_client)
    by_name = {s["function"]["name"]: s for s in schemas}

    assert {"ping", "get_current_biomarkers", "get_current_risks"} <= set(by_name)

    for tool_name in ("get_current_biomarkers", "get_current_risks"):
        params = by_name[tool_name]["function"]["parameters"]
        assert params["required"] == ["patient_id"]
        assert params["properties"]["patient_id"]["type"] == "string"

    for schema in schemas:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]


# ---------------------------------------------------------------------------
# run_case — happy flow
# ---------------------------------------------------------------------------


async def test_run_case_happy_flow_calls_real_tool_and_returns_final_answer(
    mcp_client: Client, tool_schemas: list[dict], http_client: httpx.AsyncClient
) -> None:
    case = _synthetic_case(question="What is Maya Cohen's eGFR?")
    responses = [
        _tool_call_message([("call_1", "get_current_biomarkers", {"patient_id": "P001"})]),
        _final_answer_message("Maya Cohen's most recent eGFR is 102 mL/min/1.73m2."),
    ]

    with patch("evals.runner._chat_completion", new=AsyncMock(side_effect=responses)):
        trace = await run_case(
            case, mcp_client, tool_schemas, api_key="fake-key", model="fake/model", http_client=http_client
        )

    assert trace.case_id == case.id
    assert len(trace.tool_calls) == 1
    call = trace.tool_calls[0]
    assert call.name == "get_current_biomarkers"
    assert call.arguments == {"patient_id": "P001"}
    assert call.error is None
    assert call.result["biomarkers"]["egfr_ml_min_1_73m2"] == 102  # real data from the live MCP server
    assert trace.final_answer == "Maya Cohen's most recent eGFR is 102 mL/min/1.73m2."


# ---------------------------------------------------------------------------
# run_case — regular workflow: multiple tool calls in one turn
# ---------------------------------------------------------------------------


async def test_run_case_executes_multiple_tool_calls_in_one_turn(
    mcp_client: Client, tool_schemas: list[dict], http_client: httpx.AsyncClient
) -> None:
    case = _synthetic_case(question="Compare eGFR for P001 and P002.")
    responses = [
        _tool_call_message(
            [
                ("call_1", "get_current_biomarkers", {"patient_id": "P001"}),
                ("call_2", "get_current_biomarkers", {"patient_id": "P002"}),
            ]
        ),
        _final_answer_message("P001's eGFR is 102; P002's eGFR is 74."),
    ]

    with patch("evals.runner._chat_completion", new=AsyncMock(side_effect=responses)):
        trace = await run_case(
            case, mcp_client, tool_schemas, api_key="k", model="m", http_client=http_client
        )

    assert len(trace.tool_calls) == 2
    by_patient = {c.arguments["patient_id"]: c for c in trace.tool_calls}
    assert by_patient["P001"].error is None
    assert by_patient["P002"].error is None
    assert by_patient["P001"].result["biomarkers"]["egfr_ml_min_1_73m2"] == 102
    assert by_patient["P002"].result["biomarkers"]["egfr_ml_min_1_73m2"] == 74
    assert trace.final_answer == "P001's eGFR is 102; P002's eGFR is 74."


# ---------------------------------------------------------------------------
# run_case — edge case: real tool error (unknown patient)
# ---------------------------------------------------------------------------


async def test_run_case_captures_real_tool_error_and_continues_loop(
    mcp_client: Client, tool_schemas: list[dict], http_client: httpx.AsyncClient
) -> None:
    case = _synthetic_case(case_id="unknown-patient", question="What is NOPE's eGFR?")
    responses = [
        _tool_call_message([("call_1", "get_current_biomarkers", {"patient_id": "NOPE"})]),
        _final_answer_message("I could not find a patient with ID NOPE in the system."),
    ]

    with patch("evals.runner._chat_completion", new=AsyncMock(side_effect=responses)):
        trace = await run_case(
            case, mcp_client, tool_schemas, api_key="k", model="m", http_client=http_client
        )

    # captured, not raised/crashed
    assert len(trace.tool_calls) == 1
    record = trace.tool_calls[0]
    assert record.error is not None
    assert "404" in record.error
    assert record.result is None

    # the loop continued to a second turn and got the final answer
    assert trace.final_answer == "I could not find a patient with ID NOPE in the system."

    # the error was fed back into the transcript as a tool message, not dropped
    tool_messages = [m for m in trace.raw_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    fed_back = json.loads(tool_messages[0]["content"])
    assert "error" in fed_back


# ---------------------------------------------------------------------------
# run_case — edge case: malformed tool-call arguments
# ---------------------------------------------------------------------------


async def test_run_case_handles_malformed_tool_call_arguments(
    mcp_client: Client, tool_schemas: list[dict], http_client: httpx.AsyncClient
) -> None:
    case = _synthetic_case(case_id="malformed-args")
    malformed_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_current_biomarkers", "arguments": "{not valid json"},
                        }
                    ],
                }
            }
        ]
    }
    responses = [malformed_response, _final_answer_message("Sorry, something went wrong.")]

    with patch("evals.runner._chat_completion", new=AsyncMock(side_effect=responses)):
        trace = await run_case(
            case, mcp_client, tool_schemas, api_key="k", model="m", http_client=http_client
        )

    assert len(trace.tool_calls) == 1
    record = trace.tool_calls[0]
    assert record.error is not None
    assert "could not parse tool arguments" in record.error
    assert record.result is None
    assert record.arguments == {}
    # the loop didn't crash — it continued and got the final turn
    assert trace.final_answer == "Sorry, something went wrong."


# ---------------------------------------------------------------------------
# run_case — edge case: turn cap
# ---------------------------------------------------------------------------


async def test_run_case_stops_at_max_tool_turns_without_hanging(
    mcp_client: Client, tool_schemas: list[dict], http_client: httpx.AsyncClient
) -> None:
    """A model that never returns a final answer must not loop forever."""
    case = _synthetic_case(case_id="never-stops")
    always_tool_call = _tool_call_message([("call_1", "ping", {})])
    mock_chat = AsyncMock(return_value=always_tool_call)

    with patch("evals.runner._chat_completion", new=mock_chat):
        trace = await run_case(
            case, mcp_client, tool_schemas, api_key="k", model="m", http_client=http_client
        )

    assert mock_chat.call_count == MAX_TOOL_TURNS
    assert trace.case_id == case.id
    assert len(trace.tool_calls) == MAX_TOOL_TURNS  # one ping call per turn, capped

    # a reasonable note about hitting the cap shows up somewhere in the transcript
    assert any(
        "MAX_TOOL_TURNS" in m.get("content", "")
        for m in trace.raw_messages
        if isinstance(m.get("content"), str)
    )


# ---------------------------------------------------------------------------
# make_judge_fn
# ---------------------------------------------------------------------------


def test_make_judge_fn_success_returns_model_text_content() -> None:
    judge_fn = make_judge_fn(model="fake/judge-model", api_key="fake-key")
    fake_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "PASS: looks good."}}]},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    with patch("httpx.post", return_value=fake_response) as mock_post:
        result = judge_fn("some judge prompt")

    assert result == "PASS: looks good."
    mock_post.assert_called_once()


def test_make_judge_fn_http_failure_returns_inconclusive_not_a_verdict() -> None:
    judge_fn = make_judge_fn(model="fake/judge-model", api_key="fake-key")

    with patch("httpx.post", side_effect=httpx.ConnectError("connection refused")):
        result = judge_fn("some judge prompt")

    assert not result.upper().startswith("PASS")
    assert not result.upper().startswith("FAIL")
    # confirms _parse_judge_verdict (evals/scoring.py) would treat this as
    # inconclusive/skipped rather than a false PASS or FAIL.
    assert _parse_judge_verdict(result) is None


def test_make_judge_fn_http_status_error_also_yields_inconclusive_verdict() -> None:
    """A non-2xx response (raise_for_status) is the same kind of failure as a
    connection error — also caught, also inconclusive, never crashes."""
    judge_fn = make_judge_fn(model="fake/judge-model", api_key="fake-key")
    bad_response = httpx.Response(
        500,
        text="internal server error",
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    with patch("httpx.post", return_value=bad_response):
        result = judge_fn("some judge prompt")

    assert _parse_judge_verdict(result) is None
