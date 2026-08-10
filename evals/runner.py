"""Live eval runner — drives ONE case against a real OpenRouter model + the real MCP server.

This is the network-touching half of the eval harness (`evals/scoring.py` is the
network-free half — see its module docstring). This module:

1. Discovers the MCP server's tools (`fastmcp.Client.list_tools()`) and converts
   each into an OpenAI-compatible `tools=[...]` schema.
2. Runs the OpenAI-style tool-calling loop against OpenRouter: send messages +
   tools, and whenever the model asks for a tool call, actually call the real
   MCP tool and feed the result back, until the model returns a final text
   answer (or a turn cap is hit).
3. Returns a `Trace` (imported from `evals.scoring`, not redefined here) that
   `evals/scoring.py` can grade without ever touching the network itself.

Also provides `make_judge_fn`, a simple single-turn OpenRouter call matching the
`judge_fn` contract documented in `evals/scoring.py` (see the module docstring
above `_build_judge_prompt` there).

Run standalone smoke-tests via `evals/harness.py`; this module has no `__main__`
of its own.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
from fastmcp import Client
from fastmcp.exceptions import ToolError

from evals.scoring import Case, Trace, ToolCallRecord

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Caps the tool-calling loop so a model that never stops calling tools can't
# hang the harness forever. 6 turns is generous for these cases: even the
# hardest one (`multistep-highest-t2dm`, which must call get_current_risks once
# per patient across 8 patients) plausibly needs more, but a well-behaved model
# batches multiple tool_calls into a single turn's response rather than issuing
# one per turn, so 6 round-trips is a reasonable ceiling before treating it as
# a runaway loop rather than legitimate work.
MAX_TOOL_TURNS = 6

REQUEST_TIMEOUT = 60.0

# Deliberate product/eval design decision. The name->ID directory below was added
# after a real live eval run (openai/gpt-oss-20b:free) showed the model correctly
# refusing to guess a patient_id for name-phrased questions ("What is Maya Cohen's
# eGFR?") rather than hallucinate one -- safe, but it meant most tool calls never
# fired. A real clinic assistant would have this roster from EHR/scheduling data,
# so this is an honest fix, not eval-gaming. Verified live: name resolution now
# works. Change deliberately; don't casually reword.
SYSTEM_PROMPT = """You are a clinical decision-support assistant for doctors at a single clinic. All doctors can see all patients. This clinic's patients are:
  P001 Maya Cohen · P002 David Levi · P003 Sarah Mizrahi · P004 Avraham Friedman
  P005 Yosef Katz · P006 Rivka Shapiro · P007 Noa Bar · P008 Daniel Green
When a doctor refers to a patient by name, look up their ID in this list before calling a tool — never guess an ID and never ask the doctor to supply one you can resolve here.

You have access to tools that fetch a patient's current biomarkers and compute their five clinical disease risks (CVD, T2DM, CKD, CLD, DEMENTIA) in real time. Always call the appropriate tool to get real data before answering — never state a specific lab value, risk probability, or risk band from memory or estimation.

If a tool reports that a patient is not found, say so plainly. Do not invent or guess data for unknown patients.

You provide decision support, not a diagnosis or a prescription. When asked about starting/changing a medication or treatment, note that this is a clinical decision for the doctor to make and defer to their judgement — do not issue definitive prescribing instructions as if authoritative.

When comparing risks across multiple patients, check each patient individually using the available tools (patient IDs P001 through P008)."""


# ---------------------------------------------------------------------------
# MCP tool discovery -> OpenAI tool-calling schema
# ---------------------------------------------------------------------------


def tool_to_openai_schema(tool: Any) -> dict:
    """Convert one `mcp.types.Tool` (from `Client.list_tools()`) to OpenAI's tool-calling shape.

    `Tool` exposes `.name`, `.description`, and `.inputSchema` (already a JSON
    schema dict — confirmed empirically against the running server: e.g.
    `{"type": "object", "properties": {"patient_id": {"type": "string"}},
    "required": ["patient_id"]}`), which maps directly onto `function.parameters`.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def discover_tool_schemas(mcp_client: Client) -> list[dict]:
    """List the MCP server's tools and convert all of them to OpenAI tool schemas."""
    tools = await mcp_client.list_tools()
    return [tool_to_openai_schema(t) for t in tools]


# ---------------------------------------------------------------------------
# OpenRouter HTTP calls
# ---------------------------------------------------------------------------


async def _chat_completion(
    http_client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    resp = await http_client.post(
        OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# The tool-calling loop
# ---------------------------------------------------------------------------


async def _execute_tool_call(mcp_client: Client, name: str, arguments: dict) -> ToolCallRecord:
    """Call one real MCP tool and wrap the outcome as a `ToolCallRecord`.

    Any failure — a `ToolError` from the tool itself (e.g. unknown patient) or
    anything else (e.g. the model hallucinated a tool name) — is captured as
    `error`, never raised, so one bad tool call doesn't take down the whole run.
    """
    try:
        result = await mcp_client.call_tool(name, arguments)
        return ToolCallRecord(name=name, arguments=arguments, result=result.data, error=None)
    except ToolError as exc:
        return ToolCallRecord(name=name, arguments=arguments, result=None, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - any other failure is still a recorded tool-call failure, not a crash
        return ToolCallRecord(name=name, arguments=arguments, result=None, error=str(exc))


async def run_case(
    case: Case,
    mcp_client: Client,
    tool_schemas: list[dict],
    *,
    api_key: str,
    model: str,
    http_client: httpx.AsyncClient,
) -> Trace:
    """Drive one `Case` through the OpenRouter model + real MCP tools, producing a `Trace`.

    ``http_client`` is passed in (rather than created here) so callers running
    the whole suite can reuse one connection across every case instead of
    paying a new TCP/TLS handshake per case.
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": case.question},
    ]
    tool_calls: list[ToolCallRecord] = []

    for _turn in range(MAX_TOOL_TURNS):
        response = await _chat_completion(http_client, api_key, model, messages, tool_schemas)
        message = response["choices"][0]["message"]
        messages.append(message)

        requested_calls = message.get("tool_calls")
        if not requested_calls:
            final_answer = message.get("content") or ""
            return Trace(
                case_id=case.id, tool_calls=tool_calls, final_answer=final_answer, raw_messages=messages
            )

        for tc in requested_calls:
            fn = tc["function"]
            name = fn["name"]
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                record = ToolCallRecord(
                    name=name, arguments={}, result=None, error=f"could not parse tool arguments: {exc}"
                )
            else:
                record = await _execute_tool_call(mcp_client, name, arguments)

            tool_calls.append(record)
            content = record.result if record.error is None else {"error": record.error}
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(content, default=str)}
            )

    # Turn cap hit without a final text answer — surface whatever assistant
    # text exists (usually none, since every turn above ended in tool_calls)
    # and note the cap was hit so a report reader knows this wasn't a clean stop.
    final_answer = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            final_answer = m["content"]
            break
    messages.append(
        {
            "role": "system",
            "content": f"[eval harness note: hit MAX_TOOL_TURNS={MAX_TOOL_TURNS} without a final answer]",
        }
    )
    return Trace(case_id=case.id, tool_calls=tool_calls, final_answer=final_answer, raw_messages=messages)


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------


def make_judge_fn(model: str, api_key: str) -> Callable[[str], str]:
    """Build a `judge_fn` matching `evals/scoring.py`'s documented contract.

    A plain single-turn OpenRouter chat completion, no tools involved. Uses a
    sync `httpx.Client` (one per call) since the contract is a plain
    `str -> str` function — `score_suite`/`score_case` call it synchronously,
    not from a coroutine.

    Network/HTTP failures are caught and turned into a response that doesn't
    start with PASS/FAIL, so `_parse_judge_verdict` treats it as an
    inconclusive verdict (skipped) rather than forcing a fail baked into the
    pass rate.
    """

    def judge_fn(prompt: str) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        try:
            resp = httpx.post(
                OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001 - surfaced as an unparseable verdict, not a crash
            return f"ERROR: judge call failed: {exc}"

    return judge_fn
