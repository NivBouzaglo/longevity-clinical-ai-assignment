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

import asyncio
import json
from typing import Any, Callable

import httpx
from fastmcp import Client
from fastmcp.exceptions import ToolError

from evals.scoring import Case, Trace, ToolCallRecord

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Caps the tool-calling loop so a model that never stops calling tools can't
# hang the harness forever. Set high enough for the worst case actually in
# this eval set: `multistep-highest-t2dm` may need one get_current_risks call
# per patient across all 8 patients plus a final synthesis turn (9 turns
# minimum) if the model issues one tool_call per turn rather than batching
# several into one response -- observed in practice with
# openai/gpt-oss-20b:free, which does not batch. 6 was too low for this case
# by design (it can't reach P005-P008 before hitting the cap); 10 leaves one
# turn of margin over the 9-turn worst case.
MAX_TOOL_TURNS = 10

# A "degenerate" turn -- neither a tool_call nor any text content -- means the
# model made no decision at all. Observed in practice: its own reasoning can
# show clear intent to continue (e.g. "Now P005") but the structured
# tool_calls output doesn't come through, a transient generation hiccup
# rather than a deliberate stop (a deliberate stop always has SOME final
# text). Retry the same decision point this many times before giving up and
# treating it as a genuine (empty) final answer.
MAX_DEGENERATE_RETRIES = 2

REQUEST_TIMEOUT = 60.0

# Deliberate product/eval design decision. The name->ID directory below was added
# after a real live eval run (openai/gpt-oss-20b:free) showed the model correctly
# refusing to guess a patient_id for name-phrased questions ("What is Maya Cohen's
# eGFR?") rather than hallucinate one -- safe, but it meant most tool calls never
# fired. A real clinic assistant would have this roster from EHR/scheduling data,
# so this is an honest fix, not eval-gaming. Verified live: name resolution now
# works. The second paragraph below (added after a later run) closes a gap that
# introduced: the model started using this list as authoritative for whether a
# patient EXISTS, not just for name lookup, and skipped calling the tool for an
# out-of-range ID (P999) instead of verifying against the live backend -- still
# a correct, non-fabricated answer, but not the tool-verified truth this system
# is supposed to always ground itself in. Change deliberately; don't casually reword.
SYSTEM_PROMPT = """You are a clinical decision-support assistant for doctors at a single clinic. All doctors can see all patients. This clinic's patients are:
  P001 Maya Cohen · P002 David Levi · P003 Sarah Mizrahi · P004 Avraham Friedman
  P005 Yosef Katz · P006 Rivka Shapiro · P007 Noa Bar · P008 Daniel Green
When a doctor refers to a patient by name, look up their ID in this list before calling a tool — never guess an ID and never ask the doctor to supply one you can resolve here.

This list is for name lookup only, not a verified registry. If a doctor asks about a patient ID that doesn't appear here, still call the tool to check with the system before concluding the patient doesn't exist — only report "not found" based on what the tool actually returns, never based on this list alone.

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


MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5.0


async def _chat_completion(
    http_client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """POST one chat-completion request, retrying transient failures.

    Free-tier OpenRouter models sit behind a shared upstream provider pool
    that occasionally 429s with a short `Retry-After` (seconds, not the
    account's daily-quota 429 — that one doesn't recover on its own). Retry
    those (and any response missing the expected `choices` shape, which we've
    seen from the same flaky pool) with backoff; anything else propagates
    immediately so the caller's per-case error handling can record it.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await http_client.post(
                OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data:
                raise RuntimeError(f"OpenRouter response missing 'choices': {data!r}")
            return data
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code != 429 or attempt == MAX_RETRIES - 1:
                raise
            delay = float(exc.response.headers.get("Retry-After", DEFAULT_RETRY_DELAY))
        except RuntimeError as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                raise
            delay = DEFAULT_RETRY_DELAY
        await asyncio.sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises above


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
    degenerate_retries = 0

    turn = 0
    while turn < MAX_TOOL_TURNS:
        response = await _chat_completion(http_client, api_key, model, messages, tool_schemas)
        message = response["choices"][0]["message"]
        requested_calls = message.get("tool_calls")
        content = message.get("content")

        if not requested_calls and not content and degenerate_retries < MAX_DEGENERATE_RETRIES:
            # Neither a tool call nor any text — the model made no decision.
            # A blind identical retry doesn't reliably help here: observed in
            # practice, three attempts at the same decision point returned
            # byte-identical reasoning each time ("Now P005." with no actual
            # tool_calls). Nudge with a corrective message instead, so the
            # next attempt has something new to react to rather than the
            # exact same prompt that just failed. Don't record the
            # non-message itself, don't spend a turn of the budget on it.
            degenerate_retries += 1
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You did not make a tool call or give a final answer. If you "
                        "intended to check another patient, call the appropriate tool "
                        "now with their patient_id. Otherwise, give your final answer."
                    ),
                }
            )
            continue

        messages.append(message)
        turn += 1

        if not requested_calls:
            return Trace(
                case_id=case.id, tool_calls=tool_calls, final_answer=content or "", raw_messages=messages
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
            result_payload = record.result if record.error is None else {"error": record.error}
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result_payload, default=str)}
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
