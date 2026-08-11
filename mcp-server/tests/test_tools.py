"""Integration tests for the MCP server's clinical tools.

Run:  uv run pytest mcp-server/tests -v
Note: like backend/tests/test_endpoints.py, the get_current_risks tests hit the
live MLflow model server (started separately — see GUIDE.md) and depend on the
``reset_db`` fixture to keep ``data/patient_db.db`` pristine across runs. These
tests additionally require the MCP server itself to be running and reachable at
http://127.0.0.1:9000/mcp/ (see mcp-server/README.md's "Smoke test" section),
since fastmcp.Client talks to it over real streamable-HTTP, not an in-process
transport.
"""

from __future__ import annotations

import pytest
from fastmcp import Client
from fastmcp.client.auth import BearerAuth
from fastmcp.exceptions import ToolError
from httpx import HTTPStatusError

from .conftest import MCP_URL

RISK_CODES = {"CVD", "T2DM", "CKD", "CLD", "DEMENTIA"}
BANDS = {"low", "borderline", "intermediate", "high"}


# ---------------------------------------------------------------------------
# Happy flow
# ---------------------------------------------------------------------------


async def test_list_tools_includes_ping_and_clinical_tools(client: Client) -> None:
    tools = await client.list_tools()
    names = {t.name for t in tools}
    assert {"ping", "get_current_biomarkers", "get_current_risks"} <= names


async def test_list_tools_includes_search_guidelines(client: Client) -> None:
    """search_guidelines is registered alongside the other three tools (the
    `chromadb` extra is installed in this venv, so server.py's optional-import
    guard should have wired it up).
    """
    tools = await client.list_tools()
    names = {t.name for t in tools}
    assert {"ping", "get_current_biomarkers", "get_current_risks", "search_guidelines"} <= names


async def test_biomarkers_known_patient(client: Client) -> None:
    result = await client.call_tool("get_current_biomarkers", {"patient_id": "P001"})
    body = result.data
    assert body["patient_id"] == "P001"
    # Maya Cohen's eGFR in the mock data is 102 (ground truth, same as backend tests).
    assert body["biomarkers"]["egfr_ml_min_1_73m2"] == 102


async def test_risks_known_patient_ckd_high(client: Client, reset_db) -> None:
    result = await client.call_tool("get_current_risks", {"patient_id": "P004"})
    body = result.data
    risks = body["risks"]
    assert {x["risk_code"] for x in risks} == RISK_CODES
    for x in risks:
        assert 0.0 <= x["probability"] <= 1.0
        assert x["risk_band"] in BANDS
    # P004 (Avraham Friedman) is the designed CKD patient — should read high.
    ckd = next(x for x in risks if x["risk_code"] == "CKD")
    assert ckd["risk_band"] == "high"


# ---------------------------------------------------------------------------
# Regular workflow — a second known patient, to guard against hardcoded values.
# ---------------------------------------------------------------------------


async def test_biomarkers_second_known_patient(client: Client) -> None:
    """A different patient (P002, David Levi) returns his own data, not P001's."""
    result = await client.call_tool("get_current_biomarkers", {"patient_id": "P002"})
    body = result.data
    assert body["patient_id"] == "P002"
    assert body["name"] == "David Levi"
    assert body["age_years"] == 68
    bio = body["biomarkers"]
    assert bio["egfr_ml_min_1_73m2"] == 74
    assert bio["urine_dipstick_protein"] == "trace"


async def test_risks_second_known_patient(client: Client, reset_db) -> None:
    """A different patient (P002) still returns 5 risk codes with his own name."""
    result = await client.call_tool("get_current_risks", {"patient_id": "P002"})
    body = result.data
    assert body["patient_id"] == "P002"
    assert body["name"] == "David Levi"
    assert {x["risk_code"] for x in body["risks"]} == RISK_CODES


async def test_search_guidelines_known_query_returns_expected_source(client: Client) -> None:
    """A real query against the live server surfaces the expected guideline
    file, mirroring test_biomarkers_known_patient's style of asserting on
    real returned data rather than just checking the call didn't crash.
    """
    result = await client.call_tool(
        "search_guidelines", {"query": "How is chronic kidney disease risk estimated?"}
    )
    results = result.data
    assert results
    sources = {r["source"] for r in results}
    assert "ckd_framingham.md" in sources
    for r in results:
        assert set(r.keys()) == {"text", "source", "section", "distance"}


async def test_search_guidelines_k_parameter_is_respected(client: Client) -> None:
    result = await client.call_tool(
        "search_guidelines", {"query": "cardiovascular risk factors", "k": 2}
    )
    assert len(result.data) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_biomarkers_unknown_patient_raises_tool_error(client: Client) -> None:
    with pytest.raises(ToolError) as exc_info:
        await client.call_tool("get_current_biomarkers", {"patient_id": "NOPE"})
    message = str(exc_info.value)
    assert "404" in message
    assert "NOPE" in message


async def test_risks_unknown_patient_raises_tool_error(client: Client) -> None:
    with pytest.raises(ToolError) as exc_info:
        await client.call_tool("get_current_risks", {"patient_id": "NOPE"})
    message = str(exc_info.value)
    assert "404" in message
    assert "NOPE" in message


async def test_biomarkers_missing_patient_id_raises_tool_error(client: Client) -> None:
    """Omitting the required patient_id argument is rejected by FastMCP's own
    schema validation before the tool function ever runs — surfaced to the
    client as a ToolError, not a raw crash or a silent empty response.
    """
    with pytest.raises(ToolError) as exc_info:
        await client.call_tool("get_current_biomarkers", {})
    assert "patient_id" in str(exc_info.value)


async def test_biomarkers_empty_patient_id_raises_tool_error(client: Client) -> None:
    """An empty-but-present patient_id passes argument validation (it's a valid
    str) but doesn't match any patient, so it surfaces the backend's 404.
    """
    with pytest.raises(ToolError) as exc_info:
        await client.call_tool("get_current_biomarkers", {"patient_id": ""})
    assert "404" in str(exc_info.value)


async def test_connect_with_wrong_bearer_token_is_rejected() -> None:
    with pytest.raises(HTTPStatusError) as exc_info:
        async with Client(MCP_URL, auth=BearerAuth("this-is-not-the-token")) as c:
            await c.call_tool("ping", {})
    assert exc_info.value.response.status_code == 401


async def test_connect_without_auth_is_rejected() -> None:
    with pytest.raises(HTTPStatusError) as exc_info:
        async with Client(MCP_URL) as c:
            await c.call_tool("ping", {})
    assert exc_info.value.response.status_code == 401
