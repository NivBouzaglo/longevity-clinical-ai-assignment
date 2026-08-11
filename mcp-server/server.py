"""FastMCP server — exposes the clinical backend to the assistant as MCP tools.

This skeleton BOOTS as-is: static bearer-token auth over streamable HTTP, plus one
demo tool (`ping`) so you can confirm auth + transport work end to end. Your job is
to add the two real tools (and, for bonus, a retrieval tool).

Run (from the repo root, after `uv sync`):
    uv run python mcp-server/server.py

It then listens on:  http://0.0.0.0:9000/mcp/     (note the trailing slash)
Clients must send:   Authorization: Bearer <MCP_BEARER_TOKEN>   (see repo-root .env)

Why 0.0.0.0 and port 9000: LibreChat runs in Docker and reaches this server on the
host via host.docker.internal:9000 — binding 127.0.0.1 would be unreachable from the
container. See the root GUIDE.md for the full networking + LibreChat wiring.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# `chromadb` (and therefore `search_guidelines`) is an optional extra
# (`uv sync --extra rag`) — this stays light for anyone who only needs the two
# required tools. Guard the import so a missing `chromadb` never stops the
# server from booting; we just skip registering `search_guidelines` below.
try:
    from guidelines_retrieval import search as search_guidelines_impl
except ImportError:
    search_guidelines_impl = None
    print("search_guidelines not available — install with `uv sync --extra rag`")

MCP_BEARER_TOKEN = os.getenv("MCP_BEARER_TOKEN", "dev-longevity-token-change-me")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")
HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "9000"))

# Static token auth — fine for a dev/take-home, NOT for production (tokens are plain
# text). Any request must present `Authorization: Bearer <MCP_BEARER_TOKEN>`.
verifier = StaticTokenVerifier(
    tokens={MCP_BEARER_TOKEN: {"client_id": "clinic", "scopes": ["read"]}},
    required_scopes=["read"],
)

mcp = FastMCP(name="Longevity Clinical MCP", auth=verifier)


@mcp.tool
def ping() -> dict:
    """Connectivity check: confirms the MCP server is reachable and authorized."""
    return {"ok": True, "backend_url": BACKEND_URL}


def _backend_error_message(exc: httpx.HTTPStatusError) -> str:
    """Turn a backend HTTP error response into a clear, LLM-readable message."""
    try:
        detail = exc.response.json().get("detail", exc.response.text)
    except ValueError:
        detail = exc.response.text
    return f"Backend returned {exc.response.status_code}: {detail}"


@mcp.tool
async def get_current_biomarkers(patient_id: str) -> dict:
    """Return the latest biomarker snapshot (labs + vitals) for a patient.

    Call this when you need a patient's current raw lab/vital values (e.g. eGFR,
    HbA1c, lipids, blood pressure) rather than a computed risk assessment.
    Example patient_id: 'P001'.
    """
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=15) as http:
        try:
            resp = await http.get(
                "/api/v1/get_current_biomarkers", params={"patient_id": patient_id}
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ToolError(_backend_error_message(exc)) from exc
        except httpx.RequestError as exc:
            raise ToolError(
                f"Could not reach the clinical backend at {BACKEND_URL}: {exc}"
            ) from exc
        return resp.json()


@mcp.tool
async def get_current_risks(patient_id: str) -> dict:
    """Compute and return the patient's five clinical risks, each with a trend.

    Call this when you need a patient's risk assessment (e.g. CKD, cardiovascular,
    diabetes risk) rather than raw biomarker values. This invokes an ML model server
    and may take a bit longer than fetching biomarkers. Example patient_id: 'P004'.
    """
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=15) as http:
        try:
            resp = await http.get(
                "/api/v1/get_current_risks", params={"patient_id": patient_id}
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ToolError(_backend_error_message(exc)) from exc
        except httpx.RequestError as exc:
            raise ToolError(
                f"Could not reach the clinical backend at {BACKEND_URL}: {exc}"
            ) from exc
        return resp.json()


# ---------------------------------------------------------------------------
# BONUS — a retrieval tool so answers can cite guideline text. Only registered
# if `chromadb` (the `rag` extra) was importable above.
# ---------------------------------------------------------------------------
if search_guidelines_impl is not None:

    @mcp.tool
    async def search_guidelines(query: str, k: int = 3) -> list[dict]:
        """Search the clinic's guideline corpus for snippets relevant to a query.

        Retrieves the top-k most relevant snippets from short educational summaries
        of the five risk models (CVD, T2DM, CKD, CLD, Dementia) — what each one
        estimates, which factors drive it, and which levers are modifiable. Use this
        to explain *why* a risk model flags something (e.g. "what drives CKD risk"
        or "how is dementia risk estimated") — it pairs naturally with
        `get_current_risks`, letting you ground an explanation in retrievable,
        citable text rather than just asserting it. Each result includes the source
        filename (e.g. 'ckd_framingham.md') so the explanation can be cited.
        Example query: 'what factors lower cardiovascular risk'.
        """
        # Synchronous chromadb call — fine to call directly given how small/fast
        # this corpus is, no need to spawn a thread.
        return search_guidelines_impl(query, k)


def main() -> None:
    mcp.run(transport="http", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
