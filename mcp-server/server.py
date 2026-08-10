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

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

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


# ---------------------------------------------------------------------------
# TODO — implement the two real tools by wrapping the backend endpoints.
# A tool's name + docstring + typed args are what the model reads to decide when
# and how to call it, so make them clear. Handle backend errors gracefully
# (unknown patient, backend/model server down) and return useful messages.
#
# Sketch (uncomment and finish):
#
# import httpx
#
# @mcp.tool
# async def get_current_biomarkers(patient_id: str) -> dict:
#     """Return the latest biomarker snapshot (labs + vitals) for a patient, e.g. 'P001'."""
#     async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=15) as http:
#         resp = await http.get("/api/v1/get_current_biomarkers", params={"patient_id": patient_id})
#         resp.raise_for_status()
#         return resp.json()
#
# @mcp.tool
# async def get_current_risks(patient_id: str) -> dict:
#     """Compute and return the patient's five clinical risks (with trend), e.g. 'P004'."""
#     ...
#
# BONUS — a retrieval tool so answers can cite guideline text:
# @mcp.tool
# async def search_guidelines(query: str, k: int = 3) -> list[dict]:
#     """Search the clinical guideline snippets (data/guidelines/) for grounding text."""
#     ...
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run(transport="http", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
