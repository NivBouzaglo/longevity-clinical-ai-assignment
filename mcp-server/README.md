# MCP Server (FastMCP)

Exposes the clinical backend to the assistant as **MCP tools**, protected by a
static bearer token, over streamable HTTP. `server.py` boots as-is with one demo
tool (`ping`); you add the real tools.

## What to build
1. **`get_current_biomarkers(patient_id)`** — wraps `GET /api/v1/get_current_biomarkers`. ✅ built
2. **`get_current_risks(patient_id)`** — wraps `GET /api/v1/get_current_risks`. ✅ built
3. **Bonus — `search_guidelines(query, k)`** — retrieval over `data/guidelines/`
   (embeddings + a small vector store) so the assistant can cite guideline text.
   Install extras with `uv sync --extra rag`. See [`GUIDE.md`](../GUIDE.md). ✅ built
   — see `guidelines_retrieval.py`. Registers only if `chromadb` is importable
   (guarded `try/except`), so the two required tools still work fine without
   the `rag` extra installed.

A tool's **name, docstring, and typed arguments are the model's only clue** about
when and how to call it — write them well. Validate inputs and return clear errors
when the backend is down or a patient is unknown.

## Run
```bash
uv sync                              # from repo root (installs fastmcp, httpx, python-dotenv)
uv run python mcp-server/server.py   # listens on http://0.0.0.0:9000/mcp/
```
- Binds **0.0.0.0:9000** on purpose — LibreChat (in Docker) reaches it at
  `http://host.docker.internal:9000/mcp/`. The trailing slash matters for
  `fastmcp.Client` (below) — but for LibreChat specifically, drop it: `/mcp/`
  307-redirects to `/mcp`, and LibreChat's MCP client doesn't resend the
  `Authorization` header across that redirect, which silently breaks auth. See
  `librechat/SETUP.md` trap 6 for the full story (also needs `requiresOAuth: false`).
- Auth: every request needs `Authorization: Bearer <MCP_BEARER_TOKEN>` (repo-root `.env`).

## Smoke test (local client)
```python
import asyncio
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

async def main():
    async with Client("http://127.0.0.1:9000/mcp/", auth=BearerAuth("<MCP_BEARER_TOKEN>")) as c:
        print("tools:", [t.name for t in await c.list_tools()])
        print("ping:", await c.call_tool("ping"))

asyncio.run(main())
```
Wiring this into LibreChat (the `mcpServers` block **and** the SSRF
`allowedAddresses` allowlist) is covered in [`GUIDE.md`](../GUIDE.md) — the
allowlist is the #1 reason a "connected" MCP server never actually fires.

## Notes
- `StaticTokenVerifier` lives in `fastmcp.server.auth.providers.jwt` (yes, `jwt`,
  even though the token is static) — an easy import to get wrong.
- Static tokens are for dev only; they are stored in plain text.
