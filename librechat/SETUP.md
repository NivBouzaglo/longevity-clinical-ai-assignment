# LibreChat Setup

LibreChat is the doctor-facing chat UI. You run it from its **own** repository
(you don't vendor it here) and point its built-in agent at your MCP server.

## 0. Prerequisites
Your three host services should be running first (see the root [`GUIDE.md`](../GUIDE.md)):
- FastAPI backend on `:8001`
- MLflow model server on `:5001`
- FastMCP server on `0.0.0.0:9000`

## 1. Clone LibreChat at a PINNED release
Do **not** use `main` — its config schema drifts. Pick a specific recent release
from <https://github.com/danny-avila/LibreChat/releases> that supports
`mcpSettings.allowedAddresses` and `type: streamable-http` MCP servers, and record
the tag in your submission.
```bash
git clone --branch <RELEASE_TAG> --depth 1 https://github.com/danny-avila/LibreChat.git
cd LibreChat
```

## 2. Environment
There are **two** `.env` files, and they are different:
- the **assignment repo's** `.env` (for the backend + MCP server), and
- **LibreChat's own** `.env`, which you create now — you're inside the LibreChat
  checkout from step 1.

```bash
# run this INSIDE the LibreChat checkout — uses LibreChat's .env.example
cp .env.example .env
```
In **LibreChat's** `.env` set:
- `OPENROUTER_KEY` — your OpenRouter key. It lives **only** here (LibreChat is the
  thing that talks to OpenRouter; the assignment repo never needs it).
- `MCP_BEARER_TOKEN` — must be the **same value** as the assignment repo's `.env`, so
  LibreChat can authenticate to your MCP server.

See [`env.notes.md`](env.notes.md) for the full list.

## 3. Config — wire your MCP server into `librechat.yaml` (your job)
Read these first — they explain everything below:
- **librechat.yaml configuration guide** (endpoints, OpenRouter, mounting the file):
  <https://www.librechat.ai/docs/configuration/librechat_yaml>
- **MCP servers in LibreChat** (how to integrate an MCP server):
  <https://www.librechat.ai/docs/mcp_servers>

Start from the config LibreChat ships with:
```bash
# inside the LibreChat checkout
cp librechat.example.yaml librechat.yaml
```
Then edit `librechat.yaml`:
- **Enable OpenRouter** — there's a ready `- name: 'OpenRouter'` block in that same
  file (search for `OpenRouter`). It already uses `${OPENROUTER_KEY}`. (Custom-endpoint
  fields are documented in the configuration guide above.)
- **Add your MCP server** — follow the **"Example MCP Servers Object Structure"**
  section in that same file, and the **MCP Server domain restrictions**
  (`mcpSettings`) section just above it (and the MCP docs above). Send your token via
  an `Authorization: Bearer ${MCP_BEARER_TOKEN}` header.

The comments in the `mcpSettings` section of `librechat.example.yaml` explain the
networking constraints you need to satisfy — read them closely before you fill in the
URL and the allowlist.

## 4. Run
`deploy-compose.yml` mounts `./librechat.yaml` and reads `./.env`.
```bash
docker compose -f deploy-compose.yml up -d
# If MongoDB crash-loops, add the optional override (see docker-compose.override.yml):
# docker compose -f deploy-compose.yml -f <this-repo>/librechat/docker-compose.override.yml up -d
```
Open <http://localhost:3080>, register a local account, pick the **OpenRouter**
endpoint + a tool-capable model, enable your MCP server's tools, and ask something
like *"What are Avraham Friedman's (P004) current risks, and how has his kidney risk
trended?"*

## 5. If the MCP tools never fire
The most common failure: LibreChat lists the server as "configured" but never calls a
tool. Debug in this order:
1. **Allowlist** — LibreChat blocks host/private addresses by default. Did you allow
   your server under `mcpSettings`? (Re-read that section's comments in
   `librechat.example.yaml`.)
2. **URL** — right host (a container can't reach a host service at `localhost`), right
   port, right path (streamable-HTTP is picky about the trailing slash).
3. **Bind** — is your MCP server reachable from outside its own process, or only from
   `127.0.0.1`?
4. **Reachability from inside the container:**
   ```bash
   docker compose -f deploy-compose.yml exec api wget -qO- <your-mcp-url> ; echo
   ```
   A protocol/HTTP error means the network path works; *connection refused* means the
   host/port/bind is wrong.
5. **Model** — does the selected OpenRouter model support tool/function calling?

References:
- MCP servers in LibreChat: <https://www.librechat.ai/docs/mcp_servers>
- librechat.yaml configuration: <https://www.librechat.ai/docs/configuration/librechat_yaml>
