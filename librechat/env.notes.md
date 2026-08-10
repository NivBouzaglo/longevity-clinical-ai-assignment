# LibreChat `.env` notes

LibreChat is configured through **its own `.env`** (created from LibreChat's
`.env.example`), separate from this repo's `.env`. A few values matter for this
assignment.

## Secrets LibreChat needs (already present as examples in its `.env.example`)
| Variable | What it is |
|---|---|
| `CREDS_KEY` | 64-hex-char AES key used to encrypt/decrypt stored credentials |
| `CREDS_IV` | 32-hex-char AES initialization vector |
| `JWT_SECRET` | signs login-session JWTs |
| `JWT_REFRESH_SECRET` | signs refresh JWTs |

The committed example values work for local dev, but generate your own with
LibreChat's credentials generator: <https://www.librechat.ai/toolkit/creds_generator>.

## Values YOU must set for this assignment
Add/edit these in LibreChat's `.env`:

| Variable | In stock `.env.example`? | Set to |
|---|---|---|
| `OPENROUTER_KEY` | present but **commented out** — uncomment it | the OpenRouter API key you were given |
| `MCP_BEARER_TOKEN` | **not present — add a new line** | **the same token as this repo's `.env`** — `librechat.yaml` sends it as `Authorization: Bearer` to your MCP server, so they must match |

`MCP_BEARER_TOKEN` is a **custom variable** — LibreChat has no idea what it is, and you
won't find it in `.env.example`. That's fine: `deploy-compose.yml` loads the whole
`.env` into the container (`env_file: .env`), and `librechat.yaml` interpolates any
`${VAR}` from that environment. So adding your own `MCP_BEARER_TOKEN=...` line is all it
takes for `${MCP_BEARER_TOKEN}` in the config to resolve.

```bash
# in LibreChat's .env
OPENROUTER_KEY=sk-or-...            # uncomment + set
MCP_BEARER_TOKEN=dev-longevity-token-change-me   # add this line; match the assignment repo's .env
```

> `librechat.yaml` resolves `${OPENROUTER_KEY}` and `${MCP_BEARER_TOKEN}` from
> LibreChat's `.env` at container start. If the MCP server rejects the connection
> with 401, a `MCP_BEARER_TOKEN` mismatch (or a missing line) is the first thing to check.
