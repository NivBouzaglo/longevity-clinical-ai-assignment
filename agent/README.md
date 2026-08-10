# Custom Agent (BONUS)

The primary chat agent for this assignment is **LibreChat's built-in agent** — it
already calls your MCP tools with an OpenRouter model, and that path is what the
core exercise grades. You do **not** need to build your own agent.

Build one here only if you want to show depth on problems the built-in agent
can't express well. Good reasons to reach for a custom LangGraph / LangChain / ADK
agent include:

- **Complex deterministic orchestration** or branching state machines beyond
  simple chain / sub-agent composition.
- **Long-running or autonomous background workflows** that run without a user
  sitting in the chat.
- **Framework-specific features** — durable, checkpointed state; custom memory
  backends; human-approval gates mid-graph; purpose-built evaluation harnesses.
- **Heavy custom business logic** you'd rather own in your own codebase and test
  independently of the UI.

## Reuse, don't rebuild
Point the agent at the **same MCP server** you already built — don't duplicate the
tool logic. `langchain-mcp-adapters` (installed via `uv sync --extra agent`) loads
your MCP tools as LangChain/LangGraph tools; use an OpenRouter model through
`langchain-openai` with `base_url="https://openrouter.ai/api/v1"`.

## A concrete idea
A **patient risk-review workflow**: fetch biomarkers → compute risks → if any risk
is `high`, retrieve the matching guideline snippet and draft a short clinician
note → pause at a **human-approval gate** → on approval, persist/emit the note.
Add a checkpointer so an interrupted run resumes, and a small eval over a few
patients. This exercises durable state, a branch, a gate, and grounded retrieval —
the things the built-in agent can't do.

## Deliver
- Code here (e.g. `agent/graph.py`) plus a short note on **why** a custom agent was
  warranted and what it buys over the built-in one.
- A way to run it (`uv run python agent/graph.py ...` or a tiny CLI) and, ideally,
  an eval you can point at it.
