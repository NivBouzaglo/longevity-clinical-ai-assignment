# Evaluation Harness

"It looked right once" is not done. This is where you show an **evaluation
mindset**: a small, deterministic harness that scores the assistant's answers so
every prompt/model/tool change can be measured. Building the runner is your task;
we provide the gold cases in [`cases.jsonl`](cases.jsonl).

## What to measure (three axes)
1. **Tool-call correctness** — did the assistant call the expected MCP tool with
   the right `patient_id`? (Wrong or missing tool call = fail, even if the prose
   sounds plausible.)
2. **Numeric / band faithfulness** — do the numbers and risk bands in the answer
   match the tool output? Exact for biomarker values (e.g. eGFR = 102); band +
   tolerance for risk probabilities. **This is the top clinical-safety metric** —
   a fabricated lab value or risk is the worst failure mode.
3. **Safety** — no invented data for an unknown patient; no autonomous prescribing;
   appropriate "decision support, not a diagnosis" framing.

## Run it against the model + MCP, NOT the LibreChat UI
Driving the browser UI is slow and flaky. Point the harness directly at an
OpenRouter chat model with the MCP tools attached (an OpenAI-compatible
`tools=[...]` loop, or a small client that connects to your MCP server). You get
the tool-call trace and the final text — everything you need to score all three
axes deterministically. (If you build the bonus agent, you can eval it directly.)

## Case schema (`cases.jsonl`, one JSON object per line)
| Field | Meaning |
|---|---|
| `id`, `category` | case id; one of `tool_selection` \| `numeric_faithfulness` \| `trend` \| `safety` \| `citation` \| `multi_step` |
| `question` | the doctor's message |
| `expected_tool` | tool that should be called (`any`/`none` where not applicable) |
| `patient_id` | expected patient argument |
| `expected_facts[]` | typed checks — `biomarker` (exact value+unit), `risk` (band + `approx_probability`/`tolerance`), `trend` (direction), `safety`/`no_fabrication`/`citation` (behavioural) |
| `notes` | rationale |

## Suggested scoring
- Deterministic checks for tool selection, biomarker values, and risk bands
  (parse the tool trace + regex/round the spoken numbers).
- An **LLM-as-judge** (a cheap OpenRouter model) for the behavioural `safety` and
  `citation` cases, with a rubric.
- Emit a **pass rate per category** and a list of failures. Make it re-runnable
  (`uv run python evals/harness.py`) and keep it in version control so it acts as
  a regression check.

## Bar for "strong"
Real metrics (not vibes), numeric-faithfulness weighted highest, reproducible
output, and at least one case you added yourself that caught a real regression.
