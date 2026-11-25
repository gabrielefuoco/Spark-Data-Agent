# The Agent

A bounded tool-calling loop in LangGraph. The model decides *what* to ask the
data; the graph decides what it is allowed to receive and what it may claim.

## Topology

```
START ──► agent ──(tool calls?)──► execute_tools ──┐
            │                            │         │
            │                            └─────────┘  (loop, bounded)
            │
            └──(no tool calls)──► scientific_critic ──► END
                                        │
                                        └──► handle_failure (budget, repeated calls)
```

- `agent` — the model with two tools bound. Sees: system prompt, the manifest
  digest, the conversation window, and every tool result so far.
- `execute_tools` — validates, prices, gates, executes, and judges each call.
- `scientific_critic` — a second model call that writes the interpretation.
- `handle_failure` — a transparent dead end, never a silent empty answer.

## The tools

| tool | what it guarantees |
|---|---|
| `run_safe_sql(query)` | goes through `execute_safe_spark_sql`: sqlglot AST validation, table allowlist, `LIMIT` cap. The model has no Spark access, only this door. |
| `run_spark_job(job, params)` | `circadian_features` or `topic_model`, priced before running. |

There is deliberately **no** manifest tool. The catalog numbers are 98 tokens and
never change, so they live in the system prompt: a tool call for them was a
10-15 second round-trip on every question, including "hi, what can you do?".

## Where the human enters

Before executing a tool call whose estimated cost exceeds
`HITL_COST_THRESHOLD`, the loop calls LangGraph's `interrupt()` with the query,
the estimate, the threshold and the reasons. The UI renders a card with
**Autorizza** / **Annulla** and resumes the same `thread_id` with
`Command(resume={"approved": ...})`.

A refusal is not an error: it is returned to the model as the tool's result, and
the model has to work with what it has. Approval executes exactly once — a test
proves both halves.

Cost estimation combines an AST heuristic with Spark's `EXPLAIN` size when a
session is available, and **the scale follows the dataset**: this lakehouse is
~500 MiB, so a cost unit is 500 MiB. At 50 MiB per unit a single year partition
(57 MiB) tripped the gate on a query Spark answers in half a second.

## Bounds

| bound | value | why |
|---|---|---|
| `MAX_TOOL_CALLS` | 20 | a general analysis needs ~12; beyond 20 the model is wandering |
| repeated identical call | 2 | same name and arguments twice in a row is a loop, not thinking |
| conversation budget | 800,000 tokens | of the model's 1M context |

On exhaustion the model is told to answer with what it has; if it still cannot,
`handle_failure` names the budget instead of pretending.

## Memory

`src/agent/memory.py`. The model is stateless, so "memory" is text the graph
reassembles: the whole thread, bounded by tokens rather than by turns.

- `count_tokens` uses `tiktoken`'s `cl100k_base` — at 800k an approximation is
  not worth the risk.
- `build_conversation_window` walks newest-to-oldest, keeps what fits, drops the
  oldest, never drops the newest message.
- A single oversized message is truncated to a quarter of the budget with an
  explicit marker instead of starving every other turn.
- History is skipped on a retry: the correction prompt already carries the error.

Verified across six turns where the sixth refers back to the first ("tornando ai
post del 2012").

## The numbers are not the model's

The critic writes prose. The graph appends a deterministic appendix:

```
## Appendice statistica (generata dal sistema)
| Metrica | n depressi | n controlli | p | Cohen's d | Esito |
```

The output check validates prose **and** appendix together, so a citation
requirement is satisfied by evidence rather than by the model's recollection.
This was the resolution of three failed attempts to make a flash model transcribe
one specific p-value out of fourteen results; each attempt ended with the whole
answer replaced by a compliance report.

## What the prose may and may not say

`src/agent/critic_check.py`:

- **significant** comparison → prose+appendix must carry `n`, `p`, `d`;
- any comparative or no-difference claim must rest on at least one fully cited
  assessment;
- `d` is required only for significant results — a non-significant comparison is
  acted on through its p-value;
- assertive claims about depression are rejected unless a significant, powered
  assessment supports them, in English *and* Italian;
- decimal commas are read as numbers, `p < 0.001` is accepted when true and
  tight, a restated question is not a claim.

If the critic still fails after one corrective call, the fallback is a readable
per-result summary — a degraded answer, not an error page.

## Auditing what happened

`tool_trace` records every call: name, arguments, rows returned, cost estimate,
guard payload, status. With a loop, this trace *is* the guarantee that used to be
implied by the order of the nodes.
