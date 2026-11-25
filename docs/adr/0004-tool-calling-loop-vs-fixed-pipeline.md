# ADR 0004: Tool-Calling Loop instead of a Fixed Query Pipeline

## Status

Accepted — supersedes the six-node topology described in ADR 0002 and in the
plan's "Orchestrazione LangGraph" section.

## Context

The first implementation of the agent was a fixed pipeline: an intent router
wrote **one** SQL query (or one Spark job spec), an AST validator and cost
estimator approved it, a human gate authorised expensive work, the executor ran
it, the statistical guard judged the result and the critic interpreted it. The
order of those steps was the guarantee: the guard could not be skipped because
the graph left no path around it.

That design answered one question per turn. A researcher asking "give me a
picture of how the two cohorts differ" needs a dozen queries — volumes, rhythm,
verbosity, lexicon, sessions, gaps — and no single query expresses that. The
planner had no way to look at the first result and let it decide the second
query, which is exactly what exploratory analysis is.

Measured on the real corpus: the fixed pipeline produced one result per turn; a
general question answered by the loop produced twelve tool calls, seven assessed
comparisons and five descriptive tables in about three and a half minutes.

## Decision

Replace the pipeline with a tool-calling loop. The model receives three tools —
`run_safe_sql`, `read_dataset_manifest`, `run_spark_job` — and iterates until it
can answer, bounded at `max_tool_calls` (20) with a guard against repeating the
same call twice in a row.

The guarantees that used to live in the graph's shape move into the tools and
the state:

- **The model cannot bypass SQL safety**: it has no access to Spark, only to
  `run_safe_sql`, which still runs the sqlglot AST validation, the table
  allowlist and the `LIMIT` cap.
- **The human gate sits inside the loop**: a tool call priced above the
  threshold calls `interrupt()` before executing, and a refusal is fed back as
  the tool's result.
- **Every result is judged**: each tool result passes the statistical guard and
  is stored with its verdict.
- **Auditability replaces determinism**: `tool_trace` records every call with
  its arguments, rows and guard payload. Previously "the guard ran before the
  critic" was true by construction; now it is true by archive.

## Consequences

- A general analysis is possible; it costs one model call per tool call plus
  the synthesis, and minutes of latency.
- The controlling invariant is no longer the node order but the output check:
  nothing reaches the user without an appendix of system-generated numbers, and
  a prose claim that contradicts that appendix is rejected.
- **The numbers in the final answer are never the model's recollection.** The
  critic writes the interpretation; the graph writes the statistics. This was
  forced by three failed attempts to make a flash model transcribe a specific
  p-value from fourteen results — each attempt ended with the whole answer
  replaced by a compliance report.
- Cost and latency are now variable and user-visible; a loop that wanders is
  bounded by the tool budget, not by a plan.
- The old nodes (`intent_router_planner`, `ast_validator_and_cost`,
  `query_rewriter`) are deleted rather than kept beside the loop. Two
  architectures in one repository hide each other's bugs.
- Testing moved from "assert the order of nodes" to "assert the trace and the
  invariants", which is weaker at localising a regression and stronger at
  proving what the user actually received.
