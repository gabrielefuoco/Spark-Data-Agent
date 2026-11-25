# Decisions

An index of the architecture decision records in [`docs/adr/`](adr/README.md),
plus the decisions that shaped the code but never deserved a full ADR.

## The four ADRs

| ADR | decision | cost |
|---|---|---|
| [0001](adr/0001-medallion-lakehouse-vs-flat-parquet.md) | Bronze/Silver/Gold with a dead-letter quarantine, instead of flat aggregated Parquet | three layers to maintain, and a 27% quarantine that has to be explained |
| [0002](adr/0002-langgraph-cyclical-stategraph-vs-llamaindex.md) | a LangGraph `StateGraph` instead of the LlamaIndex `AgentWorkflow` that was there | a framework to learn; control that is explicit rather than emergent |
| [0003](adr/0003-openrouter-multi-model-routing.md) | dual backend: local Ollama for development, OpenRouter for remote, with separate router and critic models | two code paths, and a configuration surface nobody exercises at once |
| [0004](adr/0004-tool-calling-loop-vs-fixed-pipeline.md) | a bounded tool loop instead of one query per turn | auditability moves from the shape of the graph to the trace and the appendix |

ADR 0004 supersedes the topology in 0002 while keeping its reasoning: LangGraph
was the right framework, the fixed pipeline inside it was not.

## Decisions made during the work, without an ADR

**The synthetic fixture.** The ticket asked for a sample of the real corpus. Real
posts about real depression cannot be published, so the generator produces a
synthetic corpus with the same shape, and `--from-corpus` covers local work.

**`p < 0.05` is not a citation.** Threshold inequalities were rejected outright at
first, which also rejected the honest `p < 0.001`. The rule is now: an inequality
counts when it is true and tight (within an order of magnitude).

**The guard counts subjects, not rows.** Forty posts from one person is one
observation. Found by review, not by a test.

**A third verdict, `descriptive`.** Two verdicts meant n-gram tables and topic
models could not be answered at all; the guard said `cannot_assess` and the critic
refused. Descriptions are now allowed and inferences are still forbidden.

**Two limits, not one.** 500 rows for what the model sees, 50,000 for what the
guard analyses, because a presentation cap must not decide statistical validity.

**The numbers are system-generated.** After three attempts to make a flash model
transcribe a p-value correctly, the appendix became deterministic. The model
interprets; the code measures.

**Keep `critic_output_check_failed`.** Twice flagged as dead telemetry by an
audit, kept deliberately: it is the only record that the guardrail fired and why,
and reading it from a checkpoint is how the fallback firing was diagnosed.

**Delete `legacy/`.** The pre-rewrite implementation lived in the repository as
evidence of the defects. Once every claim about it was written in the past tense
with a commit hash, it was removed rather than kept as a museum.

**No two architectures side by side.** When the pipeline became a loop, the old
nodes were deleted in the same change. Keeping them "just in case" hides two sets
of bugs from each other.

## Numbers that decided something

| measurement | consequence |
|---|---|
| 8,200 files across 10 directories | `spark-xml` takes one path; ingestion reads per directory and unions |
| 5,021,458 lexical rows with `top_n_ngrams=None` | default is 100 per (label, type): 400 rows, 16 KB |
| 63s per question, 47s of it Spark | view registration was re-reading five Parquet trees per tool call |
| 9s to register, 0.9s to query | registration now skipped per storage root |
| 819 subjects vs `LIMIT 500` | two limits, and `truncated` in the payload |
| 79 vs 740 subjects | `balance_ratio` in the payload, disclosure in the prose |
| 16s of model, 12s of Spark after the fix | the critic's generation is the remaining cost, and shortening it is a quality decision |
