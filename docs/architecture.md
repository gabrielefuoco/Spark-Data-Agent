# Architecture

`spark-data-agent` is a conversational analytics system over the eRISK corpus:
longitudinal Reddit posts by people labelled depressed (`label=1`) and controls
(`label=0`). A question in natural language becomes validated Spark SQL, runs
against a partitioned lakehouse, comes back with its statistical verdict, and is
only then interpreted.

The interesting constraint is not the chat window. It is that **no claim reaches
the user without a number the system generated**, and that the model cannot
bypass the SQL safety layer, the cost gate or the statistical guard.

## The three parts

```
┌─────────────────────────────┐
│ Streamlit UI                │  chat, Plotly chart, data table,
│ src/ui/app.py               │  HITL approval card, resume by thread_id
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│ Agent — LangGraph tool loop │  agent ↔ execute_tools → scientific_critic
│ src/agent/                  │  tools: run_safe_sql, run_spark_job
│                             │  memory: token-budgeted window (800k)
│                             │  guards: statistical verdict + appendix
└──────────────┬──────────────┘
               │  execute_safe_spark_sql / job executor
┌──────────────▼──────────────┐
│ PySpark Medallion Lakehouse │  bronze → silver (+quarantine) → gold
│ src/lakehouse/              │  Parquet ZSTD, partitioned, S3A-capable
│ src/analytics/              │  safe SQL, window features, Spark ML
└──────────────┬──────────────┘
               │
        local disk (/workspace/data) or s3a://lakehouse (MinIO)
```

## Components and their one job

| module | responsibility |
|---|---|
| `src/config.py` | the only place that reads the environment; everything else takes `Settings` |
| `src/spark.py` | builds the `SparkSession` (AQE on, S3A when configured) and writes ZSTD Parquet |
| `src/lakehouse/bronze.py` | immutable raw ingestion with lineage |
| `src/lakehouse/silver.py` | atomic posts, dedup, dead-letter quarantine, `partitionBy(label, year)` |
| `src/lakehouse/validation.py` | Pandera contract on the Silver schema; violations become quarantine rows |
| `src/lakehouse/gold.py` | per-subject features, top n-grams with TF-IDF, LDA topics, storage volumes |
| `src/lakehouse/manifest.py` | `gold_dataset_manifest.json`: global stats readable without Spark |
| `src/lakehouse/storage.py`, `paths.py` | S3A configuration and URI-safe path handling |
| `src/analytics/safe_sql.py` | sqlglot AST validation, table allowlist, LIMIT cap, view registration |
| `src/analytics/features.py` | circadian/session/verbosity window functions |
| `src/analytics/nlp_pipeline.py` | native Spark ML pipeline and LDA |
| `src/agent/nodes.py` | the loop nodes: agent step, tool execution, critic |
| `src/agent/guards.py` | deterministic statistics: which test, which verdict |
| `src/agent/critic_check.py` | does the prose carry the citations it claims |
| `src/agent/memory.py` | token-counted conversation window |
| `src/agent/cost.py` | SQL cost estimation feeding the human gate |
| `src/ui/app.py` | Streamlit surface |

## The lifecycle of one question

Measured on the real corpus, on a two-part question ("does post length differ
between cohorts? and the night-time share?"), 30 seconds end to end:

1. **Agent step** (`~2s`) — the model sees the system prompt, the manifest
   digest, the conversation window and the tool results so far. It answers either
   with tool calls or with a final message.
2. **Tool execution** (`~1s` per query after the first) — `run_safe_sql`
   validates the AST, prices the query, interrupts for human approval above the
   threshold, executes, and runs the statistical guard on the result.
3. **Loop** — steps 1 and 2 repeat while the model keeps calling tools, bounded
   at `MAX_TOOL_CALLS` (20) with a guard against the same call twice in a row.
4. **Critic step** (`~12s`, the dominant cost) — a second model call writes the
   interpretation, receiving every result with its verdict and the guardrails.
5. **Appendix** — the graph appends a deterministic table of every metric with
   its sample sizes, p-value, effect size and verdict. The prose is validated
   *together with* this table, so citations are never the model's recollection.

## Where the time goes

Two measurements worth knowing, because both changed the design:

- **The query engine, not the model, was the first bottleneck.** 63s total, 16s
  of model, 47s of Spark — of which 9s per call was re-registering lakehouse
  views by re-reading five Parquet trees. Registration is now skipped when the
  same storage root is already bound: a query went from ~10s to 0.4-0.9s.
- **The critic's generation is what remains** (~12s of 30s). Shortening it is a
  quality decision, not an optimisation.

## Design decisions, and what they cost

Four ADRs in [`docs/adr/`](adr/README.md). The two that shaped the code most:

- **ADR 0001** — Medallion layers with a dead-letter quarantine, because flat
  aggregated Parquet plus `collect_list` is an OOM waiting for a bigger dataset.
- **ADR 0004** — a bounded tool loop instead of a fixed query pipeline. It buys
  exploratory analysis (12 queries where there was 1) and pays for it with
  auditability: the guarantee is no longer "the graph's order" but "the trace and
  the appendix".

## Honest boundaries

- The numbers in an answer come from the system, not the model — but the model
  chooses *which* numbers to look at. A question that needs a metric nobody
  computed gets a "not assessable" answer, not a guess.
- The lakehouse is local (or local MinIO). AWS is supported by configuration and
  has never been exercised.
- The model in this repository was exercised through OpenRouter; the Ollama path
  is configured but no local model has been pulled.
