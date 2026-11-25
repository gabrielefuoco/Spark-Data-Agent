# Context: Spark Data Agent (Conversational Big Data Analytics Platform)

## Overview

Spark Data Agent is an interactive platform for the analysis of large-scale longitudinal textual data — the Reddit **eRISK** dataset for early depression detection — exposed through a conversational interface. A researcher asks a question in natural language; an agent selects or writes an analytical query; Spark executes it; the result comes back with its statistical significance attached.

## Previous implementation (removed)

Before the rewrite the repository held the original academic implementation: seven Python modules, about 1,300 lines, in Italian. It was **deleted** once the rewrite was complete; it survives only in history, at commit `e2976dc` (for example `git show e2976dc:legacy/spark_utils.py`). Nothing under `src/` ever imported it, and the ADRs keep citing it in the past tense because it is the *before* state they argue against.

What it did:

- **Pipeline**: a preprocessing script turned the raw XML/TSV into flat Parquet files in `processed_data/`; no layered lakehouse, no lineage.
- **Analytics**: PySpark functions wrapped as agent tools in `spark_utils.py` and `tools.py`, executed through a `SparkDataManager` that owned the session.
- **Agent**: LlamaIndex `AgentWorkflow` over **Gemini 2.5 Flash** — a request came in, one tool was called, one answer went out. No control loop, no self-correction, no cost gate.
- **UI**: Streamlit.
- **Known defects**, every one of them addressed by the rewrite: four blocking `.count()` calls at startup, `groupBy("subject_id").agg(collect_list("text"))` (an OOM waiting for a large enough dataset), indiscriminate `.persist()`, unpartitioned writes, `from pyspark.sql.functions import *` shadowing Python built-ins, phantom dependencies, and API keys read by hand from `src/api.env`.

## Target architecture

The rewrite is specified ticket by ticket in [`.scratch/spark-data-agent-revamp/map.md`](.scratch/spark-data-agent-revamp/map.md): a **PySpark Medallion Lakehouse** (Bronze/Silver/Gold on Parquet ZSTD with lineage and dead-letter quarantine) driven by a **cyclical LangGraph StateGraph**, with a deterministic statistical guard and a human-in-the-loop gate on expensive queries. Six tickets, in dependency order, all `ready-for-agent`.

```
                      +-----------------------------+
                      |       User / UI Streamlit   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | LangGraph Cyclical Agent    |
                      | - Planner / Router          |
                      | - Safe SQL / sqlglot AST    |
                      | - HITL Approval Gate        |
                      | - Statistical Guard (SciPy) |
                      | - Scientific Critic         |
                      +--------------+--------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
|                       PySpark Lakehouse Engine                        |
|                                                                       |
| [Bronze Layer]           [Silver Layer]            [Gold Layer]       |
| Raw TSV/XML + Lineage    Atomic Posts (1 row/post) Precomputed Cubes  |
| (_source_file, sha2)     Partitioned (label, year) Features & Manifest|
|                          Dead-letter Quarantine                       |
+-----------------------------------------------------------------------+
                                     |
                                     v
                        Storage: Local Parquet / S3 MinIO
```

The terms below describe the target. Where the current code differs, the current code is wrong, not the vocabulary.

## Core Domain Terms

- **Post (Atomic)**: A single Reddit submission or comment, maintaining 1 row = 1 record. Never aggregated with `collect_list`.
- **Subject**: A unique Reddit user profile, labeled either as depressed (`label=1`) or control (`label=0`).
- **Medallion Lakehouse**:
  - `Bronze`: Raw, immutable data ingestion with lineage columns (`_source_file`, `_ingested_at`, `_row_hash`).
  - `Silver`: Cleaned, validated, deduplicated atomic posts partitioned by `label` and `year`. Malformed records routed to `silver_quarantine.parquet`.
  - `Gold`: Precomputed analytical matrices (`gold_user_longitudinal_features`, `gold_cohort_lexical_matrix`) and static metadata catalog (`gold_dataset_manifest.json`).
- **Circadian Index**: Percentage of posts published between 00:00 and 05:00, calculated over a 30-day sliding window (`rowsBetween(-30, 0)`).
- **Statistical Significance Guard**: Deterministic post-query evaluation node ensuring sample size $N \ge 30$, running Welch's t-test or Mann-Whitney U, and computing Cohen's $d$ effect size before LLM synthesis.
- **HITL (Human-in-the-Loop)**: `interrupt_before` gate in LangGraph backed by `SqliteSaver` checkpointer that prompts user approval when Spark `EXPLAIN COST` exceeds execution thresholds.
- **Dual LLM Backend**:
  - Local mode: Ollama (`http://localhost:11434/v1`) for zero-token-cost development and unit tests.
  - Remote mode: OpenRouter API with DeepSeek (`deepseek/deepseek-chat` and `deepseek/deepseek-r1`).

## Data policy

The eRISK corpus contains posts written by real people about their own depression. It is **never** committed: `processed_data/`, `chunk/` and `ignora/` are outside version control, and any fixture shipped with the repository must be synthetic or aggregated, never sampled real text.
