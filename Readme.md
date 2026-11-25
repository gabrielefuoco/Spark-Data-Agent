# Spark Data Agent

Conversational analytics over the **eRISK** corpus: longitudinal Reddit posts by people labelled depressed (`label=1`) and controls (`label=0`), analysed with Apache Spark and exposed through an agent that has to show its statistics before it is allowed to draw a conclusion.

The interesting part is not the chat window. It is that a question in natural language becomes **SQL validated against an AST**, runs against a **partitioned lakehouse**, comes back with a **p-value and an effect size**, and stops to ask a human before anything expensive runs.

## Architecture

```
 Streamlit UI  ──  LangGraph StateGraph  ──  PySpark Medallion Lakehouse
                        │                          Bronze  raw XML/TSV + lineage
                        │                          Silver  atomic posts, partitioned
                        │                          Gold    precomputed features + manifest
                        ├── sqlglot AST validator + cost estimator
                        ├── HITL gate (interrupt + SqliteSaver checkpointer)
                        ├── safe SQL executor (SELECT only, LIMIT ≤ 500)
                        ├── statistical guard (Welch / Mann-Whitney + Cohen's d)
                        └── self-correcting rewriter (≤ 2 retries)
```

**Stack**: Spark 3.5.1 · JDK 17 · Python 3.11 · LangGraph 1.2 · Streamlit · Plotly · Pandera · sqlglot · MinIO (S3-compatible API) · Docker Compose.

## Quickstart

Docker Desktop must be running. Everything — tests, Spark jobs, the UI — executes inside the dev container: there is one Python (3.11) and one JDK (17) to keep green.

```bash
docker compose up -d                       # dev container + MinIO (bucket `lakehouse`)
docker compose run --rm --no-deps dev python -m pytest
docker compose run --rm --no-deps dev python -m ruff check .
docker compose run --rm --no-deps dev mypy
docker compose exec dev bash               # a shell inside, Spark included
```

The UI, once a model backend is configured (see *Honest scope*):

```bash
docker compose exec dev python -m streamlit run src/ui/app.py --server.address 0.0.0.0
# http://localhost:8501
```

## The data is not in this repository

The eRISK corpus is real posts written by real people about their own depression. It is **never** committed: `processed_data/`, `chunk/` and `ignora/` are gitignored, and any fixture that ships with the repository is synthetic or aggregated.

Two ways to get something to run against:

```bash
# A deterministic synthetic corpus, shaped exactly like the real one (~100 subjects, ~5k posts)
docker compose run --rm --no-deps dev python scripts/generate_dev_sample.py --seed 42

# Or place the real corpus at the repo root (chunk/chunk1..10/*.xml, risk-golden-truth-test.txt);
# the container mounts it read-only at /workspace and nothing copies it into an image.
```

## Repository layout

```
src/
  config.py            Settings: one place that reads the environment
  spark.py             SparkSession builder (AQE on) + ZSTD parquet writer
  lakehouse/           bronze, silver (quarantine), gold, manifest, storage (S3A), validation
  analytics/           window features, Spark ML topic pipeline, safe SQL engine
  agent/               LangGraph state, prompts, cost model, guards, graph
  ui/                  Streamlit + Plotly
scripts/               synthetic corpus generator
tests/                 69 tests, Spark included, no network required
docs/adr/              three architecture decision records
.scratch/              the plan: map + seven tickets, in Italian
```

## Design decisions worth reading

- [`docs/adr/0001`](docs/adr/0001-medallion-lakehouse-vs-flat-parquet.md) — why Bronze/Silver/Gold with a dead-letter quarantine instead of flat aggregated Parquet.
- [`docs/adr/0002`](docs/adr/0002-langgraph-cyclical-stategraph-vs-llamaindex.md) — why a deterministic StateGraph with an explicit cost gate instead of an agentic workflow that just passes messages through.
- [`docs/adr/0003`](docs/adr/0003-openrouter-multi-model-routing.md) — cheap routing model vs reasoning critic, and why the backend is dual.
- [`CONTEXT.md`](CONTEXT.md) — the domain vocabulary the code is written against.

Three behaviours are deliberate and easy to mistake for bugs:

1. **No `.count()` at startup.** Global statistics come from `gold_dataset_manifest.json`, written once by the ETL.
2. **Nothing is dropped silently.** Every Bronze row ends in Silver *or* in `silver/quarantine/` with an explicit `_rejection_reason`; a test asserts the counts add up.
3. **The critic is constrained, not trusted.** Below 30 unique subjects per class the guard forces `is_significant=False` regardless of the p-value, and the prompt forbids claiming a clinical signal without quoting sample size, p-value and Cohen's *d*.

## Honest scope

This is a portfolio project, and the point of it is to be accurate about what it does:

- **Storage**: writes go to a **local MinIO** instance through the S3-compatible API (`s3a://lakehouse/`). Real AWS S3 is supported by configuration and has **never been exercised**. No claim of a run on AWS is made anywhere in this repository.
- **LLM backend**: dual by design — local Ollama for development, OpenRouter for remote. In the state this was built in, **no model was pulled and no remote key was set**: the entire test suite runs offline against fake chat models, and the agent graph is exercised with an injected SQL executor. Wiring a live model is a configuration step, not a code change.
- **Clinical claims**: the corpus supports exploratory analysis, not diagnosis. The statistical guard exists to stop the agent from dressing up noise as a finding.

## Status

All seven tickets of the plan in [`.scratch/spark-data-agent-revamp/map.md`](.scratch/spark-data-agent-revamp/map.md) are closed: Docker dev environment, project hygiene and settings, medallion lakehouse, advanced PySpark analytics, data quality and statistical guard, LangGraph orchestration with HITL, and the S3/MinIO connector with the ADRs.

`69 passed`, `ruff` clean, `mypy` clean (`Success: no issues found in 24 source files`).
