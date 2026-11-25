# Spark Data Agent

Conversational analytics over the **eRISK** corpus — 397,228 Reddit posts by
people labelled depressed (`label=1`) and controls (`label=0`) — built on a
PySpark Medallion lakehouse and driven by a LangGraph agent that has to show its
statistics before it is allowed to draw a conclusion.

The point is not the chat window. The point is that a question in natural
language becomes **SQL validated against an AST**, runs on **partitioned
Parquet**, comes back with a **p-value and an effect size**, and stops to ask a
human before anything expensive runs. When the numbers do not support a claim,
the system says so in the sentence it refuses to write.

```
Q:  I depressi postano più di notte dei controlli?
A:  No. 79 soggetti depressi vs 740 controlli, Mann-Whitney U,
    p = 0.5587, Cohen's d = 0.068 — nessuna differenza rilevata.
    [appendice statistica generata dal sistema]
```

---

## Content

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [How a question becomes an answer](#how-a-question-becomes-an-answer)
- [Measured behaviour](#measured-behaviour)
- [The data](#the-data)
- [Documentation](#documentation)
- [Design principles](#design-principles)
- [Status](#status)

---

## What it does

| capability | example |
|---|---|
| **Catalog** — answered from a precomputed manifest, no Spark | "quanti soggetti per coorte?" → 1 second, 0 queries |
| **Cohort statistics** — a real hypothesis test per metric | "la lunghezza dei post differisce?" → Mann-Whitney U, p-value, Cohen's d |
| **Circadian / behavioural features** — sessions, night rhythm, verbosity z-scores | "i depressi postano più di notte?" |
| **Lexical & topic analysis** — TF-IDF top n-grams, LDA topics per cohort | "quali parole caratterizzano i depressi?" |
| **General analysis** — the model plans and runs a dozen queries itself | "dammi un quadro delle differenze tra le coorti" → 12 tool calls, 10 assessed metrics, one synthesis |
| **Ad-hoc SQL** — validated, allowlisted, capped | "quanti post nel 2014?" → 18,611 |
| **Refusals** | "DROP TABLE silver_posts" → blocked before Spark, in 4 seconds |

## Architecture

```
 Streamlit UI ──► LangGraph tool loop ──► PySpark Medallion Lakehouse
                    │                        bronze  raw XML/TSV + lineage
                    │                        silver  atomic posts + quarantine
                    │                        gold    features, n-grams, topics
                    ├── run_safe_sql        (AST + allowlist + LIMIT)
                    ├── run_spark_job       (LDA, circadian, priced)
                    ├── HITL gate           (interrupt + SqliteSaver)
                    ├── statistical guard   (subject-level, Welch/Mann-Whitney)
                    └── scientific critic   (prose + system-generated appendix)
```

**Stack**: Spark 3.5.1 · JDK 17 · Python 3.11 · LangGraph 1.2 · Streamlit ·
Plotly · Pandera · sqlglot · tiktoken · MinIO (S3-compatible) · Docker Compose.

## Quickstart

Docker Desktop must be running; everything executes inside the container, so
there is one Python and one JDK to keep green.

```bash
git clone <repo> && cd <repo>
cp .env.example .env          # add OPENROUTER_API_KEY, or point at Ollama
docker compose up -d          # dev container + MinIO (bucket `lakehouse`)

# data: a deterministic synthetic corpus (the real one is not distributed)
docker compose run --rm --no-deps dev python scripts/generate_dev_sample.py --seed 42

# or bring the real corpus: place `chunk/` and `risk-golden-truth-test.txt` at the
# repo root and follow docs/operations.md (≈10 min, ≈240 MB, gitignored)

# quality gates
docker compose run --rm --no-deps dev python -m pytest      # 148 tests, ~65s
docker compose run --rm --no-deps dev python -m ruff check .
docker compose exec -T dev mypy

# the UI
docker compose exec dev python -m streamlit run src/ui/app.py --server.address 0.0.0.0
# http://localhost:8501
```

`.env` is read by Compose when the container is created: after editing it, run
`docker compose up -d --force-recreate dev`, or the old environment stays.

## How a question becomes an answer

One line per step, and where the trust boundary is:

1. **The model plans.** It sees the system prompt, a 98-token digest of the
   dataset manifest, the conversation window and the results so far. It answers
   either with tool calls or with a final message.
2. **`run_safe_sql` executes nothing until it is safe.** sqlglot validates the
   AST, an allowlist rejects any table that is not a registered lakehouse view
   (including `csv.\`/path\`` file reads), a `LIMIT` is injected and capped, and a
   cost estimate decides whether a human must approve first.
3. **Every result is judged.** The statistical guard collapses rows per subject,
   picks Welch or Mann-Whitney after a normality screen, and returns
   `assessed` / `descriptive` / `cannot_assess` with sample sizes, p-value and
   Cohen's d — or an explicit "cannot be evaluated, and why".
4. **The loop continues or ends**, bounded at 20 tool calls and two identical
   calls in a row.
5. **The critic interprets.** A second model call writes the prose, and the graph
   appends a deterministic statistics appendix. The output check validates prose
   and appendix together, so **no number in a final answer is the model's
   recollection**.

## Measured behaviour

Six questions on the real corpus with a real model, after the optimisations:

| question | tool calls | wall clock |
|---|---|---|
| catalog ("soggetti per coorte?") | 0 | **1 s** |
| greeting / capability question | 0 | **2 s** |
| "DROP TABLE silver_posts" | 0 | **4 s** (refused) |
| one comparison ("lunghezza dei post") | 2 | 12 s |
| regex question ("quanti gatti?") | 3 | 30 s (2 gates) |
| general analysis ("quadro delle differenze") | 12 | ~4 min |

Where the time goes on a 30-second question: **model 18 s, Spark 12 s**. That
ratio was the other way round before view registration stopped re-reading five
Parquet trees on every tool call (9 s per call → 0.4 s).

## The data

The eRISK corpus is real posts by real people about their own depression. It is
**never committed**: `processed_data/`, `chunk/` and `ignora/` are gitignored and
mounted read-only in the container. Anything shipped with the repository is
synthetic or aggregated — including the test fixture, which is generated with a
seed rather than sampled.

A full ingestion of this copy produces:

| | |
|---|---|
| writings ingested | 544,447 |
| atomic posts | 397,228 |
| quarantined (mostly title-only posts) | 147,219 |
| subjects | 819 (79 depressed, 740 control) |
| period | 2006-07 → 2017-05 |

The 1:9.4 imbalance is the first thing any reader should know about this dataset,
and the system states it in every relevant answer.

## Documentation

| document | what it answers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | components, request lifecycle, where the seconds go |
| [docs/lakehouse.md](docs/lakehouse.md) | Bronze/Silver/Gold, quarantine reasons, the count invariant |
| [docs/agent.md](docs/agent.md) | the tool loop, memory, HITL, bounds, the appendix |
| [docs/statistics.md](docs/statistics.md) | the guard's methodology, thresholds, three verdicts |
| [docs/sql-safety.md](docs/sql-safety.md) | AST validation, the allowlist, the two limits |
| [docs/operations.md](docs/operations.md) | runbooks, every environment variable, troubleshooting |
| [docs/testing.md](docs/testing.md) | what the 148 tests cover, and what they cannot |
| [docs/data.md](docs/data.md) | the corpus, its policy, what it can and cannot answer |
| [docs/decisions.md](docs/decisions.md) | the four ADRs plus the decisions made without one |
| [docs/roadmap.md](docs/roadmap.md) | limits, and what I would do next |

[`CONTEXT.md`](CONTEXT.md) holds the domain vocabulary, [`docs/adr/`](docs/adr/README.md)
the decision records, and [`.scratch/spark-data-agent-revamp/`](.scratch/spark-data-agent-revamp/map.md)
the plan of record with the seven tickets that built this.

## Design principles

Four sentences that explain most of the code:

1. **The model interprets; the code measures.** Where a fact can be computed,
   the model is not asked to remember it — the statistical appendix exists
   because three attempts to make a flash model transcribe a p-value all failed.
2. **Nothing is dropped silently.** Every raw row ends in Silver or in quarantine
   with a reason, and a test asserts that the counts add up.
3. **A guardrail must be proportionate.** A check that rejects correct answers is
   worse than none: it teaches a reader to trust the fallback. Three calibration
   rounds were spent learning this (decimal commas, `p < 0.001`, restated
   questions).
4. **If you have not run it, you do not know.** Mocked tests prove logic, not
   wiring. Every bug that mattered here — `spark-xml` refusing a list of paths,
   `EXPLAIN` before views were registered, `LIMIT 500` truncating 819 subjects —
   was found by asking the system a question, not by running the suite.

## Status

Seven tickets of the plan are closed: Docker environment, project hygiene,
medallion lakehouse, advanced PySpark analytics, data quality and statistical
guard, LangGraph orchestration, S3 + ADRs. A two-axis code review then found that
three of them were "resolved" only on paper; the fixes are in the history with
their own commits, and the map records what each one closed.

```
148 tests · ruff clean · mypy clean (29 files)
src/ 5,804 lines · tests/ 3,944 · 4 ADRs · 10 documentation files
```

What is missing, what it refuses to add, and what I would do next:
[docs/roadmap.md](docs/roadmap.md).
