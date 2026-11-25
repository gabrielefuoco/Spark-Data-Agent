# Limits and Next Steps

What this system does not do, written down so that a reader — or a future
session — does not have to discover it by breaking something.

## Known limits, by cause

### The data

- **79 depressed subjects.** Any effect estimate is governed by that group. The
  payload says so, but no amount of disclosure creates subjects.
- **No clinical validation.** Labels are a research ground truth; there is no
  severity, treatment or outcome. The system refuses to phrase findings
  clinically, which means it is useful for methods, not for medicine.
- **27% of the corpus is title-only posts.** They are quarantined rather than
  silently averaged in, which is correct and also means the corpus is smaller
  than the file count suggests.
- **No streaming ingestion.** A full run reloads everything; there is no
  incremental or partitioned-by-date ingest.

### The agent

- **The model chooses which questions to ask; the code decides what it may
  claim.** A question that needs a metric nobody computed gets "not assessable",
  not a clever workaround.
- **No plan is visible before execution.** In a loop, the user sees the answer,
  not the route. `tool_trace` holds the route, but the UI does not render it yet.
- **The critic's generation is the remaining latency** (~12s of a 30s question).
  Shortening it is a quality trade, not an optimisation.
- **Memory is a token-bounded window, not long-term recall.** Older turns are
  dropped silently once 800k tokens are exceeded; in normal use that never
  happens, and if it does the session has become a document.

### The infrastructure

- **AWS is configuration only.** Everything ran against local MinIO through the
  S3-compatible API. No claim of a run on AWS is made anywhere.
- **The Ollama path is configured but unexercised.** No local model was pulled;
  every measurement used OpenRouter.
- **No CI.** Tests, lint and type checks are commands a human runs.
- **No authentication, no multi-user isolation.** One operator, one container, one
  lakehouse. There is no session model to isolate, and no account to revoke.
- **Single-node Spark.** `local[*]` in one container. The medallion design does
  not need a cluster at this size, and the code does not pretend to one.
- **Windows host caveats remain at the edges**: Docker Desktop must be started by
  hand, and bind-mount paths need `MSYS_NO_PATHCONV=1` in Git Bash.

This is a portfolio project, and being accurate about what it does is part of it:
a demonstration of method, not a production system.

## What I would do next, in order

1. **Render the trace in the UI.** The loop produces a rich audit trail — every
   query, cost, guard verdict — and the interface shows only the final prose.
   This is the largest gap between what exists and what is visible.
2. **Let the user see the plan before it runs** for expensive analyses: a "this
   will run 12 queries and take ~3 minutes" confirmation, instead of discovering
   it at the first gate.
3. **Concurrency for independent queries.** A general analysis issues twelve
   sequential queries that do not depend on each other; four at a time would cut
   wall-clock substantially, at the cost of a more complex trace.
4. **A `make`-style task file** for the ingestion runbook, which is currently a
   Python snippet in `docs/operations.md`.
5. **An integration test that runs one real question a week** against a real
   model, marked and excluded from CI by default. Everything else is mocked, and
   the two bugs that mattered most were found by using the system, not by testing
   it.
6. **A README screenshot** of the UI with the appendix and the statistical card.
   Documentation without one picture asks a reader to trust prose.

## What I would not add

- **Another agent framework.** LangGraph is one dependency and the loop is 90
  lines of topology; a second abstraction layer would buy nothing.
- **A vector store** for "semantic search over posts". The interesting questions
  in this corpus are statistical, and the guard has nothing to say about a
  similarity score.
- **Fine-tuning.** With 79 labelled subjects, any fine-tune would memorise the
  cohort.
- **A custom cache layer.** `functools.lru_cache` plus the view-registration skip
  covers what is measurable.

## How to tell whether a change is worth it

The two rules this repository kept re-learning:

1. **If you have not run it, you do not know.** Mocked tests prove logic, not
   wiring. Every bug that mattered was found by asking the system a question.
2. **Where the code can produce the fact, the model must not be asked to remember
   it.** Three failed validators and one discarded synthesis taught the same
   lesson from three directions.
