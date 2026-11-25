# Testing

148 tests, ~65 seconds, no network, no real model, no AWS.

```bash
docker compose run --rm --no-deps dev python -m pytest
docker compose run --rm --no-deps dev python -m pytest tests/test_agent_graph.py -q
docker compose run --rm --no-deps dev python -m pytest -m "not spark"   # skip Spark
```

Spark tests carry `@pytest.mark.spark` and share one session-scoped
`SparkSession` with two cores and two shuffle partitions, so the suite stays under
a minute and a half.

## What is covered where

| file | tests | covers |
|---|---|---|
| `test_agent_graph.py` | ~24 | the loop: tool dispatch, budget, repeat cap, HITL approve/refuse, appendix, fallback, guard verdicts wired end to end |
| `test_statistical_guard.py` | ~14 | unit psychology of the guard: aggregation, pseudo-replication, underpowered override, imbalance, Cohen's d |
| `test_critic_check.py` | ~20 | citation rules, forbidden claims EN/IT, decimal commas, tight inequalities, restated questions, fallback text |
| `test_safe_sql.py` | ~19 | AST blocking, allowlist (file-path exfiltration), LIMIT caps, view registration once per root |
| `test_silver.py`, `test_bronze.py`, `test_validation.py` | ~12 | lineage, quarantine reasons, the "every row accounted for" invariant, Pandera contract |
| `test_gold.py` | 1 | gold tables + manifest with zero Spark jobs at read time |
| `test_features.py`, `test_nlp_pipeline.py` | ~10 | window features on crafted dates, Spark ML pipeline, LDA distribution sanity |
| `test_storage.py`, `test_manifest_uri.py` | ~9 | S3A options, MinIO round-trip (skips without Docker), checkpoint stays local |
| `test_memory.py` | ~5 | token counting, sliding window, truncation marker, newest never dropped |
| `test_ui_app.py` | ~3 | the Streamlit app through `AppTest` with a fake graph |
| `test_config.py`, `test_generator.py`, `test_agent_models.py` | ~10 | settings defaults/overrides, deterministic sample generation, lazy model factory |

## What the tests deliberately mock

- **the model**: `FakeMessagesListChatModel` / `GenericFakeChatModel` with
  hand-written `tool_calls`. A test that reaches the network is a failing test
  even if it passes.
- **the SQL executor** in graph tests: a callable returning canned rows, so the
  loop is tested without starting Spark.
- **the Spark session** in lakehouse tests: a real one, on `tmp_path` data. Those
  are integration tests, marked `spark`.

## Something the suite could not see

Both of these were found by *using* the system, not by running it:

1. `spark-xml` rejects a list of paths, so ingesting `chunk/chunk1..10` failed on
   the first real call. The synthetic fixture writes a single chunk directory, so
   no test exercised the multi-directory layout. A regression test now does.
2. The cost estimator called `EXPLAIN` before the lakehouse views were
   registered, so every query against a view died with `TABLE_NOT_FOUND` and the
   retry loop repeated it three times.

The lesson is in the shape of the suite: mocked collaborators test *your logic*,
never *the wiring*. Keep at least one end-to-end path that a human runs.

## Manual driving

Three throwaway harnesses live in `data/` (gitignored) and are how the system is
exercised by hand:

- `manual_profile.py` — six questions across the difficulty range, printing tool
  calls, gates and latency. This is the cost profile quoted in the README.
- `manual_tool_loop.py` — one general analysis, printing the whole trace and
  auto-approving gates.
- `time_split.py` — a callback-based timing split between model calls and Spark.

They are not part of the suite on purpose: they need a network, a key and a
corpus, and a test that needs all three is a test that never runs.

## Style gates

`ruff check` (E, F, I, UP, B, SIM, line length 100), `ruff format`, and `mypy` on
`src/` with `disallow_untyped_defs`, `warn_unused_ignores` and
`warn_redundant_casts`. All three are green and are the definition of "done" for
any change.
