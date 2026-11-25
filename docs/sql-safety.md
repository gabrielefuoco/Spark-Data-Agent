# SQL Safety

The model writes SQL. Everything in this document exists because that sentence is
true.

`src/analytics/safe_sql.py`

## The three doors, in order

1. **AST validation** (`sqlglot`, dialect `spark`) — before Spark sees anything.
2. **Table allowlist** — only the lakehouse views the system registered.
3. **Row cap** — a `LIMIT` the model does not control.

Only then does `spark.sql(...)` run.

## 1. AST validation

A single statement, and its root must be `SELECT`, a set operation (`UNION`) or a
subquery wrapping one. Anything else is `DISALLOWED_STATEMENT`.

The walk over the tree rejects, wherever they appear (including inside a CTE or a
subquery):

| blocked | node types |
|---|---|
| DDL | `Create`, `Drop`, `Alter`, `TruncateTable` |
| DML | `Insert`, `Delete`, `Update`, `Merge`, `Into` |
| command escape hatches | `Command` |

Multi-statement payloads are rejected before any of this (`MULTIPLE_STATEMENTS`),
so `SELECT 1; DROP TABLE x` never reaches Spark as two statements.

## 2. The allowlist

This is the part that is easy to miss and was found by probing the running
system, not by reading the ticket.

Spark SQL lets a **table name be a file path**:

```sql
SELECT * FROM csv.`/workspace/.env.example`
SELECT * FROM text.`/workspace/src/config.py`
SELECT * FROM parquet.`/workspace/data`
SELECT * FROM read_files('/workspace/.env')
```

All four passed the original validator, because blocking `DROP` says nothing
about *which* tables a query may read. An LLM writes that query, so the agent
could have read its own `.env` — including the API keys it uses.

The rule now:

- any reference carrying a `db` or `catalog` qualifier is rejected
  (`TABLE_NOT_ALLOWED`) — that is exactly how `csv.`path``, `parquet.`path``,
  `text.`path`` and `read_files(...)` arrive;
- a bare name must be one of the registered lakehouse views, or a CTE defined in
  the same query (compared case-insensitively).
- the offending reference is quoted in the error message, so the model can
  correct itself.

`LAKEHOUSE_VIEW_NAMES` is a frozenset derived from the same mapping that registers
the views, so the two cannot drift.

## 3. Two limits, two jobs

| setting | value | bounds |
|---|---|---|
| `SQL_FETCH_LIMIT` | 50,000 | rows fetched into the driver, injected as `LIMIT` |
| `SQL_ROW_LIMIT` | 500 | rows shown to the model and available for citation |

They were one number, and that was a bug with statistical consequences: the gold
per-subject table holds 819 subjects, the query got `LIMIT 500`, and the guard
reported **52 depressed vs 448 control** — the first 500 rows in partition order.
A sample of convenience, manufactured before the guard could object.

The guard now analyses every fetched row; the model sees at most 500; the payload
records `rows_returned`, `rows_shown` and `truncated`. If the fetch ceiling is
ever hit, `truncated=true` and the critic must disclose it.

## Structured errors

`SafeSQLError` carries a machine-readable `code` so the model can react:

| code | meaning |
|---|---|
| `EMPTY_QUERY`, `MULTIPLE_STATEMENTS`, `DISALLOWED_STATEMENT` | not a single SELECT |
| `SQL_SYNTAX_ERROR` | sqlglot could not parse it |
| `TABLE_NOT_ALLOWED` | file-path or unknown table reference |
| `TABLE_NOT_FOUND` | allowed name, but no view is registered |
| `INVALID_LIMIT` | a non-literal or negative `LIMIT` |
| `SPARK_ANALYSIS_ERROR` | Spark's own analysis failure, passed through |

## Cost estimation, and the human gate

`src/agent/cost.py` combines an AST heuristic with Spark's `EXPLAIN` size when a
session exists. A query above `HITL_COST_THRESHOLD` is not executed until a human
approves it.

The scale had to be calibrated on the data: at 50 MiB per cost unit, a single
year partition (57 MiB) tripped the gate on `SELECT COUNT(*) FROM silver_posts
WHERE year = 2014` — a half-second query. At 500 MiB per unit a pruned partition
scores 0.65 and a full unfiltered scan 1.5, so the gate now separates cheap from
expensive rather than medium from slightly-less-medium.

## What the model still can do

- read any registered lakehouse view;
- write joins, window functions, CTEs, aggregations, `CASE`, regexes;
- hit the fetch ceiling and be told so;
- ask for approval and be refused.

## What it cannot do

- modify anything (no DDL/DML reaches Spark);
- read a file it names;
- see more rows than the fetch ceiling;
- exceed 500 rows in what it cites;
- skip the statistical guard — every tool result passes through it.
