# Operations

Everything runs inside Docker. There is one Python (3.11) and one JDK (17), and
Spark never executes on the Windows host.

## Start and stop

```bash
# Docker Desktop must be running; it does not start itself.
docker compose up -d            # dev container + MinIO (bucket `lakehouse` created)
docker compose ps
docker compose down             # keeps the named MinIO volume
docker compose down -v          # also drops it
```

`.env` is read by Compose at **container creation**. After editing it:

```bash
docker compose up -d --force-recreate dev
```

This has bitten before: a new model setting appeared to be ignored because the
container still carried the old environment.

## The everyday commands

```bash
docker compose run --rm --no-deps dev python -m pytest      # 148 tests, ~65s
docker compose run --rm --no-deps dev python -m ruff check .
docker compose run --rm --no-deps dev python -m ruff format src tests scripts
docker compose exec -T dev mypy                            # 29 files
docker compose exec dev bash                               # a shell with Spark
```

The UI:

```bash
docker compose exec dev python -m streamlit run src/ui/app.py --server.address 0.0.0.0
# http://localhost:8501
```

MinIO console: <http://localhost:9001> (`minioadmin` / `minioadmin`).

## Configuration

All settings live in `src/config.py` and are read from the environment or `.env`.

| variable | default | meaning |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` (local, free) or `openrouter` (remote) |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | from a container this must be `http://host.docker.internal:11434/v1` |
| `OLLAMA_MODEL` | `qwen2.5:7b` | local model name |
| `OPENROUTER_API_KEY` | — | required when backend is `openrouter`; `SecretStr`, empty means unset |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | |
| `REMOTE_ROUTER_MODEL` | `deepseek/deepseek-chat` | agent/tool-calling model |
| `REMOTE_REASONING_MODEL` | `deepseek/deepseek-r1` | critic model |
| `SPARK_STORAGE_URI` | `./data` | local path or `s3a://lakehouse` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `S3_ENDPOINT_URL` | — | S3A credentials and endpoint |
| `MIN_SUBJECTS_PER_CLASS` | 30 | below this the guard says underpowered |
| `SQL_ROW_LIMIT` | 500 | rows shown to the model |
| `SQL_FETCH_LIMIT` | 50000 | rows fetched for analysis |
| `HITL_COST_THRESHOLD` | 1.0 | cost above which a human approves |
| `MAX_TOOL_CALLS` | 20 | loop budget |
| `CONVERSATION_TOKEN_BUDGET` | 800000 | memory window |

Two model roles are separate on purpose: tool calling is cheap and frequent, the
critique is rare and benefits from a stronger model. With OpenRouter you can point
them at different models without touching code.

## Ingesting the real corpus

Place the eRISK files at the repository root (`chunk/chunk1..10/*.xml`,
`risk-golden-truth-test.txt`). The container mounts them read-only at
`/workspace`; they never enter an image and `data/` is gitignored.

```bash
docker compose exec -T -e PYSPARK_SUBMIT_ARGS="--driver-memory 4g pyspark-shell" \
  dev python /workspace/data/ingest.py
```

A minimal driver script, if you want to run it yourself:

```python
from src.config import get_settings
from src.spark import build_spark_session
from src.lakehouse.paths import LakehousePaths
from src.lakehouse.bronze import ingest_bronze_posts, ingest_bronze_subjects
from src.lakehouse.silver import build_silver
from src.lakehouse.gold import build_gold

cfg, paths = get_settings(), LakehousePaths.from_settings(get_settings())
spark = build_spark_session(cfg, shuffle_partitions=8)
ingest_bronze_posts(spark, "chunk", paths.bronze_posts)
ingest_bronze_subjects(spark, "risk-golden-truth-test.txt", paths.bronze_subjects)
build_silver(spark, paths=paths)
build_gold(spark, settings=cfg, paths=paths)
```

Expect ~10 minutes and ~240 MB under `data/`. What you should see afterwards:
544,447 bronze rows, 397,228 silver posts, 147,219 quarantined, 819 subjects.

For iteration, skip the real corpus:

```bash
docker compose run --rm --no-deps dev python scripts/generate_dev_sample.py \
  --seed 42 --subjects 100 --posts-per-subject 50
```

`--from-corpus PATH` samples the real files instead (output stays in the
gitignored `data/sample/`); `--no-broken` omits the deliberately malformed rows.

## Troubleshooting

**`error during connect: ... dockerDesktopLinuxEngine`** — Docker Desktop is not
running. Start it; on this host it never starts by itself.

**`Package 'openjdk-17-jre-headless' has no installation candidate`** — the base
image moved to a newer Debian whose only JDK is 21, which Spark 3.5 refuses. The
Dockerfile pins `python:3.11-slim-bookworm` for this reason.

**`pull access denied for minio/minio`** — the official MinIO images are no
longer on Docker Hub and `quay.io/minio/*` now requires a login. The compose file
uses the community fork `pgsty/minio` with pinned releases.

**`ImportError: cannot import name 'DateOffset'`** or **`np.NaN was removed`** —
the pandas/numpy/pyarrow pins in `pyproject.toml` are load-bearing: pandas < 3
because Spark 3.5 imports `DateOffset`, numpy < 2 because it references `np.NaN`,
pyarrow < 17 because newer builds refuse to load under numpy 1.x.

**The agent answers a catalog question in one second, then a data question slowly**
— expected. Catalog numbers come from a 98-token digest in the system prompt;
anything else costs Spark plus two model calls.

**A query that used to be fast is slow again** — check whether lakehouse view
registration is being re-done: it reads five Parquet trees (~9s) and is skipped
only when the session already registered the same storage root.

**`TABLE_NOT_FOUND` for a view that exists on disk** — the session registered a
different `SPARK_STORAGE_URI`, or Gold was rebuilt inside a live session. Reopen
the session.

## Cost of a question

Measured with a real model on the real corpus:

| question type | tool calls | wall clock |
|---|---|---|
| catalog (manifest digest) | 0 | ~1 s |
| greeting / refusal | 0 | 2-4 s |
| one comparison | 2 | ~12 s |
| general analysis (5 metrics, 5 tables) | 12 | 3-4 min |

The last row is the expensive one by design: twelve queries, ten assessed
comparisons, and an appendix with all of it.
