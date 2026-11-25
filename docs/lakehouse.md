# The Lakehouse

Three layers, one direction, and an invariant that a test enforces: **every row
that enters Bronze leaves through Silver or through quarantine.** Nothing is
dropped silently, and nothing is aggregated into a blob.

```
chunk/chunk1..10/*.xml   ─┐
risk-golden-truth-test.txt ┴─► BRONZE ──► SILVER ──► GOLD ──► manifest.json
                              (raw +     (atomic,    (precomputed
                               lineage)   partitioned) cubes)
                                   │
                                   └────► SILVER QUARANTINE
                                          (_rejection_reason)
```

## Bronze — immutable raw input

`src/lakehouse/bronze.py`. Reads the eRISK XML (`rootTag=INDIVIDUAL`,
`rowTag=WRITING`, explicit schema) and the tab-separated labels file, then adds
three lineage columns to every row:

| column | how |
|---|---|
| `_source_file` | `F.input_file_name()` |
| `_ingested_at` | `F.current_timestamp()` |
| `_row_hash` | `F.sha2` over the business columns joined with `\|\|` |

Raw values are kept raw: the `DATE` string still carries its surrounding spaces,
because trimming is a Silver decision and Bronze must stay replayable.

**One reader caveat that cost a real debugging session**: `spark-xml` accepts a
single path. Handing it a list raises `IllegalArgumentException: 'path' must be
specified for XML data`, and `recursiveFileLookup` does not help because the
source needs block locations and dies with a `NullPointerException`. A corpus
split across `chunk1..chunkN` is therefore read one directory at a time and the
frames are unioned by name.

## Silver — atomic posts, and a place for the rejects

`src/lakehouse/silver.py` and `validation.py`.

- **One row per post.** The previous implementation did
  `groupBy("subject_id").agg(collect_list("text"))`, which collapses a subject's
  entire history into a string. That is the OOM time-bomb the rewrite exists to
  remove; the word `collect_list` does not appear in `src/`.
- **Text normalised**, `date` parsed from `yyyy-MM-dd HH:mm:ss` on the trimmed
  string, labels joined with a broadcast.
- **Deterministic dedup** on `(subject_id, date, _row_hash)`.
- **Physical partitioning** on write: `.partitionBy("label", "year")`, ZSTD.
- **Pandera contract** (`PostSilverSchema`) applied inside the ETL: types, no
  nulls on keys, `label in {0,1}`, non-empty text. Violations are anti-joined out
  of Silver and land in quarantine as `contract_violation:<reason>`.

### Quarantine reasons

| reason | meaning |
|---|---|
| `missing_subject_id` | the file name carried no subject id |
| `malformed_date` | `DATE` did not parse |
| `null_or_blank_text` | empty post body (title-only posts, common in this corpus) |
| `orphan_subject_id` | the subject has no row in the labels file |
| `contract_violation:*` | a Pandera constraint failed |
| `duplicate_post` | an exact duplicate of an earlier row |

Duplicates go to quarantine rather than being filtered, so that
`bronze == silver + quarantine` holds exactly and a test can assert it.

### Real numbers from a full run

| | |
|---|---|
| Bronze rows | 544,447 (from 8,200 XML files) |
| Silver posts | 397,228 |
| Quarantine | 147,219 — 147,177 `null_or_blank_text`, 42 `duplicate_post` |
| Subjects with valid posts | 819 (740 control, 79 depressed) |
| Date range | 2006-07-13 → 2017-05-10 |

The size of the quarantine is the interesting figure: 27% of the corpus is
title-only posts. The previous pipeline let those through as empty strings.

## Gold — answers without Spark

`src/lakehouse/gold.py`. Three artefacts plus the manifest:

| table | content |
|---|---|
| `gold_user_longitudinal_features` | one row per subject: post counts, active days, sessions, mean inter-post gap, night ratio, verbosity z-scores, span |
| `gold_cohort_lexical_matrix` | top **100** n-grams per (label, type) with frequency, relative frequency, IDF and TF-IDF, ranked |
| `gold_cohort_topic_matrix` | LDA topics per cohort: mean probability, document count, top terms |
| `gold_dataset_manifest.json` | total posts, subjects and posts per class, date range, quarantine breakdown, storage volumes by layer and table |

The features come from `src/analytics/features.py` (window functions) and the
lexical/topic tables from `src/analytics/nlp_pipeline.py` — a native Spark ML
`Pipeline` (tokenizer → clinical stopwords → bigrams → CountVectorizer → IDF →
LDA k=8, maxIter=20), no Python UDFs. Keeping negation and affect markers in the
stopword list is deliberate: in clinical text, "not" is the signal.

`top_n_ngrams` defaults to 100. When it was `None` the real corpus produced
**5,021,458 rows** — the ticket asked for top n-grams, and 5M rows is not a
precomputed cube.

**The manifest is the reason metadata questions cost no Spark job.** It is
written once by the ETL and read as JSON (through `pyarrow.fs` when the lakehouse
lives on `s3a://`). The agent receives a 98-token digest of it in its system
prompt, so "how many subjects per cohort?" answers in ~1 second with zero tool
calls.

## Storage

`spark_storage_uri` decides everything: a local path, or `s3a://bucket/prefix`.
The S3A Hadoop options are attached to the session by `src/lakehouse/storage.py`
and refuse to run without endpoint and credentials rather than failing later
inside the JVM. `hadoop-aws` 3.3.4 and `aws-java-sdk-bundle` 1.12.262 are baked
into the image, matching the Hadoop shipped with Spark 3.5.1.

The SQLite checkpointer for the agent is explicitly **local**: SQLite needs file
locking and random writes, so it does not live on `s3a://`.

## Running the ETL

```bash
docker compose up -d
docker compose exec dev python -c "..."   # or a script, see docs/operations.md
```

A full ingestion of the real corpus takes roughly ten minutes with a 4 GB driver
and produces ~240 MB under `data/` (gitignored). For fast iteration, generate a
synthetic corpus instead:

```bash
docker compose run --rm --no-deps dev python scripts/generate_dev_sample.py --seed 42
```
