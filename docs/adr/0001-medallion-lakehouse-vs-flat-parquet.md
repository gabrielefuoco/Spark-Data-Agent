# ADR 0001: Medallion Lakehouse vs. Flat Parquet Tables

## Status

Accepted

## Context

The previous eRISK analytics pipeline (deleted in the rewrite; it lives in history at `e2976dc`, see `git show e2976dc:legacy/spark_utils.py`) dumped raw XML/TSV inputs into flat, unpartitioned Parquet files and collapsed every user's writing history into a single row via `groupBy("subject_id").agg(collect_list("text"))` and `concat_ws`. That design created four compounding problems:

1. **Driver/executor OOM time-bomb**: collecting unbounded post histories per subject into array/string blobs produces extreme data skew and breaks JVM memory limits as corpus size grows.
2. **Silent data loss**: malformed dates, blank texts, or orphan subjects without ground-truth labels were dropped or coerced without audit trails.
3. **Expensive startup and ad-hoc queries**: the previous `SparkDataManager` (`git show e2976dc:legacy/data_manager.py`) ran four blocking `.count()` actions at startup and indiscriminately `.persist()`ed tables without partitioning.
4. **No storage abstraction**: output paths were hardcoded to local directories with no S3A connector path.

## Decision

Adopt a three-layer **Medallion Lakehouse** stored as ZSTD-compressed Parquet, preserving **atomic post granularity (1 row = 1 post)** from ingestion through Silver:

- **Bronze (`lakehouse/bronze/`)**: Immutable raw XML posts and ground-truth TSV labels with explicit schemas and lineage columns (`_source_file`, `_ingested_at`, `_row_hash` SHA-256).
- **Silver (`lakehouse/silver/`)**: Cleaned, deduplicated atomic posts partitioned physically by `(label, year)`, validated against `PostSilverSchema` (`pandera.pyspark`). Every invalid, orphan, or duplicate row is routed to `silver/quarantine/` with an explicit `_rejection_reason` (dead-letter pattern) so `bronze_count == silver_count + quarantine_count`.
- **Gold (`lakehouse/gold/`)**: Precomputed cohort analytical tables (`gold_user_longitudinal_features`, `gold_cohort_lexical_matrix`) plus `gold_dataset_manifest.json` so startup metadata queries read JSON in sub-millisecond time with zero Spark jobs.
- **Storage layer (`src/lakehouse/storage.py`)**: All paths resolve from `Settings.spark_storage_uri`, supporting local filesystem paths and `s3a://` URIs via `hadoop-aws` 3.3.4 and `SimpleAWSCredentialsProvider`. In this repository, `s3a://` is exercised exclusively against a **local MinIO container (`pgsty/minio`)** over an S3-compatible API; real AWS S3 is supported by configuration but has not been exercised.

## Consequences

- **Positive**: Eliminates `collect_list` skew/OOM, unlocks partition pruning on `label` and `year`, makes data quality auditable via dead-letter quarantine, and removes startup Spark `.count()` latency.
- **Positive**: Local runs and `s3a://` MinIO object storage share the exact same `LakehousePaths` and Parquet ZSTD writers without AWS spend.
- **Negative**: Adds upfront ETL stages and requires pre-bundling ~280 MB of `hadoop-aws` and `aws-java-sdk-bundle` JARs in the Docker image so containers start offline without Maven downloads.
