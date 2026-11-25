"""Gold layer: precomputed analytical tables and dataset manifest.

Materializes per-subject longitudinal metrics (via ``features.py``), cohort-level
n-gram TF-IDF frequencies and LDA topic distributions (via ``nlp_pipeline.py``)
as ZSTD Parquet, alongside ``gold_dataset_manifest.json`` with storage volumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.fs as pafs
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel

from src.analytics.features import (
    DEFAULT_SESSION_GAP_SECONDS,
    DEFAULT_WINDOW_DAYS,
    summarize_user_longitudinal_features,
)
from src.analytics.nlp_pipeline import fit_topic_model
from src.config import Settings
from src.lakehouse.manifest import (
    DatasetManifest,
    build_dataset_manifest,
    resolve_manifest_filesystem,
)
from src.lakehouse.paths import LakehousePaths, join_storage_path
from src.spark import write_parquet_zstd


def _sibling_topics_path(lexical_output_path: str | Path) -> str:
    raw = str(lexical_output_path).rstrip("/")
    parent: str | Path = raw.rsplit("/", 1)[0] if "://" in raw else Path(raw).parent
    return join_storage_path(parent, "gold_cohort_topic_matrix")


def _measure_uri_bytes(uri: str, *, settings: Settings | None = None) -> int:
    """Sum byte sizes of all files under a local path or ``s3a://`` prefix without Spark."""
    fs, fs_path = resolve_manifest_filesystem(uri, settings=settings)
    info = fs.get_file_info(fs_path)
    if info.type == pafs.FileType.NotFound:
        return 0
    if info.type == pafs.FileType.File:
        return int(info.size or 0)
    entries = fs.get_file_info(pafs.FileSelector(fs_path, recursive=True, allow_not_found=True))
    return sum(int(entry.size or 0) for entry in entries if entry.type == pafs.FileType.File)


def compute_lakehouse_storage_volumes(
    paths: LakehousePaths,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Compute byte volumes per table and Medallion layer without invoking Spark."""
    table_specs: tuple[tuple[str, str, str], ...] = (
        ("bronze", "bronze_posts", paths.bronze_posts),
        ("bronze", "bronze_subjects", paths.bronze_subjects),
        ("silver", "silver_posts", paths.silver_posts),
        ("silver", "silver_quarantine", paths.silver_quarantine),
        ("gold", "gold_user_longitudinal_features", paths.gold_user_longitudinal_features),
        ("gold", "gold_cohort_lexical_matrix", paths.gold_cohort_lexical_matrix),
        ("gold", "gold_cohort_topic_matrix", paths.gold_cohort_topic_matrix),
    )

    by_table: dict[str, int] = {}
    by_layer: dict[str, int] = {"bronze": 0, "silver": 0, "gold": 0}
    for layer, name, uri in table_specs:
        size_bytes = _measure_uri_bytes(uri, settings=settings)
        by_table[name] = size_bytes
        by_layer[layer] += size_bytes

    return {
        "total_bytes": sum(by_layer.values()),
        "by_layer": by_layer,
        "by_table": by_table,
    }


def _write_manifest_with_volumes(
    manifest: DatasetManifest,
    output_path: str | Path,
    storage_volumes: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> DatasetManifest:
    """Write ``gold_dataset_manifest.json`` including precomputed storage volumes.

    Returns the augmented manifest rather than nothing: the object a caller gets
    back has to equal the one a later ``load_manifest()`` reads, otherwise the
    in-memory and on-disk versions drift apart and nothing notices.
    """
    payload = manifest.model_dump(mode="json")
    payload["storage_volumes"] = storage_volumes
    augmented = DatasetManifest.model_validate(payload)
    raw_bytes = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    fs, fs_path = resolve_manifest_filesystem(output_path, settings=settings)
    if isinstance(fs, pafs.LocalFileSystem):
        parent = str(Path(fs_path).parent)
        if parent not in {"", "."}:
            fs.create_dir(parent, recursive=True)
    with fs.open_output_stream(fs_path) as stream:
        stream.write(raw_bytes)
    return augmented


def build_user_longitudinal_features(
    silver_posts_df: DataFrame,
    output_path: str | Path,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    session_gap_seconds: int = DEFAULT_SESSION_GAP_SECONDS,
) -> DataFrame:
    """Aggregate per-subject behavioral, circadian, and verbosity metrics via ``features.py``."""
    features_df = summarize_user_longitudinal_features(
        silver_posts_df,
        window_days=window_days,
        session_gap_seconds=session_gap_seconds,
    )
    write_parquet_zstd(features_df, output_path, mode="overwrite")
    return silver_posts_df.sparkSession.read.parquet(str(output_path))


def build_cohort_lexical_matrix(
    silver_posts_df: DataFrame,
    output_path: str | Path,
    *,
    topics_output_path: str | Path | None = None,
    k: int = 8,
    max_iter: int = 20,
    min_df: float = 2.0,
    max_df: float = 0.95,
    vocab_size: int = 5000,
    top_n_ngrams: int | None = None,
    top_n_terms: int = 10,
    seed: int = 42,
) -> DataFrame:
    """Compute cohort unigram/bigram TF-IDF weights and LDA topic distributions via ``nlp_pipeline.py``."""
    combined = silver_posts_df.select(
        "label",
        F.concat_ws(" ", F.coalesce(F.col("title"), F.lit("")), F.col("text")).alias(
            "combined_text"
        ),
    )

    result = fit_topic_model(
        combined,
        input_col="combined_text",
        k=k,
        max_iter=max_iter,
        min_df=min_df,
        max_df=max_df,
        vocab_size=vocab_size,
        top_n_terms=top_n_terms,
        top_n_ngrams=top_n_ngrams,
        include_ngram_tfidf=True,
        seed=seed,
    )

    if result.cohort_ngram_tfidf is None or result.cohort_lexical_matrix is None:
        raise ValueError("build_cohort_lexical_matrix requires a 'label' column on silver_posts_df.")

    write_parquet_zstd(result.cohort_ngram_tfidf, output_path, mode="overwrite")
    topics_target = topics_output_path or _sibling_topics_path(output_path)
    write_parquet_zstd(result.cohort_lexical_matrix, topics_target, mode="overwrite")

    return silver_posts_df.sparkSession.read.parquet(str(output_path))


def build_gold(
    spark: SparkSession,
    *,
    silver_posts_path: str | Path | None = None,
    silver_quarantine_path: str | Path | None = None,
    settings: Settings | None = None,
    paths: LakehousePaths | None = None,
    lda_k: int = 8,
    lda_max_iter: int = 20,
    min_df: float = 2.0,
    top_n_ngrams: int | None = None,
    top_n_terms: int = 10,
) -> tuple[DataFrame, DataFrame, DatasetManifest]:
    """Materialize all Gold tables and the static dataset manifest with storage volumes."""
    resolved = paths or LakehousePaths.from_settings(settings)
    silver_df = spark.read.parquet(str(silver_posts_path or resolved.silver_posts))
    quarantine_df = spark.read.parquet(str(silver_quarantine_path or resolved.silver_quarantine))

    # Reused across longitudinal features, lexical/topic matrices, and manifest aggregations.
    silver_df.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        build_user_longitudinal_features(silver_df, resolved.gold_user_longitudinal_features)
        build_cohort_lexical_matrix(
            silver_df,
            resolved.gold_cohort_lexical_matrix,
            topics_output_path=resolved.gold_cohort_topic_matrix,
            k=lda_k,
            max_iter=lda_max_iter,
            min_df=min_df,
            top_n_ngrams=top_n_ngrams,
            top_n_terms=top_n_terms,
        )
        manifest = build_dataset_manifest(
            silver_df,
            quarantine_df,
            resolved.gold_dataset_manifest,
            settings=settings,
        )
    finally:
        silver_df.unpersist()

    storage_volumes = compute_lakehouse_storage_volumes(resolved, settings=settings)
    manifest = _write_manifest_with_volumes(
        manifest,
        resolved.gold_dataset_manifest,
        storage_volumes,
        settings=settings,
    )

    return (
        spark.read.parquet(resolved.gold_user_longitudinal_features),
        spark.read.parquet(resolved.gold_cohort_lexical_matrix),
        manifest,
    )
