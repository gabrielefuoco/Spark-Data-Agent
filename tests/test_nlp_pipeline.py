"""Spark ML tests for the native clinical bigram + TF-IDF + LDA pipeline."""

from __future__ import annotations

import pytest
from pyspark.ml.clustering import LDAModel
from pyspark.ml.feature import (
    CountVectorizerModel,
    IDFModel,
    NGram,
    RegexTokenizer,
    StopWordsRemover,
)
from pyspark.sql import SparkSession
from src.analytics.nlp_pipeline import (
    PRESERVED_CLINICAL_TOKENS,
    fit_topic_model,
    get_clinical_stopwords,
)


@pytest.mark.spark
def test_clinical_stopwords_preserve_negation_and_affect_markers(spark: SparkSession) -> None:
    _ = spark  # Ensures active SparkContext for StopWordsRemover.loadDefaultStopWords.
    stopwords = set(get_clinical_stopwords())

    for token in ("not", "no", "never", "nothing", "nobody", "worthless", "i", "me", "my"):
        assert token not in PRESERVED_CLINICAL_TOKENS or token not in stopwords
        assert token not in stopwords

    for generic in ("the", "and", "with", "from"):
        assert generic in stopwords


@pytest.mark.spark
def test_nlp_pipeline_fits_extracts_bigrams_and_valid_topic_distributions(
    spark: SparkSession,
) -> None:
    corpus = [
        ("s1", 1, "i feel alone every night and i cant sleep anymore"),
        ("s2", 1, "i feel alone because nothing matters and i feel worthless"),
        ("s3", 1, "cant sleep again feeling worthless and not worth anything"),
        ("s4", 1, "no energy left i feel alone and i cannot sleep"),
        ("s5", 1, "nobody cares and i feel worthless every single night"),
        ("s6", 1, "not worth trying when i cant sleep at night"),
        ("s7", 1, "nothing helps me sleep and i feel alone"),
        ("s8", 1, "worthless thoughts keep me awake so i cant sleep"),
        ("s9", 0, "good morning everyone great coffee and sunny weather today"),
        ("s10", 0, "enjoyed a great coffee before my happy weekend hike"),
        ("s11", 0, "good morning friends planning a happy weekend trip"),
        ("s12", 0, "sunny weather made our happy weekend bike ride awesome"),
        ("s13", 0, "fresh roasted coffee and good morning reading routine"),
        ("s14", 0, "awesome hiking trail during this sunny weather weekend"),
        ("s15", 0, "great coffee shop visit with old friends today"),
        ("s16", 0, "good morning workout followed by sunny weather walk"),
    ]
    df = spark.createDataFrame(corpus, schema="subject_id STRING, label INT, text STRING")

    result = fit_topic_model(
        df,
        k=8,
        max_iter=20,
        min_df=2.0,
        max_df=0.95,
        top_n_terms=5,
        seed=42,
    )

    expected_stage_types = (
        RegexTokenizer,
        StopWordsRemover,
        NGram,
        CountVectorizerModel,
        IDFModel,
        LDAModel,
    )
    assert len(result.model.stages) == len(expected_stage_types)
    for stage, expected_type in zip(result.model.stages, expected_stage_types, strict=True):
        assert isinstance(stage, expected_type)

    # Top terms per topic must be bigrams (contain a space) and retain clinical negation markers.
    term_rows = result.topic_terms.collect()
    assert len(term_rows) > 0
    terms = [row["term"] for row in term_rows]
    assert all(isinstance(term, str) and " " in term for term in terms)
    cv_model = result.model.stages[3]
    assert isinstance(cv_model, CountVectorizerModel)
    assert any(bigram in cv_model.vocabulary for bigram in ("feel alone", "not worth", "cant sleep"))

    # Per-document topicDistribution must be a valid probability distribution summing to ~1.
    doc_rows = result.document_topics.select("subject_id", "topicDistribution", "dominant_topic").collect()
    assert len(doc_rows) == len(corpus)
    for row in doc_rows:
        probs = row["topicDistribution"].toArray().tolist()
        assert len(probs) == 8
        assert all(p >= 0.0 for p in probs)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert 0 <= row["dominant_topic"] < 8

    # Cohort matrix aggregates by label and topic for Gold consumption.
    assert result.cohort_lexical_matrix is not None
    cohort_rows = result.cohort_lexical_matrix.collect()
    assert {r["label"] for r in cohort_rows} == {0, 1}
    assert all(r["top_terms"] for r in cohort_rows)
