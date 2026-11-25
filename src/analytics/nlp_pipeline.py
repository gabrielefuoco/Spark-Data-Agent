"""Distributed NLP and topic modeling via native Spark ML Pipeline.

Builds a pure-JVM Spark ML pipeline (no Python UDFs) over Silver posts:
``RegexTokenizer -> StopWordsRemover -> NGram(2) -> CountVectorizer -> IDF -> LDA``.
Negation and clinical affect/pronoun markers are explicitly preserved in the
stopword policy because clinical depression markers rely on phrases such as
``not worth``, ``feel alone``, or ``cant sleep``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.feature import (
    IDF,
    CountVectorizer,
    CountVectorizerModel,
    IDFModel,
    NGram,
    RegexTokenizer,
    StopWordsRemover,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType

# Clinical negation, pronoun, and affect tokens that must survive stopword removal.
PRESERVED_CLINICAL_TOKENS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "nor",
        "never",
        "nothing",
        "nobody",
        "none",
        "neither",
        "nowhere",
        "cannot",
        "cant",
        "dont",
        "wont",
        "didnt",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "worthless",
        "alone",
        "feel",
        "sleep",
        "i",
        "me",
        "my",
        "myself",
    }
)

# Contraction fragments left behind when splitting on \W+ (e.g., "can't" -> "can", "t").
_CONTRACTION_FRAGMENTS: tuple[str, ...] = ("s", "t", "d", "m", "ll", "ve", "re")


@dataclass(frozen=True)
class TopicModelResult:
    """Outputs of fitting the clinical bigram LDA pipeline on posts."""

    model: PipelineModel
    document_topics: DataFrame
    topic_terms: DataFrame
    cohort_lexical_matrix: DataFrame | None = None
    cohort_ngram_tfidf: DataFrame | None = None


def get_clinical_stopwords() -> list[str]:
    """Return English stopwords minus negation/affect markers, plus contraction fragments.

    Requires an active ``SparkContext`` because ``StopWordsRemover.loadDefaultStopWords``
    queries the JVM stopword registry.
    """
    default_words = StopWordsRemover.loadDefaultStopWords("english")
    filtered = [w for w in default_words if w.lower() not in PRESERVED_CLINICAL_TOKENS]
    seen = set(filtered)
    for fragment in _CONTRACTION_FRAGMENTS:
        if fragment not in seen and fragment not in PRESERVED_CLINICAL_TOKENS:
            filtered.append(fragment)
            seen.add(fragment)
    return filtered


def build_nlp_pipeline(
    *,
    input_col: str = "text",
    k: int = 8,
    max_iter: int = 20,
    min_df: float = 2.0,
    max_df: float = 0.95,
    vocab_size: int = 5000,
    seed: int = 42,
) -> Pipeline:
    """Construct the native Spark ML bigram + TF-IDF + LDA pipeline."""
    stopwords = get_clinical_stopwords()

    tokenizer = RegexTokenizer(
        inputCol=input_col,
        outputCol="words",
        pattern=r"\W+",
        gaps=True,
        toLowercase=True,
        minTokenLength=1,
    )
    remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words",
        stopWords=stopwords,
    )
    ngram = NGram(n=2, inputCol="filtered_words", outputCol="ngrams")
    cv = CountVectorizer(
        inputCol="ngrams",
        outputCol="raw_features",
        minDF=min_df,
        maxDF=max_df,
        vocabSize=vocab_size,
    )
    idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=1)
    lda = LDA(
        k=k,
        maxIter=max_iter,
        featuresCol="features",
        topicDistributionCol="topicDistribution",
        seed=seed,
    )

    return Pipeline(stages=[tokenizer, remover, ngram, cv, idf, lda])


def extract_topic_terms(model: PipelineModel, *, top_n: int = 10) -> DataFrame:
    """Extract top bigram terms and weights per LDA topic without Python UDFs."""
    cv_model = next(s for s in model.stages if isinstance(s, CountVectorizerModel))
    lda_model = next(s for s in model.stages if isinstance(s, LDAModel))

    spark = lda_model.describeTopics(maxTermsPerTopic=top_n).sparkSession
    vocab = cv_model.vocabulary
    vocab_df = spark.createDataFrame(
        [(idx, term) for idx, term in enumerate(vocab)],
        schema="term_index INT, term STRING",
    )

    topics_raw = lda_model.describeTopics(maxTermsPerTopic=top_n)
    exploded = topics_raw.select(
        F.col("topic").cast(IntegerType()).alias("topic"),
        F.posexplode(F.arrays_zip(F.col("termIndices"), F.col("termWeights"))).alias(
            "term_rank", "pair"
        ),
    ).select(
        F.col("topic"),
        (F.col("term_rank") + F.lit(1)).cast(IntegerType()).alias("term_rank"),
        F.col("pair.termIndices").cast(IntegerType()).alias("term_index"),
        F.col("pair.termWeights").cast(DoubleType()).alias("weight"),
    )

    return (
        exploded.join(F.broadcast(vocab_df), on="term_index", how="left")
        .select("topic", "term_rank", "term", "weight")
        .orderBy("topic", "term_rank")
    )


def build_cohort_topic_matrix(
    document_topics: DataFrame,
    topic_terms: DataFrame,
) -> DataFrame:
    """Aggregate per-document LDA distributions into a Gold-ready cohort matrix."""
    topic_summary = (
        document_topics.select(
            "label",
            F.posexplode(vector_to_array(F.col("topicDistribution"))).alias("topic", "topic_prob"),
        )
        .groupBy(
            F.col("label").cast(IntegerType()).alias("label"),
            F.col("topic").cast(IntegerType()).alias("topic"),
        )
        .agg(
            F.avg("topic_prob").cast(DoubleType()).alias("mean_topic_probability"),
            F.count("*").cast(LongType()).alias("document_count"),
        )
    )

    topic_labels = topic_terms.groupBy("topic").agg(
        F.concat_ws(
            ", ",
            F.expr("transform(sort_array(collect_list(struct(term_rank, term))), x -> x.term)"),
        )
        .cast(StringType())
        .alias("top_terms")
    )

    return (
        topic_summary.join(F.broadcast(topic_labels), on="topic", how="left")
        .select("label", "topic", "mean_topic_probability", "document_count", "top_terms")
        .orderBy("label", "topic")
    )


def _idf_table_from_models(
    spark_df: DataFrame,
    cv_model: CountVectorizerModel,
    idf_model: IDFModel,
    *,
    ngram_type: str,
) -> DataFrame:
    """Materialize ``(ngram_type, ngram, idf)`` from fitted Spark ML models."""
    vocab = cv_model.vocabulary
    idf_values = idf_model.idf.toArray().tolist()
    rows = [
        (ngram_type, str(term), float(weight))
        for term, weight in zip(vocab, idf_values, strict=True)
    ]
    return spark_df.sparkSession.createDataFrame(
        rows,
        schema="ngram_type STRING, ngram STRING, idf DOUBLE",
    )


def build_cohort_ngram_tfidf_matrix(
    transformed_df: DataFrame,
    model: PipelineModel,
    *,
    min_df: float = 2.0,
    max_df: float = 0.95,
    vocab_size: int = 5000,
    top_n_ngrams: int | None = 100,
) -> DataFrame:
    """Compute per-cohort unigram/bigram counts, relative frequencies, and TF-IDF weights."""
    bigram_cv = next(s for s in model.stages if isinstance(s, CountVectorizerModel))
    bigram_idf = next(s for s in model.stages if isinstance(s, IDFModel))
    bigram_idf_df = _idf_table_from_models(
        transformed_df, bigram_cv, bigram_idf, ngram_type="bigram"
    )

    try:
        uni_cv = CountVectorizer(
            inputCol="filtered_words",
            outputCol="uni_raw_features",
            minDF=min_df,
            maxDF=max_df,
            vocabSize=vocab_size,
        ).fit(transformed_df)
        uni_idf = IDF(inputCol="uni_raw_features", outputCol="uni_features", minDocFreq=1).fit(
            uni_cv.transform(transformed_df)
        )
        unigram_idf_df = _idf_table_from_models(
            transformed_df, uni_cv, uni_idf, ngram_type="unigram"
        )
    except Exception:
        # Empty unigram vocabulary under strict minDF/maxDF falls back to 0.0 IDF.
        unigram_idf_df = transformed_df.sparkSession.createDataFrame(
            [], schema="ngram_type STRING, ngram STRING, idf DOUBLE"
        )

    idf_lookup = unigram_idf_df.unionByName(bigram_idf_df)

    doc_counts = transformed_df.groupBy(F.col("label").cast(IntegerType()).alias("label")).agg(
        F.count("*").cast(LongType()).alias("cohort_doc_count")
    )

    unigrams = transformed_df.select(
        F.col("label").cast(IntegerType()).alias("label"),
        F.lit("unigram").alias("ngram_type"),
        F.explode("filtered_words").alias("ngram"),
    )
    bigrams = transformed_df.select(
        F.col("label").cast(IntegerType()).alias("label"),
        F.lit("bigram").alias("ngram_type"),
        F.explode("ngrams").alias("ngram"),
    )

    cohort_window = Window.partitionBy("label", "ngram_type")
    counts_df = (
        unigrams.unionByName(bigrams)
        .filter(F.col("ngram").isNotNull() & (F.col("ngram") != F.lit("")))
        .groupBy("label", "ngram_type", "ngram")
        .agg(F.count("*").cast(LongType()).alias("frequency"))
        .withColumn(
            "relative_frequency",
            (F.col("frequency") / F.sum("frequency").over(cohort_window)).cast(DoubleType()),
        )
    )

    rank_window = Window.partitionBy("label", "ngram_type").orderBy(
        F.col("tfidf_weight").desc(),
        F.col("frequency").desc(),
        F.col("ngram").asc(),
    )
    matrix_df = (
        counts_df.join(F.broadcast(idf_lookup), on=["ngram_type", "ngram"], how="left")
        .join(F.broadcast(doc_counts), on="label", how="left")
        .withColumn("idf_weight", F.coalesce(F.col("idf"), F.lit(0.0)).cast(DoubleType()))
        .withColumn(
            "tfidf_weight",
            (
                F.col("frequency").cast(DoubleType())
                * F.col("idf_weight")
                / F.greatest(F.col("cohort_doc_count"), F.lit(1)).cast(DoubleType())
            ).cast(DoubleType()),
        )
        .withColumn("tfidf_rank", F.row_number().over(rank_window).cast(IntegerType()))
        .select(
            "label",
            "ngram_type",
            "ngram",
            "frequency",
            "relative_frequency",
            "idf_weight",
            "tfidf_weight",
            "tfidf_rank",
        )
    )

    if top_n_ngrams is not None:
        matrix_df = matrix_df.filter(F.col("tfidf_rank") <= F.lit(top_n_ngrams))

    return matrix_df.orderBy("label", "ngram_type", "tfidf_rank")


def fit_topic_model(
    posts_df: DataFrame,
    *,
    input_col: str = "text",
    k: int = 8,
    max_iter: int = 20,
    min_df: float = 2.0,
    max_df: float = 0.95,
    vocab_size: int = 5000,
    top_n_terms: int = 10,
    top_n_ngrams: int | None = 100,
    include_ngram_tfidf: bool = False,
    seed: int = 42,
) -> TopicModelResult:
    """Fit the Spark ML LDA pipeline and return model, document topics, and top terms."""
    prepared_df = posts_df.withColumn(
        input_col,
        F.coalesce(F.col(input_col), F.lit("")),
    )

    pipeline = build_nlp_pipeline(
        input_col=input_col,
        k=k,
        max_iter=max_iter,
        min_df=min_df,
        max_df=max_df,
        vocab_size=vocab_size,
        seed=seed,
    )
    model = pipeline.fit(prepared_df)
    transformed = model.transform(prepared_df)

    prob_array = vector_to_array(F.col("topicDistribution"))

    # Drop intermediate feature vectors/token arrays unless the caller needs raw columns.
    drop_cols = [
        c
        for c in ("words", "filtered_words", "ngrams", "raw_features", "features")
        if c in transformed.columns
    ]
    document_topics = (
        transformed.drop(*drop_cols)
        .withColumn("topic_probabilities", prob_array)
        .withColumn(
            "dominant_topic_prob", F.array_max(F.col("topic_probabilities")).cast(DoubleType())
        )
        .withColumn(
            "dominant_topic",
            (F.expr("array_position(topic_probabilities, dominant_topic_prob)") - F.lit(1)).cast(
                IntegerType()
            ),
        )
    )

    topic_terms = extract_topic_terms(model, top_n=top_n_terms)
    has_label = "label" in document_topics.columns
    cohort_matrix = build_cohort_topic_matrix(document_topics, topic_terms) if has_label else None
    cohort_ngram_tfidf = (
        build_cohort_ngram_tfidf_matrix(
            transformed,
            model,
            min_df=min_df,
            max_df=max_df,
            vocab_size=vocab_size,
            top_n_ngrams=top_n_ngrams,
        )
        if include_ngram_tfidf and has_label
        else None
    )

    return TopicModelResult(
        model=model,
        document_topics=document_topics,
        topic_terms=topic_terms,
        cohort_lexical_matrix=cohort_matrix,
        cohort_ngram_tfidf=cohort_ngram_tfidf,
    )
