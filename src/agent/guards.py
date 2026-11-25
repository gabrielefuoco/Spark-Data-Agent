"""Deterministic, non-LLM statistical significance guard and critic constraints.

Evaluates query results before the scientific critic LLM synthesizes an answer.
Classifies each result purely from its data shape into one of three modes:

1. ``descriptive``: Not a two-group comparison of a continuous per-subject
   metric — including lexical/n-gram frequency tables, topic distributions,
   single counts, aggregate cohort counts, single-cohort slices, or results
   without a numeric metric. No hypothesis test is run and ``p_value`` /
   ``effect_size_cohens_d`` are ``None``.
2. ``cannot_assess``: A two-group cohort comparison of a continuous metric was
   attempted (both ``label=1`` and ``label=0`` rows are present with a numeric
   metric that is not a descriptive table), but ``subject_id`` is missing or
   empty. No hypothesis test is run and ``p_value`` / ``effect_size_cohens_d``
   are ``None``.
3. ``assessed``: A valid two-group comparison with ``subject_id``, ``label``,
   and a continuous numeric metric. Collapses multiple observations per
   ``subject_id`` into a per-subject arithmetic mean to prevent
   pseudo-replication, enforces minimum unique-subject sample sizes from
   ``Settings.min_subjects_per_class``, chooses between Welch's t-test and
   Mann-Whitney U via normality screening, computes Cohen's d, and guards
   degenerate inputs against NaN/inf or silent passes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy import stats

from src.config import Settings

ConfidenceLevel = Literal["high", "medium", "low", "underpowered", "cannot_assess", "descriptive"]
TestName = Literal["welch_t", "mann_whitney_u", "none"]
AggregationMethod = Literal["subject_mean", "none"]
AssessmentStatus = Literal["assessed", "cannot_assess", "descriptive"]

CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS = (
    "MANDATORY SCIENTIFIC CRITIC CONSTRAINTS:\n"
    "1. Answer in the SAME language as the user's question (e.g., Italian if the user "
    "asked in Italian, English if asked in English), even though these instructions are in English.\n"
    "2. If `assessment_status` is 'cannot_assess' OR `assessment_status` is 'descriptive' OR "
    "`is_significant` is false OR `is_underpowered` is true, you MUST NOT claim any cohort "
    "difference, correlation, or clinical signal associated with depression.\n"
    "3. When `assessment_status` is 'assessed', every comparative statement MUST explicitly cite "
    "the unique-subject sample sizes (`sample_size_depressed` and `sample_size_control`), raw "
    "observation counts (`observation_count_depressed` and `observation_count_control`), "
    "`aggregation_method`, `p_value`, and effect size `effect_size_cohens_d`.\n"
    "4. When `assessment_status` is 'descriptive', DO NOT refuse to answer and DO NOT invent "
    "p-values or Cohen's d: present the descriptive result directly (such as top terms/n-grams "
    "with their weights or frequencies, LDA topics with their top terms, counts, or summary rows), "
    "while explicitly noting that these are descriptive statistics without subject-level hypothesis testing.\n"
    "5. When `assessment_status` is 'cannot_assess', state `cannot_assess_reason` (or its "
    "meaning in the user's language) and explain that statistical significance cannot be "
    "evaluated without a `subject_id` dimension; do NOT fabricate p-values or Cohen's d.\n"
    "6. When the sample is underpowered, state `underpowered_reason` verbatim "
    "and frame any observed numeric difference as inconclusive."
)

_NORMALITY_ALPHA = 0.05
_SHAPIRO_MAX_N = 5000

_EXCLUDED_METRIC_COLUMNS = frozenset(
    {"label", "year", "subject_id", "first_post_at", "last_post_at", "date", "raw_date"}
)
_LEXICAL_TABLE_COLUMNS = frozenset(
    {
        "ngram",
        "ngram_type",
        "unigram",
        "bigram",
        "tfidf_weight",
        "tfidf_rank",
        "idf_weight",
        "relative_frequency",
        "frequency",
    }
)
_TOPIC_TABLE_COLUMNS = frozenset(
    {
        "topic",
        "top_terms",
        "mean_topic_probability",
        "term_rank",
        "topic_prob",
        "dominant_topic",
        "dominant_topic_prob",
    }
)
_DESCRIPTIVE_COUNT_COLUMNS = frozenset(
    {
        "count",
        "count(1)",
        "count(*)",
        "cnt",
        "row_count",
        "document_count",
        "subject_count",
        "total_posts",
        "total_subjects",
        "n_posts",
        "num_posts",
    }
)
_AGGREGATE_PREFIXES = ("count(", "sum(", "avg(", "min(", "max(")


def _validate_finite_float(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("All metric values must be finite numbers (no NaN or inf).")
    return number


class SubjectObservation(BaseModel):
    """Single numeric observation attributed to an identified subject."""

    subject_id: str
    value: float

    @field_validator("subject_id")
    @classmethod
    def check_subject_id(cls, subject_id: str) -> str:
        """Normalize and reject blank subject identifiers."""
        cleaned = subject_id.strip()
        if not cleaned:
            raise ValueError("subject_id must be a non-empty string.")
        return cleaned

    @field_validator("value")
    @classmethod
    def check_finite(cls, value: float) -> float:
        """Reject NaN/inf early so downstream SciPy calls remain deterministic."""
        return _validate_finite_float(value)


class StatisticalComparisonInput(BaseModel):
    """Input payload for comparing a continuous metric across depressed and control cohorts."""

    depressed_observations: list[SubjectObservation] = Field(default_factory=list)
    control_observations: list[SubjectObservation] = Field(default_factory=list)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_subjects_per_class: int | None = Field(default=None, ge=1)
    metric_name: str = "metric"
    has_subject_dimension: bool = True
    raw_observation_count_depressed: int | None = Field(default=None, ge=0)
    raw_observation_count_control: int | None = Field(default=None, ge=0)


class StatisticalGuardResult(BaseModel):
    """Deterministic verdict emitted by ``evaluate_statistical_guard``."""

    sample_size_depressed: int
    sample_size_control: int
    observation_count_depressed: int = 0
    observation_count_control: int = 0
    aggregation_method: AggregationMethod = "subject_mean"
    assessment_status: AssessmentStatus = "assessed"
    cannot_assess_reason: str | None = None
    descriptive_reason: str | None = None
    p_value: float | None = None
    is_significant: bool = False
    effect_size_cohens_d: float | None = None
    statistical_confidence: ConfidenceLevel
    is_underpowered: bool = False
    underpowered_reason: str | None = None
    test_used: TestName = "none"
    normality_passed: bool | None = None
    alpha: float = 0.05
    metric_name: str = "metric"


def _as_finite_float(value: object) -> float | None:
    """Return ``value`` as a float when it is a finite real number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _find_column_key(candidate_cols: Sequence[str], target_lower: str) -> str | None:
    for col in candidate_cols:
        if col.lower() == target_lower:
            return col
    return None


def _make_non_assessed_result(
    *,
    status: Literal["cannot_assess", "descriptive"],
    reason: str,
    metric_name: str,
    alpha: float = 0.05,
    obs_count_dep: int = 0,
    obs_count_ctrl: int = 0,
) -> StatisticalGuardResult:
    """Build a guard verdict with no hypothesis test and ``None`` p-value / Cohen's d."""
    return StatisticalGuardResult(
        sample_size_depressed=0,
        sample_size_control=0,
        observation_count_depressed=obs_count_dep,
        observation_count_control=obs_count_ctrl,
        aggregation_method="none",
        assessment_status=status,
        cannot_assess_reason=reason if status == "cannot_assess" else None,
        descriptive_reason=reason if status == "descriptive" else None,
        p_value=None,
        is_significant=False,
        effect_size_cohens_d=None,
        statistical_confidence=status,
        is_underpowered=False,
        underpowered_reason=None,
        test_used="none",
        normality_passed=None,
        alpha=alpha,
        metric_name=metric_name,
    )


def _aggregate_by_subject(observations: Sequence[SubjectObservation]) -> np.ndarray:
    """Collapse multiple observations per ``subject_id`` into one arithmetic mean per subject."""
    if not observations:
        return np.asarray([], dtype=np.float64)

    grouped: dict[str, list[float]] = defaultdict(list)
    for obs in observations:
        grouped[obs.subject_id].append(obs.value)

    subject_means = [
        vals[0] if len(vals) == 1 else math.fsum(vals) / len(vals) for vals in grouped.values()
    ]
    return np.asarray(subject_means, dtype=np.float64)


def _passes_normality(values: np.ndarray) -> bool:
    """Screen a sample for approximate normality without triggering constant-array warnings."""
    n = values.size
    if n < 3 or float(np.ptp(values)) == 0.0:
        return False
    if n <= _SHAPIRO_MAX_N:
        _, p_val = stats.shapiro(values)
    else:
        # Shapiro-Wilk loses calibration above 5,000 samples; D'Agostino-Pearson scales cleanly.
        _, p_val = stats.normaltest(values)
    return bool(math.isfinite(float(p_val)) and float(p_val) >= _NORMALITY_ALPHA)


def _compute_cohens_d(depressed: np.ndarray, control: np.ndarray) -> float:
    """Compute Cohen's d using pooled sample standard deviation, guarded against zero variance."""
    n1 = depressed.size
    n0 = control.size
    if n1 < 2 or n0 < 2:
        return 0.0

    var1 = float(np.var(depressed, ddof=1))
    var0 = float(np.var(control, ddof=1))
    pooled_var = ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)
    if pooled_var <= 0.0 or not math.isfinite(pooled_var):
        return 0.0

    d = (float(np.mean(depressed)) - float(np.mean(control))) / math.sqrt(pooled_var)
    return d if math.isfinite(d) else 0.0


def _classify_confidence(
    *,
    is_underpowered: bool,
    is_significant: bool,
    p_value: float,
    cohens_d: float,
) -> ConfidenceLevel:
    if is_underpowered:
        return "underpowered"
    if not is_significant:
        return "low"
    if p_value < 0.01 and abs(cohens_d) >= 0.5:
        return "high"
    return "medium"


def evaluate_statistical_guard(
    depressed_observations: Sequence[SubjectObservation],
    control_observations: Sequence[SubjectObservation],
    *,
    alpha: float = 0.05,
    min_subjects_per_class: int | None = None,
    settings: Settings | None = None,
    metric_name: str = "metric",
    has_subject_dimension: bool = True,
    raw_observation_count_depressed: int | None = None,
    raw_observation_count_control: int | None = None,
) -> StatisticalGuardResult:
    """Aggregate per-subject observations and run sample-power + hypothesis testing.

    Pure Python/NumPy/SciPy implementation with no Spark dependency so the
    LangGraph guard node (ticket 05) can call it directly on query results.

    Subject-level aggregation:
        Multiple observations sharing the same ``subject_id`` within a cohort are
        collapsed via arithmetic mean before counting ``sample_size_*`` and
        running hypothesis tests, preventing row-level pseudo-replication.

    Test-selection rule:
        1. If ``has_subject_dimension`` is ``False``, refuse to test and return an
           explicit ``assessment_status="cannot_assess"`` verdict.
        2. If either group has < 2 unique subjects or all per-subject means across
           both groups are identical, skip testing and return ``p_value=1.0``.
        3. Screen each group for normality (Shapiro-Wilk for n <= 5000,
           D'Agostino-Pearson above 5000; constant groups fail screening).
        4. Use Welch's t-test when both groups pass normality (p >= 0.05);
           otherwise use two-sided Mann-Whitney U.

    Underpowered policy:
        If either class has fewer than ``Settings.min_subjects_per_class`` unique
        subjects (default 30), ``is_significant`` is forced to ``False``
        regardless of the hypothesis test's p-value.
    """
    validated = StatisticalComparisonInput(
        depressed_observations=list(depressed_observations),
        control_observations=list(control_observations),
        alpha=alpha,
        min_subjects_per_class=min_subjects_per_class,
        metric_name=metric_name,
        has_subject_dimension=has_subject_dimension,
        raw_observation_count_depressed=raw_observation_count_depressed,
        raw_observation_count_control=raw_observation_count_control,
    )

    obs_count_dep = (
        validated.raw_observation_count_depressed
        if validated.raw_observation_count_depressed is not None
        else len(validated.depressed_observations)
    )
    obs_count_ctrl = (
        validated.raw_observation_count_control
        if validated.raw_observation_count_control is not None
        else len(validated.control_observations)
    )

    if not validated.has_subject_dimension:
        return _make_non_assessed_result(
            status="cannot_assess",
            reason=(
                f"cannot assess: no subject dimension for '{validated.metric_name}' "
                "(missing `subject_id` column in query results)."
            ),
            metric_name=validated.metric_name,
            alpha=validated.alpha,
            obs_count_dep=obs_count_dep,
            obs_count_ctrl=obs_count_ctrl,
        )

    if validated.min_subjects_per_class is not None:
        threshold = validated.min_subjects_per_class
    elif settings is not None:
        threshold = settings.min_subjects_per_class
    else:
        # pydantic-settings honours `_env_file` at runtime, but the `__init__`
        # signature Pydantic generates only exposes the model fields.
        threshold = Settings(_env_file=None).min_subjects_per_class  # type: ignore[call-arg]

    dep = _aggregate_by_subject(validated.depressed_observations)
    ctrl = _aggregate_by_subject(validated.control_observations)
    n1 = int(dep.size)
    n0 = int(ctrl.size)

    is_underpowered = n1 < threshold or n0 < threshold
    underpowered_reason = (
        f"Underpowered sample for '{validated.metric_name}': "
        f"depressed={n1} unique subjects, control={n0} unique subjects; "
        f"minimum required per class is {threshold}."
        if is_underpowered
        else None
    )

    cohens_d = _compute_cohens_d(dep, ctrl)

    all_identical = (
        n1 > 0
        and n0 > 0
        and float(np.ptp(dep)) == 0.0
        and float(np.ptp(ctrl)) == 0.0
        and float(dep[0]) == float(ctrl[0])
    )
    if n1 < 2 or n0 < 2 or all_identical:
        p_value = 1.0
        test_used: TestName = "none"
        normality_passed: bool | None = None
    else:
        normality_passed = _passes_normality(dep) and _passes_normality(ctrl)
        if normality_passed:
            _, raw_p = stats.ttest_ind(dep, ctrl, equal_var=False)
            test_used = "welch_t"
        else:
            _, raw_p = stats.mannwhitneyu(dep, ctrl, alternative="two-sided")
            test_used = "mann_whitney_u"
        p_value = float(raw_p) if math.isfinite(float(raw_p)) else 1.0

    is_significant = (not is_underpowered) and (test_used != "none") and (p_value < validated.alpha)
    confidence = _classify_confidence(
        is_underpowered=is_underpowered,
        is_significant=is_significant,
        p_value=p_value,
        cohens_d=cohens_d,
    )

    return StatisticalGuardResult(
        sample_size_depressed=n1,
        sample_size_control=n0,
        observation_count_depressed=obs_count_dep,
        observation_count_control=obs_count_ctrl,
        aggregation_method="subject_mean",
        assessment_status="assessed",
        cannot_assess_reason=None,
        p_value=p_value,
        is_significant=is_significant,
        effect_size_cohens_d=cohens_d,
        statistical_confidence=confidence,
        is_underpowered=is_underpowered,
        underpowered_reason=underpowered_reason,
        test_used=test_used,
        normality_passed=normality_passed,
        alpha=validated.alpha,
        metric_name=validated.metric_name,
    )


def _resolve_numeric_metric_column(
    rows: Sequence[dict[str, Any]],
    candidate_cols: Sequence[str],
    preferred_metric: str | None,
) -> str | None:
    if preferred_metric:
        preferred_key = _find_column_key(candidate_cols, preferred_metric.lower())
        if (
            preferred_key
            and preferred_key.lower() not in _EXCLUDED_METRIC_COLUMNS
            and any(_as_finite_float(r.get(preferred_key)) is not None for r in rows)
        ):
            return preferred_key
    for col in candidate_cols:
        if col.lower() in _EXCLUDED_METRIC_COLUMNS:
            continue
        if any(_as_finite_float(r.get(col)) is not None for r in rows):
            return col
    return None


def evaluate_query_result_guard(
    rows: Sequence[dict[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    preferred_metric: str | None = None,
    settings: Settings | None = None,
) -> StatisticalGuardResult:
    """Classify a SQL/Spark job result by its data shape and evaluate the guard.

    Decision rule (inspected purely from result columns and rows, never from
    upstream node or intent names):
    1. Empty rows -> ``descriptive`` (``descriptive: empty_result``).
    2. Lexical/n-gram frequency table (any column in ``_LEXICAL_TABLE_COLUMNS``)
       or topic distribution table (any column in ``_TOPIC_TABLE_COLUMNS``) ->
       ``descriptive``, even when ``label`` and numeric weights are present,
       because these are cohort-level descriptive summaries rather than
       per-subject metrics.
    3. Missing ``label`` column or missing numeric metric column -> ``descriptive``
       (covers single counts like ``COUNT(*)``, time-series/year summaries, and
       text/catalog listings).
    4. When ``subject_id`` is absent:
       - If the numeric metric is a cohort-level aggregate/count (column in
         ``_DESCRIPTIVE_COUNT_COLUMNS`` or SQL aggregate ``count(``/``sum(``/``avg(``)
         or only one cohort label in ``{0, 1}`` is present -> ``descriptive``.
       - If both ``label=1`` and ``label=0`` rows are present for a continuous
         metric without ``subject_id`` -> ``cannot_assess`` (a two-group
         comparison was attempted without a subject dimension).
    5. When ``subject_id``, ``label`` (both cohorts), and a continuous metric
       are present -> ``assessed`` via ``evaluate_statistical_guard``.
    """
    if not rows:
        return _make_non_assessed_result(
            status="descriptive",
            reason="descriptive: empty_result",
            metric_name=preferred_metric or "descriptive_metric",
        )

    candidate_cols: list[str] = list(columns or [])
    for row in rows:
        for key in row:
            if key not in candidate_cols:
                candidate_cols.append(key)

    cols_lower = {col.lower() for col in candidate_cols}
    if cols_lower & _LEXICAL_TABLE_COLUMNS:
        return _make_non_assessed_result(
            status="descriptive",
            reason="descriptive: lexical_frequency_table",
            metric_name=preferred_metric or "lexical_ngram",
        )

    if cols_lower & _TOPIC_TABLE_COLUMNS:
        return _make_non_assessed_result(
            status="descriptive",
            reason="descriptive: topic_distribution_table",
            metric_name=preferred_metric or "topic_distribution",
        )

    metric_col = _resolve_numeric_metric_column(rows, candidate_cols, preferred_metric)
    label_col = _find_column_key(candidate_cols, "label")
    if label_col is None:
        return _make_non_assessed_result(
            status="descriptive",
            reason="descriptive: no_cohort_label_column",
            metric_name=metric_col or preferred_metric or "descriptive_metric",
        )

    if metric_col is None:
        return _make_non_assessed_result(
            status="descriptive",
            reason="descriptive: no_numeric_metric_column",
            metric_name=preferred_metric or "descriptive_metric",
        )

    subject_col = _find_column_key(candidate_cols, "subject_id")
    metric_lower = metric_col.lower()
    if subject_col is None and (
        metric_lower in _DESCRIPTIVE_COUNT_COLUMNS or metric_lower.startswith(_AGGREGATE_PREFIXES)
    ):
        return _make_non_assessed_result(
            status="descriptive",
            reason="descriptive: cohort_aggregate_metric",
            metric_name=metric_col,
        )

    depressed: list[SubjectObservation] = []
    control: list[SubjectObservation] = []
    raw_dep_count = 0
    raw_ctrl_count = 0

    for row in rows:
        raw_label = row.get(label_col)
        val = _as_finite_float(row.get(metric_col))
        if raw_label is None or val is None:
            continue
        try:
            label_int = int(raw_label)
        except (TypeError, ValueError):
            continue
        if label_int not in (0, 1):
            continue

        if label_int == 1:
            raw_dep_count += 1
        else:
            raw_ctrl_count += 1

        if subject_col is None:
            continue
        raw_subject = row.get(subject_col)
        if raw_subject is None:
            continue
        subject_id = str(raw_subject).strip()
        if not subject_id:
            continue

        obs = SubjectObservation(subject_id=subject_id, value=val)
        if label_int == 1:
            depressed.append(obs)
        else:
            control.append(obs)

    if raw_dep_count == 0 or raw_ctrl_count == 0:
        return _make_non_assessed_result(
            status="descriptive",
            reason="descriptive: single_cohort_only",
            metric_name=metric_col,
            obs_count_dep=raw_dep_count,
            obs_count_ctrl=raw_ctrl_count,
        )

    has_subject_dimension = (subject_col is not None) and (len(depressed) + len(control) > 0)
    return evaluate_statistical_guard(
        depressed,
        control,
        settings=settings,
        metric_name=metric_col,
        has_subject_dimension=has_subject_dimension,
        raw_observation_count_depressed=raw_dep_count,
        raw_observation_count_control=raw_ctrl_count,
    )


def format_critic_guard_instructions(result: StatisticalGuardResult) -> str:
    """Format the guard verdict and mandatory constraints for the scientific critic prompt."""
    if result.assessment_status == "descriptive":
        claim_policy = (
            "DESCRIPTIVE_ONLY: Present the descriptive results directly and clearly in the "
            "user's language (top n-grams with weights/frequencies, LDA topics with top terms, "
            "counts, or summary rows). Do NOT refuse to answer and do NOT fabricate p-values or "
            "Cohen's d. FORBIDDEN: Do NOT claim any statistically significant cohort difference, "
            "correlation, or clinical signal from descriptive results."
        )
    elif result.assessment_status == "cannot_assess":
        claim_policy = (
            "FORBIDDEN: Cannot assess statistical significance (no subject dimension). "
            "Do NOT claim any depression correlation or class difference; state in the "
            "user's language that subject-level significance cannot be evaluated without "
            "`subject_id`, and do NOT fabricate p-values or Cohen's d."
        )
    elif result.is_significant and not result.is_underpowered:
        claim_policy = (
            "ALLOWED: Statistical significance gate passed; report findings strictly "
            "scoped to the observed metric."
        )
    else:
        claim_policy = (
            "FORBIDDEN: Do NOT claim any depression correlation or class difference "
            "because the result is underpowered or not statistically significant."
        )

    p_display = f"{result.p_value:.6g}" if result.p_value is not None else "N/A (not tested)"
    d_display = (
        f"{result.effect_size_cohens_d:.4f}"
        if result.effect_size_cohens_d is not None
        else "N/A (not tested)"
    )
    descriptive_line = (
        f"- descriptive_reason: {result.descriptive_reason}\n"
        if result.descriptive_reason
        else ""
    )
    cannot_assess_line = (
        f"- cannot_assess_reason: {result.cannot_assess_reason}\n"
        if result.cannot_assess_reason
        else ""
    )
    underpowered_line = (
        f"- Underpowered reason: {result.underpowered_reason}\n"
        if result.underpowered_reason
        else ""
    )
    return (
        f"{CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS}\n\n"
        f"STATISTICAL GUARD VERDICT ({result.metric_name}):\n"
        f"- assessment_status: {result.assessment_status}\n"
        f"- Claim policy: {claim_policy}\n"
        f"- sample_size_depressed: {result.sample_size_depressed}\n"
        f"- sample_size_control: {result.sample_size_control}\n"
        f"- observation_count_depressed: {result.observation_count_depressed}\n"
        f"- observation_count_control: {result.observation_count_control}\n"
        f"- aggregation_method: {result.aggregation_method}\n"
        f"- p_value: {p_display}\n"
        f"- effect_size_cohens_d: {d_display}\n"
        f"- is_significant: {result.is_significant}\n"
        f"- is_underpowered: {result.is_underpowered}\n"
        f"- statistical_confidence: {result.statistical_confidence}\n"
        f"- test_used: {result.test_used}\n"
        f"{descriptive_line}"
        f"{cannot_assess_line}"
        f"{underpowered_line}".rstrip()
    )
