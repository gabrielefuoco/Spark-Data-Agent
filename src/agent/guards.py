"""Deterministic, non-LLM statistical significance guard and critic constraints.

Evaluates two-group comparisons (depressed vs control) using SciPy before the
scientific critic LLM is allowed to interpret query results. Collapses multiple
observations per ``subject_id`` into a per-subject arithmetic mean to prevent
pseudo-replication, enforces minimum unique-subject sample sizes from
``Settings.min_subjects_per_class``, chooses between Welch's t-test and
Mann-Whitney U via normality screening, computes Cohen's d, and guards
degenerate/missing-subject inputs against NaN/inf or silent passes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy import stats

from src.config import Settings

ConfidenceLevel = Literal["high", "medium", "low", "underpowered", "cannot_assess"]
TestName = Literal["welch_t", "mann_whitney_u", "none"]
AggregationMethod = Literal["subject_mean", "none"]
AssessmentStatus = Literal["assessed", "cannot_assess"]

# Ticket 05 wires these rules into the scientific_critic node's system prompt.
CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS = (
    "MANDATORY SCIENTIFIC CRITIC CONSTRAINTS:\n"
    "1. If `assessment_status` is 'cannot_assess' OR `is_significant` is false OR "
    "`is_underpowered` is true, you MUST NOT claim any correlation, difference, or "
    "clinical signal associated with depression.\n"
    "2. Every comparative statement MUST explicitly cite the unique-subject sample sizes "
    "(`sample_size_depressed` and `sample_size_control`), raw observation counts "
    "(`observation_count_depressed` and `observation_count_control`), "
    "`aggregation_method`, `p_value`, and effect size `effect_size_cohens_d`.\n"
    "3. When `assessment_status` is 'cannot_assess', state `cannot_assess_reason` "
    "verbatim and explain that statistical significance cannot be evaluated without a "
    "`subject_id` dimension.\n"
    "4. When the sample is underpowered, state `underpowered_reason` verbatim "
    "and frame any observed numeric difference as inconclusive."
)

_NORMALITY_ALPHA = 0.05
_SHAPIRO_MAX_N = 5000


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
    p_value: float
    is_significant: bool
    effect_size_cohens_d: float
    statistical_confidence: ConfidenceLevel
    is_underpowered: bool
    underpowered_reason: str | None = None
    test_used: TestName
    normality_passed: bool | None = None
    alpha: float = 0.05
    metric_name: str = "metric"


def _aggregate_by_subject(observations: Sequence[SubjectObservation]) -> np.ndarray:
    """Collapse multiple observations per ``subject_id`` into one arithmetic mean per subject."""
    if not observations:
        return np.asarray([], dtype=np.float64)

    grouped: dict[str, list[float]] = defaultdict(list)
    for obs in observations:
        grouped[obs.subject_id].append(obs.value)

    subject_means = [
        vals[0] if len(vals) == 1 else math.fsum(vals) / len(vals)
        for vals in grouped.values()
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
        return StatisticalGuardResult(
            sample_size_depressed=0,
            sample_size_control=0,
            observation_count_depressed=obs_count_dep,
            observation_count_control=obs_count_ctrl,
            aggregation_method="none",
            assessment_status="cannot_assess",
            cannot_assess_reason=(
                f"cannot assess: no subject dimension for '{validated.metric_name}' "
                "(missing `subject_id` column in query results)."
            ),
            p_value=1.0,
            is_significant=False,
            effect_size_cohens_d=0.0,
            statistical_confidence="cannot_assess",
            is_underpowered=False,
            underpowered_reason=None,
            test_used="none",
            normality_passed=None,
            alpha=validated.alpha,
            metric_name=validated.metric_name,
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


def format_critic_guard_instructions(result: StatisticalGuardResult) -> str:
    """Format the guard verdict and mandatory constraints for the scientific critic prompt.

    Ticket 05 wires this helper directly into the ``scientific_critic`` node's
    system/state prompt so the LLM cannot override or omit statistical bounds.
    """
    if result.assessment_status == "cannot_assess":
        claim_policy = (
            "FORBIDDEN: Cannot assess statistical significance (no subject dimension). "
            "Do NOT claim any depression correlation or class difference; state that "
            "subject-level significance cannot be evaluated without `subject_id`."
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
        f"- p_value: {result.p_value:.6g}\n"
        f"- effect_size_cohens_d: {result.effect_size_cohens_d:.4f}\n"
        f"- is_significant: {result.is_significant}\n"
        f"- is_underpowered: {result.is_underpowered}\n"
        f"- statistical_confidence: {result.statistical_confidence}\n"
        f"- test_used: {result.test_used}\n"
        f"{cannot_assess_line}"
        f"{underpowered_line}".rstrip()
    )
