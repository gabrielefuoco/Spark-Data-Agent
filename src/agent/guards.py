"""Deterministic, non-LLM statistical significance guard and critic constraints.

Evaluates two-group comparisons (depressed vs control) using SciPy before the
scientific critic LLM is allowed to interpret query results. Enforces minimum
subject sample sizes from ``Settings.min_subjects_per_class``, chooses between
Welch's t-test and Mann-Whitney U via normality screening, computes Cohen's d,
and guards degenerate constant/tiny inputs against NaN/inf.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy import stats

from src.config import Settings

ConfidenceLevel = Literal["high", "medium", "low", "underpowered"]
TestName = Literal["welch_t", "mann_whitney_u", "none"]

# Ticket 05 wires these rules into the scientific_critic node's system prompt.
CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS = (
    "MANDATORY SCIENTIFIC CRITIC CONSTRAINTS:\n"
    "1. If `is_significant` is false OR `is_underpowered` is true, you MUST NOT "
    "claim any correlation, difference, or clinical signal associated with depression.\n"
    "2. Every comparative statement MUST explicitly cite the sample sizes "
    "(`sample_size_depressed` and `sample_size_control`), `p_value`, and "
    "effect size `effect_size_cohens_d`.\n"
    "3. When the sample is underpowered, state the underpowered reason verbatim "
    "and frame any observed numeric difference as inconclusive."
)

_NORMALITY_ALPHA = 0.05
_SHAPIRO_MAX_N = 5000


def _validate_finite_sequence(values: list[float]) -> list[float]:
    cleaned = [float(v) for v in values]
    if any(not math.isfinite(v) for v in cleaned):
        raise ValueError("All metric values must be finite numbers (no NaN or inf).")
    return cleaned


class StatisticalComparisonInput(BaseModel):
    """Input payload for comparing a continuous metric across depressed and control cohorts."""

    depressed_values: list[float]
    control_values: list[float]
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_subjects_per_class: int | None = Field(default=None, ge=1)
    metric_name: str = "metric"

    @field_validator("depressed_values", "control_values")
    @classmethod
    def check_finite(cls, values: list[float]) -> list[float]:
        """Reject NaN/inf early so downstream SciPy calls remain deterministic."""
        return _validate_finite_sequence(values)


class StatisticalGuardResult(BaseModel):
    """Deterministic verdict emitted by ``evaluate_statistical_guard``."""

    sample_size_depressed: int
    sample_size_control: int
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
    depressed_values: Sequence[float],
    control_values: Sequence[float],
    *,
    alpha: float = 0.05,
    min_subjects_per_class: int | None = None,
    settings: Settings | None = None,
    metric_name: str = "metric",
) -> StatisticalGuardResult:
    """Run sample-power validation, hypothesis test selection, and Cohen's d.

    Pure Python/NumPy/SciPy implementation with no Spark dependency so the
    LangGraph guard node (ticket 05) can call it directly on query results.

    Test-selection rule:
        1. If either group has < 2 observations or all observations across both
           groups are identical, skip testing and return ``p_value=1.0``.
        2. Screen each group for normality (Shapiro-Wilk for n <= 5000,
           D'Agostino-Pearson above 5000; constant groups fail screening).
        3. Use Welch's t-test when both groups pass normality (p >= 0.05);
           otherwise use two-sided Mann-Whitney U.

    Underpowered policy:
        If either class has fewer than ``Settings.min_subjects_per_class``
        subjects (default 30), ``is_significant`` is forced to ``False``
        regardless of the hypothesis test's p-value.
    """
    validated = StatisticalComparisonInput(
        depressed_values=list(depressed_values),
        control_values=list(control_values),
        alpha=alpha,
        min_subjects_per_class=min_subjects_per_class,
        metric_name=metric_name,
    )

    threshold = (
        validated.min_subjects_per_class
        if validated.min_subjects_per_class is not None
        else (settings.min_subjects_per_class if settings is not None else Settings(_env_file=None).min_subjects_per_class)
    )

    dep = np.asarray(validated.depressed_values, dtype=np.float64)
    ctrl = np.asarray(validated.control_values, dtype=np.float64)
    n1 = int(dep.size)
    n0 = int(ctrl.size)

    is_underpowered = n1 < threshold or n0 < threshold
    underpowered_reason = (
        f"Underpowered sample for '{validated.metric_name}': "
        f"depressed={n1}, control={n0}; minimum required per class is {threshold}."
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
    claim_policy = (
        "ALLOWED: Statistical significance gate passed; report findings strictly "
        "scoped to the observed metric."
        if result.is_significant and not result.is_underpowered
        else "FORBIDDEN: Do NOT claim any depression correlation or class difference "
        "because the result is underpowered or not statistically significant."
    )
    underpowered_line = (
        f"- Underpowered reason: {result.underpowered_reason}\n"
        if result.underpowered_reason
        else ""
    )
    return (
        f"{CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS}\n\n"
        f"STATISTICAL GUARD VERDICT ({result.metric_name}):\n"
        f"- Claim policy: {claim_policy}\n"
        f"- sample_size_depressed: {result.sample_size_depressed}\n"
        f"- sample_size_control: {result.sample_size_control}\n"
        f"- p_value: {result.p_value:.6g}\n"
        f"- effect_size_cohens_d: {result.effect_size_cohens_d:.4f}\n"
        f"- is_significant: {result.is_significant}\n"
        f"- is_underpowered: {result.is_underpowered}\n"
        f"- statistical_confidence: {result.statistical_confidence}\n"
        f"- test_used: {result.test_used}\n"
        f"{underpowered_line}".rstrip()
    )
