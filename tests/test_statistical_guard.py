"""Unit tests for the deterministic statistical significance guard (no Spark)."""

from __future__ import annotations

import math

import pytest
from scipy import stats
from src.agent.guards import (
    StatisticalComparisonInput,
    evaluate_statistical_guard,
    format_critic_guard_instructions,
)
from src.config import Settings


def test_cohens_d_matches_hand_computed_magnitude_and_sign() -> None:
    # depressed: mean=4.0, var=80/29; control: mean=2.0, var=80/29 -> d = sqrt(29/20) = sqrt(1.45)
    depressed = [2.0, 4.0, 6.0] * 10
    control = [0.0, 2.0, 4.0] * 10

    pos = evaluate_statistical_guard(depressed, control, metric_name="circadian_index")
    assert pos.sample_size_depressed == 30
    assert pos.sample_size_control == 30
    assert pos.is_underpowered is False
    assert pos.is_significant is True
    assert pos.effect_size_cohens_d == pytest.approx(math.sqrt(1.45), rel=1e-9)
    assert pos.p_value < 0.01
    assert pos.statistical_confidence == "high"

    # Flipping groups must negate Cohen's d with identical magnitude.
    neg = evaluate_statistical_guard(control, depressed, metric_name="circadian_index")
    assert neg.effect_size_cohens_d == pytest.approx(-math.sqrt(1.45), rel=1e-9)


def test_significance_and_p_value_ordering_across_shifts() -> None:
    # Exact symmetric normal quantiles guarantee normality and mean=10.0 without RNG variance.
    control = [10.0 + 2.0 * float(stats.norm.ppf((i + 0.5) / 40)) for i in range(40)]
    identical = list(control)
    small_shift = [v + 0.5 for v in control]
    large_shift = [v + 3.0 for v in control]

    res_ident = evaluate_statistical_guard(identical, control)
    res_small = evaluate_statistical_guard(small_shift, control)
    res_large = evaluate_statistical_guard(large_shift, control)

    assert res_ident.test_used == "welch_t"
    assert res_ident.normality_passed is True
    assert res_ident.p_value == pytest.approx(1.0)
    assert res_ident.effect_size_cohens_d == pytest.approx(0.0)
    assert res_ident.is_significant is False
    assert res_ident.statistical_confidence == "low"

    assert res_large.p_value < res_small.p_value < res_ident.p_value
    assert res_small.is_significant is False
    assert res_large.is_significant is True


def test_non_normal_skewed_sample_selects_mann_whitney_u() -> None:
    skewed_depressed = [0.0] * 25 + [100.0, 200.0, 300.0, 400.0, 500.0]
    normal_control = [1.0 + 0.05 * i for i in range(30)]

    result = evaluate_statistical_guard(skewed_depressed, normal_control)
    assert result.normality_passed is False
    assert result.test_used == "mann_whitney_u"
    assert math.isfinite(result.p_value)
    assert math.isfinite(result.effect_size_cohens_d)


def test_underpowered_sample_overrides_small_p_value() -> None:
    # Tiny 3-user cohorts with large separation produce p < 0.05 yet MUST NOT be significant.
    depressed = [100.0, 102.0, 104.0]
    control = [1.0, 2.0, 3.0]

    result = evaluate_statistical_guard(
        depressed,
        control,
        settings=Settings(_env_file=None, min_subjects_per_class=30),
        metric_name="night_post_ratio",
    )

    assert result.sample_size_depressed == 3
    assert result.sample_size_control == 3
    assert result.p_value < 0.05
    assert result.is_underpowered is True
    assert result.underpowered_reason is not None
    assert "30" in result.underpowered_reason
    assert result.is_significant is False
    assert result.statistical_confidence == "underpowered"


def test_degenerate_constant_and_single_observation_inputs_never_emit_nan_or_inf() -> None:
    same_const = evaluate_statistical_guard([5.0] * 30, [5.0] * 30)
    assert same_const.test_used == "none"
    assert same_const.p_value == 1.0
    assert same_const.effect_size_cohens_d == 0.0
    assert same_const.is_significant is False

    diff_const = evaluate_statistical_guard([5.0] * 30, [6.0] * 30)
    assert math.isfinite(diff_const.p_value)
    assert diff_const.effect_size_cohens_d == 0.0

    single = evaluate_statistical_guard([1.0], [2.0])
    assert single.test_used == "none"
    assert single.p_value == 1.0
    assert single.effect_size_cohens_d == 0.0
    assert single.is_underpowered is True
    assert single.is_significant is False


def test_non_finite_input_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        StatisticalComparisonInput(depressed_values=[1.0, float("nan")], control_values=[1.0, 2.0])

    with pytest.raises(ValueError, match="finite"):
        evaluate_statistical_guard([1.0, 2.0], [1.0, float("inf")])


def test_critic_guard_instructions_enforce_verdict_and_citations() -> None:
    underpowered = evaluate_statistical_guard([10.0, 11.0, 12.0], [1.0, 2.0, 3.0], metric_name="verbosity")
    blocked_prompt = format_critic_guard_instructions(underpowered)
    assert "FORBIDDEN" in blocked_prompt
    assert "sample_size_depressed: 3" in blocked_prompt
    assert "sample_size_control: 3" in blocked_prompt
    assert "p_value:" in blocked_prompt
    assert "effect_size_cohens_d:" in blocked_prompt

    powered_sig = evaluate_statistical_guard([2.0, 4.0, 6.0] * 10, [0.0, 2.0, 4.0] * 10, metric_name="verbosity")
    allowed_prompt = format_critic_guard_instructions(powered_sig)
    assert "ALLOWED" in allowed_prompt
    assert "sample_size_depressed: 30" in allowed_prompt
    assert "sample_size_control: 30" in allowed_prompt
