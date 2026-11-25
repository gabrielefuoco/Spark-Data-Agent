"""Unit tests for the deterministic statistical significance guard (no Spark)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from scipy import stats
from src.agent.guards import (
    StatisticalComparisonInput,
    SubjectObservation,
    evaluate_statistical_guard,
    format_critic_guard_instructions,
)
from src.config import Settings


def _obs(prefix: str, values: Sequence[float]) -> list[SubjectObservation]:
    """Build one ``SubjectObservation`` per distinct subject from a sequence of values."""
    return [
        SubjectObservation(subject_id=f"{prefix}_{i}", value=val) for i, val in enumerate(values)
    ]


def test_cohens_d_matches_hand_computed_magnitude_and_sign() -> None:
    # depressed: mean=4.0, var=80/29; control: mean=2.0, var=80/29 -> d = sqrt(29/20) = sqrt(1.45)
    depressed = _obs("dep", [2.0, 4.0, 6.0] * 10)
    control = _obs("ctrl", [0.0, 2.0, 4.0] * 10)

    pos = evaluate_statistical_guard(depressed, control, metric_name="circadian_index")
    assert pos.sample_size_depressed == 30
    assert pos.sample_size_control == 30
    assert pos.observation_count_depressed == 30
    assert pos.observation_count_control == 30
    assert pos.aggregation_method == "subject_mean"
    assert pos.assessment_status == "assessed"
    assert pos.is_underpowered is False
    assert pos.is_significant is True
    assert pos.effect_size_cohens_d == pytest.approx(math.sqrt(1.45), rel=1e-9)
    assert pos.p_value < 0.01
    assert pos.statistical_confidence == "high"

    # Flipping groups must negate Cohen's d with identical magnitude.
    neg = evaluate_statistical_guard(control, depressed, metric_name="circadian_index")
    assert neg.effect_size_cohens_d == pytest.approx(-math.sqrt(1.45), rel=1e-9)


def test_cohens_d_hand_computed_on_per_subject_means_across_multiple_rows() -> None:
    # Unequal row counts per subject whose per-subject means equal [2, 4, 6]*10 vs [0, 2, 4]*10.
    # Raw-row Cohen's d would differ, proving the calculation runs on per-subject means.
    dep_obs: list[SubjectObservation] = []
    ctrl_obs: list[SubjectObservation] = []
    for i in range(10):
        dep_obs.extend(
            [
                SubjectObservation(subject_id=f"d_{3 * i}", value=1.0),
                SubjectObservation(subject_id=f"d_{3 * i}", value=3.0),  # mean=2.0
                SubjectObservation(subject_id=f"d_{3 * i + 1}", value=2.0),
                SubjectObservation(subject_id=f"d_{3 * i + 1}", value=3.0),
                SubjectObservation(subject_id=f"d_{3 * i + 1}", value=5.0),
                SubjectObservation(subject_id=f"d_{3 * i + 1}", value=6.0),  # mean=4.0
                SubjectObservation(subject_id=f"d_{3 * i + 2}", value=6.0),  # mean=6.0
            ]
        )
        ctrl_obs.extend(
            [
                SubjectObservation(subject_id=f"c_{3 * i}", value=-1.0),
                SubjectObservation(subject_id=f"c_{3 * i}", value=1.0),  # mean=0.0
                SubjectObservation(subject_id=f"c_{3 * i + 1}", value=1.0),
                SubjectObservation(subject_id=f"c_{3 * i + 1}", value=3.0),  # mean=2.0
                SubjectObservation(subject_id=f"c_{3 * i + 2}", value=3.0),
                SubjectObservation(subject_id=f"c_{3 * i + 2}", value=4.0),
                SubjectObservation(subject_id=f"c_{3 * i + 2}", value=5.0),  # mean=4.0
            ]
        )

    res = evaluate_statistical_guard(dep_obs, ctrl_obs, metric_name="circadian_index")
    assert res.sample_size_depressed == 30
    assert res.sample_size_control == 30
    assert res.observation_count_depressed == 70
    assert res.observation_count_control == 70
    assert res.aggregation_method == "subject_mean"
    assert res.effect_size_cohens_d == pytest.approx(math.sqrt(1.45), rel=1e-9)


def test_significance_and_p_value_ordering_across_shifts() -> None:
    # Exact symmetric normal quantiles guarantee normality and mean=10.0 without RNG variance.
    control_vals = [10.0 + 2.0 * float(stats.norm.ppf((i + 0.5) / 40)) for i in range(40)]
    control = _obs("ctrl", control_vals)
    identical = _obs("dep", control_vals)
    small_shift = _obs("dep", [v + 0.5 for v in control_vals])
    large_shift = _obs("dep", [v + 3.0 for v in control_vals])

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
    skewed_depressed = _obs("dep", [0.0] * 25 + [100.0, 200.0, 300.0, 400.0, 500.0])
    normal_control = _obs("ctrl", [1.0 + 0.05 * i for i in range(30)])

    result = evaluate_statistical_guard(skewed_depressed, normal_control)
    assert result.normality_passed is False
    assert result.test_used == "mann_whitney_u"
    assert math.isfinite(result.p_value)
    assert math.isfinite(result.effect_size_cohens_d)


def test_underpowered_sample_overrides_small_p_value() -> None:
    # Tiny 3-user cohorts with large separation produce p < 0.05 yet MUST NOT be significant.
    depressed = _obs("dep", [100.0, 102.0, 104.0])
    control = _obs("ctrl", [1.0, 2.0, 3.0])

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


def test_forty_rows_from_single_subject_is_underpowered() -> None:
    # Regression test for pseudo-replication: 40 rows from 1 depressed user != 40 subjects.
    depressed_40_rows_1_subject = [
        SubjectObservation(subject_id="dep_single", value=0.90 + (i % 4) * 0.02) for i in range(40)
    ]
    control_35_subjects = _obs("ctrl", [0.10 + (i % 5) * 0.01 for i in range(35)])

    result = evaluate_statistical_guard(
        depressed_40_rows_1_subject,
        control_35_subjects,
        settings=Settings(_env_file=None, min_subjects_per_class=30),
        metric_name="night_post_ratio",
    )

    assert result.sample_size_depressed == 1
    assert result.sample_size_control == 35
    assert result.observation_count_depressed == 40
    assert result.observation_count_control == 35
    assert result.aggregation_method == "subject_mean"
    assert result.is_underpowered is True
    assert result.is_significant is False
    assert result.statistical_confidence == "underpowered"


def test_thirty_subjects_single_row_each_with_large_difference_is_powered_and_significant() -> None:
    depressed = _obs("dep", [0.75 + (i % 6) * 0.02 for i in range(30)])
    control = _obs("ctrl", [0.10 + (i % 6) * 0.02 for i in range(30)])

    result = evaluate_statistical_guard(
        depressed,
        control,
        settings=Settings(_env_file=None, min_subjects_per_class=30),
        metric_name="night_post_ratio",
    )

    assert result.sample_size_depressed == 30
    assert result.sample_size_control == 30
    assert result.observation_count_depressed == 30
    assert result.observation_count_control == 30
    assert result.is_underpowered is False
    assert result.is_significant is True
    assert result.p_value < 0.001


def test_pseudo_replication_produces_identical_p_value_for_identical_subject_means() -> None:
    dep_means = [10.0 + 1.5 * float(stats.norm.ppf((i + 0.5) / 30)) for i in range(30)]
    ctrl_means = [7.5 + 1.5 * float(stats.norm.ppf((i + 0.5) / 30)) for i in range(30)]

    # Dataset A: 1 observation per subject.
    res_single_row = evaluate_statistical_guard(
        _obs("dep", dep_means),
        _obs("ctrl", ctrl_means),
        metric_name="night_post_ratio",
    )

    # Dataset B: unequal multi-row counts per subject with exact same per-subject means.
    dep_multi: list[SubjectObservation] = []
    for i, mean in enumerate(dep_means):
        if i % 3 == 0:
            offsets = [-0.5, 0.5]
        elif i % 3 == 1:
            offsets = [-1.0, 0.0, 0.0, 1.0]
        else:
            offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]
        dep_multi.extend(
            SubjectObservation(subject_id=f"dep_{i}", value=mean + delta) for delta in offsets
        )

    ctrl_multi: list[SubjectObservation] = []
    for i, mean in enumerate(ctrl_means):
        offsets = [-0.25, 0.25] if i % 2 == 0 else [-0.5, 0.0, 0.5]
        ctrl_multi.extend(
            SubjectObservation(subject_id=f"ctrl_{i}", value=mean + delta) for delta in offsets
        )

    res_multi_row = evaluate_statistical_guard(
        dep_multi,
        ctrl_multi,
        metric_name="night_post_ratio",
    )

    assert res_single_row.sample_size_depressed == res_multi_row.sample_size_depressed == 30
    assert res_single_row.sample_size_control == res_multi_row.sample_size_control == 30
    assert res_single_row.observation_count_depressed == 30
    assert res_multi_row.observation_count_depressed == 110
    assert res_single_row.observation_count_control == 30
    assert res_multi_row.observation_count_control == 75
    assert res_single_row.aggregation_method == res_multi_row.aggregation_method == "subject_mean"
    assert res_single_row.test_used == res_multi_row.test_used
    assert res_single_row.p_value == pytest.approx(res_multi_row.p_value, rel=1e-12, abs=1e-12)
    assert res_single_row.effect_size_cohens_d == pytest.approx(
        res_multi_row.effect_size_cohens_d, rel=1e-12, abs=1e-12
    )


def test_degenerate_constant_and_single_observation_inputs_never_emit_nan_or_inf() -> None:
    same_const = evaluate_statistical_guard(_obs("d", [5.0] * 30), _obs("c", [5.0] * 30))
    assert same_const.test_used == "none"
    assert same_const.p_value == 1.0
    assert same_const.effect_size_cohens_d == 0.0
    assert same_const.is_significant is False

    diff_const = evaluate_statistical_guard(_obs("d", [5.0] * 30), _obs("c", [6.0] * 30))
    assert math.isfinite(diff_const.p_value)
    assert diff_const.effect_size_cohens_d == 0.0

    single = evaluate_statistical_guard(_obs("d", [1.0]), _obs("c", [2.0]))
    assert single.test_used == "none"
    assert single.p_value == 1.0
    assert single.effect_size_cohens_d == 0.0
    assert single.is_underpowered is True
    assert single.is_significant is False


def test_missing_subject_dimension_returns_cannot_assess_verdict() -> None:
    result = evaluate_statistical_guard(
        [],
        [],
        has_subject_dimension=False,
        raw_observation_count_depressed=40,
        raw_observation_count_control=40,
        metric_name="night_post_ratio",
    )

    assert result.assessment_status == "cannot_assess"
    assert result.cannot_assess_reason is not None
    assert "cannot assess: no subject dimension" in result.cannot_assess_reason.lower()
    assert result.statistical_confidence == "cannot_assess"
    assert result.is_significant is False
    assert result.test_used == "none"
    assert result.aggregation_method == "none"
    assert result.sample_size_depressed == 0
    assert result.sample_size_control == 0
    assert result.observation_count_depressed == 40
    assert result.observation_count_control == 40

    prompt = format_critic_guard_instructions(result)
    assert "FORBIDDEN" in prompt
    assert "cannot assess: no subject dimension" in prompt.lower()


def test_non_finite_input_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        StatisticalComparisonInput(
            depressed_observations=[
                SubjectObservation(subject_id="d1", value=1.0),
                SubjectObservation(subject_id="d2", value=float("nan")),
            ],
            control_observations=_obs("c", [1.0, 2.0]),
        )

    with pytest.raises(ValueError, match="finite"):
        evaluate_statistical_guard(_obs("d", [1.0, 2.0]), _obs("c", [1.0, float("inf")]))


def test_critic_guard_instructions_enforce_verdict_and_citations() -> None:
    underpowered = evaluate_statistical_guard(
        _obs("d", [10.0, 11.0, 12.0]),
        _obs("c", [1.0, 2.0, 3.0]),
        metric_name="verbosity",
    )
    blocked_prompt = format_critic_guard_instructions(underpowered)
    assert "FORBIDDEN" in blocked_prompt
    assert "sample_size_depressed: 3" in blocked_prompt
    assert "sample_size_control: 3" in blocked_prompt
    assert "observation_count_depressed: 3" in blocked_prompt
    assert "aggregation_method: subject_mean" in blocked_prompt
    assert "p_value:" in blocked_prompt
    assert "effect_size_cohens_d:" in blocked_prompt

    powered_sig = evaluate_statistical_guard(
        _obs("d", [2.0, 4.0, 6.0] * 10),
        _obs("c", [0.0, 2.0, 4.0] * 10),
        metric_name="verbosity",
    )
    allowed_prompt = format_critic_guard_instructions(powered_sig)
    assert "ALLOWED" in allowed_prompt
    assert "sample_size_depressed: 30" in allowed_prompt
    assert "sample_size_control: 30" in allowed_prompt
