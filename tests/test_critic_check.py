"""Unit tests for deterministic critic output validation and fallback formatting."""

from __future__ import annotations

import pytest
from src.agent.critic_check import (
    format_critic_correction_prompt,
    format_inconclusive_critic_fallback,
    validate_critic_output,
)
from src.agent.guards import StatisticalGuardResult


def _make_guard(
    *,
    n_dep: int = 35,
    n_ctrl: int = 40,
    p_value: float | None = 0.0312,
    cohens_d: float | None = 0.54,
    is_significant: bool = True,
    is_underpowered: bool = False,
    assessment_status: str = "assessed",
) -> StatisticalGuardResult:
    is_non_assessed = assessment_status in {"cannot_assess", "descriptive"}
    return StatisticalGuardResult(
        sample_size_depressed=0 if is_non_assessed else n_dep,
        sample_size_control=0 if is_non_assessed else n_ctrl,
        observation_count_depressed=0 if is_non_assessed else n_dep,
        observation_count_control=0 if is_non_assessed else n_ctrl,
        aggregation_method="none" if is_non_assessed else "subject_mean",
        assessment_status=assessment_status,  # type: ignore[arg-type]
        cannot_assess_reason=(
            "cannot assess: no subject dimension" if assessment_status == "cannot_assess" else None
        ),
        descriptive_reason=(
            "descriptive: lexical_frequency_table" if assessment_status == "descriptive" else None
        ),
        p_value=None if is_non_assessed else p_value,
        is_significant=False if is_non_assessed else is_significant,
        effect_size_cohens_d=None if is_non_assessed else cohens_d,
        statistical_confidence=(
            assessment_status  # type: ignore[arg-type]
            if is_non_assessed
            else ("underpowered" if is_underpowered else ("medium" if is_significant else "low"))
        ),
        is_underpowered=False if is_non_assessed else is_underpowered,
        underpowered_reason="Underpowered sample." if (is_underpowered and not is_non_assessed) else None,
        test_used="none" if is_non_assessed else "welch_t",
    )


@pytest.mark.parametrize(
    "p_phrase",
    [
        "p = 0.0312",
        "p=0.0312",
        "p-value: 0.0312",
        "p_value = 0.03",
        "p value is 0.031",
        "**p-value**: **0.0312**",
        "p (0.0312)",
    ],
)
def test_validate_accepts_valid_p_value_phrasings(p_phrase: str) -> None:
    guard = _make_guard(p_value=0.0312)
    text = f"Depressed=35, control=40, {p_phrase}, Cohen's d = 0.54."
    check = validate_critic_output(text, guard)
    assert check.has_p_value is True
    assert check.passed is True
    assert check.missing_items == []


def test_validate_accepts_scientific_notation_p_value() -> None:
    guard = _make_guard(p_value=1.234e-05)
    text = "Sample sizes: depressed 35, control 40; p = 1.23e-5, d = 0.54."
    check = validate_critic_output(text, guard)
    assert check.has_p_value is True
    assert check.passed is True


@pytest.mark.parametrize(
    ("p_target", "bad_p_phrase"),
    [
        (0.0031, "p < 0.05"),
        (0.0312, "p-value < 0.001"),
        (0.0312, "p < 0.0312"),
        (0.0312, "p > 0.05"),
        (0.0312, "p = 0.01"),
        (0.0312, "p = 0.05"),
        (0.0312, "p-value was small"),
    ],
)
def test_validate_rejects_missing_or_loose_or_wrong_p_value(
    p_target: float,
    bad_p_phrase: str,
) -> None:
    guard = _make_guard(p_value=p_target)
    text = f"Depressed=35, control=40, {bad_p_phrase}, Cohen's d = 0.54."
    check = validate_critic_output(text, guard)
    assert check.has_p_value is False
    assert check.passed is False
    assert check.missing_items == ["p_value"]


def test_validate_accepts_tight_true_p_value_inequalities_and_rounded_assignments() -> None:
    guard = _make_guard(n_dep=79, n_ctrl=740, p_value=0.0005936535745210291, cohens_d=0.68)
    for phrase in ("p < 0.001", "p-value <= 0.001", "p = 0.0006"):
        text = f"Depressed=79, control=740, {phrase}, Cohen's d = 0.68."
        check = validate_critic_output(text, guard)
        assert check.has_p_value is True
        assert check.passed is True
        assert check.missing_items == []


def test_validate_rejects_rounding_nonzero_p_value_to_zero() -> None:
    guard = _make_guard(p_value=0.004)
    text = "Depressed=35, control=40, p = 0.00, Cohen's d = 0.54."
    check = validate_critic_output(text, guard)
    assert check.has_p_value is False
    assert "p_value" in check.missing_items


@pytest.mark.parametrize(
    ("d_target", "d_phrase"),
    [
        (0.54, "Cohen's d = 0.54"),
        (0.54, "Cohen’s d: 0.54"),
        (0.54, "effect_size_cohens_d = 0.54"),
        (0.54, "d = 0.54"),
        (0.68, "d > 0.5"),
        (0.68, "Cohen's d >= 0.5"),
        (-0.68, "d < -0.5"),
        (-0.423, "Cohen's d of -0.42"),
        (0.0, "Cohen's d = 0.0000"),
    ],
)
def test_validate_accepts_valid_cohens_d_phrasings(d_target: float, d_phrase: str) -> None:
    guard = _make_guard(cohens_d=d_target)
    text = f"Depressed=35, control=40, p = 0.0312, {d_phrase}."
    check = validate_critic_output(text, guard)
    assert check.has_cohens_d is True
    assert check.passed is True


@pytest.mark.parametrize(
    ("d_target", "bad_d_phrase"),
    [
        (0.68, "d > 0.8"),
        (0.68, "d > 0.05"),
        (-0.68, "d > 0.5"),
        (0.54, "Cohen's d is medium"),
        (0.54, "sd = 0.54"),
        (0.54, "Cohen's d = 0.90"),
        (-0.42, "Cohen's d = 0.42"),
    ],
)
def test_validate_rejects_cohens_d_near_misses(d_target: float, bad_d_phrase: str) -> None:
    guard = _make_guard(cohens_d=d_target)
    text = f"Depressed=35, control=40, p = 0.0312, {bad_d_phrase}."
    check = validate_critic_output(text, guard)
    assert check.has_cohens_d is False
    assert check.passed is False
    assert "cohens_d" in check.missing_items


def test_validate_sample_sizes_handles_equal_counts_and_near_misses() -> None:
    guard_eq = _make_guard(n_dep=35, n_ctrl=35)
    assert (
        validate_critic_output(
            "35 depressed subjects and 35 control subjects (p = 0.0312, d = 0.54).",
            guard_eq,
        ).has_sample_sizes
        is True
    )
    # Near-miss: only mentions 35 once or only states total 70.
    assert (
        validate_critic_output(
            "35 subjects per group (p = 0.0312, d = 0.54).",
            guard_eq,
        ).has_sample_sizes
        is False
    )
    assert (
        validate_critic_output(
            "70 subjects total (p = 0.0312, d = 0.54).",
            guard_eq,
        ).has_sample_sizes
        is False
    )
    # Near-miss: decimal 0.035 does not count as 35.
    assert (
        validate_critic_output(
            "35 depressed subjects; p = 0.035, d = 0.54.",
            _make_guard(n_dep=35, n_ctrl=35, p_value=0.035),
        ).has_sample_sizes
        is False
    )


def test_validate_sample_sizes_ignores_label_tokens_and_list_numbers() -> None:
    guard_pseudo = _make_guard(n_dep=1, n_ctrl=35, is_significant=False, is_underpowered=True)
    # Mentions label=1 and 40 observations, but never quotes unique depressed count=1.
    near_miss = (
        "1. In cohort label=1 (40 posts) vs control label=0 (35 subjects), "
        "p = 0.0312, Cohen's d = 0.54."
    )
    check = validate_critic_output(near_miss, guard_pseudo)
    assert check.has_sample_sizes is False
    assert "sample_sizes" in check.missing_items

    compliant = (
        "Underpowered: sample_size_depressed=1 and sample_size_control=35 "
        "(p = 0.0312, Cohen's d = 0.54)."
    )
    assert validate_critic_output(compliant, guard_pseudo).passed is True


def test_forbidden_claim_heuristic_flags_assertions_but_allows_negated_prose() -> None:
    guard_ns = _make_guard(is_significant=False, p_value=0.25, cohens_d=0.10)

    rogue = (
        "Depressed users post at night significantly more often than controls "
        "(depressed=35, control=40, p = 0.25, Cohen's d = 0.10)."
    )
    bad_check = validate_critic_output(rogue, guard_ns)
    assert bad_check.passed is False
    assert bad_check.missing_items == []
    assert bad_check.forbidden_claims

    rogue_2 = (
        "There is a statistically significant difference between cohorts "
        "(depressed=35, control=40, p = 0.25, Cohen's d = 0.10)."
    )
    assert validate_critic_output(rogue_2, guard_ns).passed is False

    compliant_prose = (
        "No statistically significant difference was observed between depressed (35) and "
        "control (40) subjects (p = 0.25, Cohen's d = 0.10). Although significant limitations "
        "apply and the observed mean was slightly higher, this result is not statistically "
        "significant and correlation does not imply causation."
    )
    ok_check = validate_critic_output(compliant_prose, guard_ns)
    assert ok_check.forbidden_claims == []
    assert ok_check.passed is True


def test_assertive_claims_allowed_when_guard_is_significant() -> None:
    guard_sig = _make_guard(is_significant=True, is_underpowered=False)
    text = (
        "Depressed users post at night significantly more often "
        "(depressed=35, control=40, p = 0.0312, Cohen's d = 0.54)."
    )
    check = validate_critic_output(text, guard_sig)
    assert check.passed is True
    assert check.forbidden_claims == []


@pytest.mark.parametrize(
    "assertive_clause",
    [
        "depression drives night posting",
        "night posting is driven by depression",
        "depression causes late-night activity",
        "depression leads to more night posts",
        "depression increases night posting",
        "depression raises night posting frequency",
        "depression predicts night posting",
        "depressed users are more likely to post at night",
        "night posting is associated with depression",
        "night posting is linked to depression",
        "night posting is correlated with depression",
    ],
)
def test_forbidden_claim_tripwire_catches_causal_associative_and_likelihood_shapes(
    assertive_clause: str,
) -> None:
    guard_ns = _make_guard(
        n_dep=142, n_ctrl=350, p_value=0.0031, cohens_d=0.20, is_significant=False
    )
    text = (
        f"With 142 depressed and 350 control subjects, p = 0.0031, Cohen's d = 0.20, "
        f"so {assertive_clause}."
    )
    check = validate_critic_output(text, guard_ns)
    assert check.passed is False
    assert check.missing_items == []
    assert check.forbidden_claims


def test_forbidden_claim_heuristic_does_not_catch_unlisted_paraphrases() -> None:
    # The mandatory numeric citation check (sample sizes, p-value, Cohen's d) is the
    # real mechanical guardrail; the forbidden-claim regex is a surface tripwire, not a judge.
    guard_ns = _make_guard(
        n_dep=142, n_ctrl=350, p_value=0.0031, cohens_d=0.20, is_significant=False
    )
    paraphrase = (
        "With 142 depressed and 350 control subjects, p = 0.0031, Cohen's d = 0.20, "
        "night posting characterizes the depressed cohort."
    )
    check = validate_critic_output(paraphrase, guard_ns)
    assert check.forbidden_claims == []
    assert check.passed is True


def test_correction_prompt_and_deterministic_fallback_always_comply() -> None:
    for guard in (
        _make_guard(is_significant=True),
        _make_guard(is_significant=False, p_value=0.25, cohens_d=-9.67e-16),
        _make_guard(n_dep=5, n_ctrl=5, is_significant=False, is_underpowered=True),
        _make_guard(
            n_dep=0,
            n_ctrl=0,
            p_value=None,
            cohens_d=None,
            is_significant=False,
            assessment_status="cannot_assess",
        ),
        _make_guard(
            n_dep=0,
            n_ctrl=0,
            p_value=None,
            cohens_d=None,
            is_significant=False,
            assessment_status="descriptive",
        ),
    ):
        failed = validate_critic_output("Depressed users post significantly more.", guard)
        assert failed.passed is False
        correction = format_critic_correction_prompt(guard, failed)
        for item in failed.missing_items:
            assert item in correction

        fallback = format_inconclusive_critic_fallback(guard, failed)
        if guard.assessment_status in {"cannot_assess", "descriptive"}:
            assert "non viene affermata alcuna differenza" in fallback.lower()
            assert "p_value =" not in fallback
            assert "cohen's d =" not in fallback.lower()
        elif guard.is_significant and not guard.is_underpowered:
            assert "non conclusivo" not in fallback.lower()
            assert "inconclusiv" not in fallback.lower()
            assert "non viene affermata alcuna differenza" not in fallback.lower()
            assert str(guard.sample_size_depressed) in fallback
            assert str(guard.sample_size_control) in fallback
            assert guard.p_value is not None
            assert f"{guard.p_value:.6g}" in fallback
            assert guard.effect_size_cohens_d is not None
            assert f"{guard.effect_size_cohens_d:.6g}" in fallback
        else:
            assert "risultato non conclusivo" in fallback.lower()
            assert "non viene affermata alcuna differenza" in fallback.lower()
        recheck = validate_critic_output(fallback, guard)
        assert recheck.passed is True
        assert recheck.forbidden_claims == []


def test_fallback_preserves_significant_finding_and_keeps_inconclusive_for_nonsignificant() -> None:
    sig_guard = _make_guard(
        n_dep=79,
        n_ctrl=740,
        p_value=0.0005936535745210291,
        cohens_d=0.68,
        is_significant=True,
        is_underpowered=False,
    )
    sig_failed = validate_critic_output("Differenza significativa senza numeri.", sig_guard)
    assert sig_failed.passed is False
    sig_fallback = format_inconclusive_critic_fallback(sig_guard, sig_failed)
    assert "79" in sig_fallback
    assert "740" in sig_fallback
    assert f"{sig_guard.p_value:.6g}" in sig_fallback
    assert "0.68" in sig_fallback
    assert "non conclusivo" not in sig_fallback.lower()
    assert "non viene affermata alcuna differenza" not in sig_fallback.lower()
    assert validate_critic_output(sig_fallback, sig_guard).passed is True

    ns_guard = _make_guard(
        n_dep=79,
        n_ctrl=740,
        p_value=0.31,
        cohens_d=0.08,
        is_significant=False,
        is_underpowered=False,
    )
    ns_failed = validate_critic_output("Depressed users differ significantly.", ns_guard)
    assert ns_failed.passed is False
    ns_fallback = format_inconclusive_critic_fallback(ns_guard, ns_failed)
    assert "risultato non conclusivo" in ns_fallback.lower()
    assert "non viene affermata alcuna differenza" in ns_fallback.lower()
    assert validate_critic_output(ns_fallback, ns_guard).passed is True


def test_descriptive_mode_allows_italian_presentation_and_blocks_significance_claims() -> None:
    guard = _make_guard(assessment_status="descriptive")

    compliant_it = (
        "I bigrammi più frequenti nella coorte label=1 sono 'feel alone' (tfidf=0.14) e "
        "'cant sleep' (tfidf=0.11). Trattandosi di statistiche descrittive, non è possibile "
        "affermare alcuna differenza statisticamente significativa né un segnale clinico."
    )
    ok_check = validate_critic_output(compliant_it, guard)
    assert ok_check.passed is True
    assert ok_check.forbidden_claims == []

    rogue_it = (
        "I soggetti depressi usano significativamente più spesso il bigramma 'feel alone', "
        "con una differenza statisticamente significativa tra le coorti."
    )
    bad_check = validate_critic_output(rogue_it, guard)
    assert bad_check.passed is False
    assert bad_check.forbidden_claims


def test_italian_decimal_commas_are_read_as_numbers() -> None:
    """The critic answers in the question's language, so it writes 0,5587.

    Regression: the citation regexes only understood dots, so a correct Italian
    answer was rejected twice and replaced by the fallback.
    """
    guard = _make_guard(p_value=0.5587, cohens_d=0.068, is_significant=False)

    check = validate_critic_output(
        "Il test non è significativo: p = 0,5587, d = 0,068, su 35 e 40 soggetti.",
        guard,
    )

    assert check.missing_items == []


def test_italian_bare_comparative_counts_as_a_forbidden_claim() -> None:
    """'postano più di notte' asserts a difference without the adverb."""
    guard = _make_guard(is_significant=False)

    flagged = validate_critic_output(
        "I depressi postano più di notte dei controlli (p = 0.0312, d = 0.54, 35 e 40 soggetti).",
        guard,
    )
    negated = validate_critic_output(
        "I depressi non postano più di notte dei controlli (p = 0.0312, d = 0.54, 35 e 40 soggetti).",
        guard,
    )

    assert flagged.forbidden_claims
    assert negated.forbidden_claims == []


def test_restating_the_question_is_not_a_forbidden_claim() -> None:
    """The critic opens by echoing the question; that is not an assertion.

    Regression: "**Domanda:** ... pubblicano più di notte rispetto ai controlli?"
    was flagged, the check failed twice, and a correct answer became a fallback.
    """
    guard = _make_guard(is_significant=False)

    check = validate_critic_output(
        "**Domanda:** i soggetti depressi pubblicano più di notte rispetto ai controlli? "
        "**Esito:** no, p = 0.0312, d = 0.54, 35 e 40 soggetti.",
        guard,
    )

    assert check.forbidden_claims == []
