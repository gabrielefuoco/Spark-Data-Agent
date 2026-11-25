"""Deterministic, non-LLM validation of scientific critic outputs.

Verifies that the critic narrative explicitly cites the statistical guard's
p-value, Cohen's d effect size, and both cohorts' unique-subject sample sizes.
When the guard verdict is non-significant, underpowered, or ``cannot_assess``,
a narrow regex tripwire also flags common assertive comparative phrasings about
depression. This is a surface heuristic, not semantic understanding: a
paraphrase without those explicit verbs or phrases will pass, so the mandatory
numeric citation check remains the primary mechanical enforcement mechanism.
"""

from __future__ import annotations

import math
import re

from pydantic import BaseModel, Field

from src.agent.guards import StatisticalGuardResult

_NUM_TOKEN = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"

_P_VALUE_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:p[\s_-]*value|p)\b[*_`]*\)?\s*"
    rf"(?:(?:=|:|\bis\b|\bof\b|\bwas\b)\s*(?:approx\.?\s*|approximately\s*|about\s*)?|\()\s*"
    rf"[*_`]*{_NUM_TOKEN}",
    re.IGNORECASE,
)

_COHENS_D_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:(?:effect[\s_-]*size[\s_-]*)?cohen['’]?s?[\s_-]*d|d)\b[*_`]*\)?\s*"
    rf"(?:(?:=|:|\bis\b|\bof\b|\bwas\b)\s*(?:approx\.?\s*|approximately\s*|about\s*)?|\()\s*"
    rf"[*_`]*{_NUM_TOKEN}",
    re.IGNORECASE,
)

_LABEL_REF_RE = re.compile(r"\b(?:label|topic|class)\s*(?:=|:|\bis\b)?\s*\d+\b", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"(?m)^\s*\d+\.\s+")
_INT_TOKEN_RE = re.compile(r"(?<![.\d+-])(?<![eE][+-])\b(\d+)\b(?!\.\d)")

# Narrow surface tripwire for assertive comparative/causal/associative claims.
# Paired with clause-local negation checks so ordinary compliant disclaimers
# (e.g. "no statistically significant difference", "not associated with depression")
# are not flagged. A paraphrase without these explicit verbs will still pass.
_COHORT_TARGET = r"(?:depression|depressed(?:\s+(?:users?|subjects?|cohort|group|individuals?))?)"
_FORBIDDEN_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bsignificantly\s+(?:more|less|higher|lower|greater|fewer|different)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:statistically\s+)?significant\s+(?:cohort\s+)?"
        r"(?:difference|association|correlation|effect|relationship|signal|increase|decrease|shift)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:is|are|was|were)\s+(?:statistically\s+)?significant\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:confirms?|proves?|demonstrates?|establishes?)\s+(?:a\s+)?"
        r"(?:significant\s+)?(?:correlation|association|difference|link|signal)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_COHORT_TARGET}\s+(?:drives?|causes?|leads?\s+to|increases?|raises?|predicts?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:driven|caused|predicted)\s+by\s+{_COHORT_TARGET}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:associated|linked|correlated)\s+(?:with|to)\s+{_COHORT_TARGET}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_COHORT_TARGET}\s+(?:is|are|was|were|appears?|seems?)\s+"
        r"(?:(?:strongly|closely|directly|positively|negatively|significantly)\s+)?"
        r"(?:associated|linked|correlated)\s+(?:with|to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:depressed(?:\s+(?:users?|subjects?|cohort|group|individuals?))?|label\s*[=:]?\s*1)\s+"
        r"(?:are|is|were|was)\s+(?:(?:much|far|significantly)\s+)?more\s+likely\s+to\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmore\s+likely\s+(?:in|among)\s+depressed\b",
        re.IGNORECASE,
    ),
)

_NEGATION_OR_DISCLAIMER_RE = re.compile(
    r"\b(?:no|not|never|cannot|can't|neither|nor|without|lack(?:s|ing)?|"
    r"absence\s+of|fail(?:s|ed)?\s+to\s+(?:show|find|demonstrate|reach|establish)|"
    r"non-?significant|insignificant|avoid(?:\s+claiming)?|forbids?|prohibits?|"
    r"must\s+not|do\s+not|does\s+not|did\s+not|whether|if)\b",
    re.IGNORECASE,
)

_CLAUSE_SPLIT_RE = re.compile(r"[.;!?\n]+")


class CriticOutputCheck(BaseModel):
    """Deterministic validation report for a single critic response."""

    passed: bool
    has_p_value: bool
    has_cohens_d: bool
    has_sample_sizes: bool
    missing_items: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)


def _number_matches(quoted_raw: str, target: float) -> bool:
    """Compare a quoted numeric token against ``target`` within its displayed precision."""
    try:
        quoted = float(quoted_raw)
    except ValueError:
        return False
    if not math.isfinite(quoted) or not math.isfinite(target):
        return False
    if target == 0.0:
        return abs(quoted) <= 1e-6
    if quoted == 0.0 and abs(target) >= 1e-4:
        return False
    if quoted != 0.0 and (quoted < 0) != (target < 0):
        return False

    mantissa, _, exp_part = quoted_raw.lower().partition("e")
    exp = int(exp_part) if exp_part else 0
    decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    if decimals >= 2 or exp != 0:
        rounding_tol = 0.5 * (10.0 ** (exp - decimals)) + 1e-9
        tol = max(rounding_tol, 0.02 * abs(target), 1e-9)
    else:
        tol = max(1e-6, 0.02 * abs(target))
    return abs(quoted - target) <= tol


def _has_matching_metric(text: str, pattern: re.Pattern[str], target: float) -> bool:
    return any(_number_matches(match.group(1), target) for match in pattern.finditer(text))


def _has_sample_sizes(text: str, n_depressed: int, n_control: int) -> bool:
    cleaned = _LIST_ITEM_RE.sub(" ", text)
    cleaned = _LABEL_REF_RE.sub(" ", cleaned)
    ints = [int(match.group(1)) for match in _INT_TOKEN_RE.finditer(cleaned)]
    if n_depressed == n_control:
        return ints.count(n_depressed) >= 2
    return (n_depressed in ints) and (n_control in ints)


def _detect_forbidden_claims(text: str) -> list[str]:
    """Detect unnegated assertive comparative claims using a narrow clause heuristic."""
    flagged: list[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(text):
        for pattern in _FORBIDDEN_CLAIM_PATTERNS:
            for match in pattern.finditer(clause):
                prefix = clause[: match.start()]
                if _NEGATION_OR_DISCLAIMER_RE.search(prefix):
                    continue
                snippet = match.group(0)
                if snippet not in flagged:
                    flagged.append(snippet)
    return flagged


def validate_critic_output(
    critique_text: str,
    guard: StatisticalGuardResult,
) -> CriticOutputCheck:
    """Validate that ``critique_text`` obeys statistical citation and claim rules.

    Checks three mandatory numeric citations against ``guard``:
    - ``p_value``: explicit p-value assignment (e.g. ``p = 0.03``, ``p-value: 0.03``)
      matching ``guard.p_value`` within its displayed rounding precision;
      threshold-only inequalities such as ``p < 0.05`` do not satisfy this check.
    - ``cohens_d``: explicit Cohen's d assignment matching ``guard.effect_size_cohens_d``.
    - ``sample_sizes``: both ``guard.sample_size_depressed`` and
      ``guard.sample_size_control`` appearing as standalone integer counts.

    When ``guard`` forbids claims (``assessment_status == "cannot_assess"``,
    ``not is_significant``, or ``is_underpowered``), a narrow regex tripwire
    also scans each clause for unnegated assertive phrasing about depression.
    This is a surface heuristic, not semantic understanding: a paraphrase
    without those explicit verbs or phrases will pass, so the mandatory
    numeric citation check remains the primary guardrail.
    """
    text = critique_text or ""
    has_p_value = _has_matching_metric(text, _P_VALUE_RE, guard.p_value)
    has_cohens_d = _has_matching_metric(text, _COHENS_D_RE, guard.effect_size_cohens_d)
    has_sample_sizes = _has_sample_sizes(
        text,
        guard.sample_size_depressed,
        guard.sample_size_control,
    )

    missing_items: list[str] = []
    if not has_p_value:
        missing_items.append("p_value")
    if not has_cohens_d:
        missing_items.append("cohens_d")
    if not has_sample_sizes:
        missing_items.append("sample_sizes")

    claims_forbidden = (
        guard.assessment_status == "cannot_assess"
        or (not guard.is_significant)
        or guard.is_underpowered
    )
    forbidden_claims = _detect_forbidden_claims(text) if claims_forbidden else []

    passed = (not missing_items) and (not forbidden_claims)
    return CriticOutputCheck(
        passed=passed,
        has_p_value=has_p_value,
        has_cohens_d=has_cohens_d,
        has_sample_sizes=has_sample_sizes,
        missing_items=missing_items,
        forbidden_claims=forbidden_claims,
    )


def format_critic_correction_prompt(
    guard: StatisticalGuardResult,
    check: CriticOutputCheck,
) -> str:
    """Build the single corrective retry instruction naming exact missing items."""
    problems: list[str] = []
    if check.missing_items:
        problems.append(f"Missing required citations: {', '.join(check.missing_items)}.")
    if check.forbidden_claims:
        problems.append(
            "Forbidden comparative/significance claim(s) detected: "
            f"{', '.join(repr(claim) for claim in check.forbidden_claims)}."
        )
    problems_block = "\n".join(f"- {line}" for line in problems)

    claim_rule = (
        "Do NOT assert any significant difference, correlation, or clinical signal "
        "associated with depression because the statistical guard forbids claims."
        if (
            guard.assessment_status == "cannot_assess"
            or (not guard.is_significant)
            or guard.is_underpowered
        )
        else "Keep any comparative claim strictly scoped to the observed metric."
    )
    return (
        "CORRECTION REQUIRED: Your previous response failed deterministic statistical "
        "guardrail verification.\n"
        f"{problems_block}\n\n"
        "Rewrite the complete response so that it explicitly quotes:\n"
        f"- sample_size_depressed = {guard.sample_size_depressed} and "
        f"sample_size_control = {guard.sample_size_control}\n"
        f"- p_value = {guard.p_value:.6g}\n"
        f"- Cohen's d = {guard.effect_size_cohens_d:.6g}\n"
        f"{claim_rule}"
    )


def format_inconclusive_critic_fallback(
    guard: StatisticalGuardResult,
    check: CriticOutputCheck,
) -> str:
    """Build deterministic inconclusive text when the critic still fails after retry."""
    reasons: list[str] = []
    if check.missing_items:
        reasons.append(f"missing_items={', '.join(check.missing_items)}")
    if check.forbidden_claims:
        reasons.append(f"forbidden_claim_detected (count={len(check.forbidden_claims)})")
    reason_summary = "; ".join(reasons) or "guardrail check failed"

    extra_reason = guard.cannot_assess_reason or guard.underpowered_reason
    extra_line = f"- Guard detail: {extra_reason}\n" if extra_reason else ""
    d_display = (
        "0.0000" if abs(guard.effect_size_cohens_d) < 1e-12 else f"{guard.effect_size_cohens_d:.6g}"
    )

    return (
        f"Inconclusive result for `{guard.metric_name}`: the claim threshold was not met "
        "and the model critique failed statistical guardrail verification after retry, so no "
        "comparative or clinical claim about depression is asserted.\n"
        f"- sample_size_depressed: {guard.sample_size_depressed}\n"
        f"- sample_size_control: {guard.sample_size_control}\n"
        f"- p_value = {guard.p_value:.6g}\n"
        f"- Cohen's d = {d_display}\n"
        f"- assessment_status: {guard.assessment_status}, "
        f"is_significant: {guard.is_significant}, is_underpowered: {guard.is_underpowered}\n"
        f"{extra_line}"
        f"- Verification failure: {reason_summary}"
    )
