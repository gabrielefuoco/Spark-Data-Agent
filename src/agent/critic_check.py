"""Deterministic, non-LLM validation of scientific critic outputs.

Requires full citation (``n``, ``p``, ``d``) for every significant ``assessed``
result and at least one fully cited ``assessed`` result (``n``, ``p``) when any
``assessed`` result exists, while flagging unnegated assertive claims in English
and Italian whenever no significant, powered finding authorizes them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from src.agent.guards import StatisticalGuardResult
from src.agent.prompts import (
    describe_descriptive_guard_it,
    format_cohens_d,
    format_critic_fallback_closing_line,
)

_NUM_TOKEN = r"(?P<num>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
_METRIC_OP = r"(?P<op>(?:(?:is|was)\s+)?(?:<=|>=|≤|≥|<|>)|=|:|\bis\b|\bof\b|\bwas\b|\()"
_INEQUALITY_OPS = frozenset({"<", "<=", "≤", ">", ">=", "≥"})

_P_VALUE_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:p[\s_-]*value|p)\b[*_`]*\)?\s*"
    rf"{_METRIC_OP}\s*(?:approx\.?\s*|approximately\s*|about\s*)?"
    rf"[*_`]*{_NUM_TOKEN}",
    re.IGNORECASE,
)

_COHENS_D_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:(?:effect[\s_-]*size[\s_-]*)?cohen['’]?s?[\s_-]*d|d)\b[*_`]*\)?\s*"
    rf"{_METRIC_OP}\s*(?:approx\.?\s*|approximately\s*|about\s*)?"
    rf"[*_`]*{_NUM_TOKEN}",
    re.IGNORECASE,
)

_LABEL_REF_RE = re.compile(r"\b(?:label|topic|class)\s*(?:=|:|\bis\b)?\s*\d+\b", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"(?m)^\s*\d+\.\s+")
_INT_TOKEN_RE = re.compile(r"(?<![.\d+-])(?<![eE][+-])\b(\d+)\b(?!\.\d)")

# Narrow surface tripwire for assertive comparative/causal/associative claims
# in both English and Italian. Paired with clause-local negation checks so
# ordinary compliant disclaimers are not flagged.
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
    # Italian assertive significance / comparative / causal patterns
    re.compile(
        r"\bsignificativamente\s+(?:più|meno|maggior[ei]?|minor[ei]?|divers[aeio]|alt[oa]|bass[oa])\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:differenza|associazione|correlazione|relazione|effetto|aumento|calo|segnale)\s+"
        r"(?:statisticamente\s+)?significativ[aoei]\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bstatisticamente\s+significativ[aoei]\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:conferma(?:no)?|dimostra(?:no)?|prova(?:no)?|stabilisce|stabiliscono)\s+"
        r"(?:una?\s+)?(?:differenza|correlazione|associazione|legame|segnale(?:\s+clinico)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:depressione|soggetti\s+depressi|utenti\s+depressi|coorte\s+depressa)\s+"
        r"(?:causa|determina|induce|predice|aumenta)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:associat[aoei]|collegat[aoei]|correlat[aoei])\s+(?:con\s+la|alla)\s+depressione\b",
        re.IGNORECASE,
    ),
    # Bare comparatives need no adverb: "i depressi postano più di notte" is an
    # assertive claim without "significativamente". Matching the comparative
    # rather than the cohort keeps the negation guard useful: "non postano più
    # di notte" carries "non" in the prefix and is skipped.
    re.compile(r"\bpi[uù]\s+(?:di\s+notte|spesso|frequentemente|sovente)\b", re.IGNORECASE),
    re.compile(r"\bmaggior\s+(?:quota|percentuale|frequenza|incidenza)\b", re.IGNORECASE),
)

_NEGATION_OR_DISCLAIMER_RE = re.compile(
    r"\b(?:no|not|never|cannot|can't|neither|nor|without|lack(?:s|ing)?|"
    r"absence\s+of|fail(?:s|ed)?\s+to\s+(?:show|find|demonstrate|reach|establish)|"
    r"non-?significant|insignificant|avoid(?:\s+claiming)?|forbids?|prohibits?|"
    r"must\s+not|do\s+not|does\s+not|did\s+not|whether|if|"
    r"non|nessun[ao]?|mai|senza|assenza\s+di|mancanza\s+di|"
    r"impossibile\s+(?:valutare|affermare|stabilire)|non\s+risulta(?:no)?|"
    r"non\s+indica(?:no)?|non\s+dimostra(?:no)?|non\s+conferma(?:no)?|"
    r"non\s+significativ[aoei]|evitare\s+di|"
    # Hypothesis framing: the critic restates the question ("Domanda: i depressi
    # pubblicano più di notte?") before answering. That is not a claim, and
    # flagging it turned a correct answer into a fallback.
    r"domanda|domande|chiede|chiedendo|verifica|verificare|ipotesi|questione|"
    r"ask(?:s|ed|ing)?|hypothes[ie]s|whether)\b",
    re.IGNORECASE,
)

_CLAUSE_SPLIT_RE = re.compile(r"[.;!?\n]+")


def _normalise_decimal_commas(text: str) -> str:
    """Read Italian decimals ("0,5587") as the numbers they are.

    The critic answers in the language of the question. In Italian it writes
    decimal commas, every citation regex missed, and a correct answer was
    rejected twice before the fallback replaced it.
    """
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


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


def _inequality_matches(op: str, bound_raw: str, target: float) -> bool:
    """Accept an inequality only when it holds for ``target`` and is tight (within 10x)."""
    try:
        bound = float(bound_raw)
    except ValueError:
        return False
    if not math.isfinite(bound) or not math.isfinite(target):
        return False

    normalized = op.replace("≤", "<=").replace("≥", ">=")
    if normalized == "<":
        holds = target < bound
    elif normalized == "<=":
        holds = target <= bound
    elif normalized == ">":
        holds = target > bound
    elif normalized == ">=":
        holds = target >= bound
    else:
        return False
    if not holds:
        return False

    if target == 0.0 and bound == 0.0:
        return True
    if target == 0.0 or bound == 0.0:
        return False
    if (target < 0) != (bound < 0):
        return False

    larger = max(abs(target), abs(bound))
    smaller = min(abs(target), abs(bound))
    return larger <= 10.0 * smaller * (1.0 + 1e-9)


def _has_matching_metric(text: str, pattern: re.Pattern[str], target: float) -> bool:
    for match in pattern.finditer(text):
        op = match.group("op").split()[-1]
        raw_num = match.group("num")
        if op in _INEQUALITY_OPS:
            if _inequality_matches(op, raw_num, target):
                return True
        elif _number_matches(raw_num, target):
            return True
    return False


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


def _normalize_guards(
    guard: StatisticalGuardResult | Sequence[StatisticalGuardResult],
) -> list[StatisticalGuardResult]:
    if isinstance(guard, StatisticalGuardResult):
        return [guard]
    return list(guard)


def _validate_single_assessed(
    text: str,
    guard: StatisticalGuardResult,
) -> tuple[bool, bool, bool, list[str]]:
    assert guard.p_value is not None
    assert guard.effect_size_cohens_d is not None
    has_p = _has_matching_metric(text, _P_VALUE_RE, guard.p_value)
    has_d = (
        _has_matching_metric(text, _COHENS_D_RE, guard.effect_size_cohens_d)
        if guard.is_significant
        else True
    )
    has_n = _has_sample_sizes(text, guard.sample_size_depressed, guard.sample_size_control)
    missing: list[str] = []
    if not has_p:
        missing.append("p_value")
    if guard.is_significant and not has_d:
        missing.append("cohens_d")
    if not has_n:
        missing.append("sample_sizes")
    return has_p, has_d, has_n, missing


def validate_critic_output(
    critique_text: str,
    guard: StatisticalGuardResult | Sequence[StatisticalGuardResult],
) -> CriticOutputCheck:
    """Every significant ``assessed`` result must be cited in full (cohort sample sizes ``n``, ``p``-value, and Cohen's ``d``), and whenever ``assessed`` results exist any comparative or no-difference claim must cite at least one ``assessed`` result's ``n`` and ``p`` in full (failing with ``missing_items=["p_value"]`` if none is cited).

    Non-significant results never require Cohen's ``d``, ``cannot_assess`` and ``descriptive`` verdicts require no numbers, and unnegated assertive comparative, causal, or significance claims are rejected unless a significant, powered ``assessed`` result supports them.
    """
    text = _normalise_decimal_commas(critique_text or "")
    guards = _normalize_guards(guard)
    assessed = [item for item in guards if item.is_assessed]

    has_p_value = True
    has_cohens_d = True
    has_sample_sizes = True
    missing_union: set[str] = set()

    if len(assessed) == 1:
        has_p_value, has_cohens_d, has_sample_sizes, single_missing = _validate_single_assessed(
            text, assessed[0]
        )
        missing_union.update(single_missing)
    elif len(assessed) > 1:
        per_item = [(item, *_validate_single_assessed(text, item)) for item in assessed]
        sig_items = [entry for entry in per_item if entry[0].is_significant]
        if sig_items:
            for _, item_p, item_d, item_n, item_missing in sig_items:
                has_p_value = has_p_value and item_p
                has_cohens_d = has_cohens_d and item_d
                has_sample_sizes = has_sample_sizes and item_n
                missing_union.update(item_missing)
        elif not any(item_p and item_n for _, item_p, _, item_n, _ in per_item):
            any_p = any(item_p for _, item_p, _, _, _ in per_item)
            any_n = any(item_n for _, _, _, item_n, _ in per_item)
            has_p_value = any_p
            has_sample_sizes = any_n
            missing_union.add("sample_sizes" if any_p else "p_value")

    ordered_missing: list[str] = [
        key for key in ("p_value", "cohens_d", "sample_sizes") if key in missing_union
    ]

    # Cohort imbalance (`balance_ratio` / `is_imbalanced`) is intentionally kept
    # in the guard payload and critic prompt without a regex output check:
    # prose validators over-reject valid paraphrases across languages.
    claims_forbidden = not any(item.allows_claims for item in guards)
    forbidden_claims = _detect_forbidden_claims(text) if claims_forbidden else []

    passed = (not ordered_missing) and (not forbidden_claims)
    return CriticOutputCheck(
        passed=passed,
        has_p_value=has_p_value,
        has_cohens_d=has_cohens_d,
        has_sample_sizes=has_sample_sizes,
        missing_items=ordered_missing,
        forbidden_claims=forbidden_claims,
    )


def format_critic_correction_prompt(
    guard: StatisticalGuardResult | Sequence[StatisticalGuardResult],
    check: CriticOutputCheck,
) -> str:
    """Build the single corrective retry instruction naming exact missing items."""
    guards = _normalize_guards(guard)
    problems: list[str] = []
    if check.missing_items:
        problems.append(f"Missing required citations: {', '.join(check.missing_items)}.")
    if check.forbidden_claims:
        problems.append(
            "Forbidden comparative/significance claim(s) detected: "
            f"{', '.join(repr(claim) for claim in check.forbidden_claims)}."
        )
    problems_block = "\n".join(f"- {line}" for line in problems)
    scope = "" if len(guards) == 1 else " across the tool results"
    header = (
        "CORRECTION REQUIRED: Your previous response failed deterministic statistical "
        f"guardrail verification{scope}.\n"
        f"{problems_block}\n\n"
        "Rewrite the complete response in the SAME language as the user's question"
    )

    if len(guards) == 1:
        single = guards[0]
        if single.assessment_status == "descriptive":
            return (
                f"{header} so that it:\n"
                "- Presents the descriptive results faithfully (terms/weights, topics/terms, counts, or summary rows).\n"
                "- Does NOT fabricate p-values or Cohen's d.\n"
                "- Does NOT assert any cohort difference, statistical significance, correlation, or clinical signal "
                "associated with depression."
            )

        if not single.is_assessed:
            return (
                f"{header} so that it:\n"
                "- Explains that statistical significance cannot be evaluated without a `subject_id` dimension.\n"
                "- Does NOT fabricate p-values or Cohen's d.\n"
                "- Does NOT assert any significant difference, correlation, or clinical signal associated with depression."
            )

        assert single.p_value is not None
        assert single.effect_size_cohens_d is not None
        claim_rule = (
            "Keep any comparative claim strictly scoped to the observed metric."
            if single.allows_claims
            else "Do NOT assert any significant difference, correlation, or clinical signal "
            "associated with depression because the statistical guard forbids claims."
        )
        d_line = (
            f"- Cohen's d = {single.effect_size_cohens_d:.6g}\n"
            if single.is_significant
            else ""
        )
        return (
            f"{header} so that it explicitly quotes:\n"
            f"- sample_size_depressed = {single.sample_size_depressed} and "
            f"sample_size_control = {single.sample_size_control}\n"
            f"- p_value = {single.p_value:.6g}\n"
            f"{d_line}"
            f"{claim_rule}"
        )

    assessed_lines: list[str] = []
    for idx, item in enumerate(guards, start=1):
        if item.is_assessed:
            assert item.p_value is not None
            assert item.effect_size_cohens_d is not None
            status_note = (
                "SIGNIFICANT (scope claim to this metric)"
                if item.allows_claims
                else "NON-SIGNIFICANT/UNDERPOWERED (do NOT claim difference for this metric)"
            )
            d_part = (
                f", Cohen's d = {item.effect_size_cohens_d:.6g}"
                if item.is_significant
                else ""
            )
            assessed_lines.append(
                f"- Result #{idx} (`{item.metric_name}`, {status_note}): "
                f"sample_size_depressed = {item.sample_size_depressed}, "
                f"sample_size_control = {item.sample_size_control}, "
                f"p_value = {item.p_value:.6g}{d_part}"
            )
        else:
            assessed_lines.append(
                f"- Result #{idx} (`{item.metric_name}`, {item.assessment_status}): "
                "present without fabricating p-values or Cohen's d, and make NO significance/causal claim."
            )

    requirements = "\n".join(assessed_lines)
    return (
        f"{header}, citing n, p, "
        "and d for every significant result and at least one assessed result's n and p when all "
        "assessed results are non-significant, obeying every result's verdict:\n"
        f"{requirements}"
    )


def _format_guard_fallback_section(
    guard: StatisticalGuardResult,
    *,
    heading: str,
) -> str:
    if not guard.is_assessed:
        if guard.assessment_status == "descriptive":
            description = describe_descriptive_guard_it(
                guard.metric_name, guard.descriptive_reason
            )
            return f"{heading}\n{description}"

        reason = guard.cannot_assess_reason or "manca la dimensione per soggetto (`subject_id`)"
        return (
            f"{heading}\n"
            f"Impossibile valutare la significatività statistica ({reason}); "
            "non viene affermata alcuna differenza tra le coorti."
        )

    assert guard.p_value is not None
    assert guard.effect_size_cohens_d is not None
    d_display = format_cohens_d(guard.effect_size_cohens_d)
    if guard.allows_claims:
        return (
            f"{heading}\n"
            f"**Risultato statisticamente significativo** "
            f"(test `{guard.test_used}`, alpha = {guard.alpha:g}):\n"
            f"- Campione: {guard.sample_size_depressed} soggetti depressi, "
            f"{guard.sample_size_control} controlli\n"
            f"- p = {guard.p_value:.6g}, Cohen's d = {d_display}"
        )

    if guard.is_underpowered:
        detail = f" ({guard.underpowered_reason})" if guard.underpowered_reason else ""
        verdict_line = (
            f"Risultato non conclusivo — campione insufficiente{detail}: "
            "non viene affermata alcuna differenza tra le coorti."
        )
    else:
        verdict_line = (
            f"Risultato non conclusivo (nessuna significatività statistica, "
            f"test `{guard.test_used}`, alpha = {guard.alpha:g}): "
            "non viene affermata alcuna differenza tra le coorti."
        )
    return (
        f"{heading}\n"
        f"{verdict_line}\n"
        f"- Campione: {guard.sample_size_depressed} soggetti depressi, "
        f"{guard.sample_size_control} controlli\n"
        f"- p = {guard.p_value:.6g}, Cohen's d = {d_display}"
    )


def format_inconclusive_critic_fallback(
    guard: StatisticalGuardResult | Sequence[StatisticalGuardResult],
    check: CriticOutputCheck,
) -> str:
    """Build a readable Italian fallback summary when the critic still fails after retry."""
    guards = _normalize_guards(guard)
    closing_line = format_critic_fallback_closing_line(
        has_missing_items=bool(check.missing_items),
        has_forbidden_claims=bool(check.forbidden_claims),
    )

    if not guards:
        return (
            "Nessun risultato statistico disponibile per la sintesi finale.\n\n"
            f"{closing_line}"
        )

    if len(guards) == 1:
        body = _format_guard_fallback_section(
            guards[0],
            heading=f"### `{guards[0].metric_name}`",
        )
        return f"{body}\n\n{closing_line}"

    sections = [
        _format_guard_fallback_section(
            item,
            heading=f"### Risultato {idx}: `{item.metric_name}`",
        )
        for idx, item in enumerate(guards, start=1)
    ]
    return "\n\n".join(sections) + f"\n\n{closing_line}"
