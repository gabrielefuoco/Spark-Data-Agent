# Statistics

The statistical guard is the reason this project is not a chat interface over a
database. It is deterministic, it runs before any interpretation, and it has the
authority to forbid language.

`src/agent/guards.py` · `src/agent/critic_check.py`

## The unit of observation is the subject

Not the row. This was a real bug, found by a review, not by a test:

> Forty posts written by **one** person passed the power check, because the guard
> compared `min_subjects_per_class` against the number of SQL result rows. The
> p-value was then computed as if those forty rows were forty independent
> observations — pseudo-replication, in the one component built to prevent it.

Now:

1. each row carries `(subject_id, value)`;
2. multiple rows for the same subject collapse to **one mean per subject**
   (`aggregation_method="subject_mean"`);
3. `sample_size_depressed` / `sample_size_control` count **distinct subjects**;
4. the row counts are kept separately (`observation_count_*`) so a reader can see
   that the aggregation happened.

A test proves it directly: the same per-subject means distributed over 40 rows
and over 200 rows produce the **identical** p-value.

## Three verdicts, not two

| verdict | when | what the critic may do |
|---|---|---|
| `assessed` | a two-group comparison of a continuous per-subject metric | interpret, with citations |
| `descriptive` | n-gram tables, topic tables, counts, single-cohort slices | present the numbers; claiming a cohort difference is forbidden |
| `cannot_assess` | a comparison was attempted but the result has no subject dimension | state that it cannot be evaluated, and why |

`descriptive` exists because the first implementation had only the other two, and
therefore **refused to answer** "which words characterise depressed subjects" —
the n-gram table has no subjects, so the guard said `cannot_assess`, so the
critic said nothing. Two of the five advertised capabilities could not produce an
answer.

`cannot_assess` used to emit `p_value=1.0` and `d=0.0`. Those read as "tested, no
difference" rather than "no test was possible". They are now `None`. The same
correction applies to `is_imbalanced`: `None` when no comparison happened, never
`False`.

## Which test

1. **Normality screening** (Shapiro-Wilk, or D'Agostino-Pearson above 5,000
   samples) on both groups.
2. Normal → **Welch's t-test** (does not assume equal variances; the cohorts here
   are 79 vs 740).
3. Otherwise → **Mann-Whitney U**. On the real corpus this is what runs for most
   metrics, because posting behaviour is heavily skewed.
4. **Cohen's d** with the pooled standard deviation, guarded against zero
   variance and degenerate inputs so the payload never contains `NaN` or `inf`.

## Thresholds

| threshold | value | rationale |
|---|---|---|
| `MIN_SUBJECTS_PER_CLASS` | 30 | below this the verdict is `underpowered`, `is_significant=False` regardless of the p-value |
| `COHORT_IMBALANCE_RATIO_THRESHOLD` | 3.0 | at 3:1 the smaller cohort already contributes 75% of the standard-error variance, whatever the total N |
| `alpha` | 0.05 | conventional; recorded in the payload so a reader can judge |
| confidence | high / medium / low / underpowered | derived, not invented |

The real corpus is **79 vs 740 subjects, a 9.4:1 imbalance**. It passes the
30-subject rule and would have been silently presented as a clean comparison;
now the payload carries `balance_ratio`, the critic must disclose it, and the UI
warns.

## What the payload looks like

```json
{
  "sample_size_depressed": 79,
  "sample_size_control": 740,
  "observation_count_depressed": 79,
  "observation_count_control": 740,
  "aggregation_method": "subject_mean",
  "p_value": 0.0005936535745210291,
  "is_significant": true,
  "effect_size_cohens_d": -0.035531789493581344,
  "test_used": "mann_whitney_u",
  "statistical_confidence": "low",
  "is_underpowered": false,
  "balance_ratio": 9.367088607594937,
  "is_imbalanced": true,
  "assessment_status": "assessed"
}
```

## Results on the real corpus

Two comparisons worth showing, because they are the two interesting outcomes:

| metric | p | Cohen's d | reading |
|---|---|---|---|
| `mean_text_length` | 0.0006 | −0.036 | statistically significant, **practically null** — the textbook case where significance is not importance |
| `night_post_ratio` | 0.559 | 0.068 | no difference detected |

The night-posting hypothesis — the most quoted claim about this corpus — is
**not supported** by these data. The system says so, with the numbers, and
refuses to phrase it as a trend.

## Why the checker is calibrated the way it is

Three successive failures taught the same lesson: a validator that demands more
than the fault justifies ends up rejecting correct answers.

| the check rejected | because | the fix |
|---|---|---|
| `p < 0.001` | threshold inequalities were banned outright | accept an inequality when it is **true and tight** (within one order of magnitude) |
| `p = 0,5587` | the regex wanted a decimal point | normalise decimal commas |
| a restated question | "Domanda: i depressi postano più di notte?" read as a claim | hypothesis framing is not a claim |
| a six-metric synthesis | `d` missing for one of six assessments | `d` required only where the comparison is significant |
| everything | the model was asked to transcribe numbers | the graph appends them |

The last row is the real conclusion: **where the code can produce the fact, the
model should not be asked to remember it**.
