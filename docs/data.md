# The Data

## What eRISK is

A corpus of Reddit writings collected for early depression detection. Each
**subject** is one Reddit user, labelled `1` (depressed) or `0` (control) in a
ground-truth file. Each **post** is one submission or comment with a title, a
date, and a body.

It is not a dataset of numbers. It is 397,228 posts written by real people about
their own lives, some of them about their own depression, spanning eleven years.

## Layout on disk

```
chunk/
  chunk1/  subject1010_1.xml  subject1015_1.xml  ...
  ...
  chunk10/
risk-golden-truth-test.txt      # subject_id <TAB> label
writings-per-subject-all-test.txt
```

One XML file per subject, `rootTag=INDIVIDUAL`, one `WRITING` per post:

```xml
<INDIVIDUAL>
  <ID>subject1010</ID>
  <WRITING>
    <TITLE> TIL that Saigas ... </TITLE>
    <DATE> 2016-02-01 20:02:38 </DATE>
    <INFO> reddit post </INFO>
    <TEXT> ... </TEXT>
  </WRITING>
</INDIVIDUAL>
```

Two details that matter: `DATE` has **surrounding spaces**, and `subject_id` is
**not a tag** — it comes from the file name (`subject1010_1.xml` → `subject1010`).

## Shape of this particular copy

| | |
|---|---|
| XML files | 8,200 |
| Subjects in the truth file | 820 (741 control, 79 depressed) |
| Size on disk | ~184 MB |
| Writings | 544,447 |
| After cleaning | 397,228 atomic posts, 819 subjects with valid posts |
| Date range | 2006-07-13 → 2017-05-10 |

**The class imbalance is the first thing to know about this corpus**: 79 depressed
subjects against 740 controls, 9.4:1. Any comparison is dominated by the smaller
cohort, and the system now says so in the payload, in the critic's prose and in
the UI rather than letting a reader assume otherwise.

## How much of it survives cleaning

| outcome | rows | share |
|---|---|---|
| Silver posts | 397,228 | 73% |
| `null_or_blank_text` | 147,177 | 27% |
| `duplicate_post` | 42 | ~0% |

The 27% is real and worth understanding: those are submissions with a title and
an empty body. The previous implementation kept them as empty strings, which is
how a "mean text length" turns into a number about nothing.

## The data policy

The corpus is **never committed**. `processed_data/`, `chunk/` and `ignora/` are
gitignored, the container mounts them read-only, and no image layer contains
them.

Anything that ships with the repository must be **synthetic or aggregated**. That
is why:

- `scripts/generate_dev_sample.py` generates a deterministic synthetic corpus
  shaped exactly like the real one (XML files per subject, a labels TSV, and a
  few deliberately broken rows), instead of sampling real posts;
- `--from-corpus` exists for local work on the real distribution, and writes into
  gitignored `data/`;
- no test reads the real files, so the suite runs on a machine that has none.

This was a decision, not a default: the ticket asked for a 5,000-post sample of
the real corpus, and that sample would have published real people's writing about
their own depression.

## Derived artefacts

The lakehouse is reproducible from the raw files and is not committed either.
Under `data/` (gitignored) a full run produces ~240 MB:

```
data/lakehouse/
  bronze/posts, bronze/subjects
  silver/posts          (partitioned by label and year)
  silver/quarantine
  gold/gold_user_longitudinal_features
  gold/gold_cohort_lexical_matrix
  gold/gold_cohort_topic_matrix
  gold/gold_dataset_manifest.json
```

The manifest is the only part the agent reads at startup: 98 tokens of it are in
the system prompt, which is why a catalog question answers in one second.

## What the data can and cannot answer

**Can**: cohort comparisons of posting volume, rhythm, verbosity, session
structure and vocabulary; longitudinal features per subject; exploratory
questions like "how many posts mention cats?" (answered: 1,480 posts, 1,979
occurrences, 402 subjects).

**Cannot**: anything requiring clinical validation. There is no diagnosis here,
no severity, no treatment, no outcome. The labels are a research ground truth of
unknown provenance per subject, and the corpus is skewed, self-selected Reddit
data. The system's job is to keep that distinction visible in every answer, not
to pretend the numbers mean more than they do.
