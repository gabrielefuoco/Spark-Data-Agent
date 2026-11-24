# Agent Instructions

## Project specific

- **Plan of record**: `.scratch/spark-data-agent-revamp/map.md` — six tickets, in dependency order. Read the map before touching code; the current implementation is the *before* state it describes.
- **Current vs target**: `CONTEXT.md` states what the code does today and what the rewrite targets. Where they disagree, the code is wrong.
- **Data policy**: `processed_data/`, `chunk/` and `ignora/` hold the eRISK corpus — real posts from real people. They stay out of git, and any fixture committed must be synthetic or aggregated. Never `git add -A` in this repository without reading what it would stage.
- **Language**: code, comments, docstrings, commit messages and ADRs in **English**. Tracker prose stays in Italian.
- **Windows**: Spark on this host needs Java 17 and the `winutils` workaround; the revamp targets a Linux devcontainer instead of fighting it.

## Agent skills

### Issue tracker

Local markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`CONTEXT.md` and `docs/adr/`). See `docs/agents/domain.md`.
