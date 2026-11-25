# Agent Instructions

## Project specific

- **Plan of record**: `.scratch/spark-data-agent-revamp/map.md` — six tickets, in dependency order. Read the map before touching code; the current implementation is the *before* state it describes.
- **Current vs target**: `CONTEXT.md` states what the code does today and what the rewrite targets. Where they disagree, the code is wrong.
- **Data policy**: `processed_data/`, `chunk/` and `ignora/` hold the eRISK corpus — real posts from real people. They stay out of git, and any fixture committed must be synthetic or aggregated. Never `git add -A` in this repository without reading what it would stage.
- **Language**: user-facing Streamlit UI copy, HITL approval tokens (`"Autorizza"`, `"Annulla"`), and tracker prose are **Italian** (for the demo); all code, comments, docstrings, commit messages, and ADRs are **English**.
- **Windows**: Spark on this host needs Java 17 and the `winutils` workaround; the revamp targets a Linux devcontainer instead of fighting it. That container now exists (ticket 00): run everything with `docker compose run --rm --no-deps dev python -m pytest` — tests and Spark jobs execute there, never on the Windows host, so there is only one Python (3.11) and one JDK (17) to keep green.
- **Docker**: Docker Desktop must be started by hand (no daemon runs on its own); `docker compose up -d` brings up the dev container plus MinIO with a pre-created `lakehouse` bucket.

## Agent skills

### Issue tracker

Local markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`CONTEXT.md` and `docs/adr/`). See `docs/agents/domain.md`.
