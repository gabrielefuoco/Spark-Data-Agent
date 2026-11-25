# Ticket: 07 — Tipizzazione Statica con Mypy

Type: task
Status: resolved
Assignee:

Blocked by: 06

## Context

La sezione 7 della mappa prometteva `pyproject.toml` con «Ruff, Mypy e Pytest». Ruff e Pytest sono stati configurati nel ticket 01; **Mypy non è mai stato aggiunto**, e nessuno se n'è accorto perché nessun comando lo eseguiva. Risultato: ~6.400 righe con annotazioni di tipo scritte a mano e mai verificate da uno strumento.

## Task

1. Configurare `[tool.mypy]` in `pyproject.toml` con un livello di rigidità che il codice possa davvero mantenere, e aggiungere `mypy` al gruppo `dev` delle dipendenze.
2. Azzerare gli errori che emergono senza cambiare comportamento.

## Answer

**25 nov 2026** — Fatto. `mypy` gira su `src/` con `check_untyped_defs`, `disallow_untyped_defs`, `disallow_incomplete_defs`, `disallow_untyped_calls`, `warn_return_any`, `warn_unused_ignores`, `warn_redundant_casts`, `ignore_missing_imports`: sette errori iniziali, zero finali su 27 file. `mypy>=1.11` è nel gruppo `dev`.

**Due errori erano bug, non rumore di tipi:**

- `graph.py` faceva `float()` su un valore che poteva essere `None`: un `TypeError` latente alla prima query con aggregato vuoto. Sostituito con `_as_finite_float(value: object) -> float | None`.
- `_latest_user_question` era annotata `list[BaseMessage]` ma riceve l'unione dei messaggi di LangGraph; ora prende una `Sequence`.

Gli altri tre sono lacune di tipizzazione di terze parti (sqlglot, pandera, pydantic-settings) risolte al punto di chiamata, con un solo `# type: ignore[call-arg]` ristretto per `_env_file`.

**Nota operativa**: `mypy` installato a mano nel container sparisce a ogni ricreazione. Ora è una dipendenza `dev`, quindi sta nell'immagine.
