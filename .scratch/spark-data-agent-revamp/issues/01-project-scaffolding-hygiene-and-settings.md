# Ticket: 01 — Scaffolding del Progetto in Inglese e Code Hygiene (pyproject.toml, Settings e Fix Shadowing)

Type: task
Status: ready-for-agent
Assignee: 

Blocked by: none

## Context

L'attuale repository `Progetto Big Data` presenta diversi difetti di igiene del software rilevati dal Senior che tradiscono l'origine accademica del codice:
1. Nomi di file e commenti in italiano (`Agente.py`, stringhe nei tool).
2. Shadowing delle funzioni built-in di Python in `spark_utils.py` dovuto a `from pyspark.sql.functions import *` e all'import esplicito di `min, max, sum`.
3. Dipendenze fantasma non usate in `requirements.txt` (`psycopg2`, `schedule`, `openpyxl`).
4. Gestione insicura delle API key tramite lettura grezza del file `src/api.env`.

## Task

1. **Scaffolding Struttura Modulare in 100% Inglese**:
   Riorganizzare il repository con il seguente layout:
   ```text
   spark-data-agent/
   ├── .devcontainer/             # Devcontainer Linux con Java JDK 17 e PySpark
   ├── docker-compose.yml         # MinIO + Spark worker + network bridge Ollama
   ├── src/
   │   ├── config.py              # Pydantic BaseSettings per Ollama/OpenRouter e Spark
   │   ├── lakehouse/             # Pipeline Bronze/Silver/Gold ETL
   │   ├── analytics/             # Operazioni PySpark avanzate e Spark ML
   │   ├── agent/                 # Grafo LangGraph, nodi, guardie e checkpointer
   │   └── ui/                    # Streamlit app e visualizzazioni Plotly
   ├── tests/                     # Suite Pytest
   ├── docs/adr/                  # Architecture Decision Records
   └── pyproject.toml             # Configurazione ruff, mypy, pytest
   ```

2. **Configurazione pyproject.toml**:
   - Abbandonare `requirements.txt` e configurare `pyproject.toml` moderno con Poetry o uv/pip-tools.
   - Eliminare le dipendenze inutilizzate (`psycopg2`, `schedule`, `openpyxl`, `FPGrowth`).
   - Includere: `pyspark>=3.5.0`, `langgraph>=0.2.0`, `langchain-core`, `pydantic-settings`, `sqlglot`, `pandera[pyspark]`, `scipy`, `streamlit`, `plotly`.
   - Configurare i tool di qualità: Ruff (linter/formatter) e Pytest.

3. **Modulo Configurazione Centralizzato (`config.py`)**:
   - Implementare `Settings(BaseSettings)` con Pydantic V2 per leggere da variabili d'ambiente o file `.env`:
     - `LLM_BACKEND`: `"ollama"` (default per sviluppo locale a costo zero) oppure `"openrouter"` (per produzione/cloud)
     - `OLLAMA_BASE_URL`: Default `"http://localhost:11434/v1"`
     - `OLLAMA_MODEL`: Default `"qwen2.5:7b"` (o `deepseek-r1:7b` / `mistral`)
     - `OPENROUTER_API_KEY`: SecretStr (opzionale in locale)
     - `OPENROUTER_BASE_URL`: Default `"https://openrouter.ai/api/v1"`
     - `REMOTE_ROUTER_MODEL`: Default `"deepseek/deepseek-chat"` (o Qwen 2.5 72B)
     - `REMOTE_REASONING_MODEL`: Default `"deepseek/deepseek-r1"` (o DeepSeek V3)
     - `SPARK_STORAGE_URI`: Percorso lakehouse locale o bucket S3 compatibile
   - Fornire un template `.env.example` pulito.

4. **Risoluzione Shadowing Funzioni Native**:
   - Rimuovere tassativamente `from pyspark.sql.functions import *`.
   - Adottare rigorosamente la convenzione industriale `from pyspark.sql import functions as F` e `from pyspark.sql.types import ...` in tutti i moduli.
