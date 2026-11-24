# Wayfinder Master Registry (.scratch)

Mappa delle iniziative di refactoring e architettura per il repository **Spark Data Agent** (`Progetto Big Data`).

## Mappe Attive

| Mappa | Directory | Destinazione Principale |
| :--- | :--- | :--- |
| **Riprogettazione Spark Data Agent** | [`spark-data-agent-revamp/map.md`](spark-data-agent-revamp/map.md) | Architettura **PySpark Medallion Lakehouse** (Bronze/Silver/Gold su Parquet ZSTD con lineage e dead-letter quarantine) e StateGraph ciclico **LangGraph** (Ollama locale + OpenRouter DeepSeek remoto, HITL gate e statistical significance guard). |
