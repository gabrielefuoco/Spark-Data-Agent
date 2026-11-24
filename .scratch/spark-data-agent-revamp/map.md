# Map: Riprogettazione Spark Data Agent (Lakehouse Medallion + LangGraph)

## Destination

Un repository completamente refattorizzato in lingua inglese (`spark-data-agent`), basato su un'architettura **PySpark Medallion Lakehouse** (Bronze/Silver/Gold su Parquet ZSTD con lineage e quarantena) e orchestrato da un **LangGraph StateGraph ciclico** alimentato da modelli multi-provider via **OpenRouter**, con vero human-in-the-loop per job pesanti, guardie statistiche di significatività e test suite con Pytest.

## Notes

- **Input Senior**: Sezione 3 e Sezione 1.B di `Ottimizzazione CV e Architettura Progetti.md`.
- **Obiettivo prioritario**: Dimostrare padronanza ingegneristica inequivocabile nel trattamento di grandi moli di dati (Data Engineering avanzato) e nell'orchestrazione di agenti deterministici.
- **Standing preferences**:
  - Codice, commenti, docstring, commit e ADR rigorosamente in **100% inglese**.
  - Passaggio da LlamaIndex a **LangGraph** (StateGraph esplicito con checkpointer `SqliteSaver`).
  - Passaggio da Gemini cablato ad architettura duale: **Ollama locale** (sviluppo offline a costo zero) + **OpenRouter API** remota (DeepSeek Chat/R1 per routing deterministico e critica scientifica).
  - Integrazione storage cloud (connettore S3 / MinIO locale) per poter esporre AWS/Cloud nel CV.
- **Skills da consultare**: `codebase-design`, `tdd`, `implement`, `domain-modeling`.

## Decisions so far

- **Ambiente di Sviluppo Linux/Docker**: Esecuzione Spark isolata in container Docker / Devcontainer Linux (con Java JDK 17, PySpark 3.5+, MinIO e bridge per Ollama su `host.docker.internal`), azzerando le problematiche di `winutils.exe` su Windows.
- **Provider Duale**: Ollama locale per iterazioni veloci a costo zero; OpenRouter DeepSeek per esecuzione remota/produzione.
- **Fixture Dataset**: Script per generare un sample compatto (~5.000 post / 100 soggetti) per test agili offline.

## Child tickets

- [01 — Scaffolding del Progetto in Inglese e Code Hygiene (pyproject.toml, Settings e Fix Shadowing)](issues/01-project-scaffolding-hygiene-and-settings.md) — Frontier (Unblocked)
- [02 — Architettura Medallion Lakehouse (Bronze, Silver Atomico con Quarantena e Gold Tables)](issues/02-medallion-lakehouse-pipeline.md) — Blocked by 01
- [03 — Operazioni PySpark Avanzate (Window Functions Circadiane, Spark ML LDA, Broadcast Join e AST Guard)](issues/03-advanced-pyspark-operations.md) — Blocked by 02
- [04 — Data Quality con Pandera e Guardia di Significatività Statistica a Runtime](issues/04-data-quality-and-statistical-guard.md) — Blocked by 02
- [05 — Orchestrazione Ciclica con LangGraph, OpenRouter Multi-Modello e Human-in-the-Loop (HITL)](issues/05-langgraph-cyclic-orchestration-openrouter-hitl.md) — Blocked by 03, 04
- [06 — Connettore Cloud S3 / MinIO, Suite di Test Pytest e Formalizzazione ADR](issues/06-cloud-storage-s3-and-pytest-suite.md) — Blocked by 05

## Not yet specified

### 1. Risoluzione dei 4 Anti-Pattern del Codice Attuale
- **Fix OOM Time-Bomb (`collect_list`)**: Eliminare `groupBy("subject_id").agg(collect_list("text"))` e `concat_ws` in `spark_utils.py` che causavano skew estremo e OOM su larga scala. Mantenere i post atomici (1 riga = 1 post).
- **Rimozione `.toPandas()` e `.persist()` indiscriminati**: Sostituire il caching arbitrario in RAM con strategie calibrate (`MEMORY_AND_DISK`) e unpersist esplicito; eliminare i 4 `.count()` real-time all'avvio.
- **Partizionamento e Lineage**: Sostituire le scritture piatte con partizionamento fisico su disco (`partitionBy("label", "year")`) compresso in ZSTD.
- **Superamento dell'Agente Passacarte**: Rimpiazzare LlamaIndex AgentWorkflow con uno StateGraph deterministico che governa l'intero ciclo di vita dell'analisi.

### 2. Architettura PySpark Medallion Lakehouse (Parquet ZSTD)
- **Bronze Layer (`data/lakehouse/bronze/`)**: Ingestione raw XML e TSV immutabili con colonne di lineage: `_source_file` (`input_file_name()`), `_ingested_at` (`current_timestamp()`), `_row_hash` (`sha2`).
- **Silver Layer (`data/lakehouse/silver/`)**:
  - `silver_posts`: Post atomici, normalizzati, deduplicati su `(subject_id, date, _row_hash)`, partizionati per `label` e `year`.
  - `silver_quarantine.parquet` (Dead-Letter Pattern): Tabelle di scarto per post con date malformate o chiavi orfane, tracciando `_rejection_reason`.
- **Gold Layer (`data/lakehouse/gold/`)**: Tabelle pre-calcolate per query sub-secondo:
  - `gold_user_longitudinal_features`: Metriche comportamentali e temporali aggregate per utente.
  - `gold_cohort_lexical_matrix`: Top N-grammi, TF-IDF e distribuzioni topic LDA per coorte (`label=1` vs `label=0`).
  - `gold_dataset_manifest.json`: Metadati e statistiche di catalogo pre-computate per rispondere a `get_data_info` in 0.1ms.

### 3. Implementazione delle 4 Operazioni PySpark Avanzate
1. **Window Functions Avanzate (Ritmo Circadiano & Sessionizzazione)**:
   - Burstiness e Session Detection tramite `lag("date", 1).over(w)` e calcolo del delta temporale in secondi.
   - *Night-Time Posting Index (00:00–05:00)*: Calcolo della percentuale di post notturni e media mobile a 30 giorni (`rowsBetween(-30, 0)`), marcatore clinico di insonnia/depressione.
   - *Rolling Verbosity Z-Score*: Deviazione standard mobile della lunghezza post rispetto alla cronologia personale dell'utente.
2. **Distributed NLP & Spark ML Pipeline**:
   - Pipeline nativa `pyspark.ml`: `RegexTokenizer` → `StopWordsRemover` custom → `NGram(n=2)` (cattura bi-grammi clinici come *"feel alone"*, *"cant sleep"*) → `CountVectorizer` → `IDF` → **`LDA(k=8, maxIter=20)`** per estrazione di topic latenti senza `explode` selvaggi.
3. **Broadcast Joins & Mitigazione Data Skew (AQE)**:
   - Abilitazione dell'Adaptive Query Execution nella SparkSession (`adaptive.enabled`, `coalescePartitions.enabled`, `skewJoin.enabled`).
   - Utilizzo di `F.broadcast(df_truth)` nelle join tra milioni di post e la tabella soggetti.
4. **Tool Spark SQL Dinamico e Protetto (`execute_safe_spark_sql`)**:
   - Supporto a query SELECT generate dall'LLM su viste registrate.
   - Validazione AST con **sqlglot**: blocco categorico di comandi DDL/DML (DROP, ALTER, INSERT, DELETE) e iniezione automatica di `LIMIT <= 500`.

### 4. Data Quality & Statistical Guard
- **Validazione Schema in Ingestione (Pandera su PySpark)**: Contratti formali di schema (`PostSilverSchema`) eseguiti in pipeline ETL.
- **Nodo Deterministico `data_quality_and_stats_guard` a Runtime**:
  - Controllo di consistenza: verifica che ciascun gruppo abbia almeno $N \ge 30$ soggetti unici.
  - Test di Ipotesi Automatico: Esecuzione di Welch's t-test o Mann-Whitney U con calcolo dell'Effect Size (Cohen's *d*) per verificare se la differenza rilevata è statisticamente significativa ($p < 0.05$) o rumore statistico.
  - Vincolo rigido di stato per il nodo critico finale a menzionare campionamento e $p$-value.

### 5. Orchestrazione LangGraph & Human-in-the-Loop (HITL)
- **Topologia dello StateGraph**:
  - `Node 1: Intent Router & Query Planner` (Qwen 2.5 72B / GPT-4o-mini).
  - `Node 2: AST SQL Validator & Spark EXPLAIN Cost Estimator`.
  - `INTERRUPT (HITL)`: Se la query richiede scansione full-scan o ricalcolo LDA pesante, interruzione del flusso con `interrupt_before` e checkpointer `SqliteSaver`; la UI mostra tempo stimato e pulsante di approvazione.
  - `Node 3: PySpark Tool & Safe SQL Executor`.
  - `Node 4: Runtime Data Quality & Statistical Significance Guard`.
  - `Node 5: Self-Correcting Query Rewriter` (retry loop $\le 2$ tentativi su errori Spark o risultati vuoti).
  - `Node 6: Scientific Interpreter & Critic` (DeepSeek-R1 / Claude 3.5 Sonnet).
- **Configurazione OpenRouter**: Modulo centralizzato con `pydantic-settings` per gestire API key, routing di modelli e fallback.

### 6. Integrazione Cloud Storage (AWS S3 / MinIO)
- Connettore PySpark per lettura e scrittura su bucket S3 compatibile (`s3a://` o client `boto3`), con fallback trasparente su MinIO locale in Docker per consentire l'esecuzione locale senza costi.

### 7. Code Hygiene e Struttura Modulare
- Eliminazione di import selvaggi (`from pyspark.sql import functions as F`).
- `pyproject.toml` moderno con configurazione di Ruff (linter/formatter), Mypy (type checking) e Pytest.
- Suite di test con 5-10 test unitari e di integrazione Pytest.

## Out of scope

- Utilizzo di LlamaIndex o Gemini cablato staticamente.
- Generazione di grafici non riproducibili o interpretazioni cliniche prive di metrica $p$-value.
- Modifiche del dataset originale o alterazione delle etichette scientifiche di truth.
