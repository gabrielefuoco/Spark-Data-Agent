# Ticket: 02 — Architettura Medallion Lakehouse (Bronze, Silver Atomico con Quarantena e Gold Tables)

Type: task
Status: resolved
Assignee:

Blocked by: 00, 01

## Context

L'attuale pipeline in `spark_utils.py` carica i dati TSV/XML in modo monolitico, commettendo diversi gravi errori architetturali:
1. Esegue 4 chiamate bloccanti a `.count()` all'avvio dell'applicazione Streamlit, bloccando l'interfaccia per minuti.
2. Utilizza `groupBy("subject_id").agg(collect_list("text"))` e `concat_ws` creando una bomba a orologeria OOM per via dell'accumulo di milioni di post in singole righe di memoria (skew estremo).
3. Esegue `.persist()` su DataFrame giganteschi senza strategia di caching calibrata e salva i dati in formati piatti non partizionati.

## Task

Progettare ed implementare il package `src/lakehouse/` con una pipeline Medallion Lakehouse su formato Parquet compresso con ZSTD (`compression="zstd"`):

1. **Bronze Layer (`src/lakehouse/bronze.py`)**:
   - Ingestione raw immutabile dei file originali (XML dei post Reddit e TSV dei soggetti con etichette di depressione).
   - Aggiunta sistematica dei metadati di data lineage e tracciabilità:
     - `_source_file`: derivato tramite `F.input_file_name()`
     - `_ingested_at`: derivato tramite `F.current_timestamp()`
     - `_row_hash`: calcolato come `F.sha2(F.concat_ws("||", ...), 256)`
   - Scrittura su `data/lakehouse/bronze/posts/` e `data/lakehouse/bronze/subjects/`.

2. **Silver Layer con Dead-Letter Quarantine (`src/lakehouse/silver.py`)**:
   - **Silver Posts Atomici**: Mantenere la granularità atomica (1 riga = 1 post), eliminando tassativamente ogni uso di `collect_list`.
   - Normalizzazione, pulizia del testo e parsing del timestamp (`yyyy-MM-dd HH:mm:ss`).
   - Deduplicazione deterministica basata su `(subject_id, date, _row_hash)`.
   - **Dead-Letter Pattern (`silver_quarantine.parquet`)**: Isolare e salvare in una cartella di quarantena separata tutti i record con date malformate, campi obbligatori nulli o `subject_id` orfani non presenti nel file di etichette, allegando una colonna esplicita `_rejection_reason`.
   - Partizionamento fisico su disco: salvare `data/lakehouse/silver/posts/` partizionato per `label` e `year` (`.partitionBy("label", "year")`).

3. **Gold Layer con Metadati Sub-secondo (`src/lakehouse/gold.py`)**:
   - Pre-calcolare tabelle aggregate analitiche salvate in Parquet ZSTD:
     - `gold_user_longitudinal_features`: metriche temporali e comportamentali aggregate per singolo utente.
     - `gold_cohort_lexical_matrix`: distribuzioni lessicali, frequenze n-grammi e topic estratti per coorte (`label=1` vs `label=0`).
   - **`gold_dataset_manifest.json`**: Script ETL che genera un manifest JSON statico con statistiche globali (conteggio totale post, numero soggetti per classe, range temporale date, volumi storage). Questo manifest permette al tool `get_data_info` dell'agente di rispondere in 0.1ms senza svegliare cluster o sessioni Spark.

4. **Eliminazione Anti-Pattern di Startup**:
   - Rimuovere dall'inizializzazione dell'app le 4 chiamate bloccanti a `.count()`.
   - Sostituire `.persist()` indiscriminati con persistenza calibrata `StorageLevel.MEMORY_AND_DISK` usata solo quando un dataframe intermedio viene riutilizzato in rami di join divergenti, con chiamata esplicita a `.unpersist()`.

5. **Generatore di Fixture Subset per Sviluppo Locale (`scripts/generate_dev_sample.py`)**:
   - Implementare uno script CLI riproducibile per campionare un subset stratificato (es. 100 soggetti bilanciati `label=1` vs `label=0`, ~5.000 post atomici) a partire dal dataset integrale eRISK, salvandolo in `data/sample/`.
   - Questo consente di iterare a velocità massima durante lo sviluppo locale e l'esecuzione di test con PySpark ed Ollama a costo computazionale zero, lasciando l'esecuzione sul dataset integrale ai job di ingestione completa.


## Answer

### Componenti realizzati

- **`src/spark.py`**: factory `build_spark_session(settings, shuffle_partitions=...)` con configurazione da `Settings`, bind esplicito su `127.0.0.1`, log level `ERROR`, helper `stop()` e `write_parquet_zstd(..., compression="zstd", partition_by=...)` senza alcun `spark.jars.packages` a runtime.
- **`src/lakehouse/paths.py`**: dataclass `LakehousePaths` derivata da `Settings.lakehouse_root` per risolvere in modo uniforme tutti i percorsi Bronze, Silver, Gold, Manifest e Sample senza literal sparsi.
- **`src/lakehouse/bronze.py`**: ingestione raw immutabile in Parquet ZSTD per post XML (`INDIVIDUAL`/`WRITING` con schema esplicito, estrazione `subject_id` dal nome file tramite `F.input_file_name()`) e TSV delle etichette, aggiungendo `_source_file`, `_ingested_at` e `_row_hash` (`F.sha2` su `F.concat_ws("||", ...)` con `F.coalesce` sui null per evitare collisioni di shift).
- **`src/lakehouse/silver.py`**: trasformazione atomica `1 riga = 1 post` (nessun `collect_list`), pulizia whitespace, parsing `to_timestamp(trim(date), "yyyy-MM-dd HH:mm:ss")`, broadcast join con i soggetti deduplicati, deduplicazione deterministica su `(subject_id, date, _row_hash)`, persistenza calibrata `StorageLevel.MEMORY_AND_DISK` con `.unpersist()` in `finally`, scrittura partizionata `.partitionBy("label", "year")` in `silver/posts/` e instradamento di tutti i record scartati in `silver/quarantine/`.
- **`src/lakehouse/gold.py` e `src/lakehouse/manifest.py`**: materializzazione di `gold_user_longitudinal_features` (metriche temporali, active days, posts per active day, lunghezza testo, night-post ratio 00:00–05:00) e `gold_cohort_lexical_matrix` (`RegexTokenizer`, `StopWordsRemover`, `NGram(n=2)` senza LDA), più `build_dataset_manifest` / `load_manifest()` su `gold_dataset_manifest.json` in pura lettura JSON senza alcun job Spark.
- **`scripts/generate_dev_sample.py`**: CLI deterministica da `--seed` per generare un corpus sintetico eRISK-shaped (`chunk/chunk1/subjectNNN_1.xml` e `risk-golden-truth-test.txt`, default 100 soggetti bilanciati e 50 post/soggetto multi-anno) più modalità opzionale `--from-corpus PATH` con avviso su stderr che l'output è gitignored e non va mai committato.
- **Test (`tests/conftest.py`, `tests/test_generator.py`, `tests/test_bronze.py`, `tests/test_silver.py`, `tests/test_gold.py`)**: suite TDD con sessione Spark `local[2]` e `shuffle_partitions=2`.

### Decisioni progettuali e motivazioni

- **Ragioni di quarantena e invariante di accounting**: in `silver.py` sono state definite cinque ragioni esplicite (`missing_subject_id`, `malformed_date`, `null_or_blank_text`, `orphan_subject_id`, `duplicate_post`). La verifica di `missing_subject_id` precede `orphan_subject_id` perché un `subject_id` assente produce `label = NULL` nella left join. Inoltre, i duplicati esatti rilevati dalla window deterministica su `(subject_id, date, _row_hash)` vengono instradati in quarantena con `_rejection_reason = "duplicate_post"` anziché eliminati silenziosamente: in questo modo l'invariante `bronze_count == silver_count + quarantine_count` resta vera in modo assoluto anche in presenza di duplicati.
- **Scelta della fixture sintetica**: per rispettare la policy sui dati reali (`chunk/`, `processed_data/`, `ignora/`), tutti i test usano esclusivamente il generatore sintetico deterministico (`generate_synthetic_sample` in `tmp_path`), che modella sia le caratteristiche strutturali reali (spazi attorno a `<DATE>`, `<TITLE>` talvolta vuoto, distribuzione multi-anno e pattern circadiani distinti per coorte) sia i casi limite di quarantena (`--no-broken` per escluderli), senza mai includere testo reale del corpus.
- **Zero Spark job in `load_manifest()`**: `load_manifest()` legge il JSON con Pydantic (`DatasetManifest.model_validate_json`) senza importare né interrogare `SparkSession`, verificato nei test sia controllando `dagScheduler().nextJobId()` sia con monkeypatch su `SparkSession.getActiveSession`.

### Output esatto dei test e linter

Eseguito nel container `dev` (`docker compose run --rm --no-deps dev ...`):

```text
$ docker compose run --rm --no-deps dev python -m pytest -v
============================= test session starts ==============================
platform linux -- Python 3.11.18, pytest-9.2.0, pluggy-1.7.0
rootdir: /workspace
configfile: pyproject.toml
testpaths: tests
plugins: asyncio-1.5.0, langsmith-0.16.2, typeguard-4.6.1, anyio-4.16.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collected 14 items

tests/test_bronze.py ..                                                  [ 14%]
tests/test_config.py ....                                                [ 42%]
tests/test_generator.py ....                                             [ 71%]
tests/test_gold.py .                                                     [ 78%]
tests/test_silver.py ...                                                 [100%]

============================= 14 passed in 23.10s ==============================

$ docker compose run --rm --no-deps dev python -m ruff check .
All checks passed!
```
