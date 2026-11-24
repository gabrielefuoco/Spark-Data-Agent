# Ticket: 02 — Architettura Medallion Lakehouse (Bronze, Silver Atomico con Quarantena e Gold Tables)

Type: task
Status: ready-for-agent
Assignee: 

Blocked by: 01

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
