# Ticket: 06 — Connettore Cloud S3 / MinIO, Suite di Test Pytest e Formalizzazione ADR

Type: task
Status: ready-for-agent
Assignee: 

Blocked by: 05

## Nota dal ticket 00

- Il compose esiste già (`docker-compose.yml`): servizio `dev` (Spark `local[*]`, Python 3.11, JDK 17) e `minio` con bucket `lakehouse` creato dal job `minio-init`. Non riscriverlo: estendilo.
- **Le immagini ufficiali MinIO non sono più distribuibili**: `minio/minio` dà *pull access denied* su Docker Hub e `quay.io/minio/*` risponde 401. Il compose usa il fork community `pgsty/minio` + `pgsty/mc` con release pinnate. Qualsiasi ricetta trovata in rete che usi `minio/minio` non funziona.
- L'immagine ha già `spark-xml` in `pyspark/jars`; i JAR S3 (`hadoop-aws` 3.3.4 + `aws-java-sdk-bundle` 1.12.262, ~300 MB) vanno aggiunti allo stesso modo dentro il Dockerfile in questo ticket, non con `spark.jars.packages` (in container il download da Maven è lento e fragile).
- Verificato in ticket 00: Spark 3.5.1, Hadoop 3.3.4, JVM 17, scrittura Parquet ZSTD, bridge `host.docker.internal` verso Ollama.

## Context

Per completare la trasformazione di `spark-data-agent` in un flagship project di livello Enterprise, è necessario dotarlo di tre elementi che differenziano nettamente un progetto amatoriale da uno professionale:
1. Integrazione con storage cloud S3-compatibile (con fallback trasparente su MinIO locale in Docker per garantire riproducibilità a costo zero) che legittima l'esposizione di competenze Cloud/AWS nel CV.
2. Una robusta suite di test automatizzati con Pytest che copre la logica Spark, la validazione Pandera, le guardie AST e i nodi LangGraph.
3. Architecture Decision Records (ADR) formali in `docs/adr/` che documentano le scelte architetturali chiave.

## Task

1. **Connettore Cloud Storage S3 e MinIO (`src/lakehouse/storage.py`)**:
   - Configurare la SparkSession con i package Hadoop AWS (`hadoop-aws`, `aws-java-sdk-bundle`) o un client astratto di storage:
     - Supporto per protocolli `s3a://` (o `s3://` tramite boto3/fsspec).
     - Riconoscimento automatico delle credenziali configurate in `Settings` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_ENDPOINT_URL`).
   - Fornire un setup completo `docker-compose.yml` e configurazione `.devcontainer/`:
     - Container ambiente di sviluppo basato su Linux con Python 3.11 e OpenJDK 17 preinstallati (eliminando radicalmente la necessità di `winutils.exe` su Windows).
     - Servizio MinIO locale con bucket S3 pre-inizializzato (`lakehouse`) per testare scritture `s3a://` a costo zero.
     - Configurazione di rete con `extra_hosts: ["host.docker.internal:host-gateway"]` per consentire all'agente in container di interrogare l'istanza locale di Ollama in esecuzione sulla macchina host.

2. **Suite di Test Pytest (`tests/`)**:
   - Implementare una suite completa di test (5-10 test strutturati):
     - `test_lakehouse_lineage_and_quarantine.py`: verifica che i record con campi nulli o date errate vengano convogliati in `silver_quarantine.parquet` con `_rejection_reason` corretta, e che le colonne `_source_file`, `_ingested_at`, `_row_hash` siano presenti.
     - `test_window_circadian_features.py`: test con dati mock temporali per verificare il calcolo esatto della finestra temporale notturna (00:00–05:00) e del delta di burstiness.
     - `test_sqlglot_ast_security.py`: test su `execute_safe_spark_sql` per verificare il blocco categorico di iniezioni DDL/DML (`DROP TABLE`, `DELETE FROM`, `UPDATE`) e l'imposizione del vincolo `LIMIT <= 500`.
     - `test_statistical_significance_guard.py`: test del nodo deterministico con campioni sintetici noti per verificare che respinga campioni $N < 30$ e calcoli correttamente p-value e Cohen's *d*.
     - `test_agent_graph_execution.py`: test end-to-end con mock LLM per verificare il ciclo di StateGraph, il retry loop in caso di query malformata e il salvataggio dei checkpoint con `SqliteSaver`.

3. **Formalizzazione Architecture Decision Records (`docs/adr/`)**:
   - Redigere 3 ADR formali in inglese seguendo il template standard (Title, Status, Context, Decision, Consequences):
     - `docs/adr/0001-medallion-lakehouse-vs-flat-parquet.md`: Motiva l'architettura a tre livelli Bronze/Silver/Gold e la quarantena Dead-Letter rispetto a tabelle aggregate piatte con `collect_list`.
     - `docs/adr/0002-langgraph-cyclical-stategraph-vs-llamaindex.md`: Motiva la migrazione a LangGraph per controllo del flusso deterministico, self-correction e gate HITL.
     - `docs/adr/0003-openrouter-multi-model-routing.md`: Documenta la strategia di disaccoppiamento tra modelli di routing veloci a basso costo (Qwen 2.5 72B / GPT-4o-mini) e modelli di ragionamento profondo per la critica scientifica (DeepSeek-R1 / Claude 3.5 Sonnet).
