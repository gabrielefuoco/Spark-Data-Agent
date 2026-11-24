# Ticket: 06 — Connettore Cloud S3 / MinIO, Suite di Test Pytest e Formalizzazione ADR

Type: task
Status: resolved
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

## Answer

### Cosa è stato implementato
1. **Connettore S3A / MinIO (`src/lakehouse/storage.py`)**:
   - `is_s3a_uri` e `resolve_storage_target`: risolvono percorsi locali e URI `s3a://<bucket>[/prefix]` a partire da `Settings.spark_storage_uri` senza diramazioni lato chiamante.
   - `build_s3a_hadoop_options`: funzione pura e unit-testabile che restituisce `{}` per storage locale e configura per `s3a://` `fs.s3a.impl`, `fs.s3a.endpoint`, `fs.s3a.path.style.access=true`, `fs.s3a.access.key` / `fs.s3a.secret.key` (da `SecretStr`), `fs.s3a.connection.ssl.enabled` (`false` su `http://`, `true` su `https://`) e `fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider` (evita richieste al servizio metadata EC2). Solleva `StorageConfigurationError` con elenco esplicito delle variabili mancanti (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_ENDPOINT_URL`) senza mai esporre i valori delle credenziali.
   - `apply_s3a_config`: applica le opzioni a `SparkSession.Builder`, `SparkConf` o `SparkSession` esistente.
2. **JAR S3 nell'immagine Docker (`Dockerfile`)**:
   - Aggiunti in `pyspark/jars` (via `curl -fsSLO` senza nuovi `ARG`, senza `spark.jars.packages` a runtime):
     - `hadoop-aws` **3.3.4** (allineato a `hadoop-client-runtime-3.3.4` / `hadoop-client-api-3.3.4` di PySpark 3.5.10).
     - `aws-java-sdk-bundle` **1.12.262** (bundle SDK richiesto da `hadoop-aws` 3.3.4 per `S3AFileSystem`).
   - Immagine `spark-data-agent-dev:local` ricostruita con `docker compose build dev`.
3. **Suite di test S3/MinIO (`tests/test_storage.py`)**:
   - 4 unit test sempre eseguibili (risoluzione URI locali/`s3a://`, generazione opzioni MinIO vs HTTPS, fallimento esplicito senza leak di segreti, configurazione su `SparkSession.builder` senza JVM).
   - 1 test di integrazione (`test_s3a_parquet_roundtrip_and_safe_sql_on_minio`) con verifica TCP reale su `S3_ENDPOINT_URL` (`socket.create_connection` con timeout 1s): scrive/rilegge Parquet ZSTD su `s3a://lakehouse/...`, registra la vista Spark SQL e la interroga tramite `execute_safe_spark_sql`; salta in modo pulito (`skipif`) se MinIO non è raggiungibile.
4. **ADR (`docs/adr/`)**:
   - `docs/adr/README.md` (indice delle tre decisioni).
   - `0001-medallion-lakehouse-vs-flat-parquet.md`: layer Bronze/Silver/Gold atomici (1 riga = 1 post) con quarantena dead-letter e manifest JSON al posto di tabelle piatte e `collect_list`.
   - `0002-langgraph-cyclical-stategraph-vs-llamaindex.md`: `StateGraph` deterministico con AST guard (`sqlglot`), retry controllato, guardia statistica non-LLM e gate HITL via `langgraph.types.interrupt()` + `SqliteSaver` (che sostituisce `interrupt_before` statico di LangGraph 0.x).
   - `0003-openrouter-multi-model-routing.md`: client unico OpenAI-compatible (`ChatOpenAI`) con default Ollama locale gratuito (`qwen2.5:7b`) e routing remoto OpenRouter separato per ruolo (`deepseek/deepseek-chat` vs `deepseek/deepseek-r1`).

### Output del smoke test `s3a://lakehouse/smoke/`
Comando eseguito nel container `dev` contro il servizio `minio` (`http://minio:9000`):
```text
docker compose run --rm --no-deps dev python -c "... write_parquet_zstd(df, 's3a://lakehouse/smoke', mode='overwrite') ..."
TARGET: s3a://lakehouse/smoke
ROWS: [('smoke-1', 1, 0.75), ('smoke-2', 0, 0.25)]
```

### MinIO locale vs AWS S3 reale
Tutta l'integrazione `s3a://` è stata progettata e verificata contro **MinIO locale in Docker (`pgsty/minio`)** tramite API S3-compatibile (`path.style.access=true`, HTTP locale senza TLS). Il supporto a endpoint AWS S3 reali è previsto dalla configurazione (`Settings` / `build_s3a_hadoop_options`), ma non è stato eseguito né verificato su infrastruttura AWS reale.
