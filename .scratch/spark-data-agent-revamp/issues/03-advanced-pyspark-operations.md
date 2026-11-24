# Ticket: 03 — Operazioni PySpark Avanzate (Window Functions Circadiane, Spark ML LDA, Broadcast Join e AST Guard)

Type: task
Status: resolved
Assignee:

Blocked by: 02

## Context

Il Senior Advisor ha evidenziato che per posizionarsi come "AI & Data Systems Engineer" il repository non può limitarsi a banali `filter` o `groupBy.count()`, ma deve esibire feature ingegneristiche complesse che dimostrino una padronanza autentica di Apache Spark distribuito: Window functions con semantica clinica, pipeline native Spark ML, gestione del data skew tramite AQE e broadcast joins, e un motore SQL protetto per consentire query analitiche flessibili all'agente senza rischi di sicurezza.

## Task

Implementare in `src/analytics/` i moduli per le 4 operazioni PySpark avanzate:

1. **Window Functions Avanzate: Ritmo Circadiano e Sessionizzazione (`src/analytics/features.py`)**:
   - Definire una finestra di partizionamento utente ordinata temporalmente: `Window.partitionBy("subject_id").orderBy("date")`.
   - **Burstiness e Rilevamento Sessioni**: Calcolare la differenza temporale tra post consecutivi con `F.lag("date", 1).over(w)` per identificare cluster di messaggi ravvicinati (sessioni attive vs pause).
   - **Night-Time Posting Index (00:00–05:00)**: Flag binario per post pubblicati tra mezzanotte e le 5 del mattino. Calcolare la media mobile a 30 giorni (`.rowsBetween(-30, 0)`) della percentuale di post notturni per utente, utilizzabile come proxy clinico di insonnia e alterazione del ritmo circadiano.
   - **Rolling Verbosity Z-Score**: Calcolo della lunghezza media e deviazione standard mobile dei post per evidenziare bruschi cali o picchi di loquacità.

2. **Distributed NLP & Topic Modeling con Spark ML (`src/analytics/nlp_pipeline.py`)**:
   - Superare i cicli Python lenti e le UDF costose implementando una pipeline distribuita nativa `pyspark.ml.Pipeline`:
     - `RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+")`
     - `StopWordsRemover` customizzato (rimozione stopwords generiche preservando pronomi e marcatori affettivi rilevanti)
     - `NGram(n=2, inputCol="words", outputCol="ngrams")` per estrarre bi-grammi clinici (es. *"feel alone"*, *"cant sleep"*, *"worthless"*)
     - `CountVectorizer` con vocabolario filtrato su `minDF` e `maxDF` seguito da `IDF`
     - **`LDA(k=8, maxIter=20, featuresCol="features")`**: Latent Dirichlet Allocation distribuito per clusterizzare i temi emergenti nei soggetti depressi vs controllo.

3. **Ottimizzazione Esecuzione: Broadcast Joins e Adaptive Query Execution (`src/analytics/spark_session.py`)**:
   - Configurare la `SparkSession` per ottimizzare l'esecuzione distribuita e mitigare lo skew:
     - `spark.sql.adaptive.enabled = true`
     - `spark.sql.adaptive.coalescePartitions.enabled = true`
     - `spark.sql.adaptive.skewJoin.enabled = true`
   - Nelle giunzioni tra la grande tabella di post (milioni di righe) e la tabella delle etichette/soggetti (migliaia di righe), forzare l'uso esplicito del broadcast join: `posts_df.join(F.broadcast(subjects_df), on="subject_id", how="inner")`.

4. **Motore Esecuzione SQL Protetto con AST Validation (`src/analytics/safe_sql.py`)**:
   - Implementare la funzione `execute_safe_spark_sql(query: str, spark: SparkSession) -> DataFrame`.
   - Registrare le tabelle Silver e Gold come viste SQL temporanee (`createOrReplaceTempView`).
   - Integrare **`sqlglot`** per il parsing e la validazione semantica dell'AST della query prima di inviarla a Spark:
     - Bloccare categoricamente istruzioni non-SELECT (vietare `DROP`, `ALTER`, `INSERT`, `DELETE`, `CREATE`, `TRUNCATE`).
     - Rilevare o iniettare forzatamente una clausola di salvaguardia `LIMIT <= 500` per prevenire il download di dataset massivi in memoria.
     - Restituire errori strutturati chiari all'agente in caso di violazione della sintassi o dei vincoli di sicurezza.


## Answer

### Moduli e test implementati

- `src/analytics/features.py`: funzione `compute_post_window_features` con calcolo di `inter_post_gap_seconds` via `F.lag`, rilevamento sessioni (`is_new_session`, `session_index`), indice notturno 00:00–05:00 (`is_night_post`, `night_post_rolling_30d`) e verbosità mobile (`rolling_30d_text_length_mean`, `rolling_30d_text_length_std`, `verbosity_zscore`) mantenendo la granularità atomica (1 riga = 1 post).
- `src/analytics/nlp_pipeline.py`: pipeline nativa Spark ML (`RegexTokenizer` -> `StopWordsRemover` -> `NGram(n=2)` -> `CountVectorizer` -> `IDF` -> `LDA(k=8, maxIter=20)`) priva di UDF Python, esposta tramite `build_nlp_pipeline`, `fit_topic_model`, `extract_topic_terms` e `build_cohort_topic_matrix` (pronta per il matrix Gold per coorte).
- `src/spark.py`: abilitazione AQE (`spark.sql.adaptive.enabled`, `coalescePartitions.enabled`, `skewJoin.enabled`) in `build_spark_session` preservando l'API esistente (`build_spark_session`, `stop`, `write_parquet_zstd`), più helper `broadcast_join`.
- `src/analytics/safe_sql.py`: `execute_safe_spark_sql`, `validate_and_rewrite_sql`, `register_lakehouse_views` ed eccezione strutturata `SafeSQLError` con codici machine-readable (`EMPTY_QUERY`, `SQL_SYNTAX_ERROR`, `MULTIPLE_STATEMENTS`, `DISALLOWED_STATEMENT`, `INVALID_LIMIT`, `TABLE_NOT_ALLOWED`, `TABLE_NOT_FOUND`, `SPARK_ANALYSIS_ERROR`).
- Test creati: `tests/test_features.py`, `tests/test_nlp_pipeline.py`, `tests/test_safe_sql.py`.

### Decisioni tecniche rilevanti

1. **Finestra temporale a 30 giorni (`rangeBetween`) vs 30 righe (`rowsBetween`)**: il ticket citava `.rowsBetween(-30, 0)`, che però conta gli ultimi 30 post e non 30 giorni di calendario (falsando la metrica su utenti ad alta frequenza come 300 post in una notte). È stata quindi usata una finestra basata sul tempo reale (`rangeBetween(-30 * 86400, 0)` su epoch seconds), affiancata a una finestra sequenziale ordinata su `(date, _row_hash)` per `F.lag` e sessionizzazione.
2. **Guardia su varianza nulla nello z-score**: quando `rolling_30d_text_length_std <= 1e-12` (es. primo post o post con lunghezza identica nella finestra), `verbosity_zscore` viene impostato deterministicamente a `0.0` evitando `NaN`/`inf`.
3. **Policy delle stopword cliniche**: la lista parte da `StopWordsRemover.loadDefaultStopWords("english")` (che richiede `SparkContext` attivo), preserva esplicitamente negazioni, pronomi personali e marcatori affettivi (`not`, `no`, `nor`, `never`, `nothing`, `nobody`, `none`, `neither`, `nowhere`, `cannot`, `cant`, `dont`, `wont`, `didnt`, `isnt`, `arent`, `wasnt`, `werent`, `worthless`, `alone`, `feel`, `sleep`, `i`, `me`, `my`, `myself`) e filtra i residui di contrazioni generati da `\W+` (`s`, `t`, `d`, `m`, `ll`, `ve`, `re`). L'estrazione dei top termini per topic usa `describeTopics` + broadcast join sul vocabolario senza alcuna UDF Python.
4. **AQE in `src/spark.py` anziché `src/analytics/spark_session.py`**: come da indicazioni del piano, AQE e `broadcast_join` sono stati integrati nel factory esistente `src/spark.py` senza duplicare la gestione della `SparkSession`.
5. **Sicurezza SQL (`safe_sql.py`)**: validazione AST con `sqlglot` (`read="spark"`) prima di invocare Spark; accettate solo query singole `SELECT`/`WITH`/`SetOperation`/`Subquery` prive di nodi DDL/DML (`DROP`, `ALTER`, `INSERT`, `DELETE`, `CREATE`, `TRUNCATE`, `UPDATE`, `MERGE`, `INTO`, `Command`), con iniezione/riduzione automatica del `LIMIT` a `Settings.sql_row_limit` (500) e registrazione delle sole viste Parquet esistenti (rimuovendo eventuali viste stantie per i path mancanti).
6. **Hardening allowlist tabelle e blocco percorsi file (`TABLE_NOT_ALLOWED`)**: `validate_and_rewrite_sql(..., allowed_tables: frozenset[str] | None = None)` ispeziona tutti i nodi `exp.Table` nell'AST rifiutando riferimenti qualificati con `db`/`catalog` (es. `csv.`/workspace/.env.example``, `text.`...``, `parquet.`...``, `default.silver_posts`), chiamate a table-valued function come `read_files(...)` (dove `table.name == ""`), e tabelle non presenti nell'allowlist derivata da `LAKEHOUSE_VIEW_NAMES` (`_LAKEHOUSE_VIEW_ATTRS`, condivisa con `_lakehouse_view_paths`) più gli alias delle CTE (`exp.CTE`) definite nella query (confronto case-insensitive). Il messaggio di errore `SafeSQLError("TABLE_NOT_ALLOWED", ...)` include il riferimento non autorizzato per guidare il retry dell'agente.

### Esito verifiche (in container Docker)

```text
$ docker compose run --rm --no-deps dev sh -c "python -m ruff check src/analytics/safe_sql.py tests/test_safe_sql.py && python -m pytest -o addopts='' tests/test_features.py tests/test_nlp_pipeline.py tests/test_safe_sql.py"
All checks passed!
22 passed in 24.17s
```
