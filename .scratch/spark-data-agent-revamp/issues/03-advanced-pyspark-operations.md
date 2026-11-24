# Ticket: 03 — Operazioni PySpark Avanzate (Window Functions Circadiane, Spark ML LDA, Broadcast Join e AST Guard)

Type: task
Status: ready-for-agent
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
