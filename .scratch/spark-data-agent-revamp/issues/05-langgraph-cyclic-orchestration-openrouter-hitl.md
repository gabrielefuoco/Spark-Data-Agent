# Ticket: 05 — Orchestrazione Ciclica con LangGraph, OpenRouter Multi-Modello e Human-in-the-Loop (HITL)

Type: task
Status: ready-for-agent
Assignee: 

Blocked by: 03, 04

## Context

L'attuale implementazione si basa su un `AgentWorkflow` di LlamaIndex e su chiamate statiche a Gemini, agendo come semplice "passacarte": riceve la domanda dell'utente, invoca un tool e restituisce la risposta senza loop di controllo, senza capacità di auto-correzione e senza gestione dei costi computazionali.
Per posizionarsi al vertice dei candidati AI Engineer, il sistema deve essere orchestrato con **LangGraph** (StateGraph esplicito con memoria persistente), integrare modelli specializzati tramite **OpenRouter API** (routing veloce vs ragionamento profondo) e incorporare un vero gate **Human-in-the-Loop** con `SqliteSaver` per autorizzare query Spark onerose.

## Task

Progettare ed implementare il motore agentico in `src/agent/` con LangGraph e connetterlo alla UI Streamlit in `src/ui/`:

1. **State Definition & Modelli OpenRouter (`src/agent/state.py`, `src/agent/models.py`)**:
   - Definire il `DataAgentState` (TypedDict) contenente:
     - `messages`: lista messaggi conversazione
     - `user_intent`: categorizzazione della richiesta (es. catalog_metadata, lexical_analysis, circadian_study, ad_hoc_sql)
     - `generated_query`: query SQL o parametri di analisi estratti
     - `sql_cost_estimate`: stima dell'impatto computazionale (da Spark `EXPLAIN` o euristica AST)
     - `query_result`: dati estratti da Spark
     - `statistical_validation`: payload di significatività dal nodo di guardia (p-value, Cohen's d)
     - `retry_count`: contatore dei tentativi di auto-correzione
     - `critic_critique`: annotazioni scientifiche del modello critico
   - Supportare provider duale (configurabile da `Settings.LLM_BACKEND`):
     - **Modalità Locale (Ollama)**: Sviluppo ed esecuzione offline a costo zero tramite endpoint OpenAI-compatibile locale (`http://localhost:11434/v1`), utilizzando modelli come `qwen2.5:7b` o `deepseek-r1:7b` per routing e test.
     - **Modalità Remota (OpenRouter)**: Per deploy e inferenza ad alta capacità:
       - *Router & Planner*: `deepseek/deepseek-chat` (o Qwen 2.5 72B) per routing JSON veloce e deterministico.
       - *Scientific Critic*: `deepseek/deepseek-r1` per analisi statistica avanzata, evidenze cliniche e catena di ragionamento esplicita.

2. **Topologia dello StateGraph Ciclico (`src/agent/graph.py`)**:
   - **Nodo 1: Intent Router & Query Planner**: Analizza la domanda dell'utente; se la risposta è reperibile nel manifest pre-calcolato (`gold_dataset_manifest.json`), risponde istantaneamente senza invocare Spark. Se richiede elaborazione, formula la query o sceglie il job Spark appropriato.
   - **Nodo 2: AST Validator & Cost Estimator**: Valida la query con `sqlglot`. Esegue `spark.sql(query).explain(mode="cost")` per stimare se la query comporta broadcast o full-table scan pesante.
   - **Gate HITL (`interrupt_before`)**: Se la stima di costo supera una soglia configurata (es. scansione su tabella raw o ricalcolo LDA completo), lo StateGraph attiva un interrupt persistendo lo stato su `SqliteSaver(checkpointer)`.
   - **Nodo 3: PySpark Safe Executor**: Esegue la query approvata tramite il motore `execute_safe_spark_sql`.
   - **Nodo 4: Data Quality & Statistical Guard**: Esegue il Welch's t-test / Cohen's d sui risultati come definito nel Ticket 04.
   - **Nodo 5: Self-Correcting Query Rewriter (Retry Loop)**: Se l'esecuzione Spark fallisce o restituisce un dataset vuoto inaspettato, intercetta l'eccezione, aggiorna il prompt con l'errore di Spark/schema e re-instrada al planner (fino a un massimo di 2 retry prima di arrendersi con spiegazione trasparente).
   - **Nodo 6: Scientific Interpreter & Critic**: Formula l'analisi finale interpretando i dati assieme alle evidenze statistiche (p-value, intervalli di confidenza), evidenziando limitazioni del dataset e possibili bias di campionamento.

3. **Integrazione Streamlit con Resunzione Stato (`src/ui/app.py`)**:
   - Mostrare grafici interattivi con Plotly per visualizzare i risultati delle query e le distribuzioni.
   - Gestire l'interruzione HITL: quando il grafo si arresta per query onerosa, la UI renderizza una card di allerta con la query proposta, il costo stimato e i pulsanti "Autorizza Esecuzione" e "Annulla/Riformula". Al click dell'utente, l'app riprende il thread tramite il `thread_id` di LangGraph.
