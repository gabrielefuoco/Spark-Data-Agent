# Ticket: 05 — Orchestrazione Ciclica con LangGraph, OpenRouter Multi-Modello e Human-in-the-Loop (HITL)

Type: task
Status: resolved
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

## Answer

### Cosa è stato costruito

- **Stato e Factory Modelli (`src/agent/state.py`, `src/agent/models.py`)**:
  - Definito `DataAgentState` (`TypedDict` con reducer `add_messages`, `user_intent`, `generated_query`, `analysis_metric`, `sql_cost_estimate`, `query_result`, `statistical_validation`, `retry_count`, `last_error`, `planner_notes`, `hitl_decision`, `critic_system_prompt`, `critic_critique`, `final_response`, `status`) e `make_initial_state`.
  - Implementata la factory lazy `create_chat_model` / `resolve_model_config` / `build_model_factory` basata su `ChatOpenAI`: supporta sia **Ollama** locale (`http://localhost:11434/v1`) sia **OpenRouter** (`https://openrouter.ai/api/v1`, distinguendo `remote_router_model` e `remote_reasoning_model`), senza istanziare alcun client all'import o durante i test.
- **Stima dei Costi e Prompting (`src/agent/cost.py`, `src/agent/prompts.py`)**:
  - `estimate_sql_cost` esegue `spark.sql(query).explain(mode="cost")` quando è disponibile una `SparkSession`, oppure applica l'euristica deterministica su AST `sqlglot` (`estimate_ast_cost`).
  - `build_critic_system_prompt` integra `format_critic_guard_instructions(guard_result)` e `CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS` nel prompt di sistema del nodo critico.
- **StateGraph Ciclico LangGraph (`src/agent/graph.py`)**:
  - `build_agent_graph` orchestra i nodi `intent_router_planner`, `ast_validator_and_cost`, `hitl_gate`, `execute_sql`, `query_rewriter`, `statistical_guard`, `scientific_critic` e `handle_failure` con dipendenze iniettabili (`router_model`, `critic_model`, `sql_executor`, `spark`, `cost_estimator`, `manifest_loader`, `checkpointer`).
  - Le richieste `catalog_metadata` leggono `gold_dataset_manifest.json` tramite `load_manifest()` senza invocare Spark né l'executor SQL.
  - `ast_validator_and_cost` valida e riscrive la query con `validate_and_rewrite_sql` (enforce `LIMIT`) prima di stimarne il costo.
  - `execute_sql` utilizza l'executor iniettato (a runtime `make_spark_sql_executor` su `execute_safe_spark_sql`) e `statistical_guard` esegue `evaluate_statistical_guard` sui risultati.
- **UI Streamlit + Plotly (`src/ui/app.py`)**:
  - Interfaccia conversazionale con grafici Plotly (`build_result_figure`), card delle metriche statistiche, visualizzazione del manifest nella sidebar, e card HITL con query proposta, costo stimato e pulsanti `"Autorizza"` / `"Annulla"` per riprendere lo stesso `thread_id`.

### Decisioni tecniche rilevanti

- **Deviazione dall'API HITL del ticket (`interrupt()` vs `interrupt_before`)**: Il ticket menzionava `interrupt_before` a compile-time; con LangGraph `1.2.24` il gate HITL è stato implementato con `interrupt()` dinamico dentro il nodo `hitl_gate`, persistito su `SqliteSaver` e ripreso con `Command(resume=...)`. Questo consente di attivare l'interruzione condizionalmente solo quando `sql_cost_estimate["triggers_hitl"]` supera `Settings.hitl_cost_threshold`, mantenendo la ripresa deterministica sullo stesso `thread_id` anche dopo ricompilazione del grafo sullo stesso DB SQLite.
- **Euristica di costo AST**: Senza Spark attivo, `estimate_ast_cost` assegna un costo base per tabella (`gold_* = 0.2`, `silver_quarantine = 0.3`, `silver_posts = 1.2`) più penalità per `JOIN` (`+0.5`), `GROUP BY`/window (`+0.3`), pattern scan `LIKE`/`RLIKE` (`+0.25`) e scansioni non filtrate su `silver_posts` (`+0.2`), riducendo leggermente in presenza di filtri di partizione (`label`/`year`). Con `hitl_cost_threshold=1.0`, le query sui cubi Gold vengono eseguite direttamente mentre le scansioni su `silver_posts` attivano il gate HITL. Con `SparkSession` attiva viene estratto `sizeInBytes` da `explain(mode="cost")` e combinato via `max(ast_score, spark_score)`.
- **Test completamente offline con Fake LLM**: I test usano `FakeMessagesListChatModel`, `GenericFakeChatModel` e `RecordingFakeChatModel` (sottoclasse di `FakeMessagesListChatModel` che registra i messaggi ricevuti) insieme a `sql_executor` fittizi e `SqliteSaver`, senza mai toccare rete o avviare Spark. I test UI usano `streamlit.testing.v1.AppTest` iniettando un grafo compilato fittizio in `st.session_state["agent_graph"]`.
- **Semantica dei retry**: `Settings.max_query_retries` (default `2`) controlla il numero massimo di ri-pianificazioni tramite `query_rewriter` dopo errori `SafeSQLError` (anche in fase AST) o risultati inattesi vuoti (`EMPTY_RESULT`): con `max_query_retries=2` il grafo esegue fino a `1 + 2 = 3` tentativi prima di terminare su `handle_failure` con spiegazione trasparente (`status="failed"`).

### Verifica ed esito test

- Suite mirata (`tests/test_agent_models.py`, `tests/test_agent_graph.py`, `tests/test_ui_app.py`): `17 passed in 4.00s`.
- Suite completa (`docker compose run --rm --no-deps dev sh -c "python -m pytest -q && python -m ruff check ."`): `64 passed` (`[100%]`) e `All checks passed!`.

### Avvio della UI

Dal root del repository (con i container avviati tramite `docker compose up -d`):

```bash
docker compose exec dev streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 8501
```

L'interfaccia è raggiungibile su `http://localhost:8501` (la porta `8501` è già mappata in `docker-compose.yml`).

