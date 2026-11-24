# Ticket: 04 — Data Quality con Pandera e Guardia di Significatività Statistica a Runtime

Type: task
Status: resolved
Assignee:

Blocked by: 02

## Context

Un difetto critico evidenziato dal Senior Advisor nei progetti LLM accademici è l'"illusione di intelligenza statistica": l'agente esegue aggregazioni su campioni ridicoli (es. 3 utenti) e trae conclusioni cliniche categoriche (es. *"i depressi postano di più di notte"*), incorrendo in allucinazioni scientifiche.
Per posizionare il progetto come sistema affidabile di produzione, è fondamentale introdurre sia contratti di schema formali all'ingestione con Pandera sia una guardia deterministica di validazione statistica a runtime con test di ipotesi prima di consentire all'LLM di interpretare i dati.

## Task

Implementare il layer di validazione dei dati e la guardia statistica in `src/lakehouse/validation.py` e `src/agent/guards.py`:

1. **Validazione Schema Ingestione con Pandera (`src/lakehouse/validation.py`)**:
   - Definire i contratti di schema formali tramite `pandera.pyspark`:
     - `PostSilverSchema`: verifica tipi (`subject_id: StringType`, `date: TimestampType`, `text: StringType`, `label: IntegerType`), assenza di nulli sui campi chiave, e vincoli di dominio (`label in [0, 1]`, lunghezza minima del testo > 0).
   - Integrare la validazione all'interno della pipeline ETL Silver: le righe che non superano i vincoli Pandera vengono convogliate automaticamente nella tabella `silver_quarantine.parquet` con il dettaglio dell'errore di validazione.

2. **Nodo Deterministico `data_quality_and_stats_guard` (`src/agent/guards.py`)**:
   - Realizzare una guardia algoritmica non-LLM eseguita tra il tool di interrogazione Spark e il nodo critico interpretativo finale:
     - **Controllo di Consistenza del Campione**: Verificare che la query abbia restituito un campione sufficiente per entrambi i gruppi (es. $N \ge 30$ soggetti unici per classe `label=1` e `label=0`). In caso contrario, marcare lo stato con un warning esplicito di *Underpowered Sample*.
     - **Test di Ipotesi Automatico con SciPy**:
       - Se la metrica confrontata è continua (es. verbosità, percentuale post notturni), eseguire automaticamente il test di normalità o calcolare direttamente il test robusto di **Welch's t-test** (o non-parametrico **Mann-Whitney U**).
       - Calcolare la dimensione dell'effetto (**Cohen's *d***) per distinguere tra significatività statistica e rilevanza pratica.
     - **Output Strutturato**: Generare un payload deterministico contenente:
       ```json
       {
         "sample_size_depressed": 142,
         "sample_size_control": 350,
         "p_value": 0.0031,
         "is_significant": true,
         "effect_size_cohens_d": 0.68,
         "statistical_confidence": "high"
       }
       ```

3. **Vincolo di Stato e Guardrail nel Prompt del Critico**:
   - Vincolare rigidamente il prompt del nodo `scientific_critic`: l'agente ha il divieto tassativo di presentare trend come correlati alla depressione se `is_significant == false` o se $N < 30$, ed è obbligato a citare esplicitamente dimensione del campione, $p$-value e Cohen's *d* in ogni asserzione comparativa.

## Answer

### Cosa è stato implementato
- **`src/lakehouse/validation.py`**: contratto formale `PostSilverSchema` (`pandera.pyspark.DataFrameModel`) su `subject_id` (`StringType`, non nullo, lunghezza ≥ 1), `date` (`TimestampType`, non nullo), `text` (`StringType`, non nullo, lunghezza ≥ 1) e `label` (`IntegerType`, non nullo, dominio `{0, 1}`). Espone `find_post_contract_violations(posts_df, key_col=...)` per identificare le singole righe non conformi con chiave di join e motivazione per riga, più `SchemaContractError` per errori strutturali (colonne mancanti o tipi Spark errati).
- **`src/lakehouse/silver.py`**: integrato `find_post_contract_violations` in `build_silver` sulle righe candidate Silver (dopo i controlli pre-Silver e la deduplicazione). Le violazioni vengono escluse da Silver tramite `left_anti` join su `_validation_row_id` persistito e instradate in `silver/quarantine/` con `_rejection_reason` prefissato `contract_violation:<dettaglio>`, mantenendo l'invariante `Bronze == Silver + Quarantine`.
- **`src/agent/guards.py`**: guardia statistica deterministica senza dipendenza da Spark (`StatisticalComparisonInput`, `StatisticalGuardResult`, `evaluate_statistical_guard`), più `CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS` e `format_critic_guard_instructions` pronti per essere collegati al nodo `scientific_critic` nel ticket 05.
- **Test**: `tests/test_validation.py` (4 test Spark) e `tests/test_statistical_guard.py` (7 test unitari puri).

### Scelte progettuali principali
- **Dipendenze compatibili per `pandera.pyspark`**: `pandera.pyspark` importa `pyspark.pandas`, che su PySpark 3.5 richiede `pandas>=2.0,<3` (`DateOffset` rimosso in pandas 3), `numpy>=1.26,<2` (`np.NaN` rimosso in numpy 2) e `pyarrow>=14,<17` (incompatibile con numpy 1.x dalle versioni >=17), fissati in `pyproject.toml` insieme a `PYARROW_IGNORE_TIMEZONE=1` nel `Dockerfile`. L'import in `validation.py` è quindi diretto (`import pandera.pyspark as pap`) senza alcuno stub.
- **Propagazione delle violazioni alla quarantena**: poiché `pandera.pyspark` segnala errori solo a livello di colonna/check in `.pandera.errors` e non per riga, `find_post_contract_violations` esegue prima `PostSilverSchema.validate(df, lazy=True)` e, solo quando Pandera rileva errori sui dati, valuta in un singolo passaggio Spark i predicati per riga restituendo `(key_col, violation_reason)` per il successivo `left_anti` / `inner` join in `build_silver`.
- **Regola di selezione del test**: screening preliminare di normalità per gruppo (`scipy.stats.shapiro` per $N \le 5000$, `normaltest` oltre 5000, soglia $p \ge 0.05$; array costanti o con $N < 3$ falliscono lo screening senza emettere warning). Se entrambi i gruppi superano lo screening viene eseguito il Welch's $t$-test (`ttest_ind(..., equal_var=False)`); altrimenti Mann-Whitney U bilaterale. Casi degeneri ($N < 2$ o valori tutti identici) restituiscono `p_value = 1.0`, `effect_size_cohens_d = 0.0`, `test_used = "none"` senza mai produrre `NaN` o `inf`.
- **Politica *underpowered***: se una delle due classi ha meno di `Settings.min_subjects_per_class` soggetti (default `30`), `is_underpowered = True`, viene valorizzato `underpowered_reason`, `statistical_confidence = "underpowered"` e `is_significant` viene forzato a `False` indipendentemente dal $p$-value.

### Verifica (`docker compose run --rm --no-deps dev`)
- `python -m ruff check .` → `All checks passed!`
- `python -m pytest -v` → `47 passed in 53.60s` (inclusi tutti i 4 test in `tests/test_validation.py`, i 7 test in `tests/test_statistical_guard.py`, i 3 test esistenti in `tests/test_silver.py` e i test del lavoro concorrente in `src/analytics/**`).
