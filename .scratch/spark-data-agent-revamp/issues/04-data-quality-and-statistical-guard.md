# Ticket: 04 — Data Quality con Pandera e Guardia di Significatività Statistica a Runtime

Type: task
Status: ready-for-agent
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
