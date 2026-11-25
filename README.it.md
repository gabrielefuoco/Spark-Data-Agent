# Spark Data Agent

Analisi conversazionale sul corpus **eRISK** — 397.228 post Reddit di persone
etichettate come depresse (`label=1`) e controlli (`label=0`) — costruita su un
lakehouse Medallion in PySpark e pilotata da un agente LangGraph che deve
mostrare le proprie statistiche prima di poterne trarre una conclusione.

Il punto non è la finestra di chat. Il punto è che una domanda in linguaggio
naturale diventa **SQL validato contro un AST**, gira su **Parquet
partizionato**, torna con un **p-value e una dimensione dell'effetto**, e si
ferma a chiedere il permesso a una persona prima di far partire qualcosa di
costoso. Quando i numeri non sostengono un'affermazione, il sistema lo dice
nella frase che rifiuta di scrivere.

```
D:  I depressi postano più di notte dei controlli?
R:  No. 79 soggetti depressi vs 740 controlli, Mann-Whitney U,
    p = 0.5587, Cohen's d = 0.068 — nessuna differenza rilevata.
    [appendice statistica generata dal sistema]
```

---

## Indice

- [Cosa fa](#cosa-fa)
- [Architettura](#architettura)
- [Quickstart](#quickstart)
- [Come una domanda diventa una risposta](#come-una-domanda-diventa-una-risposta)
- [Comportamento misurato](#comportamento-misurato)
- [I dati](#i-dati)
- [Documentazione](#documentazione)
- [Principi di progetto](#principi-di-progetto)
- [Perimetro onesto](#perimetro-onesto)
- [Stato](#stato)

---

## Cosa fa

| capacità | esempio |
|---|---|
| **Catalogo** — risposto da un manifest precalcolato, senza Spark | "quanti soggetti per coorte?" → 1 secondo, 0 query |
| **Statistiche di coorte** — un vero test di ipotesi per metrica | "la lunghezza dei post differisce?" → Mann-Whitney U, p-value, Cohen's d |
| **Feature circadiane e comportamentali** — sessioni, ritmo notturno, z-score di verbosità | "i depressi postano più di notte?" |
| **Analisi lessicale e di topic** — top n-grammi TF-IDF, topic LDA per coorte | "quali parole caratterizzano i depressi?" |
| **Analisi generale** — il modello pianifica ed esegue una dozzina di query da solo | "dammi un quadro delle differenze tra le coorti" → 12 tool call, 10 metriche valutate, una sintesi |
| **SQL ad hoc** — validato, in allowlist, con tetto | "quanti post nel 2014?" → 18.611 |
| **Rifiuti** | "DROP TABLE silver_posts" → bloccato prima di Spark, in 4 secondi |

## Architettura

```
 Streamlit UI ──► LangGraph tool loop ──► PySpark Medallion Lakehouse
                    │                        bronze  raw XML/TSV + lineage
                    │                        silver  post atomici + quarantena
                    │                        gold    feature, n-grammi, topic
                    ├── run_safe_sql        (AST + allowlist + LIMIT)
                    ├── run_spark_job       (LDA, circadiane, con costo)
                    ├── HITL gate           (interrupt + SqliteSaver)
                    ├── statistical guard   (per soggetto, Welch/Mann-Whitney)
                    └── scientific critic   (prosa + appendice generata dal sistema)
```

**Stack**: Spark 3.5.1 · JDK 17 · Python 3.11 · LangGraph 1.2 · Streamlit ·
Plotly · Pandera · sqlglot · tiktoken · MinIO (S3-compatibile) · Docker Compose.

## Quickstart

Docker Desktop deve essere in esecuzione; tutto gira dentro il container, così
c'è un solo Python e un solo JDK da tenere verdi.

```bash
git clone <repo> && cd <repo>
cp .env.example .env          # aggiungi OPENROUTER_API_KEY, oppure punta a Ollama
docker compose up -d          # dev container + MinIO (bucket `lakehouse`)

# dati: un corpus sintetico deterministico (quello reale non è distribuito)
docker compose run --rm --no-deps dev python scripts/generate_dev_sample.py --seed 42

# oppure porta il corpus reale: metti `chunk/` e `risk-golden-truth-test.txt` nella
# radice del repo e segui docs/operations.md (≈10 min, ≈240 MB, gitignored)

# cancelli di qualità
docker compose run --rm --no-deps dev python -m pytest      # 148 test, ~65s
docker compose run --rm --no-deps dev python -m ruff check .
docker compose exec -T dev mypy

# l'interfaccia
docker compose exec dev python -m streamlit run src/ui/app.py --server.address 0.0.0.0
# http://localhost:8501
```

`.env` viene letto da Compose alla creazione del container: dopo averlo
modificato, lancia `docker compose up -d --force-recreate dev`, altrimenti resta
l'ambiente vecchio.

## Come una domanda diventa una risposta

Una riga per passo, e dove sta il confine di fiducia:

1. **Il modello pianifica.** Vede il system prompt, un digest da 98 token del
   manifest del dataset, la finestra di conversazione e i risultati fin qui.
   Risponde o con tool call, o con un messaggio finale.
2. **`run_safe_sql` non esegue niente finché non è sicuro.** sqlglot valida
   l'AST, un'allowlist rifiuta ogni tabella che non sia una vista registrata del
   lakehouse (comprese le letture di file `csv.\`/path\``), un `LIMIT` viene
   iniettato e limitato, e una stima di costo decide se serve l'approvazione di
   una persona.
3. **Ogni risultato viene giudicato.** La guardia statistica collassa le righe
   per soggetto, sceglie Welch o Mann-Whitney dopo uno screening di normalità, e
   restituisce `assessed` / `descriptive` / `cannot_assess` con dimensioni del
   campione, p-value e Cohen's d — oppure un esplicito "non valutabile, e
   perché".
4. **Il ciclo continua o si chiude**, con un tetto di 20 tool call e di due
   chiamate identiche consecutive.
5. **Il critico interpreta.** Una seconda chiamata al modello scrive la prosa, e
   il grafo appende un'appendice statistica deterministica. Il controllo
   sull'output valida prosa e appendice insieme, così **nessun numero in una
   risposta finale è un ricordo del modello**.

## Comportamento misurato

Sei domande sul corpus reale con un modello reale, dopo le ottimizzazioni:

| domanda | tool call | tempo effettivo |
|---|---|---|
| catalogo ("soggetti per coorte?") | 0 | **1 s** |
| saluto / domanda sulle capacità | 0 | **2 s** |
| "DROP TABLE silver_posts" | 0 | **4 s** (rifiutata) |
| un confronto ("lunghezza dei post") | 2 | 12 s |
| domanda con regex ("quanti gatti?") | 3 | 30 s (2 cancelli) |
| analisi generale ("quadro delle differenze") | 12 | ~4 min |

Dove va il tempo su una domanda da 30 secondi: **modello 18 s, Spark 12 s**.
Quel rapporto era invertito prima che la registrazione delle view smettesse di
rileggere cinque alberi Parquet a ogni tool call (9 s per chiamata → 0,4 s).

## I dati

Il corpus eRISK è fatto di post reali di persone reali sulla propria
depressione. **Non viene mai committato**: `processed_data/`, `chunk/` e
`ignora/` sono gitignored e montati in sola lettura nel container. Tutto ciò che
viene distribuito con il repository è sintetico o aggregato — compreso il
fixture dei test, che è generato con un seed invece che campionato.

Un'ingestione completa di questa copia produce:

| | |
|---|---|
| writings ingeriti | 544.447 |
| post atomici | 397.228 |
| in quarantena (per lo più post con solo il titolo) | 147.219 |
| soggetti | 819 (79 depressi, 740 controlli) |
| periodo | 2006-07 → 2017-05 |

Lo sbilanciamento 1:9,4 è la prima cosa che un lettore deve sapere di questo
dataset, e il sistema lo dichiara in ogni risposta pertinente.

## Documentazione

| documento | a cosa risponde |
|---|---|
| [docs/architecture.md](docs/architecture.md) | componenti, ciclo di vita di una richiesta, dove vanno i secondi |
| [docs/lakehouse.md](docs/lakehouse.md) | Bronze/Silver/Gold, motivi di quarantena, l'invariante sui conteggi |
| [docs/agent.md](docs/agent.md) | il tool loop, la memoria, l'HITL, i limiti, l'appendice |
| [docs/statistics.md](docs/statistics.md) | la metodologia della guardia, le soglie, i tre verdetti |
| [docs/sql-safety.md](docs/sql-safety.md) | validazione AST, l'allowlist, i due limiti |
| [docs/operations.md](docs/operations.md) | runbook, ogni variabile d'ambiente, troubleshooting |
| [docs/testing.md](docs/testing.md) | cosa coprono i 148 test, e cosa non possono coprire |
| [docs/data.md](docs/data.md) | il corpus, la sua policy, cosa può e non può rispondere |
| [docs/decisions.md](docs/decisions.md) | i quattro ADR più le decisioni prese senza uno |
| [docs/roadmap.md](docs/roadmap.md) | i limiti, e cosa farei dopo |

[`CONTEXT.md`](CONTEXT.md) contiene il vocabolario di dominio, [`docs/adr/`](docs/adr/README.md)
i decision record, e [`.scratch/spark-data-agent-revamp/`](.scratch/spark-data-agent-revamp/map.md)
il piano di riferimento con i sette ticket che hanno costruito questo.

## Principi di progetto

Quattro frasi che spiegano la maggior parte del codice:

1. **Il modello interpreta, il codice misura.** Dove un fatto si può calcolare,
   al modello non viene chiesto di ricordarlo — l'appendice statistica esiste
   perché tre tentativi di far trascrivere un p-value a un modello flash sono
   tutti falliti.
2. **Niente viene scartato in silenzio.** Ogni riga grezza finisce in Silver o in
   quarantena con un motivo, e un test verifica che i conteggi tornino.
3. **Un guardrail deve essere proporzionato.** Un controllo che rifiuta risposte
   corrette è peggio di nessun controllo: insegna a fidarsi del fallback. Ci
   sono voluti tre giri di calibrazione per impararlo (virgole decimali,
   `p < 0.001`, domande riformulate).
4. **Se non l'hai eseguito, non lo sai.** I test mockati dimostrano la logica,
   non il cablaggio. Ogni bug che ha contato, qui — `spark-xml` che rifiuta una
   lista di percorsi, `EXPLAIN` prima che le view fossero registrate, `LIMIT 500`
   che tronca 819 soggetti — è stato trovato facendo una domanda al sistema, non
   eseguendo la suite.

## Perimetro onesto

Questo è un progetto per il portfolio, ed essere precisi su cosa fa ne fa parte.

- **Storage**: **MinIO** locale attraverso l'API compatibile S3
  (`s3a://lakehouse`). Il vero AWS è supportato per configurazione e **non è mai
  stato esercitato**. In nessun punto di questo repository si afferma di aver
  girato su AWS.
- **LLM**: duale per progetto — Ollama in locale, OpenRouter in remoto. Ogni
  misura qui è stata fatta con OpenRouter e
  `deepseek/deepseek-v4.1-flash-1017`; nessun modello locale è stato scaricato.
  L'intera suite di test gira offline contro modelli di chat finti.
- **Affermazioni cliniche**: nessuna. Il corpus sostiene analisi esplorative, non
  diagnosi, e la guardia esiste per impedire all'agente di vestire il rumore da
  scoperta.
- **Non è produzione**: Spark su singolo nodo, nessuna CI, nessuna
  autenticazione, nessun isolamento multi-utente. È una dimostrazione di metodo.

## Stato

Sette ticket del piano sono chiusi: ambiente Docker, igiene del progetto,
lakehouse Medallion, analytics PySpark avanzate, qualità dei dati e guardia
statistica, orchestrazione LangGraph, S3 + ADR. Una code review a due assi ha poi
trovato che tre di essi erano «risolti» solo sulla carta; le correzioni sono
nella storia con i propri commit, e la mappa annota cosa ciascuno ha chiuso.

```
148 test · ruff pulito · mypy pulito (29 file)
src/ 5.804 righe · tests/ 3.944 · 4 ADR · 10 documenti
```

Cosa manca e cosa farei dopo: [docs/roadmap.md](docs/roadmap.md).
