# Handoff — stato al 24 nov 2026

Sessione chiusa qui: il lavoro riprende da una sessione nuova, guidata dall'utente.

## Fatto

- **Lettura del progetto**: 7 moduli, ~1.300 righe in italiano, stack LlamaIndex + Gemini 2.5 Flash + Streamlit + PySpark. Difetti reali confermati nel codice: 4 `.count()` bloccanti all'avvio, `groupBy(...).agg(collect_list("text"))` (OOM), `.persist()` indiscriminati, scritture piatte, `from pyspark.sql.functions import *` con shadowing di `min/max/sum`, dipendenze fantasma.
- **`.gitignore` riparato** (commit `8edfe8b`): le regole erano scritte con `./` iniziale e non matchavano nulla — un `git add -A` avrebbe messo in git 24.652 file, 16.432 dei quali il corpus eRISK. Ora lo stage è di 18 file.
- **Ticket e mappa entrati nel versionamento** (prima esistevano solo su disco).
- **`CONTEXT.md` riallineato**: descriveva al futuro un sistema inesistente. Ora dice cosa fa il codice oggi, tiene il glossario di dominio e rimanda alla mappa per il target. Aggiunta la policy sui dati.
- **`AGENTS.md`** con le regole del progetto: piano di riferimento, policy dati, inglese nel codice, caveat `winutils`.
- **Ticket 01 chiuso** (commit `a966ccc`): `pyproject.toml` con ruff e pytest, `src/config.py` con `Settings` duale ollama/openrouter, `.env.example`, scheletro `src/{lakehouse,analytics,agent,ui}` + `tests/` + `docs/adr/`, e `legacy/` con l'implementazione vecchia intatta. 4 test verdi.

## Decisione di interpretazione da ricordare

Il ticket 01 chiedeva di rinominare i moduli esistenti nella struttura nuova. Scelta diversa, scritta nel ticket: **lo scheletro nuovo nasce vuoto e il codice vecchio va in `legacy/`**, perché i ticket 02-05 riscrivono quei moduli da zero. `legacy/` si cancella a fine ticket 06.

## Prima di attaccare il ticket 02

1. **Il fixture del ticket 02 non può essere testo reale.** Il ticket prevede un campione di ~5.000 post dal dataset eRISK: sono post veri di persone con depressione. Se finiscono in un repo pubblico, il repo pubblica dati sanitari pseudonimizzati di terzi. Va deciso: fixture sintetico, oppure aggregati senza testo, oppure campione reale ma repository privato. La policy in `CONTEXT.md` dice già "sintetico o aggregato".
2. **Docker + Java 17**: risolto dal ticket 00 — `Dockerfile`, `docker-compose.yml`, `.devcontainer/` esistono e sono stati verificati a runtime (Spark 3.5.1 su JDK 17 dentro il container, MinIO con bucket `lakehouse`, bridge verso Ollama sull'host). Docker Desktop va avviato a mano: non parte da solo. I test girano nel container.
3. **`python` in questa shell non funziona** (alias Microsoft Store): il binario reale è `C:\Users\gabri\AppData\Local\Programs\Python\Python313\python.exe`. I test del ticket 01 sono stati eseguiti lì.
4. **Il ticket 06 va ridiscusso su un punto**: S3/MinIO serve a poter scrivere "AWS" nel CV, ma chi legge chiede cosa è girato davvero su AWS. La risposta onesta è "MinIO locale con API S3", e va scritta così nel README.

## Fuori da questo repository

Il repo GitHub `gabrielefuoco/Spark-Data-Agent` contiene **solo documentazione e dati, con `src/` vuota**: il codice non è mai stato pushato. Va allineato dopo il revamp, insieme al README (ticket `04` dell'effort `readme-refresh` nel monorepo `Lavoro`, oggi `wontfix`: documentare senza codice avrebbe prodotto una pagina che dichiara cose non verificabili).

## Dove guardare

- `.scratch/spark-data-agent-revamp/map.md` — il piano, i 6 ticket, i 4 anti-pattern con la loro soluzione.
- `.scratch/spark-data-agent-revamp/issues/01…06` — ognuno con stato e appunti.
- `CONTEXT.md` — stato attuale, target, glossario di dominio, policy dati.
- `legacy/README.md` — perché il codice vecchio è ancora qui.
