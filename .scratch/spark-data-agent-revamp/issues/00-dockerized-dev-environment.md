# Ticket: 00 — Ambiente di Sviluppo Dockerizzato (Linux, JDK 17, Spark locale, MinIO)

Type: task
Status: resolved
Assignee:

Blocked by: none

## Context

Il ticket 01 elencava `.devcontainer/` e `docker-compose.yml` nel layout richiesto, ma la sezione *Answer* non ne parla: i due artefatti non sono mai stati creati. Il compose è finito dentro il ticket 06, che è bloccato da 05 — cioè l'ambiente arriverebbe **dopo** il primo codice Spark, non prima. Nel frattempo lo sviluppo è su Windows con Java 17 di sistema e il `winutils` workaround, che è esattamente ciò che la mappa vuole abbandonare.

Il ticket 06 mantiene la sua parte cloud (connettore `s3a://`, `hadoop-aws`, ADR); qui si costruisce solo l'ambiente in cui i ticket 02-05 girano.

## Task

1. **Immagine di sviluppo (`Dockerfile`)**: Debian slim, Python 3.11 (non 3.13: `requires-python >=3.11`, ruff `py311`), OpenJDK 17 headless, dipendenze installate leggendo `pyproject.toml` a build time per non duplicare la lista.
2. **JAR Spark precaricati**: `spark-xml` dentro `site-packages/pyspark/jars`, così nessun `spark.jars.packages` scarica da Maven al primo avvio (fallisce offline, comunque costa secondi ad ogni cold start). I JAR S3 (`hadoop-aws`, `aws-java-sdk-bundle`, ~280 MB) restano al ticket 06, finché `storage.py` non esiste.
3. **`docker-compose.yml`**: servizio `dev` (Spark `local[*]`, un solo container — nessun worker: la scala è 5.000 post e un cluster aggiunge solo superficie di guasto), MinIO con bucket `lakehouse` inizializzato da un job `minio-init`, `extra_hosts: host.docker.internal:host-gateway` per raggiungere Ollama sull'host.
4. **`.devcontainer/devcontainer.json`** che aggancia il compose: chi apre il repo in VS Code finisce dentro lo stesso container, non in un secondo ambiente.
5. **`.dockerignore`**: il corpus eRISK (24.635 file, 330 MB) non deve mai entrare nel build context — lento, e un `COPY . .` distratto metterebbe dati sanitari di terzi in un layer.
6. **`.gitattributes`**: `eol=lf` sugli script. Un `entrypoint.sh` con CRLF da checkout Windows muore con `bad interpreter: /bin/bash^M`.
7. **I test girano dentro il container**, non sull'host: `docker compose run --rm dev python -m pytest`.

## Decisioni prese

- **Container singolo con `local[*]`**, non master+worker. Coerente con `Settings.spark_master = "local[*]"` e con la scala reale del dataset; la topologia distribuita resta una scelta del ticket 06 se e quando serve.
- **`OLLAMA_BASE_URL` cablato nel compose a `host.docker.internal`**, non derivato dal `.env` dell'host: lo stesso `.env` vale `localhost:11434` per un run sull'host, e dentro il container `localhost` è il container stesso. Derivarlo è un footgun silenzioso.
- **Docker Desktop va avviato a mano**: era fermo (pipe `dockerDesktopLinuxEngine` assente, WSL `docker-desktop` stopped). L'ambiente Linux reale è WSL2 sottostante, quindi funziona, ma non c'è un daemon che parte da solo.

## Answer

**24 nov 2026** — Fatto e verificato a runtime, non solo scritto.

**Artefatti**: `Dockerfile`, `docker-compose.yml` (servizi `dev`, `minio`, `minio-init`), `.devcontainer/devcontainer.json`, `.dockerignore`, `.gitattributes`.

**Verifiche eseguite** (Docker Desktop avviato a mano: era fermo):

- `docker compose build` verde; `docker compose up -d` alza `dev` e `minio`, e `minio-init` crea il bucket ed esce 0.
- Dentro il container: Python 3.11.18, OpenJDK 17.0.21.1, `pytest` **4 passed**, `ruff check` e `ruff format --check` puliti.
- Spark reale: sessione `local[2]`, `spark 3.5.1 | hadoop 3.3.4`, scrittura e rilettura Parquet ZSTD su 10.000 righe, e `spark-xml` che legge un XML di prova **senza** contattare Maven.
- `curl http://host.docker.internal:11434/api/tags` dal container → 200: il bridge verso Ollama sull'host risponde.
- MinIO: `/minio/health/live` → 200, bucket `lakehouse` presente.

**Problemi trovati durante la verifica** — nessuno visibile leggendo il piano:

1. **`python:3.11-slim` oggi è trixie**, che non ha `openjdk-17`: build rotta con `Package 'openjdk-17-jre-headless' has no installation candidate`. Base fissata a `python:3.11-slim-bookworm` — su trixie l'unico JDK è il 21, che Spark 3.5 rifiuta.
2. **Le immagini ufficiali MinIO non esistono più su Docker Hub** (`pull access denied for minio/minio`; `quay.io/minio/*` risponde 401). Si usa il fork community `pgsty/minio` / `pgsty/mc` con release pinnate. Il ticket 06 deve saperlo: la ricetta `minio/minio` di ogni tutorial è morta.
3. **Segreto vuoto ≠ segreto assente**: `OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}` nel compose produceva `SecretStr('')`, quindi il test «siamo in locale e gratis» falliva con `assert SecretStr('') is None`, e qualunque `if settings.openrouter_api_key` avrebbe mentito. Risolto alla radice: `Settings` converte le stringhe vuote in `None` (validator `blank_means_unset`), e il compose non passa più quella variabile — il `.env` è già visibile nel bind mount.
4. **Ruff falliva con 32 errori, tutti in `legacy/`** — cioè sul codice che il revamp vuole eliminare. `legacy` è ora escluso dal lint: è la scena del reato, non il codice da tenere pulito.
5. **`test_defaults_are_local_and_free` dipendeva dall'ambiente**: dentro il container `OLLAMA_BASE_URL` punta al bridge, quindi `"localhost" in ollama_base_url` era falso. Ora il test azzera esplicitamente la variabile e verifica il default vero.
6. **Hadoop ignora i file che iniziano con `_` o `.`**: un file di prova chiamato `_xml.xml` produce `Input path does not exist` anche se `ls` lo vede. Non è un bug dei bind mount, ma costa mezz'ora se non lo si sa.

**Comandi**:

```bash
docker compose up -d                                  # dev + MinIO
docker compose run --rm --no-deps dev python -m pytest
docker compose exec dev bash
docker compose exec dev python -m streamlit run src/ui/app.py   # dal ticket 05
```

## Note operative

- `python` in questo shell resta l'alias Microsoft Store: dentro il container è `python` normale, quindi i comandi di questo README valgono solo lì.
- Il corpus resta montato in bind read-only, fuori dall'immagine: `processed_data/`, `chunk/`, `ignora/`.
