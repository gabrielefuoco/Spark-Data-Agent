import streamlit as st
import os
import logging
import asyncio
import warnings
import sys

# Importazioni LlamaIndex
try:
    from llama_index.core.memory import ChatMemoryBuffer
except ImportError as e:
    st.error(f"Errore importazione LlamaIndex: {e}")
    st.stop()

# Importazioni dei moduli
from data_manager import get_data_manager
from agent_handler import setup_agent_workflow, get_background_event_loop, _run_workflow_and_collect

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sopprime warnings specifici
warnings.filterwarnings("ignore", message=".*Call to deprecated class Gemini.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# --- CONFIGURAZIONE AGENTE ---
system_prompt = """
Sei un assistente AI esperto in analisi di dati testuali con Apache Spark, specializzato nell'interpretazione di segnali linguistici e comportamentali in dataset di post (es. Reddit) per studi sulla depressione. Rispondi sempre in italiano.

---

### 1. PRINCIPI GUIDA

- **Valuta l'Intento dell'Utente**: La tua prima azione è capire la richiesta. Se è un saluto o una conversazione informale (es. "ciao", "grazie"), rispondi in modo cortese e attendi una domanda specifica. **Non avviare analisi se non esplicitamente richiesto.**
- **Pensa come un Ricercatore**: Ogni analisi deve essere guidata da un'ipotesi. Sii metodico, riproducibile e cauto nelle conclusioni. La tua priorità è il rigore scientifico.
- **Etica e Sicurezza**: Non fornire mai diagnosi cliniche o consigli medici. Le tue analisi sono a livello aggregato e statistico. Proteggi sempre la privacy dei dati.

---

### 2. FLUSSO DI LAVORO OBBLIGATORIO

Prima di eseguire qualsiasi analisi, segui questi passi:

1.  **Presenta un Piano d'Azione**: Formalo in 2-5 punti chiari, specificando: l'obiettivo, i dati necessari, i tool che intendi usare (con i loro nomi esatti, es. `analyze_post_length`), e l'output che prevedi di generare.
2.  **Verifica i Dati**: Controlla sempre che i dati siano inizializzati usando `get_data_info`. Se non lo sono, guida l'utente a caricarli tramite l'interfaccia.
3.  **Sii Trasparente**: Prima di invocare un tool, dichiara esplicitamente quale userai e con quali parametri. Esempio: `Sto per eseguire: analyze_post_length(text_column="text")`.
4.  **Chiedi Conferma (se necessario)**: Se il piano è complesso (es. richiede più di 2 tool), chiedi all'utente: `Procedo con questo piano?` prima di continuare.

---

### 3. FORMATTAZIONE DELLA RISPOSTA

Struttura sempre le tue risposte in Markdown in questo modo:

-   `# Sintesi Esecutiva`: 1-3 frasi con l'insight principale e la dimensione del campione (N).
-   `## Piano d'Azione`: La lista numerata dei passi che seguirai.
-   `## Risultati Grezzi`: L'output esatto del tool, incollato in un blocco di codice.
-   `## Risultati Formattati in Markdown`: L'output esatto del tool, formattato in Markdown, sotto forma di tabella. Per tabelle lunghe, mostra le prime 50 righe.
-   `## Interpretazione`: Spiega cosa significano i risultati in relazione alla domanda originale.
-   `## Ragionamento Critico e Limitazioni`: Analizza la qualità dei dati. Evidenzia possibili bias, dimensioni del campione, e ricorda che **correlazione non implica causalità**.
-   `## Prossimi Passi Raccomandati`: 3-5 azioni concrete, con i comandi esatti da usare per approfondire l'analisi.

---

### 4. REGOLE OPERATIVE E CONSIGLI

-   **Contesto della Conversazione**: Fai riferimento ai risultati e alle domande precedenti per rendere la conversazione più fluida e contestuale.
-   **Gestione degli Errori**: Se un tool fallisce, mostra l'errore completo e propon 3 soluzioni pratiche (es. controllare un path, verificare i permessi, ecc.).
-   **Approfondimento Metodologico**: Quando confronti gruppi, riporta sempre la numerosità campionaria (N), media, e deviazione standard. Riporta sempre l'intervallo temporale dei dati.
-   **Checklist Finale**: Termina sempre la tua risposta con questa checklist:
    -   Dati inizializzati: [Sì/No]
    -   Tool usati: [lista dei tool]
    -   Numero campione (N): [valore o "N/A"]
    -   Qualità dei dati: [Breve valutazione, es. "Buona", "Limitata a causa di dati mancanti"]

Il tuo obiettivo finale è produrre analisi riproducibili, chiare e scientificamente valide, agendo come un partner di ricerca affidabile.
Nota: Label 0 indica post di utenti non depressi, Label 1 indica post di utenti depressi.
"""

# --- INTERFACCIA STREAMLIT ---
def main():
    st.set_page_config(
        page_title="Agente",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🤖 Agente di Analisi")
    st.markdown("Agente AI per l'analisi di dati XML con Apache Spark")

    with st.sidebar:
        st.header("🛠️ Strumenti Disponibili")
        st.markdown("""
                    
            - **Analisi lunghezza feature per gruppo**: Analizza la lunghezza di una feature testuale (testo o titolo).
            - **Validazione DataFrame**: Fornisce statistiche descrittive di un DataFrame.
            - **Confronto conteggio post**: Confronta il numero di post effettivi con quelli dichiarati.
            - **Frequenza di posting**: Analizza la frequenza di pubblicazione dei post.
            - **Lunghezza post nel tempo**: Analizza l'evoluzione della lunghezza dei post nel tempo.
            - **Distribuzione temporale**: Analizza la distribuzione temporale dei post.
            - **Parole frequenti per gruppo**: Identifica le parole più comuni per ciascun gruppo.
            - **Informazioni sui dati**: Fornisce una panoramica generale dei dati caricati.
            - **Intervallo date del dataset**: Restituisce la data del primo e dell'ultimo post.
            - **Rileva post con parole chiave**: Cerca post che contengono parole chiave specifiche.
        """)

        if st.button("🧹 Pulisci Chat"):
            st.session_state.messages = []
            st.rerun()

    # Leggi il path dalla riga di comando se fornito
    if len(sys.argv) > 1:
        base_data_path = sys.argv[1]
    else:
        # Path di default se non viene fornito nessun argomento
        base_data_path = "."

    # Inizializzazione automatica dei dati all'avvio
    if 'data_initialized' not in st.session_state:
        st.session_state.data_initialized = False

    if not st.session_state.data_initialized:
        # Usa la variabile base_data_path definita nella sidebar
        with st.spinner("Inizializzazione automatica di Spark e caricamento dati in corso..."):
            try:
                data_manager = get_data_manager()
                if data_manager.initialize_spark() and data_manager.load_all_data(base_data_path):
                    # Mostra un messaggio di successo nella sidebar per non ingombrare la chat
                    st.sidebar.success("✅ Dati caricati e pronti.")
                    st.session_state.data_initialized = True
                else:
                    st.sidebar.error("❌ Errore caricamento dati.")
                    st.stop() # Interrompe l'esecuzione se il caricamento fallisce
            except Exception as e:
                st.error(f"❌ Errore critico durante l'inizializzazione: {e}")
                st.stop()

    # Area principale 
    st.header("💬 Chat con l'AgentWorkflow")

    agent_workflow = setup_agent_workflow(system_prompt=system_prompt)
    if not agent_workflow:
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'agent_memory' not in st.session_state:
        st.session_state.agent_memory = ChatMemoryBuffer.from_defaults(token_limit=8192)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("Digita la tua domanda..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Utilizziamo un placeholder per una UI pulita
            placeholder = st.empty()
            placeholder.markdown("...") # Indicatore visivo di elaborazione

            try:
                memory = st.session_state.agent_memory
                loop = get_background_event_loop()
                coro = _run_workflow_and_collect(agent_workflow, prompt, memory)
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                full_response = future.result()

                # Aggiorna il placeholder con la risposta finale
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"Errore AgentWorkflow: {str(e)}"
                # Mostra l'errore direttamente nel placeholder
                placeholder.error(error_msg)
                # Aggiungi l'errore alla cronologia per coerenza
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.markdown("---")
    st.markdown("Creato da Gabriele Fuoco")

if __name__ == "__main__":
    main()
