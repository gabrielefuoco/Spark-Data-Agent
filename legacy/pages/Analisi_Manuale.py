import streamlit as st
import pandas as pd
import json
from data_manager import get_data_manager
import spark_utils


ANALYSIS_FUNCTIONS = {
    "Analisi lunghezza feature per depressione": {
        "func": spark_utils.analyze_text_feature_length,
        "desc": "Analizza la lunghezza media di una feature testuale (es. post o titolo) per gruppo di depressione. L'utente può specificare la colonna di testo da analizzare."
    },
    "Confronto conteggio post (effettivo vs dichiarato)": {
        "func": spark_utils.compare_post_counts,
        "desc": "Confronta il conteggio dei post effettivi con quello dichiarato per ogni utente."
    },
    "Analisi frequenza di posting": {
        "func": spark_utils.analyze_posting_frequency,
        "desc": "Analizza la frequenza di posting. Può fornire una sintesi generale o confrontare i risultati per depressione."
    },
    "Analisi lunghezza post nel tempo": {
        "func": spark_utils.analyze_post_length_over_time,
        "desc": "Analizza come cambia la lunghezza media dei post di un utente nel tempo."
    },
    "Trova parole frequenti per depressione": {
        "func": "find_frequent_words", # Caso speciale
        "desc": "Estrae le parole più frequenti per utenti depressi e non depressi."
    },
    "Ottieni intervallo date del dataset": {
        "func": spark_utils.get_dataset_date_range,
        "desc": "Restituisce la data del primo e dell'ultimo post nell'intero dataset."
    },
    "Conta post per periodo di tempo": {
        "func": spark_utils.count_posts_by_time_period,
        "desc": "Conta i post raggruppati per anno, mese o giorno."
    },
    "Rileva post con parole chiave": {
        "func": spark_utils.detect_posts_with_keywords,
        "desc": "Cerca post che contengono una o più parole chiave specifiche."
    }
}

def run_analysis(func, data_manager, **kwargs):
    """Esegue una funzione di analisi e restituisce il risultato come DataFrame Pandas."""
    try:
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None and v != ''}

        # Gestione dei casi speciali che non restituiscono un DataFrame standard
        if func == "find_frequent_words":
            dep_df, non_dep_df = spark_utils.find_frequent_words_by_depression(
                df_model_data=data_manager.df_model_data, **clean_kwargs
            )
            st.subheader("Parole Frequenti - Depressi")
            st.dataframe(dep_df.toPandas())
            st.subheader("Parole Frequenti - Non Depressi")
            st.dataframe(non_dep_df.toPandas())
            return None  # Indica che l'output è già stato gestito

        # Determina gli argomenti corretti da passare alla funzione
        if func == spark_utils.compare_post_counts:
            func_args = {'df_posts': data_manager.df_posts, 'df_writings': data_manager.df_writings}
        elif func in [
            spark_utils.get_dataset_date_range,
        ]:
            func_args = {'df_posts': data_manager.df_posts}
        else:
            # Default per la maggior parte delle analisi
            func_args = {'df_posts': data_manager.df_posts, 'df_truth': data_manager.df_truth}

        # Combina gli argomenti e chiama la funzione
        final_args = {**func_args, **clean_kwargs}
        result_df = func(**final_args)

        return result_df.toPandas()

    except Exception as e:
        st.error(f"Si è verificato un errore durante l'analisi: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="Analisi Manuale", layout="wide")
    st.title("🔬 Analisi Manuale dei Dati")

    data_manager = get_data_manager()

    if not data_manager.is_initialized:
        st.warning("⚠️ Dati non ancora inizializzati. Assicurati che la pagina principale abbia caricato i dati correttamente.")
        st.stop()

    # --- Sidebar per controlli ---
    with st.sidebar:
        st.header("Impostazioni Analisi")
        selected_analysis_name = st.selectbox(
            "Seleziona un'analisi da eseguire:",
            options=list(ANALYSIS_FUNCTIONS.keys())
        )

        selected_analysis_obj = ANALYSIS_FUNCTIONS[selected_analysis_name]
        func_to_run = selected_analysis_obj["func"]

        # Mostra la descrizione del tool selezionato
        st.info(selected_analysis_obj["desc"])

        kwargs = {}

        # --- Interfaccia dinamica per i parametri ---
        with st.expander("Parametri della funzione", expanded=True):
            if func_to_run == spark_utils.analyze_text_feature_length:
                kwargs['text_column'] = st.selectbox("Seleziona la colonna di testo", ["text", "title"])

            elif func_to_run == spark_utils.analyze_posting_frequency:
                kwargs['group_by_depression'] = st.checkbox("Raggruppa per depressione", False)

            elif func_to_run == "find_frequent_words":
                kwargs['n'] = st.number_input("Numero di parole da trovare", min_value=5, max_value=100, value=20)
                kwargs['min_word_length'] = st.number_input("Lunghezza minima della parola", min_value=1, max_value=10, value=3)

            elif func_to_run == spark_utils.count_posts_by_time_period:
                kwargs['period'] = st.selectbox("Periodo di raggruppamento", ["year", "month", "day"])
                kwargs['group_by_depression'] = st.checkbox("Raggruppa per depressione", False)

            elif func_to_run == spark_utils.detect_posts_with_keywords:
                keywords_str = st.text_input('Parole chiave (formato JSON, es. ["ansia", "triste"])', '["depressione", "ansia"]')
                try:
                    kwargs['keywords'] = json.loads(keywords_str)
                except json.JSONDecodeError:
                    st.error("Formato JSON non valido. Assicurati di usare i doppi apici.")
                    st.stop()
            else:
                st.info("Questa funzione non richiede parametri aggiuntivi.")

        # Pulsante per eseguire l'analisi
        run_button = st.button("🚀 Esegui Analisi", type="primary")

    # --- Area principale per i risultati ---
    st.header(f"Esecuzione: `{selected_analysis_name}`")

    if run_button:
        with st.spinner("Elaborazione in corso..."):
            result = run_analysis(func_to_run, data_manager, **kwargs)

            if result is not None:
                st.subheader("Risultati")
                st.dataframe(result)


if __name__ == "__main__":
    main()