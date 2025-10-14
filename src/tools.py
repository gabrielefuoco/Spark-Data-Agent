import streamlit as st
from llama_index.core.tools import FunctionTool
import json

import spark_utils
from data_manager import get_data_manager

def analyze_feature_length_tool(feature_column: str) -> str:
    """Calcola la lunghezza media (in numero di caratteri) di una feature testuale specificata (es. 'text' o 'title') e confronta i risultati tra utenti depressi e non depressi. Questo strumento raggruppa gli utenti in base al loro stato di depressione e restituisce una tabella con le statistiche sulla lunghezza per ciascun gruppo. Richiede il parametro 'feature_column' per specificare la colonna da analizzare."""
    data_manager = get_data_manager()
    if not data_manager.is_initialized:
        return "Errore: Dati non inizializzati."

    cache_key = f"analyze_feature_length_{feature_column}"
    if cache_key not in data_manager.analysis_cache:
        with st.spinner(f"Analisi lunghezza per '{feature_column}' in corso..."):
            result_df = spark_utils.analyze_text_feature_length(
                df_posts=data_manager.df_posts,
                df_truth=data_manager.df_truth,
                text_column=feature_column
            )
            data_manager.analysis_cache[cache_key] = result_df.toPandas()
    
    result_pd = data_manager.analysis_cache[cache_key]
    return f"Risultati per 'analisi lunghezza di {feature_column}':\n{result_pd.to_string()}"

def compare_post_counts_tool() -> str:
    """Confronta il numero di post effettivamente scritti da ciascun utente con il numero di post che l'utente ha dichiarato di aver scritto. Questo strumento è utile per verificare la coerenza dei dati e identificare eventuali discrepanze tra i dati auto-riportati e quelli effettivi. Restituisce una tabella con l'ID utente, il conteggio effettivo e il conteggio dichiarato."""
    data_manager = get_data_manager()
    if not data_manager.is_initialized: return "Errore: Dati non inizializzati."

    cache_key = "compare_post_counts"
    if cache_key not in data_manager.analysis_cache:
        with st.spinner("Confronto conteggio post in corso..."):
            result_df = spark_utils.compare_post_counts(df_posts=data_manager.df_posts, df_writings=data_manager.df_writings)
            data_manager.analysis_cache[cache_key] = result_df.toPandas()
            
    result_pd = data_manager.analysis_cache[cache_key]
    return f"Risultati per 'confronto conteggio post':\n{result_pd.to_string()}"
    
def analyze_posting_frequency_tool(group_by_depression: bool = False) -> str:
    """
    Analizza la frequenza di pubblicazione dei post.
    - Se 'group_by_depression' è False (default), fornisce una sintesi statistica aggregata (media, stddev, min, max) sul comportamento di posting generale.
    - Se 'group_by_depression' è True, confronta la frequenza media di pubblicazione tra il gruppo di utenti depressi e quello di utenti non depressi.
    Utile per comprendere sia il comportamento generale sia le differenze specifiche tra i gruppi.
    """
    data_manager = get_data_manager()
    if not data_manager.is_initialized: return "Errore: Dati non inizializzati."

    cache_key = f'posting_frequency_summary_{group_by_depression}'
    if cache_key not in data_manager.analysis_cache:
        with st.spinner("Analisi frequenza di posting in corso..."):
            result_df = spark_utils.analyze_posting_frequency(
                data_manager.df_posts,
                data_manager.df_truth,
                group_by_depression=group_by_depression
            )
            data_manager.analysis_cache[cache_key] = result_df.toPandas()
            
    result = data_manager.analysis_cache[cache_key]
    return f"Sintesi statistica della frequenza di posting:\n{result.to_string()}"

def analyze_post_length_over_time_tool() -> str:
    """Analizza l'evoluzione della lunghezza dei post di ogni singolo utente nel corso del tempo. Per ogni utente, questo strumento traccia la lunghezza di ciascun post in ordine cronologico. Il risultato è una tabella che mostra l'ID utente, la data del post e la lunghezza del post, permettendo di osservare trend individuali come l'aumento o la diminuzione della verbosità nel tempo."""
    data_manager = get_data_manager()
    if not data_manager.is_initialized: return "Errore: Dati non inizializzati."

    cache_key = "post_length_over_time"
    if cache_key not in data_manager.analysis_cache:
        with st.spinner("Analisi lunghezza post nel tempo in corso..."):
            result_df = spark_utils.analyze_post_length_over_time(df_posts=data_manager.df_posts)
            data_manager.analysis_cache[cache_key] = result_df.toPandas()
            
    result_pd = data_manager.analysis_cache[cache_key]
    return f"Risultati per 'lunghezza post nel tempo':\n{result_pd.to_string()}"

def analyze_temporal_distribution_tool(period: str = 'month', group_by_depression: bool = False, limit: int = 100) -> str:
    """
    Analizza la distribuzione temporale dei post. Conta i post per periodo e ordina i risultati.
    - 'period': Definisce il livello di aggregazione ('year', 'month', 'day'). Default: 'month'.
    - 'group_by_depression': Se True, suddivide i conteggi per utenti depressi e non. Default: False.
    - 'limit': Limita il numero di risultati restituiti. Default: 100.
    """
    data_manager = get_data_manager()
    if not data_manager.is_initialized: return "Errore: Dati non inizializzati."

    cache_key = f"temporal_distribution_{period}_{group_by_depression}_{limit}"
    if cache_key not in data_manager.analysis_cache:
        with st.spinner("Analisi distribuzione temporale in corso..."):
            result_df = spark_utils.count_posts_by_time_period(
                df_posts=data_manager.df_posts,
                df_truth=data_manager.df_truth,
                period=period,
                group_by_depression=group_by_depression
            ).limit(limit)
            data_manager.analysis_cache[cache_key] = result_df.toPandas()
            
    result_pd = data_manager.analysis_cache[cache_key]
    return f"Risultati per 'distribuzione temporale':\n{result_pd.to_string()}"

def find_frequent_words_tool(n: int = 20, min_word_length: int = 3) -> str:
    """Identifica e confronta le parole più utilizzate dagli utenti depressi rispetto a quelli non depressi. Questo strumento genera due liste separate di parole: una per il gruppo depresso e una per il gruppo non depresso, ordinate per frequenza. È utile per analisi lessicali e per scoprire termini distintivi di ciascun gruppo. Accetta due parametri opzionali: 'n' (il numero di parole più frequenti da restituire per ogni gruppo, default: 20) e 'min_word_length' (la lunghezza minima che una parola deve avere per essere inclusa nell'analisi, default: 3)."""
    data_manager = get_data_manager()
    if not data_manager.is_initialized: return "Errore: Dati non inizializzati."

    cache_key = f"find_frequent_words_{n}_{min_word_length}"
    if cache_key not in data_manager.analysis_cache:
        with st.spinner("Ricerca parole frequenti in corso..."):
            dep_df, non_dep_df = spark_utils.find_frequent_words_by_depression(
                df_model_data=data_manager.df_model_data, n=n, min_word_length=min_word_length
            )
            data_manager.analysis_cache[cache_key] = (dep_df.toPandas(), non_dep_df.toPandas())

    dep_pd, non_dep_pd = data_manager.analysis_cache[cache_key]
    return f"Parole per depressi:\n{dep_pd.to_string()}\n\nParole per non depressi:\n{non_dep_pd.to_string()}"

def get_dataset_date_range_tool() -> str:
    """Recupera l'intervallo temporale coperto dall'intero dataset dei post. Questo strumento determina e restituisce la data del post più vecchio (min_date) e di quello più recente (max_date) presenti nei dati. È utile per avere una visione d'insieme del periodo di tempo analizzato."""
    data_manager = get_data_manager()
    if not data_manager.is_initialized: return "Errore: Dati non inizializzati."

    cache_key = "dataset_date_range"
    if cache_key not in data_manager.analysis_cache:
        with st.spinner("Calcolo intervallo date del dataset in corso..."):
            result_df = spark_utils.get_dataset_date_range(df_posts=data_manager.df_posts)
            data_manager.analysis_cache[cache_key] = result_df.toPandas()
            
    result_pd = data_manager.analysis_cache[cache_key]
    return f"Intervallo date del dataset:\n{result_pd.to_string()}"


def detect_posts_with_keywords_tool(keywords_json: str) -> str:
    """Ricerca e restituisce tutti i post che contengono una o più parole chiave specificate. Questo strumento è utile per filtrare e analizzare post relativi a temi specifici. Richiede un parametro obbligatorio 'keywords_json', che deve essere una stringa in formato JSON contenente una lista di parole chiave. Esempio di input valido: '[\"ansia\", \"solitudine\", \"tristezza\"]'."""
    data_manager = get_data_manager()
    if not data_manager.is_initialized: return "Errore: Dati non inizializzati."
    
    try:
        keywords = json.loads(keywords_json)
        if not isinstance(keywords, list):
            return "Errore: l'input JSON deve essere una lista di stringhe."
    except json.JSONDecodeError:
        return "Errore: Formato JSON non valido. Usa una lista di stringhe, es. '[\"parola1\", \"parola2\"]'."

    cache_key = f"detect_keywords_{keywords_json}"
    if cache_key not in data_manager.analysis_cache:
        with st.spinner(f"Ricerca post con parole chiave: {keywords}..."):
            result_df = spark_utils.detect_posts_with_keywords(
                df_posts=data_manager.df_posts,
                df_truth=data_manager.df_truth,
                keywords=keywords
            )
            data_manager.analysis_cache[cache_key] = result_df.toPandas()

    result_pd = data_manager.analysis_cache[cache_key]
    return f"Risultati per la ricerca di parole chiave:\n{result_pd.to_string()}"
    
def get_data_info_tool() -> str:
    """Fornisce una panoramica generale dei dati attualmente caricati e dello stato del sistema. Restituisce informazioni quantitative di base, come il numero di record in ciascun DataFrame (posts, truth, model_data, writings) e il numero di analisi già presenti in cache. Utile per una rapida verifica dello stato dei dati."""
    data_manager = get_data_manager()
    if not data_manager.is_initialized: return "Errore: Dati non inizializzati."

    with st.spinner("Recupero informazioni dati..."):
        info = {
            "posts_count": data_manager.df_posts.count(),
            "truth_count": data_manager.df_truth.count(),
            "model_data_count": data_manager.df_model_data.count(),
            "writings_count": data_manager.df_writings.count() if data_manager.df_writings else "Non disponibile",
            "analysis_cache_size": len(data_manager.analysis_cache)
        }
    return f"Informazioni sui dati:\n{json.dumps(info, indent=2)}"

def validate_dataframe_tool(df_name: str) -> str:
    """Esegue una validazione statistica su un DataFrame specificato, restituendone le principali statistiche descrittive. Questo strumento calcola conteggio, media, deviazione standard, minimo e massimo per le colonne numeriche. Richiede il parametro 'df_name', che deve essere una delle seguenti stringhe: 'posts', 'truth', 'writings', o 'model_data'. Utile per un'analisi esplorativa approfondita di una specifica tabella."""
    data_manager = get_data_manager()
    if not data_manager.is_initialized: return "Errore: Dati non inizializzati."

    df_map = {
        "posts": data_manager.df_posts,
        "truth": data_manager.df_truth,
        "writings": data_manager.df_writings,
        "model_data": data_manager.df_model_data
    }
    if df_name not in df_map or df_map[df_name] is None:
        return f"Errore: DataFrame '{df_name}' non trovato o non disponibile."

    with st.spinner(f"Validazione di '{df_name}' in corso..."):
        result_df = df_map[df_name].summary()
    return f"Statistiche di riepilogo per '{df_name}':\n{result_df.toPandas().to_string()}"

# ==============================================================================
# FUNZIONE PRINCIPALE PER LA CREAZIONE DELLA LISTA DI STRUMENTI
# ==============================================================================

def create_tools():
    """
    Crea e restituisce la lista completa di tutti i FunctionTool per l'agente.
    Questa funzione raccoglie tutte le funzioni definite in questo file
    e le converte nel formato richiesto da LlamaIndex.
    """
    
    # Lista di tutti i tool
    tool_functions = [
        analyze_feature_length_tool,
        compare_post_counts_tool,
        analyze_post_length_over_time_tool,
        find_frequent_words_tool,
        get_dataset_date_range_tool,
        detect_posts_with_keywords_tool,
        get_data_info_tool,
        validate_dataframe_tool,
        analyze_posting_frequency_tool,
        analyze_temporal_distribution_tool,
    ]

    # Converte ogni funzione Python in un FunctionTool di LlamaIndex
    llama_tools = []
    for func in tool_functions:
        # Il nome dello strumento viene derivato dal nome della funzione (es. 'analyze_post_length_tool' -> 'analyze_post_length')
        tool_name = func.__name__.replace('_tool', '')
        # La descrizione viene presa direttamente dal docstring della funzione
        description = func.__doc__
        
        llama_tools.append(FunctionTool.from_defaults(fn=func, name=tool_name, description=description))

    print(f"Creati {len(llama_tools)} strumenti per l'agente.")
    return llama_tools
