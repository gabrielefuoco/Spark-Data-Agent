import streamlit as st
import os
import logging


try:
    from spark_utils import create_spark_session
except ImportError as e:
    st.error(f"Errore importazione Spark: {e}")
    st.stop()

logger = logging.getLogger(__name__)

# --- GESTIONE DATI SPARK ---

class SparkDataManager:
    def __init__(self):
        self.spark = None
        self.df_posts = None
        self.df_truth = None
        self.df_writings = None
        self.df_model_data = None
        self.is_initialized = False
        self.analysis_cache = {}

    def initialize_spark(self):
        try:
            if self.spark is None:
                self.spark = create_spark_session()
                logger.info("SparkSession creata con successo")
            return True
        except Exception as e:
            logger.error(f"Errore creazione SparkSession: {e}")
            return False

    def load_all_data(self, base_data_path: str):
        try:
            # Definisci i percorsi dei dati pre-processati
            posts_path = os.path.join(base_data_path, "processed_data/posts.parquet")
            truth_path = os.path.join(base_data_path, "processed_data/truth.parquet")
            model_data_path = os.path.join(base_data_path, "processed_data/model_data.parquet")
            writings_path = os.path.join(base_data_path, "processed_data/writings.parquet")

            # Controlla se i dati pre-processati esistono
            if not os.path.exists(model_data_path):
                error_msg = f"Dati processati non trovati in {model_data_path}. Esegui prima lo script 'python preprocess.py' dal terminale."
                logger.error(error_msg)
                st.error(error_msg) # Mostra un errore anche nell'interfaccia
                return False

            # Carica direttamente i file Parquet
            with st.spinner("Caricamento dati ottimizzati in corso..."):
                self.df_posts = self.spark.read.parquet(posts_path)
                self.df_truth = self.spark.read.parquet(truth_path)
                self.df_model_data = self.spark.read.parquet(model_data_path)

                if os.path.exists(writings_path):
                    self.df_writings = self.spark.read.parquet(writings_path)

                # Metti in cache i DataFrame per performance ottimali
                self.df_posts.persist()
                self.df_truth.persist()
                self.df_model_data.persist()
                if self.df_writings is not None:
                    self.df_writings.persist()

            self.is_initialized = True
      
            logger.info("Dati ottimizzati caricati e messi in cache con successo.")

            return True

        except Exception as e:
            logger.error(f"Errore caricamento dati Parquet: {e}", exc_info=True)
            st.error(f"Errore critico nel caricamento dei dati pre-processati: {e}")
            return False

    def cleanup(self):
        if self.spark:
            self.spark.stop()

# Istanza globale del data manager per pattern singleton
@st.cache_resource
def get_data_manager():
    return SparkDataManager()
