import os
from spark_utils import (
    create_spark_session, load_xml_data, load_ground_truth,
    prepare_data_for_model, load_writings_count
)
import logging
import pandas as pd

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Script per pre-processare i dati grezzi (XML, txt) e salvarli in formati ottimizzati (Parquet).
    Questo script deve essere eseguito una sola volta prima di lanciare l'app Streamlit.
    """
    spark = create_spark_session(app_name="DataPreprocessing", memory="16g")

    base_data_path = "."

    # Definisci i percorsi di output
    posts_output_path = os.path.join(base_data_path, "processed_data/posts.parquet")
    truth_output_path = os.path.join(base_data_path, "processed_data/truth.parquet")
    model_data_output_path = os.path.join(base_data_path, "processed_data/model_data.parquet")
    writings_output_path = os.path.join(base_data_path, "processed_data/writings.parquet")

    logger.info("Avvio del pre-processing dei dati...")

    try:
        # 1. Carica tutti i dati grezzi
        chunk_base = os.path.join(base_data_path, "chunk")
        available_chunks = [item for item in os.listdir(chunk_base) if os.path.isdir(os.path.join(chunk_base, item))]
        if not available_chunks:
            raise FileNotFoundError("Nessuna cartella 'chunk' trovata.")

        xml_paths = [os.path.join(base_data_path, f"chunk/{chunk}/*.xml") for chunk in sorted(available_chunks)]
        ground_truth_path = os.path.join(base_data_path, "risk-golden-truth-test.txt")
        writings_count_path = os.path.join(base_data_path, "writings-per-subject-all-test.txt")

        df_posts_list = [load_xml_data(spark, path) for path in xml_paths]
        df_posts = df_posts_list[0]
        for df in df_posts_list[1:]:
            df_posts = df_posts.unionByName(df)
        df_truth = load_ground_truth(spark, ground_truth_path)

        if df_posts is None or df_truth is None:
            raise ValueError("Errore critico nel caricamento di df_posts o df_truth.")

        logger.info("Dati grezzi caricati con successo.")

        # 2. Prepara il DataFrame principale
        logger.info("Preparazione del DataFrame del modello...")
        df_model_data = prepare_data_for_model(df_posts, df_truth)

        # 3. Salva tutti i DataFrame necessari in formato Parquet
        logger.info(f"Salvataggio di df_posts in: {posts_output_path}")
        df_posts.write.mode("overwrite").parquet(posts_output_path)

        logger.info(f"Salvataggio di df_truth in: {truth_output_path}")
        df_truth.write.mode("overwrite").parquet(truth_output_path)

        logger.info(f"Salvataggio di df_model_data in: {model_data_output_path}")
        df_model_data.write.mode("overwrite").parquet(model_data_output_path)

        if os.path.exists(writings_count_path):
            df_writings = load_writings_count(spark, writings_count_path)
            if df_writings is not None:
                logger.info(f"Salvataggio di df_writings in: {writings_output_path}")
                df_writings.write.mode("overwrite").parquet(writings_output_path)

        logger.info("Pre-processing completato con successo!")

    except Exception as e:
        logger.error(f"Errore durante il pre-processing: {e}", exc_info=True)
    finally:
        spark.stop()
        logger.info("Sessione Spark terminata.")

if __name__ == "__main__":
    main()
