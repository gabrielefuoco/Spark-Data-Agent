from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

import os
import sys

from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, datediff, avg, stddev, count, collect_list, concat_ws, array_join, length, date_format, explode, lag, lit, min, max, sum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['PYSPARK_PYTHON'] = 'python'
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

logger.info(f"PYSPARK_PYTHON: {os.environ.get('PYSPARK_PYTHON')}")
logger.info(f"PYSPARK_DRIVER_PYTHON: {os.environ.get('PYSPARK_DRIVER_PYTHON')}")
logger.info(f"sys.executable: {sys.executable}")

# Funzione per creare una SparkSession configurata correttamente
def create_spark_session(app_name="MySparkApp", memory="16g"):
    """
    Crea una SparkSession configurata per funzionare correttamente e con supporto XML.
    """
    from pyspark.sql import SparkSession
    import os

    # Pacchetto spark-xml per la lettura dei file XML, hardcoded per problemi di compatibilità con la versione di spark installata
    spark_xml_package = "com.databricks:spark-xml_2.12:0.17.0"

    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.executor.memory", memory) \
        .config("spark.driver.memory", memory) \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.jars.packages", spark_xml_package) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark

# Funzione per caricare i dati XML dai chunk
def load_xml_data(spark: SparkSession, xml_path: str):
    """
    Carica i dati XML da un percorso usando il pacchetto spark-xml.
    Utilizza uno schema esplicito per una lettura più robusta ed efficiente.
    """
    logger.info(f"Caricamento dei dati XML da: {xml_path}")

    xml_post_schema = StructType([
        StructField("TITLE", StringType(), True),
        StructField("DATE", StringType(), True),
        StructField("INFO", StringType(), True),
        StructField("TEXT", StringType(), True)
    ])

    try:
        df = spark.read.format("xml") \
            .option("rootTag", "INDIVIDUAL") \
            .option("rowTag", "WRITING") \
            .schema(xml_post_schema) \
            .load(xml_path)
        
        df = df.withColumn("file_path", input_file_name())
        df = df.withColumn("subject_id", regexp_extract(col("file_path"), r'(subject\d+)_?\d*\.xml$', 1))
        df = df.withColumn("chunk", regexp_extract(col("file_path"), r'chunk(\d+)', 1).cast("int"))

        # Rimuove spazi bianchi prima/dopo la data per compatibilità parser Spark 3.x
        df_cleaned = df.select(
            col("subject_id"),
            col("chunk"),
            trim(col("TITLE")).alias("title"),
            to_timestamp(trim(col("DATE")), "yyyy-MM-dd HH:mm:ss").alias("date"),
            col("TEXT").alias("text")
        )
        logger.info("Dati XML caricati e processati con successo.")
        return df_cleaned
    except Exception as e:
        logger.error(f"Errore critico durante la lettura dei file XML: {e}")
        return None

# Funzione per caricare il file ground truth
def load_ground_truth(spark, path):
    """
    Carica il file ground truth con le etichette di depressione
    """
    schema = StructType([
        StructField("subject_id", StringType(), False),
        StructField("label", IntegerType(), False)
    ])
    
    return spark.read.csv(path, sep="\t", schema=schema)

# Funzione per caricare il file writings-per-subject
def load_writings_count(spark, path):
    """
    Carica il file con il conteggio dei writings per soggetto
    """
    schema = StructType([
        StructField("subject_id", StringType(), False),
        StructField("num_writings", IntegerType(), False)
    ])
    
    return spark.read.csv(path, sep="\t", schema=schema)

def prepare_data_for_model(df_posts, df_truth, text_filter=True, title_filter=True):
    """
    Prepara i dati per il modello di classificazione
    
    Parameters:
    - df_posts: DataFrame con i post
    - df_truth: DataFrame con le etichette ground truth
    - text_filter: Se True, filtra i post con testo valido (default: True)
    - title_filter: Se True, filtra i post con titolo valido (default: True)
    """
    # Verifica che entrambi i DataFrame abbiano dati
    logger.info(f"Post DataFrame count: {df_posts.count()}")
    logger.info(f"Truth DataFrame count: {df_truth.count()}")
    
    # Costruisci la condizione di filtro dinamicamente
    filter_condition = lit(True)  # Inizia con True
    
    if text_filter:
        filter_condition = filter_condition & \
                          col("text").isNotNull() & \
                          (col("text") != "") & \
                          (length(col("text")) > 0)
    
    if title_filter:
        filter_condition = filter_condition & \
                          col("title").isNotNull() & \
                          (col("title") != "")
    
    # Applica il filtro
    df_posts_valid = df_posts.filter(filter_condition)
    
    logger.info(f"Post validi dopo filtro: {df_posts_valid.count()}")
    
    # Verifica la sovrapposizione dei subject_id
    post_subjects = df_posts_valid.select("subject_id").distinct()
    truth_subjects = df_truth.select("subject_id").distinct()
    
    logger.info(f"Subject_id unici nei post: {post_subjects.count()}")
    logger.info(f"Subject_id unici nella truth: {truth_subjects.count()}")
    
    # Trova l'intersezione
    common_subjects = post_subjects.intersect(truth_subjects)
    logger.info(f"Subject_id comuni: {common_subjects.count()}")
    
    # Aggregare tutti i post per utente
    df_user_posts = df_posts_valid.groupBy("subject_id").agg(
        collect_list("text").alias("texts"),
        collect_list("title").alias("titles")
    )
    
    logger.info(f"Utenti aggregati: {df_user_posts.count()}")
    
    # Unisci tutti i testi in un unico campo
    df_user_posts = df_user_posts.withColumn(
        "combined_texts",
        concat_ws(" ", col("texts"))
    ).withColumn(
        "combined_titles",
        concat_ws(" ", col("titles"))
    ).withColumn(
        "all_text",
        concat_ws(" ", col("combined_texts"), col("combined_titles"))
    )
    
    # Join con le etichette
    df_model_data = df_user_posts.join(
        df_truth,
        on="subject_id",
        how="left_outer" #  left_outer perchè alcuni soggetti potrebbero non avere etichetta
    ).select(
        "subject_id",
        "all_text",
        "label"
    )
    
    logger.info(f"Dati modello dopo join: {df_model_data.count()}")
    
    return df_model_data

def get_top_words(df_words, label_value, n=20, min_length=3):
    """
    Estrae le parole più frequenti per un gruppo specifico
    
    Parameters:
    - df_words: DataFrame con le parole tokenizzate
    - label_value: Valore dell'etichetta (0 o 1)
    - n: Numero di parole top da restituire (default: 20)
    - min_length: Lunghezza minima della parola (default: 3)
    """
    # Conversioen a int perchè l'agente potrebbe iserire float
    n = int(n)
    min_length = int(min_length)

    # Filtra per etichetta
    df_filtered = df_words.filter(col("label") == label_value)
    
    # Esplodi le liste di parole in righe separate
    df_exploded = df_filtered.select("subject_id", explode("filtered_words").alias("word"))
    
    # Filtra le parole troppo corte e i caratteri non validi
    df_clean_words = df_exploded.filter(
        (length(col("word")) >= min_length) &
        col("word").rlike("^[a-zA-Z]+$")  # Solo lettere
    )
    
    # Conta le occorrenze di ciascuna parola
    word_counts = df_clean_words.groupBy("word").count().orderBy(col("count").desc()).limit(n)
    
    return word_counts

def _get_aggregated_by_label(df_main: DataFrame, df_truth: DataFrame, aggregations: list):
    """FUNZIONE HELPER: Unisce un DataFrame con df_truth e calcola aggregazioni per gruppo."""
    df_joined = df_main.join(df_truth, "subject_id", "inner")
    return df_joined.groupBy("label").agg(*aggregations)


def analyze_text_feature_length(df_posts, df_truth, text_column="text", **kwargs):
    """
    Calcola la lunghezza media di una feature testuale per utenti depressi vs non depressi.

    Parameters:
    - df_posts: DataFrame con i post.
    - df_truth: DataFrame con le etichette ground truth.
    - text_column: Nome della colonna di testo da analizzare (es. "text" o "title").
    """
    df_valid = df_posts.filter(
        col(text_column).isNotNull() &
        (col(text_column) != "") &
        (length(col(text_column)) > 0)
    )
    df_with_length = df_valid.withColumn("text_length", length(col(text_column)))
    aggregations = [
        avg("text_length").alias("avg_length"),
        stddev("text_length").alias("stddev_length"),
        count("*").alias("post_count"),
        min("text_length").alias("min_length"),
        max("text_length").alias("max_length")
    ]
    return _get_aggregated_by_label(df_with_length, df_truth, aggregations)

def find_frequent_words_by_depression(df_model_data, n=20, min_word_length=3):
    """
    Trova le parole più frequenti per utenti depressi vs non depressi
    
    Parameters:
    - df_model_data: DataFrame pre-processato con 'all_text' e 'label'
    - n: Numero di parole top da restituire (default: 20)
    - min_word_length: Lunghezza minima della parola (default: 3)
    """
    # Tokenizza il testo
    tokenizer = Tokenizer(inputCol="all_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # Applica la pipeline di preprocessing
    df_words = remover.transform(tokenizer.transform(df_model_data))
    
    # Trova le parole più frequenti per depressi (label=1)
    depressed_words = get_top_words(df_words, 1, n, min_word_length)
    
    # Trova le parole più frequenti per non depressi (label=0)
    non_depressed_words = get_top_words(df_words, 0, n, min_word_length)
    
    return (depressed_words, non_depressed_words)


def compare_post_counts(df_posts, df_writings, group_by_column="subject_id", **kwargs):
    """
    Confronta il numero di post effettivi con il conteggio dichiarato per utente
    
    Parameters:
    - df_posts: DataFrame con i post
    - df_writings: DataFrame con il conteggio dichiarato
    - group_by_column: Colonna per il raggruppamento (default: "subject_id")
    """
    # Conta i post effettivi per utente
    actual_counts = df_posts.groupBy(group_by_column).count().withColumnRenamed("count", "actual_count")
    
    # Join con il conteggio dichiarato
    result = actual_counts.join(df_writings, group_by_column, "left_outer")
    
    # Aggiungi una colonna con la differenza
    result = result.withColumn(
        "difference", 
        col("actual_count") - col("num_writings")
    )
    
    return result

def analyze_posting_frequency_improved(df_posts, subject_id_column="subject_id", date_column="date"):
    """
    Analizza la frequenza di posting per utente con controlli migliorati
    
    Parameters:
    - df_posts: DataFrame con i post
    - subject_id_column: Nome della colonna subject_id (default: "subject_id")
    - date_column: Nome della colonna data (default: "date")
    """
    # Filtrare i dati validi
    df_valid_dates = df_posts.filter(
        col(date_column).isNotNull() &
        col(subject_id_column).isNotNull()
    )
    
    logger.info(f"Post con date valide: {df_valid_dates.count()}")
    
    # Contare i post per utente prima di procedere
    user_post_counts = df_valid_dates.groupBy(subject_id_column).count()
    users_with_multiple_posts = user_post_counts.filter(col("count") > 1)
    
    logger.info(f"Utenti con più di un post: {users_with_multiple_posts.count()}")
    
    if users_with_multiple_posts.count() == 0:
        logger.warning("Nessun utente con più di un post trovato")
        return df_posts.select(subject_id_column, lit(0.0).alias("avg_days_between_posts"),
                              lit(0.0).alias("stddev_days_between_posts"),
                              lit(0).alias("post_count")).limit(0)
    
    # Procedere solo con utenti che hanno più di un post
    valid_users = users_with_multiple_posts.select(subject_id_column)
    df_filtered = df_valid_dates.join(valid_users, subject_id_column)
    
    # Ordinare i post per utente e data
    window_spec = Window.partitionBy(subject_id_column).orderBy(date_column)
    
    # Calcolare la differenza in giorni tra post consecutivi
    df_with_diff = df_filtered.withColumn(
        "prev_date",
        lag(date_column).over(window_spec)
    ).withColumn(
        "days_between_posts",
        when(col("prev_date").isNotNull(),
             datediff(col(date_column), col("prev_date")))
        .otherwise(None)
    )
    
    # Calcolare la media dei giorni tra i post per utente
    result = df_with_diff.filter(col("days_between_posts").isNotNull()).groupBy(subject_id_column).agg(
        avg("days_between_posts").alias("avg_days_between_posts"),
        stddev("days_between_posts").alias("stddev_days_between_posts"),
        count("*").alias("post_count")
    )
    
    return result

def analyze_posting_frequency(df_posts, df_truth, group_by_depression: bool = False, subject_id_column="subject_id", date_column="date", **kwargs):
    """
    Analizza la frequenza di posting. Se group_by_depression è True, confronta i risultati tra i gruppi.
    Altrimenti, fornisce una sintesi statistica sull'intero dataset.
    """
    df_frequency = analyze_posting_frequency_improved(df_posts, subject_id_column, date_column)

    if group_by_depression:
        aggregations = [
            avg("avg_days_between_posts").alias("avg_days_between_posts"),
            stddev("avg_days_between_posts").alias("stddev_days_between_posts"),
            count("*").alias("user_count")
        ]
        return _get_aggregated_by_label(df_frequency, df_truth, aggregations)
    else:
        return df_frequency.describe(['avg_days_between_posts', 'stddev_days_between_posts', 'post_count'])

def analyze_post_length_over_time(df_posts, subject_id_column="subject_id", date_column="date", text_column="text", date_format_pattern="yyyy-MM", **kwargs):
    """
    Analizza come cambia la lunghezza dei post nel tempo per utente
    
    Parameters:
    - df_posts: DataFrame con i post
    - subject_id_column: Nome della colonna subject_id (default: "subject_id")
    - date_column: Nome della colonna data (default: "date")
    - text_column: Nome della colonna testo (default: "text")
    - date_format_pattern: Pattern per il formato data (default: "yyyy-MM")
    """
    # Assicurati che ci sia una colonna data valida
    df_valid = df_posts.filter(
        col(date_column).isNotNull() &
        col(text_column).isNotNull() &
        (col(text_column) != "")
    )
    
    logger.info(f"Post validi per analisi temporale: {df_valid.count()}")
    
    # Aggiungi una colonna con la lunghezza del testo
    df_with_length = df_valid.withColumn("text_length", length(col(text_column)))
    
    # Estrai periodo dalla data
    df_with_date = df_with_length.withColumn(
        "time_period",
        date_format(col(date_column), date_format_pattern)
    )
    
    # Calcola la lunghezza media per periodo e utente
    result = df_with_date.groupBy(subject_id_column, "time_period").agg(
        avg("text_length").alias("avg_length"),
        count("*").alias("post_count")
    ).orderBy(subject_id_column, "time_period")
    
    return result

def validate_dataframe(df, df_name):
    """
    Valida un DataFrame e stampa statistiche di base
    """
    logger.info(f"\n=== Validazione {df_name} ===")
    logger.info(f"Numero di righe: {df.count()}")
    logger.info(f"Numero di colonne: {len(df.columns)}")
    logger.info(f"Colonne: {df.columns}")
    
    # Controlla valori null
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    logger.info("Valori null per colonna:")
    null_counts.show()
    
    return df

def get_dataset_date_range(df_posts, **kwargs):
    """
    Restituisce l'intervallo temporale del dataset (data minima e massima).
    """
    result = df_posts.agg(
        min("date").alias("min_date"),
        max("date").alias("max_date")
    )
    # Conversiomne a string per evitare problemi con pandas
    result = result.withColumn("min_date", col("min_date").cast("string")) \
                   .withColumn("max_date", col("max_date").cast("string"))
    return result

def count_posts_by_time_period(df_posts, df_truth=None, period: str = 'month', group_by_depression: bool = False, **kwargs):
    """
    Conta i post per periodo di tempo. Se group_by_depression è True, raggruppa anche per etichetta di depressione.
    """
    if group_by_depression and df_truth is None:
        raise ValueError("df_truth deve essere fornito per raggruppare per depressione.")

    if period.lower() not in ['year', 'month', 'day']:
        raise ValueError("Il periodo deve essere 'year', 'month', o 'day'")

    format_map = { 'year': 'yyyy', 'month': 'yyyy-MM', 'day': 'yyyy-MM-dd' }
    df_with_period = df_posts.withColumn("time_period", date_format(col("date"), format_map[period.lower()]))

    if group_by_depression:
        df_joined = df_with_period.join(df_truth, "subject_id")
        result = df_joined.groupBy("time_period", "label").count().orderBy("time_period", "label")
    else:
        result = df_with_period.groupBy("time_period").count().orderBy("time_period")

    return result

def detect_posts_with_keywords(df_posts, df_truth, keywords: list, **kwargs):
    """
    Conta, per ogni gruppo (depresso/non depresso), quanti post contengono
    almeno una delle parole chiave specificate.
    """
    # Unisci i post con le etichette di verità
    df_joined = df_posts.join(df_truth, "subject_id")

    # Crea una condizione SQL per la ricerca delle parole chiave (case-insensitive)
    condition = col("text").rlike(f"(?i)({'|'.join(keywords)})")

    # Applica la condizione per creare una colonna flag
    df_with_flag = df_joined.withColumn("has_keyword", when(condition, 1).otherwise(0))

    # Raggruppa per etichetta e calcola i conteggi
    result = df_with_flag.groupBy("label").agg(
        count("*").alias("total_posts"),
        sum("has_keyword").alias("posts_with_keyword")
    ).withColumn(
        "percentage_with_keyword",
        (col("posts_with_keyword") / col("total_posts")) * 100
    )

    return result