"""Script para ejecutar el pipeline completo y generar archivos parquet"""

import pandas as pd
import sys
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocess.cleaning import clean_data
from preprocess.aggregation import create_daily_product_aggregation, create_product_global_metrics
from preprocess.preprocessing import create_product_lag_features, create_time_features, prepare_modeling_data


def load_data(filepath: str) -> pd.DataFrame:
    """Cargar datos desde archivo parquet"""
    try:
        df = pd.read_parquet(filepath)
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error cargando datos: {e}")
        raise


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """Guardar datos procesados en parquet"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath, index=False)
        logger.info(f"Datos guardados en: {filepath} ({df.shape[0]} filas, {df.shape[1]} columnas)")
    except Exception as e:
        logger.error(f"Error guardando datos: {e}")
        raise


def run_pipeline():
    """Ejecutar pipeline completo"""
    logger.info("=== INICIANDO PIPELINE ===")

    # Rutas base
    base_path = os.path.dirname(__file__)           # src/pipeline
    project_root = os.path.dirname(base_path)       # src
    data_root = os.path.dirname(project_root)       # raíz del proyecto

    input_path = os.path.join(data_root, "data", "raw", "data_sample.parquet")
    processed_dir = os.path.join(data_root, "data", "processed")

    if not os.path.exists(input_path):
        logger.error(f"No se encuentra el archivo raw: {input_path}")
        return

    try:
        # 1. CLEANING
        logger.info("Paso 1: Cleaning")
        df_raw = load_data(input_path)
        df_clean = clean_data(df_raw)
        save_processed_data(df_clean, os.path.join(processed_dir, "data_cleaned.parquet"))

        # 2. AGGREGATION
        logger.info("Paso 2: Aggregation")
        daily_agg = create_daily_product_aggregation(df_clean)
        product_global = create_product_global_metrics(df_clean)

        df_with_agg = df_clean.merge(daily_agg, on=["product_sku", "parsed_date"], how="left")
        df_with_agg = df_with_agg.merge(product_global, on="product_sku", how="left")
        save_processed_data(df_with_agg, os.path.join(processed_dir, "data_processed.parquet"))

        # 3. PREPROCESSING
        logger.info("Paso 3: Preprocessing")
        df_with_lags = create_product_lag_features(df_with_agg)
        df_with_time = create_time_features(df_with_lags)
        df_final = prepare_modeling_data(df_with_time)
        save_processed_data(df_final, os.path.join(processed_dir, "data_training.parquet"))

        # Resumen final
        logger.info("=== PIPELINE FINALIZADO ===")
        logger.info(f"Archivos generados en {processed_dir}:")
        logger.info(" - data_cleaned.parquet")
        logger.info(" - data_processed.parquet")
        logger.info(" - data_training.parquet")

    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()