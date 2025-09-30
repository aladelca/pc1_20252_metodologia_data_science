"""Pipeline principal de preprocesamiento"""

import pandas as pd
import os
import logging
from .cleaning import clean_data
from .aggregation import create_product_lag_features, prepare_modeling_data

logger = logging.getLogger(__name__)

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
    """Guardar datos procesados"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath, index=False)
        logger.info(f"Datos guardados en: {filepath}")
    except Exception as e:
        logger.error(f"Error guardando datos: {e}")
        raise

def run_preprocessing_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """Ejecutar pipeline completo de preprocesamiento"""
    logger.info("Iniciando pipeline de preprocesamiento")
    
    # 1. Cargar datos
    logger.info("Cargando datos raw...")
    df_raw = load_data(input_path)
    
    # 2. Limpieza
    logger.info("Aplicando limpieza...")
    df_clean = clean_data(df_raw)
    
    # 3. Crear features de lag
    logger.info("Creando features temporales...")
    df_with_lags = create_product_lag_features(df_clean)
    
    # 4. Preparar para modelado
    logger.info("Preparando datos para modelado...")
    df_final = prepare_modeling_data(df_with_lags)
    
    # 5. Guardar resultados
    logger.info("Guardando datos procesados...")
    save_processed_data(df_final, output_path)
    
    logger.info("Pipeline completado exitosamente")
    return df_final