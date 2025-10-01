"""Script para ejecutar el pipeline completo y generar archivos parquet"""

import pandas as pd
import sys
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
    """Guardar datos procesados"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath, index=False)
        logger.info(f"Datos guardados en: {filepath}")
    except Exception as e:
        logger.error(f"Error guardando datos: {e}")
        raise

def generate_raw_process():
    """Ejecutar pipeline completo y generar todos los archivos parquet"""
    logger.info("Iniciando generación de datos procesados...")
    
    # Configurar paths - desde src/pipeline hasta data
    base_path = os.path.dirname(__file__)  # src/pipeline
    project_root = os.path.dirname(base_path)  # src
    data_root = os.path.dirname(project_root)  # raíz del proyecto
    
    input_path = os.path.join(data_root, 'data', 'raw', 'data_sample.parquet')
    processed_dir = os.path.join(data_root, 'data', 'processed')
    
    logger.info(f"Ruta data: {os.path.join(data_root, 'data')}")
    logger.info(f"Ruta input: {input_path}")
    
    # Verificar que existe el archivo raw
    if not os.path.exists(input_path):
        logger.error(f"No se encuentra el archivo raw: {input_path}")
        return
    
    try:
        # 1. CARGAR DATOS RAW
        logger.info("Cargando datos raw...")
        df_raw = load_data(input_path)
        logger.info(f"   Datos raw: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")
        
        # 2. LIMPIEZA - Guardar data_cleaned.parquet
        logger.info("Aplicando limpieza...")
        df_clean = clean_data(df_raw)
        cleaned_path = os.path.join(processed_dir, 'data_cleaned.parquet')
        save_processed_data(df_clean, cleaned_path)
        logger.info(f"data_cleaned.parquet: {df_clean.shape[0]} filas")
        
        # 3. AGREGACIONES - Guardar data_processed.parquet (con agregaciones)
        logger.info("Creando agregaciones...")
        daily_agg = create_daily_product_aggregation(df_clean)
        product_global = create_product_global_metrics(df_clean)
        
        # Unir agregaciones con datos limpios
        df_with_agg = df_clean.merge(daily_agg, on=['product_sku', 'parsed_date'], how='left')
        df_with_agg = df_with_agg.merge(product_global, on='product_sku', how='left')
        
        processed_path = os.path.join(processed_dir, 'data_processed.parquet')
        save_processed_data(df_with_agg, processed_path)
        logger.info(f"data_processed.parquet: {df_with_agg.shape[0]} filas")
        
        # 4. FEATURE ENGINEERING - Aplicar lags y features temporales
        logger.info("Creando features de lag...")
        df_with_lags = create_product_lag_features(df_with_agg)
        
        # 5. PREPARACIÓN MODELADO - Guardar data_training.parquet
        logger.info("Preparando datos para modelado...")
        df_final = prepare_modeling_data(df_with_lags)
        training_path = os.path.join(processed_dir, 'data_training.parquet')
        save_processed_data(df_final, training_path)
        logger.info(f"data_training.parquet: {df_final.shape[0]} filas")
        
        # RESUMEN FINAL
        logger.info("\nRESUMEN FINAL DE ARCHIVOS GENERADOS:")
        logger.info(f"   data_cleaned.parquet    - {df_clean.shape[0]:>6} filas, {df_clean.shape[1]:>2} columnas (Limpieza básica)")
        logger.info(f"   data_processed.parquet  - {df_with_agg.shape[0]:>6} filas, {df_with_agg.shape[1]:>2} columnas (Con agregaciones)")
        logger.info(f"   data_training.parquet   - {df_final.shape[0]:>6} filas, {df_final.shape[1]:>2} columnas (Listo para modelado)")
        
        logger.info(f"\nPipeline completado exitosamente!")
        logger.info(f"Archivos guardados en: {processed_dir}")
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        raise

if __name__ == "__main__":
    generate_raw_process()