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
from preprocess.aggregation import create_product_lag_features, create_time_features, prepare_modeling_data
from preprocess.preprocessing import load_data, save_processed_data

def generate_raw_process():
    """Ejecutar pipeline completo usando TUS funciones reales"""
    
    logger.info("INICIANDO PIPELINE - DÍA/PRODUCTO/TRANSACCIÓN")
    
    # 1. Cargar datos raw
    logger.info("Cargando datos raw...")
    try:
        df_raw = load_data('data/raw/data_sample.parquet')
        logger.info(f"Datos raw cargados: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")
    except Exception as e:
        logger.error(f"Error cargando datos: {e}")
        return
    
    # 2. Limpieza
    logger.info("Aplicando limpieza...")
    try:
        df_clean = clean_data(df_raw)
        logger.info(f"✅ Limpieza completada: {df_clean.shape[0]} filas")
        logger.info(f"   - Transacciones completadas: {df_clean['is_completed_transaction'].sum()}")
        logger.info(f"   - Productos únicos: {df_clean['product_sku'].nunique()}")
        
        # Guardar datos limpios
        save_processed_data(df_clean, 'data/processed/data_cleaned.parquet')
        logger.info("Datos limpios guardados")
        
    except Exception as e:
        logger.error(f"Error en limpieza: {e}")
        return
    
    # 3. Crear features de lag por producto (TU función)
    logger.info("Creando features lag por producto...")
    try:
        df_lagged = create_product_lag_features(df_clean)
        logger.info(f"Features lag creados: {df_lagged.shape[0]} filas")
        
        # Contar features lag creados
        lag_features = [col for col in df_lagged.columns if 'lag' in col or 'rolling' in col or 'trend' in col]
        logger.info(f"   - Features lag creados: {len(lag_features)}")
        
    except Exception as e:
        logger.error(f"Error creando features lag: {e}")
        return
    
    # 4. Crear características temporales
    logger.info("Creando características temporales...")
    try:
        df_temporal = create_time_features(df_lagged)
        logger.info(f"Características temporales creadas")
        
        # Guardar datos procesados completos
        save_processed_data(df_temporal, 'data/processed/data_processed.parquet')
        logger.info("Datos procesados guardados")
        
    except Exception as e:
        logger.error(f"Error creando características temporales: {e}")
        return
    
    # 5. Preparar datos finales para modelado (TU función)
    logger.info("Preparando datos para modelado...")
    try:
        df_model = prepare_modeling_data(df_temporal)  # ← TU función real
        logger.info(f"Datos para modelado preparados: {df_model.shape[0]} filas")
        logger.info(f"   - Columnas finales: {df_model.shape[1]}")
        
        # Guardar datos para entrenamiento
        save_processed_data(df_model, 'data/processed/data_training.parquet')
        logger.info("Datos para entrenamiento guardados")
        
    except Exception as e:
        logger.error(f"Error preparando datos para modelado: {e}")
        return
    
    # 6. Resumen final
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    #logger.info("=" * 60)
    #logger.info("RESUMEN FINAL:")
    #logger.info(f"   - Datos raw: {df_raw.shape[0]} filas")
    #logger.info(f"   - Datos limpios: {df_clean.shape[0]} filas")
    #logger.info(f"   - Datos con features: {df_temporal.shape[0]} filas")
    #logger.info(f"   - Datos para modelado: {df_model.shape[0]} filas")
    #logger.info(f"   - Transacciones con unidades > 0: {df_clean['is_completed_transaction'].sum()}")
    #logger.info(f"   - Productos únicos: {df_model['product_sku'].nunique()}")
    #logger.info("=" * 60)

if __name__ == "__main__":
    generate_raw_process()