"""Limpieza de datos - Día/Producto/Transacción"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza específica manteniendo granularidad detallada"""
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    logger.info("Iniciando limpieza - Día/Producto/Transacción")
    
    # 1. Eliminar filas sin información crítica para granularidad Grupo 1
    critical_cols = ['transaction_id', 'product_sku', 'parsed_date']
    df_clean = df_clean.dropna(subset=critical_cols)
    logger.info(f"Filas después de eliminar nulos críticos: {len(df_clean)}/{initial_rows}")
    
    # 2. Convertir fecha
    df_clean['parsed_date'] = pd.to_datetime(df_clean['parsed_date'])
    
    # 3. Definir variable objetivo: unidades vendidas
    df_clean['units_sold'] = df_clean['product_quantity'].clip(lower=0)
    df_clean['is_completed_transaction'] = (df_clean['units_sold'] > 0)
    
    # 4. Limpiar valores numéricos
    numeric_cols = ['product_quantity', 'product_price_usd', 'product_revenue_usd']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            if col == 'product_quantity':
                df_clean[col] = df_clean[col].clip(lower=0)
            else:
                df_clean[col] = df_clean[col].fillna(0)
    
    # 5. Validar SKUs
    df_clean = df_clean[df_clean['product_sku'].str.len() > 0]
    
    # 6. Eliminar duplicados en granularidad Grupo 1
    dup_count = df_clean.duplicated(subset=['transaction_id', 'product_sku', 'parsed_date']).sum()
    if dup_count > 0:
        logger.warning(f"Eliminando {dup_count} duplicados")
        df_clean = df_clean.drop_duplicates(
            subset=['transaction_id', 'product_sku', 'parsed_date'], 
            keep='first'
        )
    
    logger.info(f"Limpieza completada: {len(df_clean)} filas")
    logger.info(f"Transacciones completadas: {df_clean['is_completed_transaction'].sum()}")
    logger.info(f"Productos únicos: {df_clean['product_sku'].nunique()}")
    
    return df_clean