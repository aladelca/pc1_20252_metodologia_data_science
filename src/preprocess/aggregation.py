"""AGREGACIÓN - Día/Producto/Transacción
SOLO funciones de agrupamiento/aggregation para crear métricas históricas"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_daily_product_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    AGREGACIÓN: Métricas diarias por producto para construir features históricas
    Nivel: Día + Producto (para luego hacer merge con transacciones individuales)
    """
    logger.info("Calculando agregaciones diarias por producto...")
    
    # Agrupar por día y producto para tener histórico de comportamiento
    daily_agg = df.groupby(['product_sku', 'parsed_date']).agg({
        'units_sold': ['sum', 'mean', 'std', 'count'],
        'transaction_id': 'nunique',
        'visitor_id': 'nunique',
        'product_price_usd': 'mean'
    }).reset_index()
    
    # Limpiar nombres de columnas
    daily_agg.columns = [
        'product_sku', 'parsed_date',
        'daily_units_total', 'daily_avg_units', 'daily_std_units', 'daily_records_count',
        'daily_unique_transactions', 'daily_unique_visitors', 'daily_avg_price'
    ]
    
    # Llenar NaNs
    daily_agg['daily_std_units'] = daily_agg['daily_std_units'].fillna(0)
    
    logger.info(f"Agregación diaria-producto: {len(daily_agg)} registros")
    return daily_agg

def create_product_global_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    AGREGACIÓN: Métricas globales por producto (todo el histórico)
    Nivel: Producto
    """
    logger.info("Calculando métricas globales por producto...")
    
    product_global = df.groupby('product_sku').agg({
        'units_sold': ['sum', 'mean', 'std', 'count'],
        'product_price_usd': ['mean', 'std', 'min', 'max'],
        'parsed_date': ['min', 'max', 'nunique'],
        'transaction_id': 'nunique',
        'visitor_id': 'nunique'
    }).reset_index()
    
    # Limpiar nombres
    product_global.columns = [
        'product_sku', 'global_units_total', 'global_avg_units', 'global_std_units', 'global_records_count',
        'global_avg_price', 'global_std_price', 'global_min_price', 'global_max_price',
        'first_transaction_date', 'last_transaction_date', 'active_days_count',
        'global_unique_transactions', 'global_unique_visitors'
    ]
    
    # Calcular métricas derivadas
    product_global['global_sales_consistency'] = (
        product_global['global_records_count'] / product_global['active_days_count']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    product_global['product_lifetime_days'] = (
        product_global['last_transaction_date'] - product_global['first_transaction_date']
    ).dt.days + 1
    
    logger.info(f"Métricas globales para {len(product_global)} productos")
    return product_global

def create_weekly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    AGREGACIÓN: Tendencias semanales por producto
    Nivel: Semana + Producto
    """
    logger.info("Calculando tendencias semanales por producto...")
    
    df_weekly = df.copy()
    df_weekly['year_week'] = df_weekly['parsed_date'].dt.strftime('%Y-%U')
    
    weekly_agg = df_weekly.groupby(['product_sku', 'year_week']).agg({
        'units_sold': ['sum', 'mean'],
        'transaction_id': 'nunique',
        'visitor_id': 'nunique',
        'product_price_usd': 'mean'
    }).reset_index()
    
    weekly_agg.columns = [
        'product_sku', 'year_week',
        'weekly_units_total', 'weekly_avg_units',
        'weekly_unique_transactions', 'weekly_unique_visitors',
        'weekly_avg_price'
    ]
    
    logger.info(f"Tendencias semanales: {len(weekly_agg)} registros")
    return weekly_agg