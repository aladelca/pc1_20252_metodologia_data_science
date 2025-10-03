"""Funciones de preprocesamiento de datos"""

import pandas as pd
import numpy as np
import logging
from .cleaning import clean_data
from .aggregation import create_daily_product_aggregation, create_product_global_metrics

logger = logging.getLogger(__name__)

def create_product_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crear variables LAG a nivel de PRODUCTO para predecir transacciones futuras"""
    df_lagged = df.copy()
    
    # Ordenar por producto y fecha
    df_lagged = df_lagged.sort_values(['product_sku', 'parsed_date'])
    
    product_features = []
    
    for product_sku in df_lagged['product_sku'].unique():
        product_data = df_lagged[df_lagged['product_sku'] == product_sku].copy()
        
        # 1. LAGS TEMPORALES
        for lag in [1, 3, 7, 14]:
            product_data[f'product_units_lag_{lag}'] = product_data['units_sold'].shift(lag)
        
        # 2. PROMEDIOS MÓVILES
        product_data['product_units_rolling_mean_7'] = (
            product_data['units_sold'].shift(1).rolling(window=7, min_periods=1).mean()
        )
        product_data['product_units_rolling_mean_30'] = (
            product_data['units_sold'].shift(1).rolling(window=30, min_periods=1).mean()
        )
        
        # 3. TENDENCIAS HISTÓRICAS
        product_data['product_units_trend_7'] = (
            product_data['units_sold'].shift(1).rolling(window=7).mean() -
            product_data['units_sold'].shift(1).rolling(window=7).mean().shift(7)
        )
        
        # 4. FRECUENCIA DE TRANSACCIONES
        if 'is_completed_transaction' in product_data.columns:
            product_data['product_transaction_freq_7'] = (
                product_data['is_completed_transaction'].shift(1).rolling(window=7).sum()
            )
        else:
            product_data['product_transaction_freq_7'] = np.nan
        
        # 5. DÍAS DESDE ÚLTIMA TRANSACCIÓN
        product_data['days_since_last_transaction'] = (
            product_data['parsed_date'] - product_data['parsed_date'].shift(1)
        ).dt.days
        
        product_features.append(product_data)
    
    df_with_lags = pd.concat(product_features, ignore_index=True)
    
    logger.info(f"Features temporales creados: {[col for col in df_with_lags.columns if 'lag' in col or 'rolling' in col or 'trend' in col]}")
    
    return df_with_lags


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crear características temporales generales"""
    df_featured = df.copy()
    
    # Variables de fecha
    df_featured['year'] = df_featured['parsed_date'].dt.year
    df_featured['month'] = df_featured['parsed_date'].dt.month
    df_featured['day'] = df_featured['parsed_date'].dt.day
    df_featured['dayofweek'] = df_featured['parsed_date'].dt.dayofweek
    df_featured['weekofyear'] = df_featured['parsed_date'].dt.isocalendar().week
    df_featured['is_weekend'] = (df_featured['dayofweek'] >= 5).astype(int)
    df_featured['is_month_start'] = df_featured['parsed_date'].dt.is_month_start.astype(int)
    df_featured['is_month_end'] = df_featured['parsed_date'].dt.is_month_end.astype(int)
    
    # Features cíclicos
    df_featured['month_sin'] = np.sin(2 * np.pi * df_featured['month'] / 12)
    df_featured['month_cos'] = np.cos(2 * np.pi * df_featured['month'] / 12)
    df_featured['dayofweek_sin'] = np.sin(2 * np.pi * df_featured['dayofweek'] / 7)
    df_featured['dayofweek_cos'] = np.cos(2 * np.pi * df_featured['dayofweek'] / 7)
    
    logger.info("Features temporales de calendario creados")
    
    return df_featured


def prepare_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preparar datos finales para modelado"""
    df_model = df.copy()
    
    # 1. Features temporales
    df_model = create_time_features(df_model)
    
    # 2. Filtrar transacciones
    if 'is_completed_transaction' in df_model.columns:
        df_model = df_model[df_model['is_completed_transaction']]
    else:
        df_model = df_model[df_model['units_sold'] > 0]
        
    # 3. Selección de features
    feature_columns = [
        'units_sold',
        'year', 'month', 'day', 'dayofweek', 'is_weekend',
        'is_month_start', 'is_month_end',
        'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
        'product_units_lag_1', 'product_units_lag_3', 'product_units_lag_7', 'product_units_lag_14',
        'product_units_rolling_mean_7', 'product_units_rolling_mean_30',
        'product_units_trend_7', 'product_transaction_freq_7', 'days_since_last_transaction'
    ]
    
    available_features = [col for col in feature_columns if col in df_model.columns]
    
    # IDs para tracking
    id_columns = ['product_sku', 'parsed_date', 'product_name']
    
    final_columns = id_columns + available_features
    df_final = df_model[final_columns].copy()
    
    # Drop de nulos
    initial_rows = len(df_final)
    df_final = df_final.dropna(subset=available_features)
    
    logger.info(f"Dataset final: {len(df_final)} filas ({initial_rows - len(df_final)} eliminadas por nulos)")
    logger.info(f"Features finales: {len(available_features)}")
    
    return df_final
