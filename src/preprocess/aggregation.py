# Agregación día/producto - Grupo 2

import pandas as pd
import numpy as np


def identify_product_columns(df):
    """Busca columnas con info de productos."""
    product_mapping = {}

    # Buscar columnas típicas
    # TODO: quizás agregar más variantes después
    product_id_candidates = ['productSKU', 'product_sku', 'sku', 'productName',
                             'product_name', 'product_id', 'productId']
    product_name_candidates = ['productName', 'product_name', 'name']
    product_category_candidates = ['productCategory', 'product_category', 'category',
                                   'categoryName', 'category_name']
    product_brand_candidates = ['productBrand', 'product_brand', 'brand']

    # Buscar columnas de ID de producto
    for candidate in product_id_candidates:
        if candidate in df.columns:
            product_mapping['product_id'] = candidate
            break

    # Buscar columnas de nombre de producto
    for candidate in product_name_candidates:
        if candidate in df.columns:
            product_mapping['product_name'] = candidate
            break

    # Buscar columnas de categoría
    for candidate in product_category_candidates:
        if candidate in df.columns:
            product_mapping['product_category'] = candidate
            break

    # Buscar columnas de marca
    for candidate in product_brand_candidates:
        if candidate in df.columns:
            product_mapping['product_brand'] = candidate
            break

    # Si no hay ID de producto, usar el nombre como ID (no es ideal pero funciona)
    if 'product_id' not in product_mapping and 'product_name' in product_mapping:
        product_mapping['product_id'] = product_mapping['product_name']

    print(f"Encontré estas columnas de producto: {product_mapping}")
    return product_mapping


def create_product_identifier(df: pd.DataFrame,
                              product_columns):
    """
    Crea un identificador único de producto combinando múltiples campos.

    Args:
        df: DataFrame con los datos
        product_columns: Mapeo de columnas de producto

    Returns:
        DataFrame con columna product_id creada/actualizada
    """
    df_agg = df.copy()

    # Crear identificador de producto
    if 'product_id' in product_columns:
        # Usar ID existente, limpiando valores nulos
        df_agg['product_id'] = df_agg[product_columns['product_id']].fillna(
            'unknown_product')
    else:
        # Crear ID combinando campos disponibles
        id_parts = []

        if 'product_name' in product_columns:
            id_parts.append(
                df_agg[product_columns['product_name']].fillna('unknown'))

        if 'product_category' in product_columns:
            id_parts.append(
                df_agg[product_columns['product_category']].fillna('unknown'))

        if id_parts:
            df_agg['product_id'] = id_parts[0].astype(str)
            for part in id_parts[1:]:
                df_agg['product_id'] += '_' + part.astype(str)
        else:
            # Si no hay información de producto, usar un ID genérico
            df_agg['product_id'] = 'generic_product'

    # Limpiar el product_id
    df_agg['product_id'] = df_agg['product_id'].astype(
        str).str.strip().str.lower()
    df_agg['product_id'] = df_agg['product_id'].replace(
        ['', 'nan', 'none', 'null'], 'unknown_product')

    return df_agg


def calculate_daily_product_metrics(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """Calcula las métricas agregadas por día y producto.

    Parameters:
    df: DataFrame con los datos ya limpios
    date_column: Nombre de la columna de fecha

    Returns:
    DataFrame con las métricas agregadas
    """
    # Buscar todas las columnas numéricas que podemos agregar
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Métricas específicas a agregar
    metrics_to_aggregate = {
        'revenue': ['sum', 'mean', 'count'],
        'quantity': ['sum', 'mean'],
        'visits': ['sum'],
        'hits': ['sum'],
        'pageviews': ['sum'],
        'transactions': ['sum'],
        'transactionRevenue': ['sum'],
        'totalTransactionRevenue': ['sum']
    }

    # Construir diccionario de agregación
    agg_dict = {}

    for metric, agg_funcs in metrics_to_aggregate.items():
        # Buscar columnas que contengan el nombre de la métrica
        matching_columns = [
            col for col in numeric_columns if metric.lower() in col.lower()]

        for col in matching_columns:
            for func in agg_funcs:
                if func == 'count':
                    agg_dict[f'{col}_{func}'] = pd.NamedAgg(
                        column=col, aggfunc='count')
                elif func == 'sum':
                    agg_dict[f'{col}_{func}'] = pd.NamedAgg(
                        column=col, aggfunc='sum')
                elif func == 'mean':
                    agg_dict[f'{col}_{func}'] = pd.NamedAgg(
                        column=col, aggfunc='mean')

    # Fallback: si no encontramos nada específico, agregamos todo lo numérico
    # (puede ser mucho pero mejor tener de más que de menos)
    if not agg_dict:
        for col in numeric_columns:
            agg_dict[f'{col}_sum'] = pd.NamedAgg(column=col, aggfunc='sum')
            agg_dict[f'{col}_mean'] = pd.NamedAgg(column=col, aggfunc='mean')
            agg_dict[f'{col}_count'] = pd.NamedAgg(column=col, aggfunc='count')
            # print(f"Agregando {col}")  # debug line - commented out

    # Agregar por fecha y producto
    try:
        df_aggregated = df.groupby([date_column, 'product_id']).agg(
            **agg_dict).reset_index()
    except Exception as e:
        print(f"Error en agregación: {e}")
        # Fallback: agregación simple
        df_aggregated = df.groupby(
            [date_column, 'product_id']).size().reset_index(name='record_count')

    return df_aggregated


def add_time_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Añade características temporales al DataFrame.

    Args:
        df: DataFrame con datos agregados
        date_column: Nombre de la columna de fecha

    Returns:
        DataFrame con características temporales añadidas
    """
    df_features = df.copy()

    # Asegurar que la fecha es datetime
    df_features[date_column] = pd.to_datetime(df_features[date_column])

    # Características temporales
    df_features['year'] = df_features[date_column].dt.year
    df_features['month'] = df_features[date_column].dt.month
    df_features['day'] = df_features[date_column].dt.day
    df_features['day_of_week'] = df_features[date_column].dt.dayofweek
    df_features['week_of_year'] = df_features[date_column].dt.isocalendar().week
    df_features['quarter'] = df_features[date_column].dt.quarter

    # Indicadores de día de semana
    df_features['is_weekend'] = df_features['day_of_week'].isin(
        [5, 6]).astype(int)
    df_features['is_monday'] = (df_features['day_of_week'] == 0).astype(int)
    df_features['is_friday'] = (df_features['day_of_week'] == 4).astype(int)

    return df_features


def create_lagged_features(df: pd.DataFrame,
                           target_columns,
                           lags=[1, 7, 14, 30],
                           date_column='date'):
    """
    Crea características de lag para series de tiempo.

    Args:
        df: DataFrame con datos agregados
        target_columns: Columnas para las que crear lags
        lags: Lista de lags a crear
        date_column: Nombre de la columna de fecha

    Returns:
        DataFrame con características de lag
    """
    df_lagged = df.copy()

    # Asegurar que está ordenado por fecha y producto
    df_lagged = df_lagged.sort_values(['product_id', date_column])

    for target_col in target_columns:
        if target_col in df_lagged.columns:
            for lag in lags:
                # Crear lag por producto
                df_lagged[f'{target_col}_lag_{lag}'] = (
                    df_lagged.groupby('product_id')[target_col]
                    .shift(lag)
                )

                # Crear rolling mean
                if lag >= 7:
                    df_lagged[f'{target_col}_rolling_mean_{lag}'] = (
                        df_lagged.groupby('product_id')[target_col]
                        .rolling(window=lag, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )

    return df_lagged


def filter_products_by_activity(df: pd.DataFrame,
                                min_days: int = 30,
                                min_transactions: int = 10,
                                date_column: str = 'date') -> pd.DataFrame:
    """Filtra productos con poca actividad.

    Solo mantenemos productos que tengan suficientes datos para entrenar.

    Args:
        df: DataFrame agregado
        min_days: Mínimo número de días 
        min_transactions: Mínimo número de transacciones
        date_column: Nombre de la columna de fecha

    Returns:
        DataFrame filtrado
    """
    # Calculamos stats básicas por producto
    stats = df.groupby('product_id').agg({
        date_column: 'nunique',  # días únicos
        'product_id': 'count'  # registros totales
    }).rename(columns={date_column: 'unique_days', 'product_id': 'total_records'})

    # Filtrar solo los productos que tienen suficiente data
    good_products = stats[
        (stats['unique_days'] >= min_days) &
        (stats['total_records'] >= min_transactions)
    ].index

    df_filtered = df[df['product_id'].isin(good_products)].copy()

    print(
        f"Productos con suficiente actividad: {len(good_products)} de {len(stats)}")

    return df_filtered


def aggregate_daily_product_data(df: pd.DataFrame,
                                 date_column: str = 'date',
                                 min_days: int = 30,
                                 add_features: bool = True) -> pd.DataFrame:
    """
    Función principal para agregar datos por día y producto (Grupo 2).

    Args:
        df: DataFrame con datos limpios
        date_column: Nombre de la columna de fecha
        min_days: Mínimo días de actividad para incluir producto
        add_features: Si añadir características temporales y lags

    Returns:
        DataFrame agregado por día y producto
    """
    print("Iniciando agregación día/producto para Grupo 2")

    # 1. Identificar columnas de producto
    product_columns = identify_product_columns(df)

    if not product_columns:
        print("No se encontraron columnas de producto. Creando producto genérico.")
        df['product_id'] = 'generic_product'
    else:
        # 2. Crear identificador de producto
        df = create_product_identifier(df, product_columns)

    # 3. Calcular métricas agregadas
    df_aggregated = calculate_daily_product_metrics(df, date_column)

    # 4. Filtrar productos con suficiente actividad
    df_aggregated = filter_products_by_activity(
        df_aggregated, min_days=min_days, date_column=date_column)

    if add_features:
        # 5. Añadir características temporales
        df_aggregated = add_time_features(df_aggregated, date_column)

        # 6. Crear características de lag
        numeric_columns = [col for col in df_aggregated.columns
                           if col not in [date_column, 'product_id'] and
                           df_aggregated[col].dtype in ['int64', 'float64']]

        if numeric_columns:
            # Usar las primeras métricas para lags
            # Limitar para evitar demasiadas características
            target_columns = numeric_columns[:3]
            df_aggregated = create_lagged_features(
                df_aggregated, target_columns, date_column=date_column)

    print(f"Agregación completada. Shape final: {df_aggregated.shape}")
    print(f"Productos únicos: {df_aggregated['product_id'].nunique()}")
    print(
        f"Rango de fechas: {df_aggregated[date_column].min()} - {df_aggregated[date_column].max()}")

    return df_aggregated
