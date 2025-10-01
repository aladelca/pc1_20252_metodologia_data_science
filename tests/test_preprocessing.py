
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Agregar el directorio raíz del proyecto al sys.path para importar módulos
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.preprocess.cleaning import clean_data
from src.preprocess.aggregation import aggregate_data_by_product

@pytest.fixture
def raw_data_sample() -> pd.DataFrame:
    """
    Provides a sample raw DataFrame for testing, mimicking the real data structure.
    """
    data = {
        'transaction_date': ['20170801', '20170801', '20170802', '20170802'],
        'transaction_id': ['123', '123', '124', '125'],
        'product_sku': ['SKU1', 'SKU2', 'SKU1', 'SKU1'],
        'product_quantity': [1, 2, 1, 3],
        'product_price_usd': [10.0, 5.0, 10.5, 10.5],
        'product_revenue_usd': [10.0, 10.0, 10.5, 31.5],
        'transaction_revenue_usd': [20.0, 20.0, 10.5, 31.5],
        'transaction_tax_usd': [2.0, 2.0, 1.0, 3.0],
        'transaction_shipping_usd': [5.0, 5.0, 5.0, 5.0],
        'total_visits': [1, 1, 1, 1],
        'total_hits': [5, 5, 3, 4],
        'total_pageviews': [3, 3, 2, 2],
        'time_on_site_seconds': [100, 100, np.nan, 60],
        'device_category': ['desktop', '(not set)', 'mobile', 'desktop'],
        'channel_grouping': ['Organic Search', 'Direct', np.nan, 'Referral']
    }
    return pd.DataFrame(data)

def test_clean_data(raw_data_sample: pd.DataFrame):
    """
    Tests the clean_data function to ensure it correctly cleans the raw data.
    """
    cleaned_df = clean_data(raw_data_sample.copy())

    # Test de conversion de datos
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['transaction_date'])

    # Test reemplazo para '(not set)'
    assert '(not set)' not in cleaned_df['device_category'].unique()

    # Test para objetos NaN
    assert cleaned_df['channel_grouping'].isnull().sum() == 0
    assert 'unknown' in cleaned_df['channel_grouping'].unique()

    # Test comprobar columnas numéricas completas
    assert cleaned_df['time_on_site_seconds'].isnull().sum() == 0

def test_aggregate_data_by_product(raw_data_sample: pd.DataFrame):
    """
    Tests de la función aggregate_data_by_product para el grupo 2.
    """
    cleaned_df = clean_data(raw_data_sample.copy())
    aggregated_df = aggregate_data_by_product(cleaned_df)

    # Resultados esperados luego de agregarlos
    # Day 1, SKU1: 1 transacción
    # Day 1, SKU2: 1 transacción
    # Day 2, SKU1: 2 transacciones
    assert len(aggregated_df) == 3

    # Check aggregation para Day 2, SKU1
    day2_sku1_agg = aggregated_df[
        (aggregated_df['transaction_date'] == '2017-08-02') & 
        (aggregated_df['product_sku'] == 'SKU1')
    ].iloc[0]

    assert day2_sku1_agg['total_product_quantity'] == 4  # 1 + 3
    assert pytest.approx(day2_sku1_agg['total_product_revenue_usd']) == 42.0  # 10.5 + 31.5
    assert pytest.approx(day2_sku1_agg['avg_product_price_usd']) == 10.5 # (10.5, 10.5)
    assert day2_sku1_agg['total_transactions'] == 2 # 124, 125
