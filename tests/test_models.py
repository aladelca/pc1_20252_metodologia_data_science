
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Agregar el directorio raíz del proyecto al sys.path para importar módulos
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.models.ml import create_features
from src.models.deep_learning import create_sequences

@pytest.fixture
def sample_time_series() -> pd.DataFrame:
    """
    Rango de fechas para el DataFrame de Test.
    """
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    data = {
        'value': range(30)
    }
    return pd.DataFrame(data, index=dates)


def test_create_features_shape_and_columns(sample_time_series: pd.DataFrame):
    """
    Tests de create_features crean las características correctas y eliminan filas NaNs.
    """
    featured_df = create_features(sample_time_series['value'], 'value')
    
    # Luego de eliminar los NaNs el DataFrame debe ser mas corto
    assert len(featured_df) < len(sample_time_series)
    
    # Check las columnas esperadas
    expected_cols = ['dayofweek', 'month', 'year', 'dayofyear', 'lag_1', 'lag_7', 'lag_14', 'roll_mean_7', 'roll_mean_14']
    for col in expected_cols:
        assert col in featured_df.columns

def test_create_features_short_series(sample_time_series: pd.DataFrame):
    """
    Tests de create_features manejan una serie de tiempo más corta que la ventana más grande.
    """
    short_series = sample_time_series.head(10)
    featured_df = create_features(short_series['value'], 'value')
    assert featured_df.empty  # Se espera un DataFrame vacío debido a dropna()


def test_create_sequences_shape_and_content():
    """
    Tests de forma y contenido de las secuencias creadas por create_sequences.
    """
    dataset = np.arange(100).reshape(-1, 1)
    look_back = 10
    
    X, Y = create_sequences(dataset, look_back)
    
    # Shape check
    expected_X_shape = (100 - look_back - 1, look_back)
    expected_Y_shape = (100 - look_back - 1,)
    assert X.shape == expected_X_shape
    assert Y.shape == expected_Y_shape
    
    # Content check
    assert np.array_equal(X[0], np.arange(0, 10))
    assert Y[0] == 10
    assert np.array_equal(X[1], np.arange(1, 11))
    assert Y[1] == 11

def test_create_sequences_edge_cases():
    """
    Tests create_sequences con casos extremos, para un conjunto de datos corto.
    """
    # Dataset más corto que look_back
    dataset = np.arange(5).reshape(-1, 1)
    look_back = 10
    X, Y = create_sequences(dataset, look_back)
    assert X.shape[0] == 0
    assert Y.shape[0] == 0
    
    # look_back of 1
    dataset = np.arange(10).reshape(-1, 1)
    look_back = 1
    X, Y = create_sequences(dataset, look_back)
    assert X.shape == (8, 1)
    assert Y.shape == (8,)
