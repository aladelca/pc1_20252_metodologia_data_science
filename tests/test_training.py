# GROUP04
# tests/test_training.py - Tests unitarios para training.py
import os
import sys
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import pytest
import joblib

# --- asegurar que src/ esté en el path ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from pipeline.training import (
    evaluate,
    make_features,
    train_xgboost,
)


# ---------------------
# Fixtures
# ---------------------
@pytest.fixture
def tmp_training_paths(tmp_path, request):
    """Rutas temporales para pruebas de training."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    uid = uuid.uuid4().hex[:8]
    testname = request.node.name.replace(os.sep, "_")
    
    model_file = models_dir / f"{testname}_model_{uid}.pkl"
    data_file = data_dir / f"{testname}_data_{uid}.parquet"
    
    return {
        "models_dir": models_dir,
        "data_dir": data_dir,
        "model_file": model_file,
        "data_file": data_file,
    }


@pytest.fixture
def synthetic_brand_series():
    """Serie temporal sintética para una marca con tendencia y estacionalidad."""
    dates = pd.date_range("2016-01-01", periods=200, freq="D", tz="UTC")
    
    # Generar serie con tendencia + estacionalidad semanal + ruido
    trend = np.linspace(100, 150, 200)
    seasonal = 20 * np.sin(np.arange(200) * 2 * np.pi / 7)
    noise = np.random.RandomState(42).normal(0, 5, 200)
    
    values = trend + seasonal + noise
    values = np.maximum(values, 0)  # No negativos
    
    series = pd.Series(values, index=dates, name="events")
    return series


@pytest.fixture
def synthetic_brand_daily_data():
    """DataFrame con datos día/marca para testing."""
    dates = pd.date_range("2016-01-01", periods=200, freq="D")
    
    rows = []
    for d in dates:
        # Generar eventos para (not set)
        n_events = int(100 + 20 * np.sin(d.dayofyear * 2 * np.pi / 7) + np.random.randn() * 5)
        n_events = max(0, n_events)
        rows.append({
            "transaction_date": d.strftime("%Y%m%d"),
            "product_brand": "(not set)",
            "events": n_events,
        })
    
    df = pd.DataFrame(rows)
    # Convertir transaction_date a datetime
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y%m%d", utc=True)
    return df


# ---------------------
# Tests unitarios
# ---------------------
def test_evaluate():
    """Test función de evaluación de métricas."""
    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([95, 160, 195, 240, 310])
    
    metrics = evaluate(y_true, y_pred)
    
    # Verificar que devuelve las métricas esperadas
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE%" in metrics
    
    # Verificar que son valores numéricos positivos
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] > 0
    assert metrics["MAPE%"] > 0
    
    # MAE debe ser menor que RMSE (por definición)
    assert metrics["MAE"] <= metrics["RMSE"]


def test_evaluate_perfect_prediction():
    """Test evaluación con predicción perfecta."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])
    
    metrics = evaluate(y_true, y_pred)
    
    # Todas las métricas deben ser 0
    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["MAPE%"] == 0


def test_make_features(synthetic_brand_series):
    """Test creación de features."""
    result = make_features(synthetic_brand_series)
    
    # Verificar que se crea DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Verificar columnas esperadas
    assert "target" in result.columns
    assert "lag_1" in result.columns
    assert "lag_7" in result.columns
    assert "ma_7" in result.columns
    assert "dow" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns
    
    # Verificar que hay filas con NaN al inicio (por lags)
    assert result.isna().sum().sum() > 0
    
    # Verificar dimensiones
    assert len(result) == len(synthetic_brand_series)


def test_make_features_lag_values(synthetic_brand_series):
    """Test que los lags tienen valores correctos."""
    result = make_features(synthetic_brand_series)
    
    # lag_1 debe ser el valor anterior
    # (índice 10 de lag_1 debe ser igual a índice 9 de target)
    idx = 10
    assert result["lag_1"].iloc[idx] == result["target"].iloc[idx - 1]
    
    # lag_7 debe ser el valor de 7 días atrás
    idx = 30
    assert result["lag_7"].iloc[idx] == result["target"].iloc[idx - 7]


def test_make_features_moving_average():
    """Test cálculo de media móvil."""
    # Serie simple para verificar cálculo
    dates = pd.date_range("2016-01-01", periods=20, freq="D", tz="UTC")
    series = pd.Series(range(20), index=dates, name="events")
    
    result = make_features(series, lags=(1,), mas=(3,))
    
    # ma_3 en índice 3 debe ser el promedio de índices 0, 1, 2
    # (porque se usa shift(1), entonces usa los 3 valores anteriores al actual)
    idx = 4
    expected_ma = (series.iloc[1] + series.iloc[2] + series.iloc[3]) / 3
    assert abs(result["ma_3"].iloc[idx] - expected_ma) < 1e-6


def test_make_features_calendar_variables(synthetic_brand_series):
    """Test variables calendario."""
    result = make_features(synthetic_brand_series)
    
    # dow debe estar entre 0 y 6
    assert result["dow"].min() >= 0
    assert result["dow"].max() <= 6
    
    # month debe estar entre 1 y 12
    assert result["month"].min() >= 1
    assert result["month"].max() <= 12
    
    # is_weekend debe ser 0 o 1
    assert set(result["is_weekend"].unique()) <= {0, 1}


def test_make_features_custom_params():
    """Test make_features con parámetros personalizados."""
    dates = pd.date_range("2016-01-01", periods=50, freq="D", tz="UTC")
    series = pd.Series(range(50), index=dates, name="events")
    
    result = make_features(series, lags=(1, 2), mas=(3,))
    
    # Debe tener solo los lags especificados
    assert "lag_1" in result.columns
    assert "lag_2" in result.columns
    assert "lag_3" not in result.columns
    
    # Debe tener solo las MAs especificadas
    assert "ma_3" in result.columns
    assert "ma_7" not in result.columns


def test_train_xgboost(tmp_training_paths, synthetic_brand_series):
    """Test entrenamiento de XGBoost."""
    test_days = 30
    model_path = tmp_training_paths["model_file"]
    
    # Entrenar
    metrics = train_xgboost(
        series=synthetic_brand_series,
        test_days=test_days,
        model_pkl_path=model_path
    )
    
    # Verificar que devuelve métricas
    assert isinstance(metrics, dict)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE%" in metrics
    
    # Verificar que las métricas son razonables
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] >= metrics["MAE"]
    
    # Verificar que se guardó el modelo
    assert model_path.exists()
    
    # Verificar que se puede cargar el modelo
    loaded_model = joblib.load(model_path)
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")


def test_train_xgboost_model_can_predict(tmp_training_paths, synthetic_brand_series):
    """Test que el modelo entrenado puede hacer predicciones."""
    model_path = tmp_training_paths["model_file"]
    
    # Entrenar
    train_xgboost(
        series=synthetic_brand_series,
        test_days=30,
        model_pkl_path=model_path
    )
    
    # Cargar modelo
    model = joblib.load(model_path)
    
    # Crear features para predicción
    feats = make_features(synthetic_brand_series).dropna()
    X = feats.drop(columns=["target"])
    
    # Hacer predicción
    predictions = model.predict(X.iloc[-10:])
    
    # Verificar que devuelve predicciones
    assert len(predictions) == 10
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    assert all(p >= 0 for p in predictions)  # No debe predecir negativos


def test_train_xgboost_different_test_sizes(tmp_training_paths, synthetic_brand_series):
    """Test entrenamiento con diferentes tamaños de test."""
    model_path1 = tmp_training_paths["models_dir"] / "model_test10.pkl"
    model_path2 = tmp_training_paths["models_dir"] / "model_test50.pkl"
    
    metrics1 = train_xgboost(synthetic_brand_series, test_days=10, model_pkl_path=model_path1)
    metrics2 = train_xgboost(synthetic_brand_series, test_days=50, model_pkl_path=model_path2)
    
    # Ambos deben completarse sin error
    assert model_path1.exists()
    assert model_path2.exists()
    
    # Ambos deben devolver métricas
    assert "MAE" in metrics1
    assert "MAE" in metrics2


def test_train_xgboost_minimum_data():
    """Test con datos mínimos (debe manejar gracefully)."""
    # Serie muy corta
    dates = pd.date_range("2016-01-01", periods=60, freq="D", tz="UTC")
    series = pd.Series(range(60), index=dates, name="events")
    
    # Intentar entrenar con test_days pequeño
    model_path = Path("temp_model.pkl")
    
    try:
        metrics = train_xgboost(series, test_days=10, model_pkl_path=model_path)
        # Si completa, verificar que devuelve métricas
        assert isinstance(metrics, dict)
    finally:
        # Limpiar
        if model_path.exists():
            model_path.unlink()


def test_train_xgboost_metrics_quality(tmp_training_paths, synthetic_brand_series):
    """Test que las métricas están en rangos razonables."""
    model_path = tmp_training_paths["model_file"]
    
    metrics = train_xgboost(
        series=synthetic_brand_series,
        test_days=30,
        model_pkl_path=model_path
    )
    
    # Las métricas deben ser finitas (no inf, no nan)
    assert np.isfinite(metrics["MAE"])
    assert np.isfinite(metrics["RMSE"])
    assert np.isfinite(metrics["MAPE%"])
    
    # MAPE debe estar en porcentaje razonable (0-100)
    # (puede ser >100 si hay errores muy grandes, pero esperamos <200)
    assert 0 <= metrics["MAPE%"] < 200


def test_train_xgboost_reproducibility(tmp_training_paths, synthetic_brand_series):
    """Test que el entrenamiento es reproducible (por random_state)."""
    model_path1 = tmp_training_paths["models_dir"] / "model1.pkl"
    model_path2 = tmp_training_paths["models_dir"] / "model2.pkl"
    
    # Entrenar dos veces con la misma serie
    metrics1 = train_xgboost(synthetic_brand_series, test_days=30, model_pkl_path=model_path1)
    metrics2 = train_xgboost(synthetic_brand_series, test_days=30, model_pkl_path=model_path2)
    
    # Las métricas deben ser idénticas (por random_state=42)
    assert abs(metrics1["MAE"] - metrics2["MAE"]) < 1e-6
    assert abs(metrics1["RMSE"] - metrics2["RMSE"]) < 1e-6


def test_make_features_handles_short_series():
    """Test que make_features maneja series cortas."""
    dates = pd.date_range("2016-01-01", periods=10, freq="D", tz="UTC")
    series = pd.Series(range(10), index=dates, name="events")
    
    # Debe completarse sin error
    result = make_features(series)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10


def test_evaluate_handles_zeros():
    """Test que evaluate maneja correctamente valores cero."""
    y_true = np.array([0, 0, 100, 200])
    y_pred = np.array([10, 5, 95, 205])
    
    metrics = evaluate(y_true, y_pred)
    
    # Debe completarse sin dividir por cero
    assert np.isfinite(metrics["MAE"])
    assert np.isfinite(metrics["RMSE"])
    # MAPE puede ser alto pero finito
    assert np.isfinite(metrics["MAPE%"])


# ---------------------
# FIN GROUP 04 tests
# ---------------------
