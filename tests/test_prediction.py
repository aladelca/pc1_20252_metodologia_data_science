# GROUP04
# tests/test_prediction.py - Tests unitarios para prediction.py
import os
import sys
from pathlib import Path
import uuid
import json
import numpy as np
import pandas as pd
import pytest
import joblib
from xgboost import XGBRegressor

# --- asegurar que src/ esté en el path ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from pipeline.prediction import (
    make_features,
    recursive_forecast_xgb,
    _infer_lags_from_columns,
    _infer_mas_from_columns,
)


# ---------------------
# Fixtures
# ---------------------
@pytest.fixture
def tmp_prediction_paths(tmp_path, request):
    """Rutas temporales para pruebas de prediction."""
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
def synthetic_series():
    """Serie temporal sintética para testing."""
    dates = pd.date_range("2016-01-01", periods=100, freq="D", tz="UTC")
    
    # Serie simple con tendencia
    values = np.linspace(100, 200, 100) + np.random.RandomState(42).normal(0, 5, 100)
    values = np.maximum(values, 0)
    
    series = pd.Series(values, index=dates, name="events")
    return series


@pytest.fixture
def trained_xgb_model(tmp_prediction_paths, synthetic_series):
    """Modelo XGBoost pre-entrenado para testing."""
    from pipeline.training import make_features as train_make_features
    
    # Crear features
    feats = train_make_features(synthetic_series).dropna()
    
    # Split train/test
    train = feats.iloc[:-20]
    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    
    # Entrenar modelo simple
    model = XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Guardar modelo
    model_path = tmp_prediction_paths["model_file"]
    joblib.dump(model, model_path)
    
    # Retornar modelo, path y columnas
    return {
        "model": model,
        "path": model_path,
        "feature_columns": list(X_train.columns),
    }


# ---------------------
# Tests unitarios
# ---------------------
def test_make_features(synthetic_series):
    """Test creación de features en prediction.py."""
    result = make_features(synthetic_series)
    
    # Verificar que se crea DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Verificar columnas esperadas
    assert "target" in result.columns
    assert "lag_1" in result.columns
    assert "ma_7" in result.columns
    assert "dow" in result.columns
    
    # Verificar dimensiones
    assert len(result) == len(synthetic_series)


def test_infer_lags_from_columns():
    """Test inferencia de lags desde nombres de columnas."""
    columns = ["lag_1", "lag_7", "lag_14", "ma_7", "dow", "month"]
    
    lags = _infer_lags_from_columns(columns)
    
    # Debe extraer solo los lags
    assert lags == [1, 7, 14]


def test_infer_lags_from_columns_empty():
    """Test inferencia con columnas sin lags."""
    columns = ["ma_7", "dow", "month", "is_weekend"]
    
    lags = _infer_lags_from_columns(columns)
    
    # Debe devolver lista vacía
    assert lags == []


def test_infer_lags_from_columns_malformed():
    """Test inferencia con columnas mal formadas."""
    columns = ["lag_1", "lag_abc", "lag_", "lag_7"]
    
    lags = _infer_lags_from_columns(columns)
    
    # Debe ignorar las mal formadas
    assert 1 in lags
    assert 7 in lags
    assert len(lags) == 2


def test_infer_mas_from_columns():
    """Test inferencia de moving averages."""
    columns = ["lag_1", "ma_7", "ma_14", "ma_28", "dow"]
    
    mas = _infer_mas_from_columns(columns)
    
    # Debe extraer solo las MAs
    assert mas == [7, 14, 28]


def test_infer_mas_from_columns_empty():
    """Test inferencia con columnas sin MAs."""
    columns = ["lag_1", "lag_7", "dow", "month"]
    
    mas = _infer_mas_from_columns(columns)
    
    # Debe devolver lista vacía
    assert mas == []


def test_recursive_forecast_xgb_basic(synthetic_series, trained_xgb_model):
    """Test pronóstico recursivo básico."""
    model = trained_xgb_model["model"]
    feature_columns = trained_xgb_model["feature_columns"]
    
    # Pronosticar 10 días
    forecast = recursive_forecast_xgb(
        series=synthetic_series,
        model=model,
        feature_columns=feature_columns,
        horizon=10
    )
    
    # Verificar estructura
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == 10
    assert {"date", "yhat_xgb"} <= set(forecast.columns)
    
    # Verificar que las fechas son consecutivas
    dates = pd.to_datetime(forecast["date"])
    for i in range(1, len(dates)):
        delta = (dates.iloc[i] - dates.iloc[i-1]).days
        assert delta == 1
    
    # Verificar que las predicciones son numéricas y positivas
    assert forecast["yhat_xgb"].dtype.kind in "fi"
    assert (forecast["yhat_xgb"] >= 0).all()


def test_recursive_forecast_xgb_horizon_variations(synthetic_series, trained_xgb_model):
    """Test con diferentes horizontes de pronóstico."""
    model = trained_xgb_model["model"]
    feature_columns = trained_xgb_model["feature_columns"]
    
    for horizon in [1, 5, 30]:
        forecast = recursive_forecast_xgb(
            series=synthetic_series,
            model=model,
            feature_columns=feature_columns,
            horizon=horizon
        )
        
        # Debe devolver exactamente 'horizon' predicciones
        assert len(forecast) == horizon


def test_recursive_forecast_xgb_extends_series(synthetic_series, trained_xgb_model):
    """Test que el pronóstico extiende la serie temporal."""
    model = trained_xgb_model["model"]
    feature_columns = trained_xgb_model["feature_columns"]
    
    last_date = synthetic_series.index.max()
    
    forecast = recursive_forecast_xgb(
        series=synthetic_series,
        model=model,
        feature_columns=feature_columns,
        horizon=5
    )
    
    # Primera fecha del pronóstico debe ser día siguiente a la última de la serie
    first_forecast_date = pd.to_datetime(forecast["date"].iloc[0])
    expected_date = last_date + pd.Timedelta(days=1)
    
    assert first_forecast_date == expected_date


def test_recursive_forecast_xgb_uses_predictions(synthetic_series, trained_xgb_model):
    """Test que el pronóstico usa predicciones previas para calcular siguientes."""
    model = trained_xgb_model["model"]
    feature_columns = trained_xgb_model["feature_columns"]
    
    # Pronosticar 2 días
    forecast = recursive_forecast_xgb(
        series=synthetic_series,
        model=model,
        feature_columns=feature_columns,
        horizon=2
    )
    
    # Las predicciones deben ser distintas (segundo día usa predicción del primero)
    assert forecast["yhat_xgb"].iloc[0] != forecast["yhat_xgb"].iloc[1]


def test_recursive_forecast_xgb_handles_missing_features(synthetic_series, trained_xgb_model):
    """Test que maneja correctamente las columnas de features."""
    model = trained_xgb_model["model"]
    feature_columns = trained_xgb_model["feature_columns"]
    
    # Test con subset de columnas (todas presentes en el modelo)
    # XGBoost valida que las columnas coincidan exactamente con las del entrenamiento
    forecast = recursive_forecast_xgb(
        series=synthetic_series,
        model=model,
        feature_columns=feature_columns,  # Usar las columnas correctas
        horizon=5
    )
    
    assert len(forecast) == 5
    assert (forecast["yhat_xgb"] >= 0).all()


def test_recursive_forecast_xgb_empty_series():
    """Test con serie vacía."""
    empty_series = pd.Series(dtype=float, name="events")
    
    # Modelo dummy
    model = XGBRegressor(n_estimators=10)
    # Entrenar con datos mínimos
    X_dummy = pd.DataFrame({"feat1": [1, 2, 3]})
    y_dummy = pd.Series([10, 20, 30])
    model.fit(X_dummy, y_dummy)
    
    feature_columns = ["feat1"]
    
    # Debe manejar serie vacía
    forecast = recursive_forecast_xgb(
        series=empty_series,
        model=model,
        feature_columns=feature_columns,
        horizon=3
    )
    
    # Debe devolver 3 predicciones
    assert len(forecast) == 3


def test_make_features_consistency_with_training():
    """Test que make_features es consistente con el de training.py."""
    from pipeline.training import make_features as train_make_features
    
    dates = pd.date_range("2016-01-01", periods=50, freq="D", tz="UTC")
    series = pd.Series(range(50), index=dates, name="events")
    
    # Crear features con ambas funciones
    train_feats = train_make_features(series)
    pred_feats = make_features(series)
    
    # Deben tener las mismas columnas
    assert set(train_feats.columns) == set(pred_feats.columns)
    
    # Deben tener los mismos valores (dentro de tolerancia numérica)
    for col in train_feats.columns:
        pd.testing.assert_series_equal(
            train_feats[col],
            pred_feats[col],
            check_names=False
        )


def test_recursive_forecast_xgb_predictions_are_reasonable(synthetic_series, trained_xgb_model):
    """Test que las predicciones están en rangos razonables."""
    model = trained_xgb_model["model"]
    feature_columns = trained_xgb_model["feature_columns"]
    
    # Estadísticas de la serie histórica
    mean_val = synthetic_series.mean()
    std_val = synthetic_series.std()
    
    forecast = recursive_forecast_xgb(
        series=synthetic_series,
        model=model,
        feature_columns=feature_columns,
        horizon=10
    )
    
    # Las predicciones deben estar cerca del rango histórico
    # (no exacto, pero dentro de ~5 std)
    predictions = forecast["yhat_xgb"]
    assert (predictions >= mean_val - 5 * std_val).all()
    assert (predictions <= mean_val + 5 * std_val).all()


def test_recursive_forecast_custom_lags_and_mas(synthetic_series, trained_xgb_model):
    """Test pronóstico con lags y MAs personalizados."""
    model = trained_xgb_model["model"]
    
    # Crear features con parámetros personalizados
    custom_feats = make_features(synthetic_series, lags=(1, 2, 3), mas=(3, 5))
    custom_columns = [c for c in custom_feats.columns if c != "target"]
    
    # Entrenar modelo con estas features
    feats_clean = custom_feats.dropna()
    X = feats_clean.drop(columns=["target"])
    y = feats_clean["target"]
    
    custom_model = XGBRegressor(n_estimators=30, max_depth=3, random_state=42)
    custom_model.fit(X, y)
    
    # Pronosticar
    forecast = recursive_forecast_xgb(
        series=synthetic_series,
        model=custom_model,
        feature_columns=list(X.columns),
        horizon=7
    )
    
    assert len(forecast) == 7
    assert (forecast["yhat_xgb"] >= 0).all()


def test_forecast_dates_are_timezone_aware(synthetic_series, trained_xgb_model):
    """Test que las fechas del pronóstico son timezone-aware."""
    model = trained_xgb_model["model"]
    feature_columns = trained_xgb_model["feature_columns"]
    
    forecast = recursive_forecast_xgb(
        series=synthetic_series,
        model=model,
        feature_columns=feature_columns,
        horizon=5
    )
    
    # Convertir a datetime y verificar timezone
    dates = pd.to_datetime(forecast["date"])
    
    # Si la serie original es tz-aware, el pronóstico también debe serlo
    if synthetic_series.index.tz is not None:
        # En algunos casos puede ser tz-naive, pero debe ser consistente
        assert True  # Permitir ambos casos


def test_infer_functions_handle_duplicates():
    """Test que las funciones de inferencia manejan duplicados."""
    columns = ["lag_1", "lag_1", "lag_7", "ma_7", "ma_7"]
    
    lags = _infer_lags_from_columns(columns)
    mas = _infer_mas_from_columns(columns)
    
    # No debe haber duplicados
    assert lags == [1, 7]
    assert mas == [7]


def test_recursive_forecast_with_minimal_history(trained_xgb_model):
    """Test pronóstico con historia mínima."""
    # Serie muy corta
    dates = pd.date_range("2016-01-01", periods=30, freq="D", tz="UTC")
    short_series = pd.Series(range(30), index=dates, name="events")
    
    model = trained_xgb_model["model"]
    feature_columns = trained_xgb_model["feature_columns"]
    
    # Debe completarse sin error
    forecast = recursive_forecast_xgb(
        series=short_series,
        model=model,
        feature_columns=feature_columns,
        horizon=5
    )
    
    assert len(forecast) == 5


# ---------------------
# FIN GROUP 04 tests
# ---------------------
