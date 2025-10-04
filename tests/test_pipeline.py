import numpy as np
from src.pipeline.preprocess import load_and_preprocess, split_train_test
from src.pipeline.training import evaluate
from src.pipeline.prediction import run_prediction


def test_full_pipeline(monkeypatch, tmp_path):
    """
    Test de integración que valida:
    - carga y preprocesamiento de datos
    - entrenamiento de modelo
    - predicciones y métricas
    """

    # Paso 1: Preprocesamiento
    df_no_quant, df_quant = load_and_preprocess(
        "data/aggregated/day_category.parquet", "Accessories"
    )
    assert not df_no_quant.empty
    assert not df_quant.empty

    # Paso 2: Split de datos
    X_train, X_test, y_train, y_test = split_train_test(
        df_no_quant, df_quant, train_size=0.8
    )
    assert len(X_train) > 0
    assert len(X_test) > 0

    # Paso 3: Mockear modelo para predicción rápida
    class MockModel:
        def predict(self, X):
            return np.ones(len(X))  # predicción constante
    
    def mock_load(path):
        return MockModel()
    
    # Patch de joblib.load en prediction.py
    monkeypatch.setattr("src.pipeline.prediction.joblib.load", mock_load)

    # Paso 4: Ejecutar predicción
    X_test2, y_test2, preds = run_prediction()
    assert len(preds) == len(X_test2)
    assert isinstance(preds[0], (int, float))

    # Paso 5: Evaluar métricas
    metrics = evaluate(y_test2, preds)
    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE"}
    assert all(v >= 0 for v in metrics.values())