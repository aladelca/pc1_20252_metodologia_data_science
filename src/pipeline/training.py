import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.pipeline.preprocess import load_and_preprocess, split_train_test


def evaluate(y_true, y_pred):
    """Calcula métricas de evaluación para un
    modelo de predicción de series temporales.

    Se incluyen el Error Absoluto Medio (MAE),
    la Raíz del Error Cuadrático Medio (RMSE)
    y el Error Porcentual Absoluto Medio (MAPE).
    En el cálculo de MAPE se agrega 1 al denominador
    para evitar divisiones por cero.

    Parameters
    y_true (array-like o pd.Series): Valores reales observados.
    y_pred (array-like o pd.Series): Valores predichos por el modelo.

    Returns
    dict Diccionario con las métricas calculadas:
    - "MAE": float, error absoluto medio.
    - "RMSE": float, raíz del error cuadrático medio.
    - "MAPE": float, error porcentual absoluto medio (%).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def main():
    # Cargar datos
    df_no_quant, df_quant = load_and_preprocess(
        "data/aggregated/day_category.parquet", "Accessories"
        )

    # Split 80/20 sin solapamiento
    X_train, X_test, y_train, y_test = (
        split_train_test(df_no_quant, df_quant, train_size=0.8)
    )

    # Entrenar modelo
    gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
    gb.fit(X_train, y_train)

    # Evaluar en test
    preds = gb.predict(X_test)
    metrics = evaluate(y_test, preds)

    print("📊 Métricas Baseline:", metrics)

    # Guardar modelo
    joblib.dump(gb, "src/models/gb_model.pkl")
    print("✅ Modelo guardado en models/gb_model.pkl")


if __name__ == "__main__":
    main()
