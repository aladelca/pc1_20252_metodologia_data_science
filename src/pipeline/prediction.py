
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
from pathlib import Path
from typing import List

# --- Funciones de Ingeniería de Características (Replicadas de training.py) ---
# Es crucial que sean idénticas para consistencia entre entrenamiento y predicción.
def create_features_for_prediction(df_hist: pd.Series, future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Crea un DataFrame de características para fechas futuras basado en datos históricos.
    """
    target_col = df_hist.name
    
    # Combina el historial con placeholders para el futuro para calcular lags/rolling
    full_series = df_hist.copy()
    future_series = pd.Series(np.nan, index=future_dates, name=target_col)
    combined = pd.concat([full_series, future_series])

    # Crea el DataFrame de características sobre el índice combinado
    df_feat = combined.to_frame()
    df_feat.columns = ['value'] # Columna temporal

    # Características de tiempo
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['weekofyear'] = df_feat.index.isocalendar().week.astype(int)
    df_feat['quarter'] = df_feat.index.quarter

    # Características de lag y ventana móvil
    for lag in [1, 7, 14, 21, 28]:
        df_feat[f'lag_{lag}'] = combined.shift(lag)
        
    for win in [7, 14, 28]:
        df_feat[f'roll_mean_{win}'] = combined.shift(1).rolling(window=win).mean()
        df_feat[f'roll_std_{win}'] = combined.shift(1).rolling(window=win).std()
        df_feat[f'roll_max_{win}'] = combined.shift(1).rolling(window=win).max()
        df_feat[f'roll_min_{win}'] = combined.shift(1).rolling(window=win).min()

    # Retorna solo las filas correspondientes a las fechas futuras
    return df_feat.loc[future_dates].drop(columns=['value'])

# --- Pipeline de Predicción ---
def run_prediction(product_sku: str, prediction_days: int) -> pd.DataFrame:
    """
    Ejecuta el pipeline de predicción para un SKU y un número de días futuros.
    """
    print("--- Iniciando Pipeline de Predicción ---")

    # 1. Cargar Modelos y Datos Históricos
    project_root = Path(__file__).resolve().parents[2]
    processed_data_path = project_root / 'data' / 'processed' / 'group2_data.parquet'
    models_path = project_root / 'models'

    print("Cargando modelos y datos históricos...")
    clf = xgb.XGBClassifier()
    clf.load_model(models_path / f'{product_sku}_clf.json')

    reg = xgb.XGBRegressor()
    reg.load_model(models_path / f'{product_sku}_reg.json')

    df = pd.read_parquet(processed_data_path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # 2. Preparar Serie de Tiempo Histórica
    ts_one_product = df[df["product_sku"] == product_sku].copy()
    ts_one_product.set_index('transaction_date', inplace=True)
    ts_one_product = ts_one_product.asfreq('D').fillna(0)
    target_col = 'total_product_quantity'
    ts_hist = ts_one_product[target_col]

    # 3. Generar Características para Fechas Futuras
    last_date = ts_hist.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')
    
    print(f"Generando características para {prediction_days} días futuros...")
    X_future = create_features_for_prediction(ts_hist, future_dates)

    # 4. Realizar Predicciones
    print("Realizando predicciones...")
    preds_is_sale = clf.predict(X_future)
    preds_quantity = reg.predict(X_future)

    # 5. Combinar y Formatear Resultados
    final_preds = preds_quantity * preds_is_sale
    final_preds[final_preds < 0] = 0 # Asegurar no negatividad

    df_preds = pd.DataFrame({
        'fecha': future_dates,
        'sku': product_sku,
        'prediccion_cantidad': np.round(final_preds).astype(int)
    })

    print("--- Pipeline de Predicción Finalizado ---")
    return df_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de predicción para el modelo de dos etapas.")
    parser.add_argument(
        '--product-sku',
        type=str,
        default='GGOEGDHQ015399',
        help='El SKU del producto para el cual generar predicciones.'
    )
    parser.add_argument(
        '--prediction-days',
        type=int,
        default=7,
        help='El número de días futuros a predecir.'
    )
    args = parser.parse_args()

    predictions_df = run_prediction(args.product_sku, args.prediction_days)
    
    print("\n--- Predicciones ---")
    print(predictions_df.to_string(index=False))
