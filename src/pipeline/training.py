# training.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path

def run_training():
    print("🚀 Entrenamiento RÁPIDO - Solo generar .pkl")
    
    # 1. Paths
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "data_training.parquet"
    models_path = project_root / "artifacts" / "trained_models"
    models_path.mkdir(parents=True, exist_ok=True)

    # 2. Cargar datos YA PREPROCESADOS
    df = pd.read_parquet(data_path)
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])

    # 3. Filtrar (EXACTO como tu experimentación)
    df = df[df["units_sold"] <= 15].copy()
    product_counts = df["product_sku"].value_counts()
    products_50plus = product_counts[product_counts >= 50].index
    df = df[df["product_sku"].isin(products_50plus)]

    print(f"📊 Datos para entrenamiento: {len(df)} filas")

    # 4. SOLO columnas numéricas (evitar el error de promo_id)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_features if col not in ["units_sold"]]
    
    print(f"🎯 Usando {len(features)} features numéricas")

    # 5. Entrenar modelo (parámetros de tu experimentación)
    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    print("🔄 Entrenando modelo...")
    X = df[features]
    y = df["units_sold"]
    model.fit(X, y)

    # 6. Guardar modelo
    model_path = models_path / "xgboost_model.pkl"
    joblib.dump((model, features), model_path)

    print(f"✅ Modelo guardado en: {model_path}")
    print("🎯 LISTO para prediction.py")

if __name__ == "__main__":
    run_training()