
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import joblib
from pathlib import Path

# --- Funciones de Ingeniería de Características (del notebook) ---
def create_features_enhanced(df, target_col):
    df_feat = df.to_frame().copy()
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['weekofyear'] = df_feat.index.isocalendar().week.astype(int)
    df_feat['quarter'] = df_feat.index.quarter
    
    for lag in [1, 7, 14, 21, 28]:
        df_feat[f'lag_{lag}'] = df_feat[target_col].shift(lag)
        
    for win in [7, 14, 28]:
        df_feat[f'roll_mean_{win}'] = df_feat[target_col].shift(1).rolling(window=win).mean()
        df_feat[f'roll_std_{win}'] = df_feat[target_col].shift(1).rolling(window=win).std()
        df_feat[f'roll_max_{win}'] = df_feat[target_col].shift(1).rolling(window=win).max()
        df_feat[f'roll_min_{win}'] = df_feat[target_col].shift(1).rolling(window=win).min()
        
    df_feat.dropna(inplace=True)
    return df_feat

# --- Pipeline de Entrenamiento ---
def run_training(product_sku: str):
    """
    Ejecuta el pipeline de entrenamiento para un SKU de producto específico.
    """
    print("--- Iniciando Pipeline de Entrenamiento ---")
    
    # 1. Cargar Datos
    project_root = Path(__file__).resolve().parents[2]
    processed_data_path = project_root / 'data' / 'processed' / 'group2_data.parquet'
    models_path = project_root / 'models'
    models_path.mkdir(exist_ok=True)

    print(f"Cargando datos desde: {processed_data_path}")
    df = pd.read_parquet(processed_data_path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # 2. Preparar Serie de Tiempo para el producto
    print(f"Preparando datos para el producto: {product_sku}")
    ts_one_product = df[df["product_sku"] == product_sku].copy()
    ts_one_product.set_index('transaction_date', inplace=True)
    ts_one_product = ts_one_product.asfreq('D').fillna(0)
    target_col = 'total_product_quantity'
    ts = ts_one_product[target_col]

    # 3. Ingeniería de Características
    print("Creando características...")
    df_featured = create_features_enhanced(ts, target_col)
    X = df_featured.drop(columns=[target_col])
    y = df_featured[target_col]

    # 4. Entrenamiento del Modelo de Clasificación
    print("Entrenando modelo de clasificación...")
    y_clf = (y > 0).astype(int)
    neg, pos = np.bincount(y_clf)
    scale_pos_weight_value = neg / pos

    clf = xgb.XGBClassifier(
        n_estimators=500,
        eval_metric='logloss',
        learning_rate=0.01,
        scale_pos_weight=scale_pos_weight_value,
        random_state=42,
        # early_stopping_rounds no se usa aquí para simplificar el script,
        # pero se podría añadir con un eval_set si se hace split.
    )
    clf.fit(X, y_clf, verbose=False)
    clf_path = models_path / f'{product_sku}_clf.json'
    clf.save_model(clf_path)
    print(f"Clasificador guardado en: {clf_path}")

    # 5. Entrenamiento del Modelo de Regresión
    print("Entrenando modelo de regresión...")
    X_reg = X[y > 0]
    y_reg = y[y > 0]

    reg = xgb.XGBRegressor(
        n_estimators=500,
        eval_metric='rmse',
        learning_rate=0.01,
        random_state=42
    )
    reg.fit(X_reg, y_reg, verbose=False)
    reg_path = models_path / f'{product_sku}_reg.json'
    reg.save_model(reg_path)
    print(f"Regressor guardado en: {reg_path}")

    print("--- Pipeline de Entrenamiento Finalizado ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento para el modelo de dos etapas.")
    parser.add_argument(
        '--product-sku',
        type=str,
        default='GGOEGDHQ015399', # SKU del producto con más ventas como default
        help='El SKU del producto para el cual entrenar el modelo.'
    )
    args = parser.parse_args()
    
    run_training(args.product_sku)
