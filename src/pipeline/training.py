#-------------------GROUP 04 training pipeline-------------------
"""
Training pipeline for GA Group 04 — Día/Marca.
Focus: single-brand daily series ("events") with XGBoost as winning model.

Uso:
  python training.py --data brand_daily_group04.parquet --brand "(not set)" \
    --outdir models_group04 --model_name xgb_group04.pkl

Guarda:
  - ../src/models/<model_name>.pkl         # Modelo XGBoost (ruta fija solicitada)
  - <outdir>/xgb_feature_columns.json      # Columnas de features para predicción
  - <outdir>/xgb_test_preds.csv            # Predicciones en test set
  - <outdir>/metrics.csv                   # Métricas MAE, RMSE, MAPE
  - <outdir>/meta.json                     # Metadata (marca, métricas, config)
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Agregar src/ al path para importar módulos locales
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess.preprocessing import TimeSeriesPreprocessor, TSConfig  # noqa: E402


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}


def make_features(series, lags=(1, 2, 3, 5, 7, 14, 21, 28), mas=(7, 14, 28)):
    df = pd.DataFrame({"target": series})
    for L in lags:
        df[f"lag_{L}"] = series.shift(L)
    for w in mas:
        df[f"ma_{w}"] = series.shift(1).rolling(w).mean()
    idx = df.index
    df["dow"] = idx.dayofweek
    df["dom"] = idx.day
    df["week"] = idx.isocalendar().week.astype(int)
    df["month"] = idx.month
    df["year"] = idx.year
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    return df


def train_xgboost(series, test_days, outdir: Path, model_pkl_path: Path):
    feats = make_features(series).dropna()
    train = feats.iloc[:-test_days]
    test = feats.iloc[-test_days:]

    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = evaluate(y_test, pred)

    # === Guardados ===
    # 1) Modelo pkl en ../src/models/<model_name>.pkl (ruta fija solicitada)
    model_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_pkl_path)

    # 2) Artefactos auxiliares en outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "xgb_feature_columns.json").write_text(
        json.dumps(list(X_train.columns), indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {"date": X_test.index, "y": y_test.values, "yhat_xgb": pred}
    ).to_csv(outdir / "xgb_test_preds.csv", index=False)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Ruta al parquet con columnas: transaction_date, product_brand, events")
    parser.add_argument("--brand", default="(not set)")
    parser.add_argument("--outdir", default="models_group04",
                        help="Carpeta para métricas/artefactos (no el .pkl)")
    # NUEVO: nombre del modelo y ruta fija ../src/models
    default_models_dir = (Path(__file__).resolve().parent / ".." / "src" / "models").resolve()
    parser.add_argument("--model_name", default="xgb_model.pkl",
                        help="Nombre del archivo del modelo .pkl (se guardará en ../src/models)")
    parser.add_argument("--test_days", type=int, default=30)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    models_dir = default_models_dir
    model_pkl_path = models_dir / args.model_name  # ../src/models/nombreModelo.pkl

    cfg = TSConfig(
        date_col="transaction_date",
        brand_col="product_brand",
        target_metric="events",
        freq="D",
    )
    pre = TimeSeriesPreprocessor(cfg)
    df = pre.load_data(args.data)
    series = pre.select_brand_series(df, brand=args.brand, metric=cfg.target_metric)

    print(f"\n{'='*70}")
    print(f"Entrenando XGBoost para marca: {args.brand}")
    print(f"Días de test: {args.test_days}")
    print(f"Modelo se guardará en: {model_pkl_path}")
    print(f"Artefactos auxiliares en: {outdir.resolve()}")
    print(f"{'='*70}\n")

    xgb_metrics = train_xgboost(series, test_days=args.test_days,
                                outdir=outdir, model_pkl_path=model_pkl_path)

    meta = {
        "brand": args.brand,
        "target_metric": cfg.target_metric,
        "test_days": args.test_days,
        "model": "XGBoost",
        "model_pkl": str(model_pkl_path),
        "metrics": xgb_metrics,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"modelo": "XGBoost", **xgb_metrics}]).to_csv(outdir / "metrics.csv", index=False)
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== Entrenamiento completado ===")
    print("Modelo: XGBoost (winner)")
    print("Métricas:", xgb_metrics)
    print(f"PKL guardado en: {model_pkl_path}")
    print(f"Artefactos guardados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()
#-------------------FINISH GROUP 04 training pipeline-------------------
