#-------------------GROUP 04 prediction pipeline-------------------
"""
Prediction pipeline for GA Group 04 — Día/Marca.
Recursive H-step forecast with trained XGBoost.

Uso:
  # Modelo en ../src/models/xgb_group04.pkl
  python prediction.py --data brand_daily_group04.parquet \
    --brand "(not set)" \
    --model_name xgb_group04.pkl \
    --artifactsdir models_group04 \
    --horizon 30 \
    --out predictions_xgb.csv

También puedes pasar el JSON de columnas explícitamente:
  python prediction.py --data brand_daily_group04.parquet \
    --brand "(not set)" \
    --model_name xgb_group04.pkl \
    --feature_cols path/a/xgb_feature_columns.json
"""
import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

# Agregar src/ al path para importar módulos locales
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess.preprocessing import TimeSeriesPreprocessor, TSConfig  # noqa: E402


def make_features(series: pd.Series, lags=(1,2,3,5,7,14,21,28), mas=(7,14,28)) -> pd.DataFrame:
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
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    return df


def _infer_lags_from_columns(cols: List[str]) -> List[int]:
    lags = []
    for c in cols:
        if c.startswith("lag_"):
            try:
                lags.append(int(c.split("_")[1]))
            except Exception:
                pass
    return sorted(set(lags))


def _infer_mas_from_columns(cols: List[str]) -> List[int]:
    mas = []
    for c in cols:
        if c.startswith("ma_"):
            try:
                mas.append(int(c.split("_")[1]))
            except Exception:
                pass
    return sorted(set(mas))


def recursive_forecast_xgb(series: pd.Series, model, feature_columns: List[str], horizon: int) -> pd.DataFrame:
    series_ext = series.copy()
    preds, dates = [], []

    lags = _infer_lags_from_columns(feature_columns)
    mas  = _infer_mas_from_columns(feature_columns)

    for _ in range(horizon):
        # siguiente fecha (respeta tz si existe)
        next_date = (series_ext.index.max() + pd.Timedelta(days=1)) if len(series_ext) \
                    else pd.Timestamp.today().normalize()

        tmp = series_ext.copy()
        tmp.loc[next_date] = np.nan

        fe = make_features(
            tmp,
            lags=tuple(lags) if lags else (1,2,3,5,7,14,21,28),
            mas=tuple(mas) if mas else (7,14,28)
        )

        x_next = fe.drop(columns=["target"]).iloc[[-1]]
        # Alinear exactamente a las columnas de entrenamiento
        x_next = x_next.reindex(columns=feature_columns, fill_value=0.0)

        yhat = float(model.predict(x_next)[0])
        preds.append(yhat)
        dates.append(next_date)

        # Actualizar serie con la predicción (para que lags/MA futuras la usen)
        series_ext.loc[next_date] = yhat

    return pd.DataFrame({"date": dates, "yhat_xgb": preds})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Parquet con transaction_date, product_brand, events")
    parser.add_argument("--brand", default="(not set)")
    # NUEVO: nombre del modelo en ../src/models
    parser.add_argument("--model_name", default="xgb_model.pkl",
                        help="Nombre del .pkl dentro de ../src/models (ej: xgb_group04.pkl)")
    # Dónde está el JSON de columnas (si no pasas --feature_cols)
    parser.add_argument("--artifactsdir", default="models_group04",
                        help="Carpeta donde está xgb_feature_columns.json")
    # O bien ruta explícita al JSON de columnas
    parser.add_argument("--feature_cols", default=None,
                        help="Ruta directa a xgb_feature_columns.json (opcional)")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--out", default="predictions_xgb.csv")
    args = parser.parse_args()

    # Resolver ruta fija del modelo: ../src/models/<model_name>.pkl
    model_dir_fixed = (Path(__file__).resolve().parent / ".." / "src" / "models").resolve()
    model_path = model_dir_fixed / args.model_name

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    print(f"Cargando modelo XGBoost desde: {model_path}")
    model = joblib.load(model_path)

    # Cargar columnas de features
    if args.feature_cols:
        feature_cols_path = Path(args.feature_cols)
    else:
        feature_cols_path = Path(args.artifactsdir) / "xgb_feature_columns.json"

    if not feature_cols_path.exists():
        raise FileNotFoundError(
            f"No se encontró el JSON de columnas en: {feature_cols_path}\n"
            f"Indica --feature_cols <ruta_json> o --artifactsdir <carpeta>"
        )

    feature_columns = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    print(f"Columnas de features cargadas desde: {feature_cols_path}")

    # Cargar datos y construir serie de una marca
    cfg = TSConfig(date_col="transaction_date", brand_col="product_brand", target_metric="events", freq="D")
    pre = TimeSeriesPreprocessor(cfg)
    df = pre.load_data(args.data)
    series = pre.select_brand_series(df, brand=args.brand, metric=cfg.target_metric)

    # Pronóstico recursivo
    print(f"Pronosticando {args.horizon} días para la marca '{args.brand}' ...")
    fc = recursive_forecast_xgb(series, model, feature_columns, args.horizon)
    fc.to_csv(args.out, index=False)

    print(f"\nPredicción guardada ({len(fc)} días) en: {Path(args.out).resolve()}")
    print(fc.head())


if __name__ == "__main__":
    main()
#-------------------FINISH GROUP 04 prediction pipeline-------------------
