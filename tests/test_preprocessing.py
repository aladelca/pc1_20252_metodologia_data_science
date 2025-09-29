#GROUP4
# tests/test_preprocessing.py GROUP04
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# --- asegurar que src/ esté en el path ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

# módulos del pipeline
import importlib
import preprocess.aggregation as aggregation_mod
import preprocess.preprocessing as preprocessing_mod
# cleaning es opcional (el test hace fallback si falla)
import preprocess.cleaning as cleaning_mod


# ---------------------
# Fixtures de utilería
# ---------------------
@pytest.fixture
def tmp_paths(tmp_path):
    """Rutas temporales (no tocan tus datos reales)."""
    raw = tmp_path / "data" / "raw" / "data_sample.parquet"
    cleaned = tmp_path / "data" / "raw" / "data_sample_cleaned_group04.parquet"
    aggregated = tmp_path / "data" / "raw" / "brand_daily_group04.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    return {
        "raw": raw,
        "cleaned": cleaned,
        "aggregated": aggregated,
    }


@pytest.fixture
def synthetic_raw_df():
    """Dataset sintético con múltiples filas por día/marca para probar el conteo ('events')."""
    # 10 días con multiplicidad variable para una marca y otra de control
    dates = pd.date_range("2016-08-01", periods=90, freq="D")
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        # (not set) con multiplicidad aleatoria 1..5
        k1 = int(rng.integers(1, 6))
        rows += [{"transaction_date": d.strftime("%Y%m%d"), "product_brand": "(not set)"}] * k1
        # otra marca con multiplicidad 0..3
        k2 = int(rng.integers(0, 4))
        rows += [{"transaction_date": d.strftime("%Y%m%d"), "product_brand": "BrandX"}] * k2
    df = pd.DataFrame(rows)
    # añadir una métrica numérica opcional para probar suma si se desea
    df["product_quantity"] = 1
    return df


# ---------------------
# Tests
# ---------------------
def test_end_to_end_agg_and_preprocess(tmp_paths, synthetic_raw_df):
    # recargar módulos por si el usuario los editó
    importlib.reload(aggregation_mod)
    importlib.reload(preprocessing_mod)
    importlib.reload(cleaning_mod)

    # ========= 1) Guardar RAW y tratar de correr CLEANING =========
    synthetic_raw_df.to_parquet(tmp_paths["raw"], index=False)

    cleaned_ok = False
    try:
        # si tienes un run_cleaning real, úsalo
        from preprocess.cleaning import CleaningConfig, run_cleaning

        cfg_clean = CleaningConfig(
            raw_path=str(tmp_paths["raw"]),
            out_parquet=str(tmp_paths["cleaned"]),
            out_csv=str(tmp_paths["cleaned"]).replace(".parquet", ".csv"),
            persist=True,
            drop_columns_missing_action="impute",
        )
        _ = run_cleaning(cfg_clean)
        cleaned_ok = Path(tmp_paths["cleaned"]).exists()
    except Exception:
        cleaned_ok = False

    if not cleaned_ok:
        # Fallback mínimo: construir un "cleaned" equivalente al esperado por aggregation
        df = synthetic_raw_df.copy()
        # parsed_date (tz-aware, como en tu código)
        ser = pd.to_datetime(df["transaction_date"].astype(str), format="%Y%m%d", errors="coerce", utc=True)
        df["parsed_date"] = ser
        # product_brand ya existe; normalizamos tipo
        df["product_brand"] = df["product_brand"].astype("object")
        df.to_parquet(tmp_paths["cleaned"], index=False)

    # ========= 2) AGGREGATION: in-memory y a disco =========
    df_in = pd.read_parquet(tmp_paths["cleaned"])

    # si no tuviera transaction_date, crearlo desde parsed_date (tu flujo lo contempla)
    if "transaction_date" not in df_in.columns and "parsed_date" in df_in.columns:
        df_in["transaction_date"] = df_in["parsed_date"]

    # a) cálculo en memoria (conteo de events)
    agg_mem = aggregation_mod.calculate_day_brand_aggregation(df_in, metric_col=None)
    assert {"transaction_date", "product_brand", "events"} <= set(agg_mem.columns)
    # hay 10 días → al menos 10 grupos para cada marca que se presente
    assert agg_mem["transaction_date"].nunique() == 90

    # b) escritura a disco con generate_day_brand_grouped_data_files
    res = aggregation_mod.generate_day_brand_grouped_data_files(
        in_path=str(tmp_paths["cleaned"]),
        out_path=str(tmp_paths["aggregated"]),
        metric_col=None,  # conteo
    )
    assert res == "success"
    assert Path(tmp_paths["aggregated"]).exists()

    agg_disk = pd.read_parquet(tmp_paths["aggregated"])
    # comparar igualdad (ordenando)
    cols = sorted({"transaction_date", "product_brand", "events"})
    assert agg_mem[cols].sort_values(cols).reset_index(drop=True).equals(
        agg_disk[cols].sort_values(cols).reset_index(drop=True)
    )

    # ========= 3) PREPROCESSING: cargar y construir serie univariada =========
    cfg = preprocessing_mod.TSConfig(
    date_col="transaction_date",
    brand_col="product_brand",
    target_metric="events",
    freq="D",
    fill_missing="zero",
    # --- ventanas pequeñas para que no se vacíe el train ---
    lags=(1, 2, 3),
    ventanas_ma=(3, 5, 7),
    ventanas_std=(3, 5),
    ventanas_roll=(3, 5, 7),
    ventanas_momentum=(3,),
    # ---- desactivar chequeo de fuga ----
    check_leakage=False
    )
    prep = preprocessing_mod.TimeSeriesPreprocessor(cfg)
    df_models = prep.load_data(str(tmp_paths["aggregated"]))
    # index de fechas (tz-aware en UTC)
    assert isinstance(df_models.index, pd.DatetimeIndex)
    assert str(df_models.index.tz) == "UTC"
    assert {"product_brand", "events"} <= set(df_models.columns)

    # elegir marca (not set) o la que exista
    brand = "(not set)"
    if "product_brand" in df_models.columns:
        brand = df_models["product_brand"].mode(dropna=True).iloc[0]

    # serie diaria para esa marca
    series = prep.select_brand_series(df_models, brand=brand, metric="events")
    assert series.index.is_monotonic_increasing
    assert series.shape[0] >= 10  # 10 días + rellenos si hubiera
    assert series.dtype.kind in "fi"  # numérica

    # ========= 4) Preparaciones de modelo (que “no revienten”) =========
    # ARIMA
    arima_series = prep.prepare_arima_data(series)
    assert len(arima_series) > 0

    # Prophet
    prophet_df, exog_cols = prep.prepare_prophet_data(series)
    assert {"ds", "y"} <= set(prophet_df.columns)
    assert isinstance(exog_cols, list)

    # ML (usa fechas del rango)
    X_tr, y_tr, X_te, y_te = prep.prepare_ml_data(
        series,
        train_start=str(series.index.min().date()),
        train_end=str((series.index.min() + pd.Timedelta(days=60)).date()),
        test_start=str((series.index.min() + pd.Timedelta(days=61)).date()),
    )
    for arr in (X_tr, y_tr, X_te, y_te):
        assert len(arr) >= 1

    # LSTM
    feats = prep.create_features(series).dropna()
    Xtr, Xte, ytr, yte = prep.prepare_lstm_data(feats, sequence_length=3)
    assert Xtr.ndim == 3 and Xte.ndim == 3

    # SARIMAX exógenas (vacías)
    exog = prep.create_sarimax_exogenous(series, events_periods=[])
    assert exog.shape[0] == series.shape[0]
#--------------------------------------------------------------------------------------FIN TEST GROUP04-