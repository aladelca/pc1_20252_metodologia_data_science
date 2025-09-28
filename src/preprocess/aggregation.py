import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# ===========================================================
# aggregation.py — Grupo 04 (Día/Marca)
# Objetivo: devolver TODAS las filas con:
#   - 'day' (día del mes, "01"… "31")
#   - 'product_brand'
#   sin agrupar ni deduplicar (se preserva la multiplicidad).
#
# Input : ../data/raw/data_sample_cleaned_group04.parquet
# Output: ../data/raw/brand_daily_group04.parquet
# ===========================================================

@dataclass
class AggregationConfig:
    # IO
    input_path: str = "../data/raw/data_sample_cleaned_group04.parquet"
    out_parquet: Optional[str] = "../data/raw/brand_daily_group04.parquet"
    persist: bool = True

    # Columnas clave del input limpio
    date_col: str = "parsed_date"
    brand_col: str = "product_brand"

def _ensure_dir(path: Optional[str]):
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_cleaned(cfg: AggregationConfig) -> pd.DataFrame:
    if not os.path.exists(cfg.input_path):
        raise FileNotFoundError(f"No existe el archivo de entrada: {cfg.input_path}")
    return pd.read_parquet(cfg.input_path)

def aggregate_day_brand_minimal(df: pd.DataFrame, cfg: AggregationConfig) -> pd.DataFrame:
    """
    Devuelve todas las filas con sólo dos columnas:
      - 'day'  -> día del mes en formato "01"…"31"
      - 'product_brand'
    NO agrupa, NO elimina duplicados: preserva el conteo original.
    """
    out = df.copy()

    # Fecha
    if cfg.date_col not in out.columns:
        similar = [c for c in out.columns if cfg.date_col.lower() in c.lower()]
        if not similar:
            raise ValueError(f"No se encontró columna de fecha '{cfg.date_col}'.")
        out = out.rename(columns={similar[0]: cfg.date_col})
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="coerce", utc=True)
    out = out.dropna(subset=[cfg.date_col])

    # Marca
    if cfg.brand_col not in out.columns:
        similar = [c for c in out.columns if "brand" in c.lower()]
        if not similar:
            raise ValueError(f"No se encontró columna de marca '{cfg.brand_col}'.")
        out = out.rename(columns={similar[0]: cfg.brand_col})
    out[cfg.brand_col] = out[cfg.brand_col].astype("object")

    # Día del mes como string 2-dígitos (preserva todas las filas)
    out["day"] = out[cfg.date_col].dt.day.astype(int).astype(str).str.zfill(2)

    # Sólo columnas requeridas, sin drop_duplicates
    out = out[["day", cfg.brand_col]].reset_index(drop=True)
    return out

# Alias de compatibilidad (si algún notebook llama a este nombre)
def aggregate_brand_daily(df: pd.DataFrame, cfg: AggregationConfig) -> pd.DataFrame:
    return aggregate_day_brand_minimal(df, cfg)

def run_aggregation(cfg: AggregationConfig = None) -> pd.DataFrame:
    cfg = cfg or AggregationConfig()
    df = load_cleaned(cfg)
    day_brand = aggregate_day_brand_minimal(df, cfg)

    if cfg.persist:
        _ensure_dir(cfg.out_parquet)
        day_brand.to_parquet(cfg.out_parquet, index=False)

    return day_brand

if __name__ == "__main__":
    out = run_aggregation()
    print("Agregación Día/Marca (mantiene multiplicidad) completada:", out.shape)
    print(out.head())
# ===========================================================
# FIN GROUP 04 aggregation module (day-only, no dedupe)
# ===========================================================
