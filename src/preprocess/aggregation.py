import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# GROUP 04 aggregation module

def calculate_day_brand_aggregation(df: pd.DataFrame, metric_col: str | None = None) -> pd.DataFrame:
    """
    Agrega por 'transaction_date' (día) y 'product_brand'.
    - Si metric_col es None o no existe, devuelve el conteo de filas ('events').
    - Si metric_col existe (y es numérica o convertible), devuelve la suma de esa métrica.
    """
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["transaction_date"])
    df["transaction_date"] = df["transaction_date"].dt.normalize()

    if "product_brand" not in df.columns:
        raise ValueError("No se encontró la columna 'product_brand' en el DataFrame de entrada.")
    df["product_brand"] = df["product_brand"].astype("object")

    if metric_col and metric_col in df.columns:
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
        df_metric = df.dropna(subset=[metric_col])
        agg = (
            df_metric
            .groupby(["transaction_date", "product_brand"], as_index=False)[metric_col]
            .sum()
            .rename(columns={metric_col: f"sum_{metric_col}"})
        )
    else:
        agg = (
            df.groupby(["transaction_date", "product_brand"])
              .size()
              .reset_index(name="events")
        )
    return agg


def generate_day_brand_grouped_data_files(
    in_path: str = "../../data/raw/data_sample.parquet",
    out_path: str = "../../data/raw/day_brand.parquet",
    metric_col: str | None = None,
) -> pd.DataFrame:
    """
    Lee el parquet de entrada, agrega por día y marca, guarda el resultado **y devuelve el DataFrame**.

    Parámetros
    ------En base al feedback de Pull Request
    ----------
    in_path : str
        Ruta del parquet de entrada.
    out_path : str
        Ruta donde guardar el parquet agregado.
    metric_col : str | None
        Métrica a sumar; si None -> cuenta 'events'.

    Retorna
    -------
    pd.DataFrame
        El DataFrame agregado (mismo que se guarda en `out_path`).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = pd.read_parquet(in_path)
    grouped = calculate_day_brand_aggregation(data, metric_col=metric_col)

    grouped.to_parquet(out_path, index=False)
    return grouped

# ===========================================================
# FIN GROUP 04 aggregation module (day-only, no dedupe)
# ===========================================================