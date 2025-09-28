import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
#GROUP 04 aggregation module)

def calculate_day_brand_aggregation(df: pd.DataFrame, metric_col: str | None = None) -> pd.DataFrame:
    """
    Agrega por 'transaction_date' (día) y 'product_brand'.
    - Si metric_col es None o no existe, devuelve el conteo de filas ('events').
    - Si metric_col existe (y es numérica o convertible), devuelve la suma de esa métrica.

    Parámetros
    ----------
    df : pd.DataFrame
        Debe contener 'transaction_date' (object o datetime) y 'product_brand'.
    metric_col : str | None
        Nombre de la métrica a sumar (opcional).

    Retorna
    -------
    pd.DataFrame
        DataFrame agregado con columnas:
        - 'transaction_date' (normalizada a fecha)
        - 'product_brand'
        - 'events' (si no hay métrica) o 'sum_<metric_col>' (si se indicó métrica)
    """
    # Asegurar fecha datetime y normalizar a día
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["transaction_date"])
    df["transaction_date"] = df["transaction_date"].dt.normalize()

    # Asegurar columna de marca como texto (conserva "(not set)")
    if "product_brand" not in df.columns:
        raise ValueError("No se encontró la columna 'product_brand' en el DataFrame de entrada.")
    df["product_brand"] = df["product_brand"].astype("object")

    # Si se especifica una métrica válida, sumamos; si no, contamos eventos
    if metric_col and metric_col in df.columns:
        # Intentar convertir a numérico para sumar
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
        # Eliminar nulos de la métrica antes de sumar
        df_metric = df.dropna(subset=[metric_col])
        agg = (
            df_metric
            .groupby(["transaction_date", "product_brand"], as_index=False)[metric_col]
            .sum()
            .rename(columns={metric_col: f"sum_{metric_col}"})
        )
    else:
        # Conteo de filas por día/marca
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
) -> str:
    """
    Lee el parquet de entrada, agrega por día y marca, y guarda el resultado.
    - Si metric_col es None -> guarda conteo de 'events'
    - Si metric_col existe -> guarda 'sum_<metric_col>'
    """
    # Crear directorio destino
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Leer datos
    data = pd.read_parquet(in_path)

    # Agregar
    grouped = calculate_day_brand_aggregation(data, metric_col=metric_col)

    # Guardar
    grouped.to_parquet(out_path, index=False)
    return "success"


# ===========================================================
# FIN GROUP 04 aggregation module (day-only, no dedupe)
# ===========================================================
