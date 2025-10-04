# Limpieza de datos - Grupo 2
import pandas as pd
import numpy as np


def clean_date_column(df, date_column="transaction_date"):
    """Convierte fechas de formato YYYYMMDD a datetime."""
    df_clean = df.copy()
    if date_column in df_clean.columns:
        if df_clean[date_column].dtype == "object":
            df_clean[date_column] = pd.to_datetime(
                df_clean[date_column], format="%Y%m%d", errors="coerce")
        else:
            df_clean[date_column] = pd.to_datetime(
                df_clean[date_column].astype(str), format="%Y%m%d", errors="coerce")
        df_clean = df_clean.dropna(subset=[date_column])
    return df_clean


def clean_revenue_columns(df):
    """Convierte revenue de microdólares a dólares."""
    df_clean = df.copy()
    revenue_cols = ["transactionRevenue",
                    "sessionQualityDim", "totalTransactionRevenue"]

    for col in revenue_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            df_clean[col] = df_clean[col].fillna(0) / 1000000

    return df_clean


def clean_product_data(df):
    """Limpia datos de productos."""
    df_clean = df.copy()

    # Limpiar product_id
    if "product_name" in df_clean.columns:
        df_clean["product_id"] = df_clean["product_name"].fillna("Unknown")
        df_clean["product_id"] = df_clean["product_id"].str.strip()
    elif "v2ProductName" in df_clean.columns:
        df_clean["product_id"] = df_clean["v2ProductName"].fillna("Unknown")
        df_clean["product_id"] = df_clean["product_id"].str.strip()

    return df_clean


def clean_data(df, date_column="date", revenue_strategy='convert', missing_strategy='fill'):
    """Función principal de limpieza."""
    print("Limpiando datos...")

    df_clean = df.copy()
    df_clean = clean_date_column(df_clean, date_column)
    df_clean = clean_revenue_columns(df_clean)
    df_clean = clean_product_data(df_clean)

    print(f"Datos antes: {df.shape}")
    print(f"Datos después: {df_clean.shape}")

    return df_clean
