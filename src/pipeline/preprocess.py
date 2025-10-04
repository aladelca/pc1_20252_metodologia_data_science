import pandas as pd


def load_and_preprocess(filepath, category, lags=[1, 2, 7, 14]):
    df = pd.read_parquet(filepath)
    df['product_category'] = df['product_category'].astype('category')

    # Filtrar por categoría
    df_cat = df[df["product_category"] == category].copy()

    # Agrupar por fecha
    df_cat = (
        df_cat.groupby("transaction_date")["product_quantity"]
        .sum().reset_index()
    )
    df_cat["transaction_date"] = pd.to_datetime(df_cat["transaction_date"])
    df_cat = df_cat.set_index("transaction_date")

    # Crear lags
    for lag in lags:
        df_cat[f"lag_{lag}"] = df_cat["product_quantity"].shift(lag)

    # Variables de calendario
    df_cat["day_of_week"] = df_cat.index.dayofweek
    df_cat["month"] = df_cat.index.month

    # Eliminar NaN
    df_cat = df_cat.dropna()

    df_no_quant = df_cat.drop(columns=["product_quantity"])
    df_quant = df_cat["product_quantity"]

    return df_no_quant, df_quant


def split_train_test(df_no_quant, df_quant, train_size=0.8):
    """Divide en 80% train y 20% test SIN solapamiento."""
    split_index = int(len(df_no_quant) * train_size)
    X_train, X_test = (
        df_no_quant.iloc[:split_index], df_no_quant.iloc[split_index:]
    )
    y_train, y_test = df_quant.iloc[:split_index], df_quant.iloc[split_index:]
    return X_train, X_test, y_train, y_test
