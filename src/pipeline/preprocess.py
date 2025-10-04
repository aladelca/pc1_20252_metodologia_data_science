import pandas as pd


def load_and_preprocess(filepath, category, lags=[1, 2, 7, 14]):
    """
    Carga un archivo Parquet de ventas, lo filtra por categoría y genera
    un conjunto de características con variables rezagadas y de calendario.

    Parameters
    filepath (str or Path): Ruta al archivo Parquet de entrada
    con los datos de ventas.
    category (str): Categoría de producto a filtrar.
    lags (list of int, optional): Lista de rezagos (lags) a generar
    para `product_quantity`. Por defecto: [1, 2, 7, 14].

    Returns
    df_no_quant (pd.DataFrame): DataFrame con variables de calendario
    y lags, excluyendo `product_quantity`.
    df_quant (pd.Series): Serie con la cantidad diaria de productos vendidos
    (`product_quantity`).
    """
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
    """Divide un conjunto de datos en subconjuntos de entrenamiento y prueba.

    La división se realiza de forma secuencial (sin mezcla de datos),
    respetando el orden temporal de los registros
    para evitar fuga de información.

    Parameters
    df_no_quant (pd.DataFrame): DataFrame con las variables explicativas
    (sin `product_quantity`).
    df_quant (pd.Series): Serie con la variable objetivo (`product_quantity`).
    train_size (float, optional): Proporción de los datos destinados
    a entrenamiento. Valor por defecto: 0.8 (80% train, 20% test).

    Returns
    X_train (pd.DataFrame): Subconjunto de entrenamiento de
    las variables explicativas.
    X_test (pd.DataFrame): Subconjunto de prueba de las variables explicativas.
    y_train (pd.Series): Subconjunto de entrenamiento de la variable objetivo.
    y_test (pd.Series): Subconjunto de prueba de la variable objetivo.
    """
    split_index = int(len(df_no_quant) * train_size)
    X_train, X_test = (
        df_no_quant.iloc[:split_index], df_no_quant.iloc[split_index:]
    )
    y_train, y_test = df_quant.iloc[:split_index], df_quant.iloc[split_index:]
    return X_train, X_test, y_train, y_test
