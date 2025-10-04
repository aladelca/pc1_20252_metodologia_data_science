import pandas as pd


def clean_for_day_and_category_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame para la agregación por día y categoría.

    Parameters:
    df (pd.DataFrame): DataFrame de entrada con columnas 'transaction_date',
    'product_category' y 'product_quantity'.

    Returns:
    pd.DataFrame: DataFrame limpio listo para la agregación.
    """
    # Asegurarse de que 'transaction_date' es de tipo datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    # Eliminar filas con valores nulos en 'product_quantity'
    df = df.dropna(subset=['product_quantity'])
    # Eliminar filas con valores nulos en 'product_category'
    df = df.dropna(subset=['product_category'])
    # Remover todas las filas cuyo valor no representa a una categoria válida
    df = df[df['product_category'] != '(not set)']
    df = df[df['product_category'] != '${productitem.product.origCatName}']
    # Convertir la columna 'product_category' a tipo categórica
    df['product_category'] = df['product_category'].astype('category')

    return df
