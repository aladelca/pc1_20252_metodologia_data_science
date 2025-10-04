import pandas as pd


def aggregate_by_day_and_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el total de 'product_quantity' por 'transaction_date' y
    'product_category'.
    Se espera que la data este limpia y lista para agrupación.

    Parameters:
    df (pd.DataFrame): DataFrame de entrada con columnas 'transaction_date',
    'product_category' y 'product_quantity'.

    Returns:
    pd.DataFrame: DataFrame agregado con columnas 'transaction_date',
    'product_category' y la sumatoria de 'product_quantity'.
    """
    # Agrupar por 'transaction_date' y 'product_category',
    # sumando 'product_quantity'
    aggregated_df = (df.groupby(['transaction_date', 'product_category'],
                                as_index=False,
                                observed=True)['product_quantity']
                     .sum().reset_index())

    return aggregated_df
