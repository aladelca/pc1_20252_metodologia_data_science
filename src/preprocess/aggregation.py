
import pandas as pd

def aggregate_data_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los datos de la transacción por fecha y producto, según lo requerido para el Grupo 2.

    Args:
        df: Los datos de la transacción limpios como el DataFrame de Pandas.

    Returns:
        DataFrame con datos agregados por fecha y producto.
    """
    # Define las reglas de agregación
    aggregation_rules = {
        'product_quantity': 'sum',
        'product_revenue_usd': 'sum',
        'product_price_usd': 'mean',
        'transaction_id': 'nunique'
    }

    # Agrupar por fecha y SKU del producto
    df_agg = df.groupby(['transaction_date', 'product_sku']).agg(aggregation_rules).reset_index()

    # Cambiar el nombre de las columnas para mayor claridad
    df_agg.rename(columns={
        'product_quantity': 'total_product_quantity',
        'product_revenue_usd': 'total_product_revenue_usd',
        'product_price_usd': 'avg_product_price_usd',
        'transaction_id': 'total_transactions'
    }, inplace=True)

    return df_agg
