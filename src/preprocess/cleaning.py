
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los datos de la transacción sin procesar.

    Args:
        df: La raw data de pandas DataFrame.

    Returns:
        La data limpia.
    """
    # Replasamos los valores del placeholder
    df.replace(['(not set)', 'not available in demo dataset', '(none)'], pd.NA, inplace=True)

    # Convertir date column
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%Y%m%d')

    # Convertir columnas numéricas e imputar NaNs
    numeric_cols = [
        'transaction_revenue_usd', 'transaction_tax_usd', 'transaction_shipping_usd',
        'product_quantity', 'product_price_usd', 'product_revenue_usd',
        'total_visits', 'total_hits', 'total_pageviews', 'time_on_site_seconds'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Imputar NaNs en columnas categóricas
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('unknown')

    return df
