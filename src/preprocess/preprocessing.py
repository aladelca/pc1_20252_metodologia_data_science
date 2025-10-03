import pandas as pd

def fill_missing_dates_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    # Normalizar a fecha (sin hora) y asegurar tipo numérico
    df['transaction_date'] = df['transaction_date'].dt.normalize()  # type: ignore[attr-defined]
    df['product_quantity'] = pd.to_numeric(df['product_quantity'],
                                           errors='coerce').fillna(0)

    # Relleno por categoría en el rango activo
    filled_frames = []
    for category, group in df.groupby('product_category', observed=True):
        group = group.sort_values('transaction_date')
        start_date = group['transaction_date'].min()
        end_date = group['transaction_date'].max()
        if pd.isna(start_date) or pd.isna(end_date):
            continue

        full_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Crear serie con índice de fecha y reindexar con 0
        ts = group.set_index('transaction_date')['product_quantity']
        # Asegurar frecuencia diaria antes de reindexar (no introduce NaN aún)
        ts = ts.asfreq('D')
        ts = ts.reindex(full_range, fill_value=0)

        df_cat = pd.DataFrame({
            'transaction_date': ts.index,
            'product_category': category,
            'product_quantity': ts.astype(float).values  # mantener
                                                         # tipo consistente
        })
        filled_frames.append(df_cat)

    if filled_frames:
        df = pd.concat(filled_frames, ignore_index=True)
    else:
        # Si no hubo frames (caso borde), garantizar esquema vacío correcto
        df = df[['transaction_date', 'product_category',
                 'product_quantity']].copy()

    # Tipos finales y orden
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    # Si es de tipo category originalmente, preservarlo
    try:
        original_dtype = df['product_category'].dtype
        df['product_category'] = df['product_category'].astype(original_dtype)
    except Exception:
        pass

    df['product_quantity'] = pd.to_numeric(df['product_quantity'],
                                           errors='coerce').fillna(0)
    df = df.sort_values(['product_category', 'transaction_date'])

    return df

