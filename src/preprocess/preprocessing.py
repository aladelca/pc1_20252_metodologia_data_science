import pandas as pd
import os
from src.preprocess.aggregation import aggregate_by_day_and_category
from src.preprocess.cleaning import clean_for_day_and_category_aggregation


def fill_missing_dates_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena fechas faltantes con ceros en la serie temporal de cada categoría.

    La función asegura que para cada `product_category` se complete el rango
    de fechas activo (desde la primera hasta la última fecha observada) con
    frecuencia diaria. Si en un día no existen registros, se asigna un valor
    de `0` en `product_quantity`. También se normalizan las fechas y se
    garantiza que `product_quantity` sea de tipo numérico.

    Parameters
    df (pd.DataFrame): DataFrame de entrada con las columnas:
    - `transaction_date` (datetime-like): fechas de transacciones.
    - `product_category` (str o category): categoría del producto.
    - `product_quantity` (numérico): cantidad vendida.

    Returns
    pd.DataFrame
    DataFrame resultante con:
    - Fechas continuas por categoría (sin huecos).
    - Columna `product_quantity` rellenada con `0`
    donde no había registros.
    - Tipos consistentes (`datetime64` para fechas,
    `float` para cantidades).
    """

    # Normalizar a fecha (sin hora) y asegurar tipo numérico
    df['transaction_date'] = (
        df['transaction_date'].dt.normalize()  # type: ignore[attr-defined]
      )
    df['product_quantity'] = pd.to_numeric(
        df['product_quantity'], errors='coerce'
    ).fillna(0)

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
        df = (
            df[['transaction_date', 'product_category', 'product_quantity']]
            .copy()
            )

    # Tipos finales y orden
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    # Si es de tipo category originalmente, preservarlo
    try:
        original_dtype = df['product_category'].dtype
        df['product_category'] = df['product_category'].astype(original_dtype)
    except Exception:
        pass

    df['product_quantity'] = pd.to_numeric(
        df['product_quantity'], errors='coerce'
        ).fillna(0)
    df = df.sort_values(['product_category', 'transaction_date'])

    return df


def preprocessing() -> str:
    """
    Preprocesa los datos de ventas y genera archivos agregados
    por día y categoría.

    Parameters
    None

    Returns
    str: Cadena de confirmación. Retorna `"success"`
    si el proceso se ejecuta correctamente.
    """
    # Crear el directorio si no existe
    if not os.path.exists('data/aggregated'):
        os.makedirs('data/aggregated')

    # Leer el archivo Parquet
    df = pd.read_parquet('data/raw/data_sample.parquet')
    # Limpiar los datos
    df = clean_for_day_and_category_aggregation(df)
    # Agrupar y agregar los datos
    grouped_data = aggregate_by_day_and_category(df)
    # Llenar los dias que no se tengan ventas con 0, pero solo dentro
    # del rango de fechas valido para cada categoria
    grouped_data = fill_missing_dates_with_zero(grouped_data)
    # Guardar el DataFrame agrupado en un nuevo archivo Parquet
    grouped_data.to_parquet('data/aggregated/day_category.parquet')

    return "success"


if __name__ == "__main__":
    stat_preproc = preprocessing()
