import pandas as pd
import os

def calculate_day_category_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el total de 'product_quantity' por 'transaction_date' y 'product_category'.

    Parameters:
    df (pd.DataFrame): DataFrame de entrada con columnas 'transaction_date', 'product_category' y 'product_quantity'.

    Returns:
    pd.DataFrame: DataFrame agregado con columnas 'transaction_date', 'product_category' y la sumatoria de 'product_quantity'.
    """
    # Asegurarse de que 'transaction_date' es de tipo datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    # Eliminar filas con valores nulos en 'product_quantity'
    df = df.dropna(subset=['product_quantity'])
    # Remover todas las filas cuyo 'product_category' es '(not set)' o '${productitem.product.origCatName}'
    df = df[df['product_category'] != '(not set)']
    df = df[df['product_category'] != '${productitem.product.origCatName}']

    # Agrupar por 'transaction_date' y 'product_category', sumando 'product_quantity'
    aggregated_df = df.groupby(['transaction_date', 'product_category'], as_index=False)['product_quantity'].sum().reset_index()

    return aggregated_df


def generate_day_category_grouped_data_files() -> str:
    """
    Genera los archivos de datos agrupados por día y categoría.
    """
    # Crear el directorio si no existe
    if not os.path.exists('../../data/aggregated'):
        os.makedirs('../../data/aggregated')

    # Leer el archivo Parquet
    data = pd.read_parquet('../../data/raw/data_sample.parquet')
    # Agrupar y agregar los datos
    grouped_data = calculate_day_category_aggregation(data)
    # Guardar el DataFrame agrupado en un nuevo archivo Parquet
    grouped_data.to_parquet('../../data/aggregated/day_category.parquet')

    return "success"


generate_day_category_grouped_data_files()