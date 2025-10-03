import pandas as pd
import os
from src.preprocess.aggregation import aggregate_by_day_and_category
from src.preprocess.cleaning import clean_for_day_and_category_aggregation
from src.preprocess.preprocessing import fill_missing_dates_with_zero

def preprocessing() -> str:
    """
    Genera los archivos de datos agrupados por día y categoría.
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