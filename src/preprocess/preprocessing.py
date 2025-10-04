
import pandas as pd
from pathlib import Path
import sys

# Agregar el directorio raíz del proyecto al sys.path para importar módulos
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.preprocess.cleaning import clean_data
from src.preprocess.aggregation import aggregate_data_by_product

def preprocess_for_group2(input_path: str, output_path: str):
    """
    Preprocesamiento pipeline para Grupo 2.
    Lee los datos sin procesar, los limpia, los agrega por fecha y producto, y guarda el resultado.

    Args:
        input_path: Path del raw parquet file.
        output_path: Path para guardar el parquet file procesado.
    """
    print("Empezando el preprocesamiento...")
    
    # Read data
    print(f"Leyendo la data de {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Clean data
    print("Limpiando la data...")
    df_cleaned = clean_data(df)
    
    # Aggregate data
    print("Se agrega la data por fecha y producto...")
    df_aggregated = aggregate_data_by_product(df_cleaned)
    
    # Save processed data
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Guardando la data en {output_path}...")
    df_aggregated.to_parquet(output_path, index=False)
    
    print("Preprocesamiento completado!")
    return df_aggregated

if __name__ == '__main__':
    # Esto permite ejecutar el script directamente
    # Ejemplo desde el directorio raíz:
    # python src/preprocess/preprocessing.py data/raw/data_sample.parquet data/processed/group2_data.parquet
    import argparse

    parser = argparse.ArgumentParser(description='Run the Group 2 preprocessing pipeline.')
    parser.add_argument('input_path', type=str, help='Path to the raw input data file.')
    parser.add_argument('output_path', type=str, help='Path to save the processed output file.')

    args = parser.parse_args()

    preprocess_for_group2(args.input_path, args.output_path)
