#!/usr/bin/env python3
"""
Script para ejecutar el pipeline de entrenamiento
"""

from pipeline.G2_LACC_simple_training import SimpleTrainingPipeline
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / 'src'))


def main():
    """Ejecutar pipeline de entrenamiento."""

    print("EJECUTANDO PIPELINE DE ENTRENAMIENTO")
    print("=" * 60)

    # Configuración
    data_path = "data/raw/data_sample.parquet"

    # Verificar que existen los datos
    if not Path(data_path).exists():
        print(f"Error: No se encontró el archivo de datos: {data_path}")
        return 1

    try:
        # Crear y ejecutar pipeline
        pipeline = SimpleTrainingPipeline()
        results = pipeline.run_pipeline(data_path)

        print("\nPIPELINE DE ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)

        # Mostrar resumen
        total_products = len(results)
        successful_trainings = sum(
            1 for product_models in results.values()
            for model_result in product_models.values()
            if model_result.get('success', False)
        )

        print(f"Productos procesados: {total_products}")
        print(f"Entrenamientos exitosos: {successful_trainings}")
        print(f"Resultados guardados en: models/trained/simple_results.json")

        return 0

    except Exception as e:
        print(f"\nERROR EN EL PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
