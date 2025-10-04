#!/usr/bin/env python3
"""
Script para ejecutar el pipeline de predicción
"""

from pipeline.G2_LACC_simple_prediction import SimplePredictionPipeline
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / 'src'))


def main():
    """Ejecutar pipeline de predicción."""

    print("EJECUTANDO PIPELINE DE PREDICCIÓN")
    print("=" * 60)

    # Configuración
    data_path = "data/raw/data_sample.parquet"
    forecast_horizon = 7  # Predecir 7 días

    # Verificar que existen los datos
    if not Path(data_path).exists():
        print(f"Error: No se encontró el archivo de datos: {data_path}")
        return 1

    try:
        # Crear y ejecutar pipeline
        pipeline = SimplePredictionPipeline()
        predictions = pipeline.run_pipeline(
            data_path, forecast_days=forecast_horizon)

        print("\nPIPELINE DE PREDICCIÓN COMPLETADO EXITOSAMENTE")
        print("=" * 60)

        # Mostrar resumen
        total_predictions = len(predictions)
        unique_products = len(set(p['product_id']
                              for p in predictions)) if predictions else 0

        print(f"Predicciones generadas: {total_predictions}")
        print(f"Productos únicos: {unique_products}")
        print(f"Horizonte de predicción: {forecast_horizon} días")
        print(f"Archivos en: results/predictions/")

        return 0

    except Exception as e:
        print(f"\nERROR EN EL PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
