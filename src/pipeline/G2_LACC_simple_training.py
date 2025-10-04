# Pipeline de Entrenamiento Simple - Grupo 2

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.G2_LACC_simple_models import SimpleMovingAverage, LinearTrend, SeasonalNaive
from preprocess.aggregation import aggregate_daily_product_data
from preprocess.cleaning import clean_data


class SimpleTrainingPipeline:
    """Pipeline simple de entrenamiento para Grupo 2."""

    def __init__(self, output_dir: str = "models/trained"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models = {
            'moving_average': SimpleMovingAverage,
            'linear_trend': LinearTrend,
            'seasonal_naive': SeasonalNaive
        }

    def run_pipeline(self, data_path: str):
        """Ejecutar pipeline completo."""
        print("=== PIPELINE DE ENTRENAMIENTO GRUPO 2 ===")

        # 1. Cargar datos
        print("Cargando datos...")
        df = pd.read_parquet(data_path)
        df_clean = clean_data(df)

        # 2. Agregar por día/producto
        print("Agregando datos...")
        df_processed = aggregate_daily_product_data(
            df_clean,
            date_column='parsed_date',
            min_days=5,  # Mínimo 5 días para test
            add_features=False
        )

        print(f"Datos procesados: {df_processed.shape}")
        print(f"Productos: {df_processed['product_id'].nunique()}")

        # 3. Entrenar modelos
        print("Entrenando modelos...")
        results = {}

        # Solo primeros 5 para test
        for product in df_processed['product_id'].unique()[:5]:
            print(f"  Producto: {product}")
            product_data = df_processed[df_processed['product_id'] == product]
            product_data = product_data.sort_values('parsed_date')

            if len(product_data) < 5:
                continue

            # Dividir datos
            train_size = int(len(product_data) * 0.8)
            train_data = product_data.iloc[:train_size]['product_revenue_usd_sum']
            test_data = product_data.iloc[train_size:]['product_revenue_usd_sum']

            results[product] = {}

            # Entrenar cada modelo
            for model_name, model_class in self.models.items():
                try:
                    # Configurar parámetros
                    if model_name == 'moving_average':
                        model = model_class(window=min(7, len(train_data)//2))
                    elif model_name == 'seasonal_naive':
                        model = model_class(
                            season_length=min(7, len(train_data)//2))
                    else:
                        model = model_class()

                    # Entrenar
                    model.fit(train_data)

                    # Evaluar si hay datos de test
                    if len(test_data) > 0:
                        predictions = model.predict(len(test_data))
                        mae = mean_absolute_error(
                            test_data.values, predictions)
                        rmse = np.sqrt(mean_squared_error(
                            test_data.values, predictions))
                    else:
                        mae = 0.0
                        rmse = 0.0

                    results[product][model_name] = {
                        'mae': mae,
                        'rmse': rmse,
                        'success': True
                    }

                    print(f"    {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}")

                except Exception as e:
                    print(f"    {model_name}: Error - {e}")
                    results[product][model_name] = {
                        'success': False, 'error': str(e)}

        # 4. Guardar resultados
        output_file = self.output_dir / "simple_results.json"
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'group': 'Grupo_2',
            'products': results,
            'summary': {
                'total_products': len(results),
                'data_shape': df_processed.shape
            }
        }

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\nResultados guardados en: {output_file}")
        print("=== PIPELINE COMPLETADO ===")

        return results


def main():
    """Ejecutar pipeline."""
    pipeline = SimpleTrainingPipeline()
    results = pipeline.run_pipeline("data/raw/data_sample.parquet")
    return results


if __name__ == "__main__":
    main()
