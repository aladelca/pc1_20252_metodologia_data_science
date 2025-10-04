# Pipeline de Predicción Simple - Grupo 2

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

from models.G2_LACC_simple_models import SimpleMovingAverage, LinearTrend, SeasonalNaive
from preprocess.aggregation import aggregate_daily_product_data
from preprocess.cleaning import clean_data


class SimplePredictionPipeline:
    """Pipeline simple de predicción para Grupo 2."""

    def __init__(self):
        self.models = {
            'moving_average': SimpleMovingAverage,
            'linear_trend': LinearTrend,
            'seasonal_naive': SeasonalNaive
        }

    def run_pipeline(self, data_path: str, forecast_days: int = 7):
        """Ejecutar pipeline de predicción."""
        print("=== PIPELINE DE PREDICCIÓN GRUPO 2 ===")

        # 1. Cargar y procesar datos
        print("Procesando datos...")
        df = pd.read_parquet(data_path)
        df_clean = clean_data(df)

        df_processed = aggregate_daily_product_data(
            df_clean,
            date_column='parsed_date',
            min_days=5,
            add_features=False
        )

        print(f"Datos procesados: {df_processed.shape}")

        # 2. Generar predicciones
        print(f"Generando predicciones para {forecast_days} días...")
        predictions = []

        # Primeros 10 productos
        for product in df_processed['product_id'].unique()[:10]:
            print(f"  Prediciendo: {product}")

            product_data = df_processed[df_processed['product_id'] == product]
            product_data = product_data.sort_values('parsed_date')

            if len(product_data) < 5:
                continue

            series = product_data['product_revenue_usd_sum']

            # Usar el mejor modelo (moving average por simplicidad)
            try:
                model = SimpleMovingAverage(window=min(7, len(series)//2))
                model.fit(series)

                forecast = model.predict(forecast_days)

                # Generar fechas futuras
                last_date = product_data['parsed_date'].max()
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )

                for i, (date, value) in enumerate(zip(future_dates, forecast)):
                    predictions.append({
                        'product_id': product,
                        'forecast_date': date.strftime('%Y-%m-%d'),
                        'forecast_day': i + 1,
                        'predicted_value': float(value),
                        'model': 'moving_average'
                    })

                print(f"    ✅ Predicción exitosa")

            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue

        # 3. Guardar predicciones
        output_dir = Path("results/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_file = output_dir / f"simple_forecasts_{timestamp}.csv"
        df_pred = pd.DataFrame(predictions)
        df_pred.to_csv(csv_file, index=False)

        # Parquet
        parquet_file = output_dir / f"simple_forecasts_{timestamp}.parquet"
        df_pred.to_parquet(parquet_file, index=False)

        # Resumen JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'unique_products': len(df_pred['product_id'].unique()) if len(predictions) > 0 else 0,
            'forecast_horizon': forecast_days,
            'files': [str(csv_file), str(parquet_file)]
        }

        summary_file = output_dir / f"forecast_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nPredicciones guardadas:")
        print(f"  CSV: {csv_file}")
        print(f"  Parquet: {parquet_file}")
        print(f"  Resumen: {summary_file}")
        print("=== PIPELINE COMPLETADO ===")

        return predictions


def main():
    """Ejecutar pipeline de predicción."""
    pipeline = SimplePredictionPipeline()
    predictions = pipeline.run_pipeline(
        "data/raw/data_sample.parquet", forecast_days=7)
    return predictions


if __name__ == "__main__":
    main()
