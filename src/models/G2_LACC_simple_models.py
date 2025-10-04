# Modelos simples para Grupo 2 - Estudiante

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SimpleMovingAverage:
    """Modelo 1: Promedio móvil simple."""

    def __init__(self, window=7):
        self.window = window
        self.name = "Moving_Average"
        self.data = None

    def fit(self, series):
        self.data = series.values
        return self

    def predict(self, periods):
        if self.data is None:
            return np.zeros(periods)

        # Usar últimos valores para predecir
        recent = self.data[-self.window:]
        avg = np.mean(recent)

        # Predicción simple: repetir promedio
        return np.full(periods, avg)


class LinearTrend:
    """Modelo 2: Tendencia lineal simple."""

    def __init__(self):
        self.name = "Linear_Trend"
        self.slope = 0
        self.intercept = 0
        self.last_value = 0

    def fit(self, series):
        data = series.values
        if len(data) < 2:
            self.slope = 0
            self.intercept = data[0] if len(data) > 0 else 0
        else:
            # Calcular tendencia simple
            x = np.arange(len(data))
            self.slope = (data[-1] - data[0]) / (len(data) - 1)
            self.intercept = data[0]
            self.last_value = data[-1]
        return self

    def predict(self, periods):
        predictions = []
        for i in range(periods):
            pred = self.last_value + (self.slope * (i + 1))
            predictions.append(max(0, pred))  # No negativos
        return np.array(predictions)


class SeasonalNaive:
    """Modelo 3: Naive estacional simple."""

    def __init__(self, season_length=7):
        self.season_length = season_length
        self.name = "Seasonal_Naive"
        self.seasonal_pattern = None

    def fit(self, series):
        data = series.values
        if len(data) < self.season_length:
            self.seasonal_pattern = data
        else:
            # Obtener patrón estacional (últimos 7 días)
            self.seasonal_pattern = data[-self.season_length:]
        return self

    def predict(self, periods):
        if self.seasonal_pattern is None:
            return np.zeros(periods)

        # Repetir patrón estacional
        predictions = []
        for i in range(periods):
            idx = i % len(self.seasonal_pattern)
            predictions.append(self.seasonal_pattern[idx])

        return np.array(predictions)


def train_and_evaluate_models(series, test_size=0.2):
    """Entrenar y evaluar los 3 modelos."""

    # Dividir datos
    split_point = int(len(series) * (1 - test_size))
    train_data = series[:split_point]
    test_data = series[split_point:]

    if len(train_data) < 7 or len(test_data) < 1:
        return None

    # Crear modelos
    models = [
        SimpleMovingAverage(window=7),
        LinearTrend(),
        SeasonalNaive(season_length=7)
    ]

    results = {}

    for model in models:
        try:
            # Entrenar
            model.fit(train_data)

            # Predecir
            predictions = model.predict(len(test_data))

            # Evaluar
            mae = mean_absolute_error(test_data.values, predictions)
            rmse = np.sqrt(mean_squared_error(test_data.values, predictions))

            results[model.name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'predictions': predictions
            }

        except Exception as e:
            print(f"Error en {model.name}: {e}")
            continue

    return results


def select_best_model(results):
    """Seleccionar mejor modelo basado en MAE."""
    if not results:
        return None

    best_model = min(results.keys(), key=lambda x: results[x]['mae'])
    return best_model, results[best_model]
