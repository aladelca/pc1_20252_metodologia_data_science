import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """Calculate basic regression metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero in MAPE
    mape_mask = y_true != 0
    if np.any(mape_mask):
        mape = np.mean(
            np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
    else:
        mape = float('inf')

    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mape,
        'r2': r2_score(y_true, y_pred)
    }


def summarize_model_performance(results_dict):
    """Summarize model performance from results dictionary."""
    summary = {}
    for model_name, metrics in results_dict.items():
        if isinstance(metrics, dict) and 'metrics' in metrics:
            model_metrics = metrics['metrics']
            summary[model_name] = {
                'mae': model_metrics.get('mae', float('inf')),
                'rmse': model_metrics.get('rmse', float('inf')),
                'mape': model_metrics.get('mape', float('inf')),
                'r2': model_metrics.get('r2', -1)
            }
        elif isinstance(metrics, dict):
            summary[model_name] = {
                'mae': metrics.get('mae', float('inf')),
                'rmse': metrics.get('rmse', float('inf')),
                'mape': metrics.get('mape', float('inf')),
                'r2': metrics.get('r2', -1)
            }
    return summary
