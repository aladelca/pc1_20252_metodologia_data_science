import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_predictions(actual, predicted, dates=None, title="Predicciones vs Valores Reales", figsize=(12, 6)):
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if dates is not None:
        x_axis = pd.to_datetime(dates)
    else:
        x_axis = range(len(actual))
    
    ax.plot(x_axis, actual, label='Valores Reales', marker='o', linewidth=2)
    ax.plot(x_axis, predicted, label='Predicciones', marker='s', linewidth=2, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Valor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if dates is not None:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def plot_time_series(data, date_col, value_col, title="Serie de Tiempo", figsize=(12, 6)):
    """Plot time series data."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])
    
    ax.plot(data[date_col], data[value_col], linewidth=2, marker='o', markersize=4)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Valor')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_model_comparison(metrics_df, metric='rmse', title=None, figsize=(10, 6)):
    """Plot comparison of model performance."""
    if title is None:
        title = f'Comparacion de Modelos - {metric.upper()}'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    model_means = metrics_df.groupby('model')[metric].mean().sort_values()
    
    bars = ax.bar(range(len(model_means)), model_means.values, alpha=0.7)
    
    for i, (model, value) in enumerate(model_means.items()):
        ax.text(i, value + value * 0.01, f'{value:.3f}', 
               ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Modelo')
    ax.set_ylabel(metric.upper())
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(model_means)))
    ax.set_xticklabels(model_means.index, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig