"""
Pipeline principal de preprocessing para el Grupo 2 - Día/Producto.

Este módulo integra las funciones de limpieza y agregación para crear
un pipeline completo de preprocessing de datos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

from .cleaning import clean_data
from .aggregation import aggregate_daily_product_data

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Clase principal para el preprocessing de datos del Grupo 2.
    
    Maneja el pipeline completo de limpieza, agregación y preparación
    de datos para modelado de series de tiempo día/producto.
    """
    
    def __init__(self, 
                 date_column: str = 'date',
                 min_days_activity: int = 30,
                 min_transactions: int = 10):
        """
        Inicializa el preprocessor.
        
        Args:
            date_column: Nombre de la columna de fecha
            min_days_activity: Mínimo días de actividad para incluir producto
            min_transactions: Mínimo número de transacciones por producto
        """
        self.date_column = date_column
        self.min_days_activity = min_days_activity
        self.min_transactions = min_transactions
        self.is_fitted = False
        self.product_stats_ = None
        self.date_range_ = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga datos desde archivo parquet.
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            DataFrame con los datos crudos
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
            if file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Formato de archivo no soportado: {file_path.suffix}")
            
            logger.info(f"Datos cargados exitosamente. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
    
    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y valida los datos.
        
        Args:
            df: DataFrame con datos crudos
            
        Returns:
            DataFrame limpio
        """
        logger.info("Iniciando limpieza de datos")
        
        # Aplicar limpieza
        df_clean = clean_data(
            df=df,
            date_column=self.date_column,
            revenue_strategy='convert',
            missing_strategy='fill'
        )
        
        # Validaciones adicionales
        if df_clean.empty:
            raise ValueError("DataFrame quedó vacío después de la limpieza")
        
        if self.date_column not in df_clean.columns:
            raise ValueError(f"Columna de fecha '{self.date_column}' no encontrada")
        
        logger.info(f"Limpieza completada. Shape: {df_clean.shape}")
        return df_clean
    
    def aggregate_data(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega datos por día y producto.
        
        Args:
            df_clean: DataFrame limpio
            
        Returns:
            DataFrame agregado
        """
        logger.info("Iniciando agregación por día/producto")
        
        df_aggregated = aggregate_daily_product_data(
            df=df_clean,
            date_column=self.date_column,
            min_days=self.min_days_activity,
            add_features=True
        )
        
        if df_aggregated.empty:
            raise ValueError("No hay datos después de la agregación")
        
        # Guardar estadísticas
        self.product_stats_ = {
            'total_products': df_aggregated['product_id'].nunique(),
            'total_days': df_aggregated[self.date_column].nunique(),
            'date_range': (df_aggregated[self.date_column].min(), 
                          df_aggregated[self.date_column].max())
        }
        
        self.date_range_ = self.product_stats_['date_range']
        
        logger.info(f"Agregación completada. Productos: {self.product_stats_['total_products']}")
        return df_aggregated
    
    def create_time_series_format(self, df_aggregated: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Convierte datos agregados a formato de series de tiempo por producto.
        
        Args:
            df_aggregated: DataFrame agregado
            
        Returns:
            Diccionario con series de tiempo por producto
        """
        logger.info("Creando formato de series de tiempo")
        
        time_series_data = {}
        
        # Crear serie de tiempo para cada producto
        for product_id in df_aggregated['product_id'].unique():
            product_data = df_aggregated[
                df_aggregated['product_id'] == product_id
            ].copy()
            
            # Ordenar por fecha
            product_data = product_data.sort_values(self.date_column)
            
            # Establecer fecha como índice
            product_data = product_data.set_index(self.date_column)
            
            # Eliminar columna product_id ya que está implícita
            if 'product_id' in product_data.columns:
                product_data = product_data.drop('product_id', axis=1)
            
            time_series_data[product_id] = product_data
        
        logger.info(f"Series de tiempo creadas para {len(time_series_data)} productos")
        return time_series_data
    
    def save_processed_data(self, 
                          df_aggregated: pd.DataFrame,
                          output_dir: str = 'data/processed',
                          save_individual_series: bool = True) -> None:
        """
        Guarda los datos procesados.
        
        Args:
            df_aggregated: DataFrame agregado
            output_dir: Directorio de salida
            save_individual_series: Si guardar series individuales por producto
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar dataset agregado completo
        aggregated_file = output_path / 'daily_product_aggregated.parquet'
        df_aggregated.to_parquet(aggregated_file, index=False)
        logger.info(f"Dataset agregado guardado en: {aggregated_file}")
        
        # Guardar estadísticas
        stats_file = output_path / 'preprocessing_stats.json'
        import json
        with open(stats_file, 'w') as f:
            json.dump(self.product_stats_, f, indent=2, default=str)
        
        if save_individual_series:
            # Crear directorio para series individuales
            series_dir = output_path / 'individual_series'
            series_dir.mkdir(exist_ok=True)
            
            time_series_data = self.create_time_series_format(df_aggregated)
            
            for product_id, product_data in time_series_data.items():
                # Limpiar nombre de archivo
                safe_product_id = str(product_id).replace('/', '_').replace('\\', '_')
                series_file = series_dir / f'{safe_product_id}.parquet'
                product_data.to_parquet(series_file)
            
            logger.info(f"Series individuales guardadas en: {series_dir}")
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta el preprocessor y transforma los datos.
        
        Args:
            df: DataFrame con datos crudos
            
        Returns:
            DataFrame procesado y agregado
        """
        # Limpiar datos
        df_clean = self.clean_and_validate(df)
        
        # Agregar datos
        df_aggregated = self.aggregate_data(df_clean)
        
        self.is_fitted = True
        
        return df_aggregated
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma nuevos datos usando el preprocessor ajustado.
        
        Args:
            df: DataFrame con datos crudos
            
        Returns:
            DataFrame procesado
        """
        if not self.is_fitted:
            raise ValueError("El preprocessor debe ser ajustado primero con fit_transform()")
        
        # Aplicar las mismas transformaciones
        df_clean = self.clean_and_validate(df)
        df_aggregated = self.aggregate_data(df_clean)
        
        return df_aggregated


def preprocess_data(input_file: str,
                   output_dir: str = 'data/processed',
                   date_column: str = 'date',
                   min_days_activity: int = 30) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """
    Función de conveniencia para preprocessing completo.
    
    Args:
        input_file: Archivo de entrada con datos crudos
        output_dir: Directorio de salida para datos procesados
        date_column: Nombre de la columna de fecha
        min_days_activity: Mínimo días de actividad por producto
        
    Returns:
        Tupla con (DataFrame procesado, instancia del preprocessor)
    """
    # Crear preprocessor
    preprocessor = DataPreprocessor(
        date_column=date_column,
        min_days_activity=min_days_activity
    )
    
    # Cargar datos
    df_raw = preprocessor.load_data(input_file)
    
    # Procesar datos
    df_processed = preprocessor.fit_transform(df_raw)
    
    # Guardar resultados
    preprocessor.save_processed_data(df_processed, output_dir)
    
    return df_processed, preprocessor


if __name__ == "__main__":
    # Ejemplo de uso
    input_file = "data/raw/data_sample.parquet"
    output_dir = "data/processed"
    
    try:
        df_processed, preprocessor = preprocess_data(
            input_file=input_file,
            output_dir=output_dir
        )
        
        print(f"Preprocessing completado exitosamente!")
        print(f"Shape final: {df_processed.shape}")
        print(f"Productos únicos: {df_processed['product_id'].nunique()}")
        
    except Exception as e:
        logger.error(f"Error en preprocessing: {e}")
        raise
