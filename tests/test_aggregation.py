"""Tests UNITARIOS para aggregation.py"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from preprocess.aggregation import create_product_lag_features, create_time_features, prepare_modeling_data


class TestCreateProductLagFeatures:
    """Tests unitarios para create_product_lag_features"""
    
    def test_creates_lag_columns(self):
        """Test que crea las columnas de lag esperadas"""
        test_data = pd.DataFrame({
            'product_sku': ['P001'] * 5,
            'parsed_date': pd.date_range('2024-01-01', periods=5),
            'units_sold': [1, 2, 3, 4, 5],
            'is_completed_transaction': [True] * 5
        })
        
        result = create_product_lag_features(test_data)
        
        expected_columns = [
            'product_units_lag_1', 'product_units_lag_3', 
            'product_units_lag_7', 'product_units_lag_14'
        ]
        
        for col in expected_columns:
            assert col in result.columns
    
    def test_lag_values_are_correct(self):
        """Test que los valores de lag son correctos"""
        test_data = pd.DataFrame({
            'product_sku': ['P001'] * 3,
            'parsed_date': pd.date_range('2024-01-01', periods=3),
            'units_sold': [10, 20, 30],
            'is_completed_transaction': [True] * 3
        })
        
        result = create_product_lag_features(test_data)
        
        # Verificar valores específicos
        assert pd.isna(result['product_units_lag_1'].iloc[0])  # Primer valor NaN
        assert result['product_units_lag_1'].iloc[1] == 10     # Lag 1 del segundo registro
        assert result['product_units_lag_1'].iloc[2] == 20     # Lag 1 del tercer registro
    
    def test_lags_are_product_specific(self):
        """Test que los lags son específicos por producto"""
        test_data = pd.DataFrame({
            'product_sku': ['P001'] * 2 + ['P002'] * 2,
            'parsed_date': ['2024-01-01', '2024-01-02'] * 2,
            'units_sold': [1, 2, 10, 20],  # P002 tiene valores diferentes
            'is_completed_transaction': [True] * 4
        })
        
        result = create_product_lag_features(test_data)
        
        p001_data = result[result['product_sku'] == 'P001']
        p002_data = result[result['product_sku'] == 'P002']
        
        # Los lags deben ser independientes por producto
        assert p001_data['product_units_lag_1'].iloc[1] == 1   # Lag de P001
        assert p002_data['product_units_lag_1'].iloc[1] == 10  # Lag de P002


class TestCreateTimeFeatures:
    """Tests unitarios para create_time_features"""
    
    def test_creates_time_features(self):
        """Test que crea características temporales"""
        test_data = pd.DataFrame({
            'parsed_date': pd.to_datetime(['2024-01-01', '2024-01-15'])
        })
        
        result = create_time_features(test_data)
        
        expected_features = [
            'year', 'month', 'day', 'dayofweek', 'is_weekend',
            'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_time_feature_values(self):
        """Test que los valores de características temporales son correctos"""
        test_data = pd.DataFrame({
            'parsed_date': pd.to_datetime(['2024-01-01'])  # Lunes
        })
        
        result = create_time_features(test_data)
        
        assert result['year'].iloc[0] == 2024
        assert result['month'].iloc[0] == 1
        assert result['day'].iloc[0] == 1
        assert result['dayofweek'].iloc[0] == 0  # Lunes = 0
        assert result['is_weekend'].iloc[0] == 0  # Lunes no es fin de semana


class TestPrepareModelingData:
    """Tests unitarios para prepare_modeling_data"""
    
    def test_filters_completed_transactions(self):
        """Test que filtra solo transacciones completadas"""
        test_data = pd.DataFrame({
            'transaction_id': ['T001', 'T002'],
            'is_completed_transaction': [True, False],
            'units_sold': [1, 0],
            'product_units_lag_1': [0, 0],
            'year': [2024, 2024]
        })
        
        result = prepare_modeling_data(test_data)
        
        assert len(result) == 1
        assert result['transaction_id'].iloc[0] == 'T001'
    
    def test_removes_rows_with_null_features(self):
        """Test que elimina filas con features nulos"""
        test_data = pd.DataFrame({
            'transaction_id': ['T001', 'T002'],
            'is_completed_transaction': [True, True],
            'units_sold': [1, 2],
            'product_units_lag_1': [0, None],  # T002 tiene nulo
            'year': [2024, 2024]
        })
        
        result = prepare_modeling_data(test_data)
        
        assert len(result) == 1
        assert result['transaction_id'].iloc[0] == 'T001'
    
    def test_includes_required_columns(self):
        """Test que incluye todas las columnas requeridas"""
        test_data = pd.DataFrame({
            'transaction_id': ['T001'],
            'product_sku': ['P001'],
            'parsed_date': ['2024-01-01'],
            'product_name': ['Test Product'],
            'is_completed_transaction': [True],
            'units_sold': [1],
            'product_units_lag_1': [0],
            'year': [2024],
            'month': [1]
        })
        
        result = prepare_modeling_data(test_data)
        
        required_columns = [
            'transaction_id', 'product_sku', 'parsed_date', 'product_name',
            'units_sold', 'product_units_lag_1', 'year', 'month'
        ]
        
        for col in required_columns:
            assert col in result.columns