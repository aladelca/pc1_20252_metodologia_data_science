"""Tests UNITARIOS para cleaning.py"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from preprocess.cleaning import clean_data


class TestCleanData:
    """Tests unitarios para la función clean_data"""
    
    def test_removes_nulls_in_critical_columns(self):
        """Test que clean_data elimina filas con nulos en columnas críticas"""
        test_data = pd.DataFrame({
            'transaction_id': ['T001', 'T002', None, 'T004'],
            'product_sku': ['P001', None, 'P003', 'P004'],
            'parsed_date': ['2024-01-01', '2024-01-02', '2024-01-03', None],
            'product_quantity': [1, 2, 0, 1]
        })
        
        result = clean_data(test_data)
        
        assert result['transaction_id'].notna().all()
        assert result['product_sku'].notna().all()
        assert result['parsed_date'].notna().all()
        assert len(result) == 1
    
    def test_units_sold_non_negative(self):
        """Test que units_sold no tiene valores negativos"""
        test_data = pd.DataFrame({
            'transaction_id': ['T001', 'T002'],
            'product_sku': ['P001', 'P002'],
            'parsed_date': ['2024-01-01', '2024-01-02'],
            'product_quantity': [2, -1]
        })
        
        result = clean_data(test_data)
        
        assert (result['units_sold'] >= 0).all()
        assert result['units_sold'].tolist() == [2, 0]
    
    def test_removes_duplicates(self):
        """Test que elimina duplicados en granularidad"""
        test_data = pd.DataFrame({
            'transaction_id': ['T001', 'T001', 'T002'],
            'product_sku': ['P001', 'P001', 'P002'],
            'parsed_date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'product_quantity': [1, 1, 2]
        })
        
        result = clean_data(test_data)
        
        assert len(result) == 2
        assert set(result['transaction_id']) == {'T001', 'T002'}
    
    def test_creates_required_columns(self):
        """Test que crea las columnas units_sold y is_completed_transaction"""
        test_data = pd.DataFrame({
            'transaction_id': ['T001'],
            'product_sku': ['P001'],
            'parsed_date': ['2024-01-01'],
            'product_quantity': [1]
        })
        
        result = clean_data(test_data)
        
        assert 'units_sold' in result.columns
        assert 'is_completed_transaction' in result.columns
        assert result['is_completed_transaction'].iloc[0] == True
    
    def test_converts_parsed_date_to_datetime(self):
        """Test que convierte parsed_date a datetime"""
        test_data = pd.DataFrame({
            'transaction_id': ['T001'],
            'product_sku': ['P001'],
            'parsed_date': ['2024-01-01'],
            'product_quantity': [1]
        })
        
        result = clean_data(test_data)
        
        assert pd.api.types.is_datetime64_any_dtype(result['parsed_date'])