"""Tests UNITARIOS para preprocessing.py"""

import pytest
import pandas as pd
import tempfile
import os
import sys

sys.path.append(os.path.join(os.path.dirname.__file__, '../src'))

from preprocess.preprocessing import load_data, save_processed_data


class TestLoadData:
    """Tests unitarios para load_data"""
    
    def test_load_data_success(self):
        """Test que load_data carga correctamente un archivo parquet"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            test_data = pd.DataFrame({
                'transaction_id': ['T001', 'T002'],
                'product_sku': ['P001', 'P002']
            })
            test_data.to_parquet(tmp_file.name, index=False)
            
            try:
                result = load_data(tmp_file.name)
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 2
                assert list(result.columns) == ['transaction_id', 'product_sku']
            finally:
                os.unlink(tmp_file.name)
    
    def test_load_data_nonexistent_file(self):
        """Test que load_data lanza error con archivo inexistente"""
        with pytest.raises(Exception):
            load_data('nonexistent_file.parquet')


class TestSaveProcessedData:
    """Tests unitarios para save_processed_data"""
    
    def test_save_processed_data_creates_file(self):
        """Test que save_processed_data crea el archivo correctamente"""
        test_data = pd.DataFrame({
            'transaction_id': ['T001'],
            'units_sold': [1]
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, 'test_output.parquet')
            
            save_processed_data(test_data, output_path)
            
            assert os.path.exists(output_path)
            
            # Verificar que los datos se guardaron correctamente
            loaded_data = pd.read_parquet(output_path)
            assert len(loaded_data) == 1
            assert loaded_data['transaction_id'].iloc[0] == 'T001'
    
    def test_save_processed_data_creates_directory(self):
        """Test que save_processed_data crea el directorio si no existe"""
        test_data = pd.DataFrame({'test': [1]})
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, 'new_directory', 'output.parquet')
            
            save_processed_data(test_data, output_path)
            
            assert os.path.exists(output_path)