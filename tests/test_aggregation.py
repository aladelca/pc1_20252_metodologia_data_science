# GROUP04
# tests/test_aggregation.py - Tests unitarios para aggregation.py
import os
import sys
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import pytest

# --- asegurar que src/ esté en el path ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from preprocess.aggregation import (
    calculate_day_brand_aggregation,
    generate_day_brand_grouped_data_files,
)


# ---------------------
# Fixtures
# ---------------------
@pytest.fixture
def tmp_aggregation_paths(tmp_path, request):
    """Rutas temporales para pruebas de aggregation."""
    base = tmp_path / "data" / "raw"
    base.mkdir(parents=True, exist_ok=True)
    
    uid = uuid.uuid4().hex[:8]
    testname = request.node.name.replace(os.sep, "_")
    
    input_file = base / f"{testname}_input_{uid}.parquet"
    output_file = base / f"{testname}_output_{uid}.parquet"
    
    return {"base": base, "input": input_file, "output": output_file}


@pytest.fixture
def synthetic_transaction_data():
    """Dataset sintético con múltiples transacciones por día/marca."""
    dates = pd.date_range("2016-08-01", periods=10, freq="D")
    rows = []
    
    for d in dates:
        # BrandA con 3 transacciones por día
        rows += [{"transaction_date": d, "product_brand": "BrandA", "product_quantity": 10}] * 3
        # BrandB con 2 transacciones por día
        rows += [{"transaction_date": d, "product_brand": "BrandB", "product_quantity": 20}] * 2
        # (not set) con 5 transacciones por día
        rows += [{"transaction_date": d, "product_brand": "(not set)", "product_quantity": 5}] * 5
    
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_data_with_nulls():
    """Dataset con valores nulos para testing."""
    data = {
        "transaction_date": ["2016-08-01", "2016-08-01", "2016-08-02", None, "2016-08-03"],
        "product_brand": ["BrandA", "BrandA", "BrandB", "BrandC", None],
        "product_quantity": [10, 20, 30, 40, 50],
    }
    return pd.DataFrame(data)


# ---------------------
# Tests unitarios
# ---------------------
def test_calculate_day_brand_aggregation_count_events(synthetic_transaction_data):
    """Test agregación por conteo de eventos (default)."""
    result = calculate_day_brand_aggregation(synthetic_transaction_data, metric_col=None)
    
    # Verificar estructura
    assert {"transaction_date", "product_brand", "events"} <= set(result.columns)
    
    # Verificar tipos
    assert pd.api.types.is_datetime64_any_dtype(result["transaction_date"])
    assert result["product_brand"].dtype == "object"
    
    # Verificar conteo correcto
    # BrandA tiene 3 eventos por día, 10 días = 10 registros de BrandA
    brand_a = result[result["product_brand"] == "BrandA"]
    assert len(brand_a) == 10
    assert brand_a["events"].iloc[0] == 3
    
    # BrandB tiene 2 eventos por día
    brand_b = result[result["product_brand"] == "BrandB"]
    assert len(brand_b) == 10
    assert brand_b["events"].iloc[0] == 2
    
    # (not set) tiene 5 eventos por día
    not_set = result[result["product_brand"] == "(not set)"]
    assert len(not_set) == 10
    assert not_set["events"].iloc[0] == 5


def test_calculate_day_brand_aggregation_sum_metric(synthetic_transaction_data):
    """Test agregación sumando una métrica específica."""
    result = calculate_day_brand_aggregation(
        synthetic_transaction_data,
        metric_col="product_quantity"
    )
    
    # Verificar estructura
    assert {"transaction_date", "product_brand", "sum_product_quantity"} <= set(result.columns)
    
    # Verificar suma correcta
    # BrandA: 3 transacciones * 10 quantity = 30 por día
    brand_a = result[result["product_brand"] == "BrandA"]
    assert brand_a["sum_product_quantity"].iloc[0] == 30
    
    # BrandB: 2 transacciones * 20 quantity = 40 por día
    brand_b = result[result["product_brand"] == "BrandB"]
    assert brand_b["sum_product_quantity"].iloc[0] == 40


def test_calculate_day_brand_aggregation_handles_nulls(synthetic_data_with_nulls):
    """Test que la agregación maneja correctamente valores nulos."""
    result = calculate_day_brand_aggregation(synthetic_data_with_nulls, metric_col=None)
    
    # Debe filtrar filas con transaction_date nulo
    assert len(result) <= len(synthetic_data_with_nulls)
    
    # No debe haber fechas nulas en el resultado
    assert result["transaction_date"].notna().all()
    
    # BrandA con fecha válida debe aparecer
    assert "BrandA" in result["product_brand"].values


def test_calculate_day_brand_aggregation_missing_brand_column():
    """Test que lanza error si falta columna product_brand."""
    data = pd.DataFrame({
        "transaction_date": ["2016-08-01", "2016-08-02"],
        "some_column": [1, 2],
    })
    
    with pytest.raises(ValueError, match="product_brand"):
        calculate_day_brand_aggregation(data, metric_col=None)


def test_calculate_day_brand_aggregation_timezone_aware():
    """Test que las fechas se convierten a timezone-aware (UTC)."""
    data = pd.DataFrame({
        "transaction_date": ["2016-08-01", "2016-08-02"],
        "product_brand": ["BrandA", "BrandA"],
    })
    
    result = calculate_day_brand_aggregation(data, metric_col=None)
    
    # Verificar que el índice es timezone-aware
    assert result["transaction_date"].dt.tz is not None
    assert str(result["transaction_date"].dt.tz) == "UTC"


def test_calculate_day_brand_aggregation_normalizes_dates():
    """Test que las fechas se normalizan a medianoche."""
    data = pd.DataFrame({
        "transaction_date": pd.to_datetime(["2016-08-01 14:30:00", "2016-08-01 18:45:00"], utc=True),
        "product_brand": ["BrandA", "BrandA"],
    })
    
    result = calculate_day_brand_aggregation(data, metric_col=None)
    
    # Ambas transacciones deben agruparse en el mismo día
    assert len(result) == 1
    assert result["events"].iloc[0] == 2
    
    # La fecha debe estar normalizada (sin hora)
    assert result["transaction_date"].iloc[0].hour == 0


def test_calculate_day_brand_aggregation_multiple_brands():
    """Test agregación con múltiples marcas."""
    data = pd.DataFrame({
        "transaction_date": ["2016-08-01"] * 10,
        "product_brand": ["BrandA"] * 3 + ["BrandB"] * 4 + ["BrandC"] * 3,
    })
    
    result = calculate_day_brand_aggregation(data, metric_col=None)
    
    # Debe haber 3 grupos (1 día, 3 marcas)
    assert len(result) == 3
    
    # Verificar conteos
    assert result[result["product_brand"] == "BrandA"]["events"].iloc[0] == 3
    assert result[result["product_brand"] == "BrandB"]["events"].iloc[0] == 4
    assert result[result["product_brand"] == "BrandC"]["events"].iloc[0] == 3


def test_generate_day_brand_grouped_data_files(tmp_aggregation_paths, synthetic_transaction_data):
    """Test generación de archivos agregados end-to-end."""
    # Guardar datos de entrada
    synthetic_transaction_data.to_parquet(tmp_aggregation_paths["input"], index=False)
    
    # Ejecutar agregación y guardado
    result_df = generate_day_brand_grouped_data_files(
        in_path=str(tmp_aggregation_paths["input"]),
        out_path=str(tmp_aggregation_paths["output"]),
        metric_col=None,
    )
    
    # Verificar que devuelve DataFrame
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0
    
    # Verificar que se guardó el archivo
    assert tmp_aggregation_paths["output"].exists()
    
    # Cargar archivo guardado y comparar
    saved_df = pd.read_parquet(tmp_aggregation_paths["output"])
    
    # Verificar estructura
    assert {"transaction_date", "product_brand", "events"} <= set(saved_df.columns)
    
    # Verificar que el DataFrame devuelto es igual al guardado
    pd.testing.assert_frame_equal(
        result_df.sort_values(["transaction_date", "product_brand"]).reset_index(drop=True),
        saved_df.sort_values(["transaction_date", "product_brand"]).reset_index(drop=True)
    )


def test_generate_day_brand_grouped_data_files_with_metric(tmp_aggregation_paths, synthetic_transaction_data):
    """Test generación con métrica específica."""
    synthetic_transaction_data.to_parquet(tmp_aggregation_paths["input"], index=False)
    
    result_df = generate_day_brand_grouped_data_files(
        in_path=str(tmp_aggregation_paths["input"]),
        out_path=str(tmp_aggregation_paths["output"]),
        metric_col="product_quantity",
    )
    
    # Verificar que tiene la columna de suma
    assert "sum_product_quantity" in result_df.columns
    
    # Verificar que el archivo existe
    assert tmp_aggregation_paths["output"].exists()


def test_generate_day_brand_grouped_data_creates_directory(tmp_path, synthetic_transaction_data):
    """Test que crea el directorio de salida si no existe."""
    input_file = tmp_path / "input.parquet"
    output_file = tmp_path / "nested" / "folder" / "output.parquet"
    
    synthetic_transaction_data.to_parquet(input_file, index=False)
    
    # El directorio nested/folder no existe
    assert not output_file.parent.exists()
    
    # Ejecutar
    result_df = generate_day_brand_grouped_data_files(
        in_path=str(input_file),
        out_path=str(output_file),
        metric_col=None,
    )
    
    # Verificar que se creó el directorio y el archivo
    assert output_file.exists()
    assert isinstance(result_df, pd.DataFrame)


def test_aggregation_preserves_not_set_brand(synthetic_transaction_data):
    """Test que (not set) se preserva correctamente."""
    result = calculate_day_brand_aggregation(synthetic_transaction_data, metric_col=None)
    
    # (not set) debe estar presente
    assert "(not set)" in result["product_brand"].values
    
    # Verificar conteo correcto para (not set)
    not_set = result[result["product_brand"] == "(not set)"]
    assert len(not_set) == 10  # 10 días
    assert not_set["events"].iloc[0] == 5  # 5 eventos por día


def test_aggregation_empty_dataframe():
    """Test comportamiento con DataFrame vacío."""
    data = pd.DataFrame(columns=["transaction_date", "product_brand"])
    
    result = calculate_day_brand_aggregation(data, metric_col=None)
    
    # Debe devolver DataFrame vacío pero con estructura correcta
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert {"transaction_date", "product_brand", "events"} <= set(result.columns)


def test_aggregation_single_row():
    """Test con una sola fila."""
    data = pd.DataFrame({
        "transaction_date": ["2016-08-01"],
        "product_brand": ["BrandA"],
    })
    
    result = calculate_day_brand_aggregation(data, metric_col=None)
    
    assert len(result) == 1
    assert result["events"].iloc[0] == 1


def test_aggregation_consistency_memory_vs_disk(tmp_aggregation_paths, synthetic_transaction_data):
    """Test que el cálculo en memoria es consistente con el guardado en disco."""
    synthetic_transaction_data.to_parquet(tmp_aggregation_paths["input"], index=False)
    
    # Cálculo en memoria
    df_input = pd.read_parquet(tmp_aggregation_paths["input"])
    mem_result = calculate_day_brand_aggregation(df_input, metric_col=None)
    
    # Cálculo y guardado
    disk_result = generate_day_brand_grouped_data_files(
        in_path=str(tmp_aggregation_paths["input"]),
        out_path=str(tmp_aggregation_paths["output"]),
        metric_col=None,
    )
    
    # Ambos deben ser idénticos
    cols = sorted({"transaction_date", "product_brand", "events"})
    pd.testing.assert_frame_equal(
        mem_result[cols].sort_values(cols).reset_index(drop=True),
        disk_result[cols].sort_values(cols).reset_index(drop=True)
    )


# ---------------------
# FIN GROUP 04 tests
# ---------------------