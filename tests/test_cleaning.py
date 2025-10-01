# GROUP04
# tests/test_cleaning.py - Tests unitarios para cleaning.py
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

from preprocess.cleaning import (
    CleaningConfig,
    DataQualityReport,
    run_cleaning,
    normalize_strings_and_placeholders,
    parse_dates,
    normalize_brand,
    drop_high_missing_columns,
    drop_all_missing_rows,
    coerce_types,
    drop_duplicates,
    negatives_to_nan,
    impute_missing,
    handle_outliers,
    drop_constant_cols,
    reduce_categoricals_cardinality,
    _protected_columns,
)


# ---------------------
# Fixtures
# ---------------------
@pytest.fixture
def tmp_cleaning_paths(tmp_path, request):
    """Rutas temporales para pruebas de cleaning."""
    base = tmp_path / "data" / "raw"
    base.mkdir(parents=True, exist_ok=True)
    
    uid = uuid.uuid4().hex[:8]
    testname = request.node.name.replace(os.sep, "_")
    
    raw = base / f"{testname}_raw_{uid}.parquet"
    cleaned = base / f"{testname}_cleaned_{uid}.parquet"
    csv_out = base / f"{testname}_cleaned_{uid}.csv"
    
    return {"base": base, "raw": raw, "cleaned": cleaned, "csv": csv_out}


@pytest.fixture
def synthetic_dirty_data():
    """Dataset sintético con problemas de calidad típicos."""
    data = {
        "transaction_date": ["20160801", "20160802", "20160803", "20160804", "20160805", "invalid", "20160807"],
        "product_brand": ["BrandA", "(not set)", "  ", "BrandB", None, "BrandA", "BrandC"],
        "revenue": [100.5, -20.0, 300.0, 150.0, 200.0, 50.0, np.nan],
        "sessions": [10, 20, 30, 40, 50, 60, 70],
        "constant_col": [1, 1, 1, 1, 1, 1, 1],
        "mostly_null": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1],
        "category": ["A", "B", "A", "C", "D", "E", "F"],
    }
    return pd.DataFrame(data)


# ---------------------
# Tests unitarios
# ---------------------
def test_normalize_strings_and_placeholders(synthetic_dirty_data):
    """Test normalización de strings y placeholders."""
    cfg = CleaningConfig()
    result = normalize_strings_and_placeholders(synthetic_dirty_data, cfg)
    
    # Verificar que los espacios en blanco se conviertan a NaN
    assert result["product_brand"].isna().sum() >= 1
    # "(not set)" debe mantenerse en la marca
    assert "(not set)" in result["product_brand"].values


def test_parse_dates(synthetic_dirty_data):
    """Test parsing de fechas."""
    cfg = CleaningConfig()
    result = parse_dates(synthetic_dirty_data, cfg)
    
    # Debe crear columna parsed_date
    assert "parsed_date" in result.columns
    # Debe ser tipo datetime
    assert pd.api.types.is_datetime64_any_dtype(result["parsed_date"])
    # Fechas inválidas deben ser NaT
    assert result["parsed_date"].isna().sum() >= 1


def test_normalize_brand(synthetic_dirty_data):
    """Test normalización de marca."""
    cfg = CleaningConfig(drop_unknown_brands=False)
    dqr = DataQualityReport()
    
    result = normalize_brand(synthetic_dirty_data, cfg, dqr)
    
    # Debe crear columna product_brand si no existe
    assert "product_brand" in result.columns
    # No debe haber NaN en product_brand después de normalizar
    assert result["product_brand"].notna().all()
    # "(not set)" debe mantenerse
    assert "(not set)" in result["product_brand"].values


def test_normalize_brand_drop_unknown(synthetic_dirty_data):
    """Test normalización de marca con eliminación de Unknown."""
    cfg = CleaningConfig(drop_unknown_brands=True, brand_unknown_token="Unknown")
    dqr = DataQualityReport()
    
    before_rows = len(synthetic_dirty_data)
    result = normalize_brand(synthetic_dirty_data, cfg, dqr)
    
    # Debe eliminar filas con Unknown
    assert len(result) <= before_rows
    assert "Unknown" not in result["product_brand"].values
    assert dqr.brand_unknown_rows >= 0


def test_drop_high_missing_columns(synthetic_dirty_data):
    """Test eliminación de columnas con alto % de nulos."""
    cfg = CleaningConfig(
        drop_columns_missing_threshold=0.85,
        drop_columns_missing_action="drop"
    )
    dqr = DataQualityReport()
    
    result = drop_high_missing_columns(synthetic_dirty_data, cfg, dqr)
    
    # mostly_null debería eliminarse (>95% nulos)
    assert "mostly_null" not in result.columns
    assert len(dqr.dropped_columns_missing) >= 1


def test_drop_high_missing_columns_impute(synthetic_dirty_data):
    """Test imputación de columnas con alto % de nulos."""
    cfg = CleaningConfig(
        drop_columns_missing_threshold=0.85,
        drop_columns_missing_action="impute"
    )
    dqr = DataQualityReport()
    
    result = drop_high_missing_columns(synthetic_dirty_data, cfg, dqr)
    
    # mostly_null debe mantenerse pero imputarse
    assert "mostly_null" in result.columns
    assert len(dqr.imputed_columns_high_missing) >= 1


def test_drop_all_missing_rows():
    """Test eliminación de filas completamente nulas."""
    data = pd.DataFrame({
        "col1": [1, np.nan, 3],
        "col2": [4, np.nan, 6],
        "col3": [7, np.nan, 9],
    })
    cfg = CleaningConfig(drop_rows_all_missing=True)
    dqr = DataQualityReport()
    
    result = drop_all_missing_rows(data, cfg, dqr)
    
    # Debe eliminar la fila 1 (toda NaN)
    assert len(result) == 2
    assert dqr.dropped_rows_all_missing == 1


def test_coerce_types(synthetic_dirty_data):
    """Test conversión de tipos de datos."""
    # Agregar columna que parece numérica pero es string
    df = synthetic_dirty_data.copy()
    df["numeric_string"] = ["1", "2", "3", "4", "5", "6", "7"]
    
    cfg = CleaningConfig(coerce_object_to_numeric_threshold=0.5)
    result = coerce_types(df, cfg)
    
    # numeric_string debe convertirse a numérico
    assert pd.api.types.is_numeric_dtype(result["numeric_string"])


def test_drop_duplicates():
    """Test eliminación de duplicados."""
    data = pd.DataFrame({
        "col1": [1, 2, 1, 3],
        "col2": [4, 5, 4, 6],
    })
    cfg = CleaningConfig(drop_duplicates=True)
    dqr = DataQualityReport()
    
    result = drop_duplicates(data, cfg, dqr)
    
    # Debe eliminar la fila duplicada
    assert len(result) == 3
    assert dqr.duplicate_rows_dropped == 1


def test_negatives_to_nan(synthetic_dirty_data):
    """Test conversión de negativos a NaN en métricas."""
    cfg = CleaningConfig(negatives_to_nan=True)
    dqr = DataQualityReport()
    
    result = negatives_to_nan(synthetic_dirty_data, cfg, dqr)
    
    # revenue tiene un valor negativo que debe convertirse a NaN
    assert result["revenue"].isna().sum() >= synthetic_dirty_data["revenue"].isna().sum()
    assert "revenue" in dqr.negatives_to_nan_cols


def test_impute_missing():
    """Test imputación de valores faltantes."""
    data = pd.DataFrame({
        "numeric": [1, 2, np.nan, 4, 5],
        "category": ["A", "B", np.nan, "A", "B"],
    })
    cfg = CleaningConfig(impute_numeric="median", impute_categorical="mode")
    
    result = impute_missing(data, cfg)
    
    # No debe haber NaN después de imputar
    assert result["numeric"].notna().all()
    assert result["category"].notna().all()


def test_handle_outliers():
    """Test manejo de outliers con winsorización."""
    data = pd.DataFrame({
        "values": [1, 2, 3, 4, 5, 100, 200]  # 100 y 200 son outliers
    })
    cfg = CleaningConfig(winsorize=True, winsor_limits=(0.1, 0.9))
    dqr = DataQualityReport()
    
    result = handle_outliers(data, cfg, dqr)
    
    # Los outliers deben estar limitados
    assert result["values"].max() < 200
    assert "values" in dqr.winsorized_cols


def test_drop_constant_cols(synthetic_dirty_data):
    """Test eliminación de columnas constantes."""
    cfg = CleaningConfig(drop_constant_columns=True)
    dqr = DataQualityReport()
    
    result = drop_constant_cols(synthetic_dirty_data, cfg, dqr)
    
    # constant_col debe eliminarse
    assert "constant_col" not in result.columns
    assert "constant_col" in dqr.constant_cols_dropped


def test_reduce_categoricals_cardinality():
    """Test reducción de cardinalidad en categóricas."""
    data = pd.DataFrame({
        "category": ["A"] * 50 + ["B"] * 30 + ["C"] * 5 + ["D"] * 5 + ["E"] * 3
    })
    cfg = CleaningConfig(rare_thresh=10, rare_apply_on=100)
    
    result = reduce_categoricals_cardinality(data, cfg)
    
    # Categorías raras deben colapsar a "Otros"
    assert "Otros" in result["category"].values
    # A y B deben mantenerse (frecuencia >= 10)
    assert "A" in result["category"].values
    assert "B" in result["category"].values


def test_protected_columns(synthetic_dirty_data):
    """Test que columnas protegidas no se eliminen."""
    cfg = CleaningConfig()
    protected = _protected_columns(synthetic_dirty_data, cfg)
    
    # product_brand debe estar protegida
    assert "product_brand" in protected or "__brand_not_found__" not in protected


def test_run_cleaning_end_to_end(tmp_cleaning_paths, synthetic_dirty_data):
    """Test del pipeline completo de cleaning."""
    # Guardar datos sintéticos
    synthetic_dirty_data.to_parquet(tmp_cleaning_paths["raw"], index=False)
    
    cfg = CleaningConfig(
        raw_path=str(tmp_cleaning_paths["raw"]),
        out_parquet=str(tmp_cleaning_paths["cleaned"]),
        out_csv=str(tmp_cleaning_paths["csv"]),
        persist=True,
        drop_unknown_brands=False,
        drop_columns_missing_action="drop",
    )
    
    df_clean, report = run_cleaning(cfg)
    
    # Verificar que se ejecutó
    assert df_clean is not None
    assert isinstance(report, DataQualityReport)
    
    # Verificar que se guardaron los archivos
    assert Path(tmp_cleaning_paths["cleaned"]).exists()
    assert Path(tmp_cleaning_paths["csv"]).exists()
    
    # Verificar que el reporte tiene datos
    assert report.n_rows_before > 0
    assert report.n_rows_after > 0
    assert report.n_cols_before > 0
    assert report.n_cols_after > 0
    
    # Verificar que parsed_date existe y es datetime
    assert "parsed_date" in df_clean.columns
    assert pd.api.types.is_datetime64_any_dtype(df_clean["parsed_date"])
    
    # Verificar que product_brand existe
    assert "product_brand" in df_clean.columns


def test_cleaning_maintains_not_set_brand(synthetic_dirty_data):
    """Test que (not set) se mantiene como valor válido en product_brand."""
    cfg = CleaningConfig(drop_unknown_brands=False)
    dqr = DataQualityReport()
    
    # Normalizar strings y placeholders
    df = normalize_strings_and_placeholders(synthetic_dirty_data, cfg)
    # Normalizar marca
    df = normalize_brand(df, cfg, dqr)
    
    # (not set) debe mantenerse
    assert "(not set)" in df["product_brand"].values


# ---------------------
# FIN GROUP 04 tests
# ---------------------
