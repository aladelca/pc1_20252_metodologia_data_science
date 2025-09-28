import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set

import numpy as np
import pandas as pd

# ===========================================================
# cleaning.py — Grupo 04 (Día/Marca)
# Limpieza integral para GA Transactions
#   - Ruta de entrada:  ../data/raw/data_sample.parquet
#   - Rutas de salida:  ../data/raw/data_sample_cleaned_group04.parquet / .csv
# ===========================================================

@dataclass
class CleaningConfig:
    # IO
    raw_path: str = "../data/raw/data_sample.parquet"
    out_parquet: str = "../data/raw/data_sample_cleaned_group04.parquet"
    out_csv: str = "../data/raw/data_sample_cleaned_group04.csv"
    persist: bool = True

    # Normalización de strings
    strip_whitespace: bool = True
    to_lower_categoricals: bool = False
    placeholders: List[str] = field(default_factory=lambda: [
        "", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "?",
        "(not set)", "(not provided)", "(not-set)", "(not_provided)", "not set", "not_set"
    ])

    # Candidatas de fecha
    date_candidates: List[str] = field(default_factory=lambda: [
        "parsed_date", "transaction_date", "date", "timestamp", "visitStartTime", "datetime"
    ])

    # Marca
    brand_candidates: List[str] = field(default_factory=lambda: [
        "product_brand", "productBrand", "brand", "Brand", "product_brand_name"
    ])
    brand_placeholders: List[str] = field(default_factory=lambda: [
        "(not set)", "", " ", "na", "n/a", "none", "null", "NULL"
    ])
    brand_unknown_token: str = "Unknown"
    drop_unknown_brands: bool = False  # si True, se eliminan filas con brand desconocida

    # Reglas de nulos
    drop_columns_missing_threshold: float = 0.95  # columnas con >95% nulos -> acción
    drop_columns_missing_action: str = "drop"     # 'drop' | 'impute'
    drop_rows_all_missing: bool = True            # elimina filas 100% nulas
    coerce_object_to_numeric_threshold: float = 0.50
    # Filtrado adicional de filas inválidas clave
    drop_rows_with_invalid_date: bool = True      # elimina filas con parsed_date NaT si existe

    # Imputación
    impute_numeric: str = "median"               # 'median' | 'mean' | 'zero' | 'none'
    impute_categorical: str = "mode"             # 'mode' | 'unknown' | 'none'
    unknown_token: str = "Desconocido"

    # Métricas que no deben ser negativas
    non_negative_keywords: List[str] = field(default_factory=lambda: [
        "revenue","transaction","pageviews","sessions","qty","quantity",
        "units","purchases","price","value","amount"
    ])
    negatives_to_nan: bool = True

    # Duplicados
    drop_duplicates: bool = True
    subset_dupes: Optional[List[str]] = None

    # Outliers
    winsorize: bool = True
    winsor_limits: Tuple[float, float] = (0.01, 0.99)  # 1-99%

    # Cardinalidad de categóricas (colapsar niveles raros)
    rare_thresh: int = 20
    rare_apply_on: int = 100

    # Columnas constantes
    drop_constant_columns: bool = True


@dataclass
class DataQualityReport:
    n_rows_before: int = 0
    n_cols_before: int = 0
    n_rows_after: int = 0
    n_cols_after: int = 0
    dropped_columns_missing: List[str] = field(default_factory=list)
    dropped_rows_all_missing: int = 0
    duplicate_rows_dropped: int = 0
    negatives_to_nan_cols: List[str] = field(default_factory=list)
    winsorized_cols: List[str] = field(default_factory=list)
    constant_cols_dropped: List[str] = field(default_factory=list)
    brand_unknown_rows: int = 0
    rows_dropped_invalid_date: int = 0
    imputed_columns_high_missing: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "n_rows_before": [self.n_rows_before],
            "n_cols_before": [self.n_cols_before],
            "n_rows_after": [self.n_rows_after],
            "n_cols_after": [self.n_cols_after],
            "dropped_columns_missing": [self.dropped_columns_missing],
            "dropped_rows_all_missing": [self.dropped_rows_all_missing],
            "duplicate_rows_dropped": [self.duplicate_rows_dropped],
            "negatives_to_nan_cols": [self.negatives_to_nan_cols],
            "winsorized_cols": [self.winsorized_cols],
            "constant_cols_dropped": [self.constant_cols_dropped],
            "brand_unknown_rows": [self.brand_unknown_rows],
        })


# ---------------------------
# Helpers
# ---------------------------
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _parse_date_column(df: pd.DataFrame, date_col: str) -> pd.Series:
    if date_col == "transaction_date" and not np.issubdtype(df[date_col].dtype, np.datetime64):
        ser = df[date_col].astype(str).str.zfill(8)
        return pd.to_datetime(ser, format="%Y%m%d", errors="coerce", utc=True)
    return pd.to_datetime(df[date_col], errors="coerce", utc=True)

def _winsorize(s: pd.Series, low: float, high: float) -> pd.Series:
    if s.dropna().empty:
        return s
    ql, qh = s.quantile([low, high])
    return s.clip(lower=ql, upper=qh)

def _protected_columns(df: pd.DataFrame, cfg: CleaningConfig) -> Set[str]:
    """
    Columnas que NO deben eliminarse/alterarse por reglas automáticas:
    - fecha parseada 'parsed_date' (si existe)
    - cualquier candidata de fecha presente
    - la columna de marca (si existe); si no existe, se creará 'product_brand'
    """
    protected = set()
    # fechas
    for c in cfg.date_candidates:
        if c in df.columns:
            protected.add(c)
    # marca (primera encontrada entre los candidatos)
    brand_col = _detect_col(df, cfg.brand_candidates)
    if brand_col is not None:
        protected.add(brand_col)
    else:
        # si no existe, la crearemos luego; proteger el nombre objetivo
        protected.add("product_brand")
    # también proteger 'parsed_date' explícitamente por si se crea en parse_dates
    protected.add("parsed_date")
    return protected


# ---------------------------
# Pipeline de limpieza
# ---------------------------
def load_raw(cfg: CleaningConfig) -> pd.DataFrame:
    return pd.read_parquet(cfg.raw_path)

def normalize_strings_and_placeholders(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    brand_col = _detect_col(out, cfg.brand_candidates)  # detectar columna de marca si existe

    # set de placeholders completo y una versión sin "(not set)" para la marca
    placeholders_all = set(cfg.placeholders)
    placeholders_without_notset = placeholders_all - {"(not set)"}

    for c in obj_cols:
        # strip y minusculizar si aplica
        if cfg.strip_whitespace:
            out[c] = out[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
        if cfg.to_lower_categoricals:
            out[c] = out[c].apply(lambda x: x.lower() if isinstance(x, str) else x)

        # placeholders -> NaN (pero NO tocar "(not set)" en la marca)
        if c == brand_col or c == "product_brand":
            out[c] = out[c].apply(lambda x: np.nan if isinstance(x, str) and x in placeholders_without_notset else x)
        else:
            out[c] = out[c].apply(lambda x: np.nan if isinstance(x, str) and x in placeholders_all else x)

    return out


def parse_dates(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    out = df.copy()
    date_col = _detect_col(out, cfg.date_candidates)
    if date_col is not None:
        parsed = _parse_date_column(out, date_col)
        out["parsed_date"] = parsed
    return out

def drop_rows_invalid_date(df: pd.DataFrame, cfg: CleaningConfig, dqr: DataQualityReport) -> pd.DataFrame:
    """Eliminar filas con fecha inválida (parsed_date NaT) si está habilitado.
    Mantiene la columna de fecha, sólo filtra filas con NaT para no romper la serie temporal.
    """
    if not cfg.drop_rows_with_invalid_date:
        return df
    if "parsed_date" not in df.columns:
        return df
    before = len(df)
    df = df[~df["parsed_date"].isna()].copy()
    dqr.rows_dropped_invalid_date = before - len(df)
    return df

def normalize_brand(df: pd.DataFrame, cfg: CleaningConfig, dqr: DataQualityReport) -> pd.DataFrame:
    out = df.copy()
    brand_col = _detect_col(out, cfg.brand_candidates)

    # Si no existe ninguna columna de marca, crear product_brand = "Unknown"
    if brand_col is None:
        out["product_brand"] = cfg.brand_unknown_token
        brand_col = "product_brand"

    before = len(out)
    out[brand_col] = out[brand_col].astype("object").fillna(cfg.brand_unknown_token)

    # IMPORTANTE: NO reemplazar "(not set)" → se mantiene tal cual
    placeholders_no_notset = set(cfg.brand_placeholders) - {"(not set)"}
    out[brand_col] = out[brand_col].apply(
        lambda x: cfg.brand_unknown_token
        if (isinstance(x, str) and x.strip() in placeholders_no_notset)
        else x
    )

    # Crear columna canónica para downstream siempre
    out["product_brand"] = out[brand_col].astype("object")

    if cfg.drop_unknown_brands:
        out = out[out[brand_col] != cfg.brand_unknown_token]
        dqr.brand_unknown_rows = before - len(out)

    return out


def drop_high_missing_columns(
    df: pd.DataFrame,
    cfg: CleaningConfig,
    dqr: DataQualityReport,
    protected: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Gestiona columnas con alto porcentaje de nulos, respetando columnas protegidas
    (fechas y marca). Acción configurable:
    - 'drop': elimina las columnas (previo respeto a protegidas)
    - 'impute': imputa valores (numéricas -> media; categóricas -> moda)
    Si no se provee el set de protegidas, se calcula automáticamente a partir del df y cfg.
    """
    protected = protected or _protected_columns(df, cfg)
    missing_pct = df.isna().mean()
    targets = [c for c in missing_pct[missing_pct > cfg.drop_columns_missing_threshold].index.tolist()
               if c not in protected]
    if not targets:
        return df

    action = (cfg.drop_columns_missing_action or "drop").lower()
    if action == "drop":
        df = df.drop(columns=targets)
        dqr.dropped_columns_missing = targets
        return df

    # action == 'impute': imputar por tipo de dato
    out = df.copy()
    imputed_cols: List[str] = []
    for c in targets:
        ser = out[c]
        # Numéricas -> media (si no se puede calcular, fallback 0)
        if pd.api.types.is_numeric_dtype(ser):
            mean_val = ser.mean()
            if pd.isna(mean_val):
                mean_val = 0
            out[c] = ser.fillna(mean_val)
            imputed_cols.append(c)
        else:
            # Categóricas/otros -> moda (si no hay moda, fallback unknown_token)
            mode = ser.mode(dropna=True)
            fillv = mode.iloc[0] if not mode.empty else cfg.unknown_token
            out[c] = ser.fillna(fillv)
            imputed_cols.append(c)
    dqr.imputed_columns_high_missing = imputed_cols
    return out

def drop_all_missing_rows(df: pd.DataFrame, cfg: CleaningConfig, dqr: DataQualityReport) -> pd.DataFrame:
    if not cfg.drop_rows_all_missing:
        return df
    before = len(df)
    df = df.dropna(how="all")
    dqr.dropped_rows_all_missing = before - len(df)
    return df

def coerce_types(
    df: pd.DataFrame,
    cfg: CleaningConfig,
    protected: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Convierte objetos a numérico cuando aplica, sin tocar columnas protegidas."""
    protected = protected or _protected_columns(df, cfg)
    out = df.copy()
    for c in out.columns:
        if c in protected:
            continue  # no convertir marca/fechas a numérico
        if out[c].dtype == "object":
            coerced = pd.to_numeric(out[c], errors="coerce")
            if coerced.notna().mean() >= cfg.coerce_object_to_numeric_threshold:
                out[c] = coerced
    return out

def drop_duplicates(df: pd.DataFrame, cfg: CleaningConfig, dqr: DataQualityReport) -> pd.DataFrame:
    if not cfg.drop_duplicates:
        return df
    before = len(df)
    df = df.drop_duplicates(subset=cfg.subset_dupes)
    dqr.duplicate_rows_dropped = before - len(df)
    return df

def negatives_to_nan(df: pd.DataFrame, cfg: CleaningConfig, dqr: DataQualityReport) -> pd.DataFrame:
    out = df.copy()
    if not cfg.negatives_to_nan:
        return out
    affected = []
    num_cols = out.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if any(k in c.lower() for k in cfg.non_negative_keywords):
            negs = (out[c] < 0).sum()
            if negs > 0:
                out.loc[out[c] < 0, c] = np.nan
                affected.append(c)
    dqr.negatives_to_nan_cols = affected
    return out

def impute_missing(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    out = df.copy()
    # numéricas
    for c in out.select_dtypes(include=[np.number]).columns:
        if out[c].isna().any():
            if cfg.impute_numeric == "median":
                out[c] = out[c].fillna(out[c].median())
            elif cfg.impute_numeric == "mean":
                out[c] = out[c].fillna(out[c].mean())
            elif cfg.impute_numeric == "zero":
                out[c] = out[c].fillna(0)
    # categóricas / bool
    for c in out.select_dtypes(include=["object", "category", "bool"]).columns:
        if out[c].isna().any():
            if cfg.impute_categorical == "mode":
                mode = out[c].mode(dropna=True)
                fillv = mode.iloc[0] if not mode.empty else cfg.unknown_token
                out[c] = out[c].fillna(fillv)
            elif cfg.impute_categorical == "unknown":
                out[c] = out[c].fillna(cfg.unknown_token)
    return out

def handle_outliers(df: pd.DataFrame, cfg: CleaningConfig, dqr: DataQualityReport) -> pd.DataFrame:
    out = df.copy()
    if not cfg.winsorize:
        return out
    for c in out.select_dtypes(include=[np.number]).columns:
        out[c] = _winsorize(out[c], *cfg.winsor_limits)
        dqr.winsorized_cols.append(c)
    return out

def drop_constant_cols(
    df: pd.DataFrame,
    cfg: CleaningConfig,
    dqr: DataQualityReport,
    protected: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Elimina columnas constantes excepto las protegidas."""
    if not cfg.drop_constant_columns:
        return df
    protected = protected or _protected_columns(df, cfg)
    nunique = df.nunique(dropna=False)
    const_cols = [c for c in nunique[nunique <= 1].index.tolist() if c not in protected]
    if const_cols:
        dqr.constant_cols_dropped = const_cols
        df = df.drop(columns=const_cols)
    return df

def reduce_categoricals_cardinality(
    df: pd.DataFrame,
    cfg: CleaningConfig,
    protected: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Colapsa categorías raras en 'Otros' sin afectar columnas protegidas."""
    protected = protected or _protected_columns(df, cfg)
    out = df.copy()
    cat_cols = [c for c in out.select_dtypes(include=["object", "category"]).columns if c not in protected]
    for c in cat_cols:
        n_unique = out[c].nunique(dropna=True)
        if 0 < n_unique <= cfg.rare_apply_on:
            vc = out[c].value_counts(dropna=True)
            rare = vc[vc < cfg.rare_thresh].index
            if len(rare) > 0:
                out[c] = out[c].where(~out[c].isin(rare), other="Otros")
    return out


# ---------------------------
# Runner
# ---------------------------
def run_cleaning(cfg: CleaningConfig = None):
    cfg = cfg or CleaningConfig()
    dqr = DataQualityReport()

    # Load
    df = load_raw(cfg)
    dqr.n_rows_before, dqr.n_cols_before = df.shape

    # Set de columnas protegidas (marca/fechas)
    protected = _protected_columns(df, cfg)

    # Normalize placeholders/strings
    df = normalize_strings_and_placeholders(df, cfg)

    # Parse dates -> 'parsed_date' si es posible (protegerla también)
    df = parse_dates(df, cfg)
    protected.add("parsed_date")

    # Normalize brand (Unknown o drop) — y crear columna canónica product_brand
    df = normalize_brand(df, cfg, dqr)
    detected_brand = _detect_col(df, cfg.brand_candidates)
    if detected_brand:
        protected.add(detected_brand)
    protected.add("product_brand")

    # Filtrar filas con fecha inválida si corresponde (sin eliminar la columna)
    df = drop_rows_invalid_date(df, cfg, dqr)

    # Drop columns/rows con nulos (respetando protegidas)
    df = drop_high_missing_columns(df, cfg, dqr, protected)
    df = drop_all_missing_rows(df, cfg, dqr)

    # Coerce types (no tocar protegidas)
    df = coerce_types(df, cfg, protected)

    # Duplicados
    df = drop_duplicates(df, cfg, dqr)

    # No negativos -> NaN
    df = negatives_to_nan(df, cfg, dqr)

    # Imputación
    df = impute_missing(df, cfg)

    # Outliers (winsorize 1-99%)
    df = handle_outliers(df, cfg, dqr)

    # Columnas constantes (respetando protegidas)
    df = drop_constant_cols(df, cfg, dqr, protected)

    # Reducir cardinalidad (no tocar la marca)
    df = reduce_categoricals_cardinality(df, cfg, protected)

    # Final shape
    dqr.n_rows_after, dqr.n_cols_after = df.shape

    # Persist
    if cfg.persist:
        _ensure_dir(cfg.out_parquet)
        df.to_parquet(cfg.out_parquet, index=False)
        df.to_csv(cfg.out_csv, index=False)

    return df, dqr


if __name__ == "__main__":
    df_clean, report = run_cleaning()
    print("Limpieza completada.")
    print(df_clean.shape)
    print(report.to_frame())
# ------------------------------------------------------------------
# FINISH GROUP 04 cleaning module
# ------------------------------------------------------------------
