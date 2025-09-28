import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa import seasonal
from statsmodels.tsa.stattools import adfuller

# ------------------------------------------------------------------
# Preprocessing for GA Transactions — Group 4 (Día/Marca)
# ------------------------------------------------------------------

@dataclass
class TSConfig:
    date_col: Optional[str] = None              # 'parsed_date' o 'transaction_date'
    brand_col: Optional[str] = None             # 'product_brand'
    target_metric: Optional[str] = None         # p.ej. 'events', 'transactions', etc.

    brand_placeholders: Tuple[str, ...] = ("", " ", "na", "n/a", "none", "null", "NULL", None)
    brand_unknown_token: str = "Unknown"
    drop_unknown_brands: bool = False

    lags: Tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    ventanas_ma: Tuple[int, ...] = (5, 10, 20, 50, 100)
    ventanas_std: Tuple[int, ...] = (5, 10, 20, 50)
    ventanas_roll: Tuple[int, ...] = (5, 10, 20)
    ventanas_cortas: Tuple[int, ...] = (5, 10)
    ventanas_largas: Tuple[int, ...] = (20, 50)
    ventanas_momentum: Tuple[int, ...] = (5, 10, 20)
    incluir_fechas: bool = True
    incluir_tecnicas: bool = True
    incluir_momentum: bool = True

    freq: str = "D"
    fill_missing: str = "zero"  # 'zero' | 'ffill' | 'none'
    #Aviso Posible fuga de información
    check_leakage: bool = True
    leakage_eq_threshold: float = 0.90  # umbral para avisar (puedes cambiarlo)



# ======================
# Utils
# ======================
def adf_test(series: pd.Series, title: str = "") -> bool:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) < 10:
        return False
    try:
        result = adfuller(s, autolag='AIC')
        pvalue = result[1]
        return pvalue < 0.05
    except Exception:
        return False

def preparar_datos_prophet(y: pd.Series, exog: Dict[str, Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
    df_p = pd.DataFrame({'ds': y.index, 'y': y.values})
    if exog:
        for name, vec in exog.items():
            s = pd.Series(vec, index=y.index)
            df_p[name] = s.values
    return df_p

def crear_variable_exogena_segura(series: pd.Series, periods: List[Tuple[pd.Timestamp, pd.Timestamp]], name: str) -> pd.Series:
    idx = series.index
    out = pd.Series(0, index=idx, dtype=int)
    for (start, end) in periods:
        mask = (idx >= pd.to_datetime(start, utc=True)) & (idx <= pd.to_datetime(end, utc=True))
        out.loc[mask] = 1
    out.name = name
    return out

def create_sequences(features: np.ndarray, target: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def verificar_no_data_leakage(df: pd.DataFrame, target_col: str, threshold: float = 0.90) -> None:
    if target_col not in df.columns:
        return
    tgt = df[target_col]
    for c in df.columns:
        if c == target_col:
            continue
        eq_rate = (df[c] == tgt).mean()
        if np.isfinite(eq_rate) and eq_rate >= threshold:
            print(f"[warning] Posible fuga de información en '{c}' (={eq_rate:.2%} iguales a target). Revísalo.")


# ======================
# Clase principal
# ======================
class TimeSeriesPreprocessor:
    """Preprocesamiento multi-modelo adaptado a GA Día/Marca."""
    def __init__(self, config: TSConfig = None):
        self.config = config or TSConfig()
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.is_fitted = False

    # --------- Carga y esquema GA ---------
    def load_data(self, file_path: str) -> pd.DataFrame:
        data = pd.read_parquet(file_path)

        # Detectar/forzar fecha
        date_col = self.config.date_col or self._detect_date_col(data)
        if date_col not in data.columns:
            raise ValueError(f"No se encontró columna de fecha (busqué '{date_col}'). Define config.date_col.")

        # Parseo robusto (tz-aware safe)
        is_dt = pd.api.types.is_datetime64_any_dtype(data[date_col])
        if not is_dt:
            if date_col == 'transaction_date':
                ser = data[date_col].astype(str).str.zfill(8)
                data[date_col] = pd.to_datetime(ser, format="%Y%m%d", errors='coerce', utc=True)
            else:
                data[date_col] = pd.to_datetime(data[date_col], errors='coerce', utc=True)
        else:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce', utc=True)

        data = data.dropna(subset=[date_col]).sort_values(date_col)
        data = data.set_index(date_col)
        data.index.name = 'Date'  # tz-aware, UTC

        # Normalizar marca si existe
        brand_col = self.config.brand_col or self._detect_brand_col(data)
        if brand_col in data.columns:
            data[brand_col] = data[brand_col].astype("object").fillna(self.config.brand_unknown_token)
            data[brand_col] = data[brand_col].apply(
                lambda x: self.config.brand_unknown_token
                if (isinstance(x, str) and x.strip() in self.config.brand_placeholders)
                else x
            )
            if self.config.drop_unknown_brands:
                data = data[data[brand_col] != self.config.brand_unknown_token]

        return data

    def _detect_date_col(self, df: pd.DataFrame) -> str:
        if 'parsed_date' in df.columns:
            return 'parsed_date'
        if 'transaction_date' in df.columns:
            return 'transaction_date'
        candidates = [c for c in df.columns if any(k in c.lower() for k in ['date','time','timestamp','datetime'])]
        if candidates:
            return candidates[0]
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.name or 'Date'
        return 'Date'

    def _detect_brand_col(self, df: pd.DataFrame) -> str:
        for name in ['product_brand','productBrand','brand','Brand','product_brand_name']:
            if name in df.columns:
                return name
        for c in df.columns:
            if 'brand' in c.lower():
                return c
        return '__brand_not_found__'

    def select_brand_series(self, df: pd.DataFrame, brand: Optional[str], metric: Optional[str]) -> pd.Series:
        brand_col = self.config.brand_col or self._detect_brand_col(df)
        if brand_col not in df.columns:
            raise ValueError("Columna de marca no detectada. Define config.brand_col (p.ej., 'product_brand').")
        metric_col = metric or self.config.target_metric
        if metric_col is None or metric_col not in df.columns:
            raise ValueError("Define 'target_metric' en config o pásala al método.")

        sub = df[df[brand_col] == brand] if brand is not None else df.copy()
        y = pd.to_numeric(sub[metric_col], errors='coerce')

        # Resample diario
        daily = y.resample(self.config.freq).sum(min_count=1)
        if self.config.fill_missing == 'zero':
            daily = daily.fillna(0)
        elif self.config.fill_missing == 'ffill':
            daily = daily.ffill()
        daily.name = metric_col
        return daily

    # --------- Chequeos básicos ---------
    def check_stationarity(self, series: pd.Series, title: str = '') -> bool:
        return adf_test(series, title)

    def decompose_series(self, series: pd.Series, period: int = 7, model: str = 'additive'):
        series = pd.to_numeric(series, errors='coerce').dropna()
        return seasonal.seasonal_decompose(series, period=period, model=model)

    # --------- Feature engineering genérico ---------
    def _crear_features_completas(self, target_series: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({'target': target_series})
        for L in self.config.lags:
            df[f'lag_{L}'] = target_series.shift(L)
        for w in self.config.ventanas_ma:
            df[f'ma_{w}'] = target_series.shift(1).rolling(w).mean()
        for w in self.config.ventanas_std:
            df[f'std_{w}'] = target_series.shift(1).rolling(w).std()
        for w in self.config.ventanas_roll:
            df[f'rollsum_{w}'] = target_series.shift(1).rolling(w).sum()
        for w in self.config.ventanas_momentum:
            df[f'mom_{w}'] = target_series / target_series.shift(w) - 1.0
        if self.config.incluir_fechas:
            idx = df.index
            df['dow'] = idx.dayofweek
            df['dom'] = idx.day
            df['week'] = idx.isocalendar().week.astype(int)
            df['month'] = idx.month
            df['year'] = idx.year
            df['is_weekend'] = (df['dow'].isin([5,6])).astype(int)
        return df

    def create_features(self, series: Union[pd.Series, pd.DataFrame], target_col: str = None) -> pd.DataFrame:
        target_series = series if isinstance(series, pd.Series) else series[target_col]
        feats = self._crear_features_completas(target_series)
        feats['target_diff'] = feats['target'].diff()
        if self.config.check_leakage:
            verificar_no_data_leakage(feats, 'target', threshold=self.config.leakage_eq_threshold)
        return feats


    # --------- Preparaciones por tipo de modelo ---------
    def prepare_arima_data(self, series: Union[pd.Series, pd.DataFrame], start_date: str = '2000-01-01') -> pd.Series:
        if isinstance(series, pd.DataFrame):
            num_cols = series.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("No hay columnas numéricas en el DataFrame para ARIMA.")
            series = series[num_cols[0]]

        # Asegurar que la comparación sea tz-aware
        start_dt = pd.to_datetime(start_date, utc=True)
        # Filtrado por rango
        filtered = series[series.index >= start_dt]

        is_stat = self.check_stationarity(filtered, 'Original Series')
        if not is_stat:
            diff = filtered.diff().dropna()
            self.check_stationarity(diff, 'Differenced Series')
            return diff
        return filtered

    def prepare_prophet_data(self, series: Union[pd.Series, pd.DataFrame], events_periods: List[Tuple[pd.Timestamp, pd.Timestamp]] = None):
        if isinstance(series, pd.DataFrame):
            num_cols = series.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("No hay columnas numéricas en el DataFrame para Prophet.")
            series = series[num_cols[0]]

        # Prophet: suele preferir índice tz-naive
        y = series.copy()
        if getattr(y.index, "tz", None) is not None:
            y.index = y.index.tz_convert(None)

        exog_vars = {}
        if events_periods:
            # construir con el índice original del usuario (naive)
            exog_vars['event_1'] = crear_variable_exogena_segura(
                y, [(pd.to_datetime(a), pd.to_datetime(b)) for (a, b) in events_periods], 'event_1'
            )
        prophet_df = preparar_datos_prophet(y, exog_vars)
        return prophet_df, list(exog_vars.keys())

    def prepare_ml_data(self, series: Union[pd.Series, pd.DataFrame], 
                        train_start: str = '2021-01-01', train_end: str = '2023-12-31',
                        test_start: str = '2024-01-01', target_col: str = None):
        if isinstance(series, pd.DataFrame):
            if target_col is None:
                num_cols = series.select_dtypes(include=[np.number]).columns
                if len(num_cols) == 0:
                    raise ValueError("No hay columnas numéricas para ML.")
                target_col = num_cols[0]
            target_series = series[target_col]
        else:
            target_series = series

        # Fechas tz-aware para cortes
        t_start = pd.to_datetime(train_start, utc=True)
        t_end   = pd.to_datetime(train_end,   utc=True)
        te_start= pd.to_datetime(test_start,  utc=True)

        feats = self.create_features(target_series, target_col=None)
        train_data = feats.loc[t_start:t_end]
        test_data  = feats.loc[te_start:]

        train_data = train_data.dropna()
        test_data = test_data.dropna()

        feature_cols = [c for c in train_data.columns if c not in ['target', 'target_diff']]
        X_train = train_data[feature_cols]
        y_train = train_data['target_diff']
        X_test = test_data[feature_cols]
        y_test = test_data['target_diff']
        return X_train, y_train, X_test, y_test

    def prepare_lstm_data(self, series: pd.DataFrame, sequence_length: int = 60, train_ratio: float = 0.8,
                          feature_columns: List[str] = None, target_column: str = None):
        if target_column is None:
            num_cols = series.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("No hay columnas numéricas para LSTM.")
            target_column = num_cols[0]
        if feature_columns is None:
            feature_columns = [c for c in series.columns if c != target_column]
        features = series[feature_columns].values
        target = series[target_column].values.reshape(-1, 1)
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target)
        X, y = create_sequences(features_scaled, target_scaled.flatten(), sequence_length)
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        self.is_fitted = True
        return X_train, X_test, y_train, y_test

    def inverse_transform_lstm(self, predictions: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call prepare_lstm_data first.")
        return self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    def create_sarimax_exogenous(self, series: pd.Series, events_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]):
        exog_df = pd.DataFrame(index=series.index)
        for i, period in enumerate(events_periods):
            var_name = f'event_{i+1}'
            exog_df[var_name] = crear_variable_exogena_segura(series, [period], var_name)
        return exog_df

    # --------- Orquestador cómodo ---------
    def preprocess_for_model(self, data_path: str, model_type: str, brand: Optional[str] = None, **kwargs):
        df = self.load_data(data_path)
        if self.config.target_metric is not None:
            series = self.select_brand_series(df, brand=brand, metric=self.config.target_metric)
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("No hay columnas numéricas en el dataset.")
            series = df[num_cols[0]].resample(self.config.freq).sum(min_count=1).fillna(0)
        if model_type == 'arima':
            return self.prepare_arima_data(series, **kwargs)
        elif model_type == 'prophet':
            return self.prepare_prophet_data(series, **kwargs)
        elif model_type == 'ml':
            return self.prepare_ml_data(series, **kwargs)
        elif model_type == 'lstm':
            feats = self.create_features(series).dropna()
            return self.prepare_lstm_data(feats, **kwargs)
        elif model_type == 'sarimax':
            exog = self.create_sarimax_exogenous(series, kwargs.get('events_periods', []))
            return series, exog
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# --------- Helper de módulo ---------
def preprocess_for_model(data_path: str, model_type: str, **kwargs):
    preprocessor = TimeSeriesPreprocessor()
    data = preprocessor.load_data(data_path)
    if model_type == 'arima':
        return preprocessor.prepare_arima_data(data, **kwargs)
    elif model_type == 'prophet':
        return preprocessor.prepare_prophet_data(data, **kwargs)
    elif model_type == 'ml':
        return preprocessor.prepare_ml_data(data, **kwargs)
    elif model_type == 'lstm':
        return preprocessor.prepare_lstm_data(data, **kwargs)
    elif model_type == 'sarimax':
        series = data.iloc[:, 0] if len(data.columns) else pd.Series(dtype=float, index=data.index)
        exog = preprocessor.create_sarimax_exogenous(series, kwargs.get('events_periods', []))
        return series, exog
    else:
        raise ValueError(f"Unknown model type: {model_type}")
# ------------------------------------------------------------------
# FINISH GROUP 04 preprocessing module
# ------------------------------------------------------------------
