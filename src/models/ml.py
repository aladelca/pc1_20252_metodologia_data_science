import pandas as pd

def create_features(df: pd.Series, target_col: str) -> pd.DataFrame:
    """
    Creates time series features from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        target_col (str): The name of the target column.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    df_feat = df.to_frame().copy()
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['dayofyear'] = df_feat.index.dayofyear
    for lag in [1, 7, 14]:
        df_feat[f'lag_{lag}'] = df_feat[target_col].shift(lag)
    for win in [7, 14]:
        df_feat[f'roll_mean_{win}'] = df_feat[target_col].shift(1).rolling(window=win).mean()
    df_feat.dropna(inplace=True)
    return df_feat
