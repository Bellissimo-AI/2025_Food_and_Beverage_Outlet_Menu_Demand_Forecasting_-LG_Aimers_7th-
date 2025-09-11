
import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame, kr_holidays, use_shift: bool=True, log_transform: bool=True) -> pd.DataFrame:
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['요일'] = df['영업일자'].dt.weekday
    df['month'] = df['영업일자'].dt.month

    if log_transform:
        df['매출수량'] = np.log1p(np.maximum(df['매출수량'], 0))

    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_day'] = np.sin(2 * np.pi * df['요일'] / 7)
    df['cos_day'] = np.cos(2 * np.pi * df['요일'] / 7)

    df['is_holiday'] = df['영업일자'].apply(lambda x: 1 if x in kr_holidays else 0)

    shift_val = 1 if use_shift else 0
    shifted_sales = df['매출수량'].shift(shift_val) if use_shift else df['매출수량']

    df['rolling_mean_7'] = shifted_sales.rolling(7, min_periods=1).mean().fillna(0)
    df['rolling_median_7'] = shifted_sales.rolling(7, min_periods=1).median().fillna(0)
    df['rolling_mean_14'] = shifted_sales.rolling(14, min_periods=1).mean().fillna(0)
    df['rolling_mean_28'] = shifted_sales.rolling(28, min_periods=1).mean().fillna(0)
    df['rolling_std_7'] = shifted_sales.rolling(7, min_periods=1).std().fillna(0)

    return df
