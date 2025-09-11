
import math
import pandas as pd

def time_series_date_kfold_real_dates(df: pd.DataFrame, n_folds: int=10):
    df = df.sort_values("영업일자").reset_index(drop=True)
    folds = []

    unique_dates = df['영업일자'].sort_values().unique()
    n_dates = len(unique_dates)
    fold_sizes = math.ceil(n_dates / n_folds)

    for i in range(n_folds):
        val_start_idx = i * fold_sizes
        val_end_idx = min((i+1) * fold_sizes, n_dates) - 1

        val_dates = unique_dates[val_start_idx:val_end_idx+1]
        val_df = df[df['영업일자'].isin(val_dates)]
        train_df = df[~df['영업일자'].isin(val_dates)]

        folds.append((train_df, val_df))

        print(f"Fold {i+1}:")
        print(f"  Train 기간: {train_df['영업일자'].min()} ~ {train_df['영업일자'].max()}")
        print(f"  Val   기간: {val_df['영업일자'].min()} ~ {val_df['영업일자'].max()}\n")

    return folds
