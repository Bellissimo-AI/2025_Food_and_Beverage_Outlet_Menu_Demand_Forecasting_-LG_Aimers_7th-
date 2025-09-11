
from torch.utils.data import Dataset
import torch

FEATURE_COLS = [
    '매출수량', 'rolling_mean_7', 'rolling_median_7',
    'rolling_mean_14', 'rolling_mean_28', 'rolling_std_7',
    'sin_day', 'cos_day', 'sin_month', 'cos_month', 'is_holiday'
]

TARGET_COL = ['매출수량']

class SalesDataset(Dataset):
    def __init__(self, df, lookback=28, predict=7):
        self.samples = []
        for menu_id, g in df.groupby('menu_id'):
            arr_X = g[FEATURE_COLS].values
            arr_y = g[TARGET_COL].values
            for i in range(len(g) - lookback - predict + 1):
                X_seq = arr_X[i:i+lookback]
                y_seq = arr_y[i+lookback:i+lookback+predict, 0]
                self.samples.append((X_seq, y_seq, menu_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_seq, y_seq, menu_id = self.samples[idx]
        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
            torch.tensor(menu_id, dtype=torch.long),
        )
