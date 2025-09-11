
import os
import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import holidays

from .config import TrainConfig
from .seed import set_seed
from .features import make_features
from .data import map_menu_ids
from .dataset import SalesDataset, FEATURE_COLS
from .model import MultiOutputLSTMWithEmb
from .losses import ScaledSMAPELoss
from .kfold import time_series_date_kfold_real_dates
from .optimizers import RAdam, Lookahead

def preprocess_train(df: pd.DataFrame):
    kr_holidays = holidays.KR()
    processed = []
    for _, g in df.groupby('영업장명_메뉴명'):
        g_proc = make_features(g, kr_holidays, use_shift=True, log_transform=True)
        processed.append(g_proc)
    full = pd.concat(processed, ignore_index=True)
    full['영업일자'] = pd.to_datetime(full['영업일자'])
    return full

def fit_scaler(df: pd.DataFrame):
    features_to_scale = ['매출수량', 'rolling_mean_7', 'rolling_median_7',
                         'rolling_mean_14', 'rolling_mean_28', 'rolling_std_7']
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df, scaler, features_to_scale

def train_folds(train_raw: pd.DataFrame, out_dir: str, cfg: TrainConfig):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(42)

    # map menu ids first (keeps consistent emb_num)
    train_raw, menu2id, menu_list = map_menu_ids(train_raw)

    # feature engineering
    train_proc = preprocess_train(train_raw)

    # scaling
    train_proc, scaler, scaled_cols = fit_scaler(train_proc)

    folds = time_series_date_kfold_real_dates(train_proc, cfg.folds)

    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device(cfg.resolve_device())

    for fold_idx, (train_df, val_df) in enumerate(folds, 1):
        print(f"\n===== Fold {fold_idx}/{cfg.folds} =====")
        print(f"Train 기간: {train_df['영업일자'].min()} ~ {train_df['영업일자'].max()}")
        print(f"Val 기간:   {val_df['영업일자'].min()} ~ {val_df['영업일자'].max()}")

        train_dataset = SalesDataset(train_df, cfg.lookback, cfg.predict)
        val_dataset = SalesDataset(val_df, cfg.lookback, cfg.predict)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

        model = MultiOutputLSTMWithEmb(
            input_dim=len(FEATURE_COLS), emb_num=len(menu_list), emb_dim=cfg.emb_dim,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, output_dim=cfg.predict, dropout_p=cfg.dropout_p
        ).to(device)

        criterion = ScaledSMAPELoss(zero_weight=cfg.zero_weight)
        base_optimizer = RAdam(model.parameters(), lr=cfg.lr)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

        for epoch in range(cfg.epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch, menu_id in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                menu_id = menu_id.to(device)

                optimizer.zero_grad()
                output = model(X_batch, menu_id)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch, menu_id in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    menu_id = menu_id.to(device)
                    output = model(X_batch, menu_id)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)

            tqdm.write(f"[Fold {fold_idx} | Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        model_path = os.path.join(models_dir, f"model_fold{fold_idx}.pth")
        torch.save(model.state_dict(), model_path)

    # persist scaler and artifacts
    joblib.dump({'scaler': scaler, 'scaled_cols': scaled_cols, 'menu2id': menu2id, 'menu_list': menu_list},
                os.path.join(out_dir, "scaler.pkl"))
    print("✅ Saved models and scaler.")
