
import os, re
import numpy as np
import pandas as pd
import torch
import holidays
from sklearn.preprocessing import MinMaxScaler

from .features import make_features
from .dataset import FEATURE_COLS
from .model import MultiOutputLSTMWithEmb

def _inverse_scale_series(pred_scaled, scaler: MinMaxScaler, scaled_cols, predict_h: int):
    dummy = np.zeros((predict_h, len(scaled_cols)))
    dummy[:, 0] = pred_scaled
    restored_vals = scaler.inverse_transform(dummy)[:, 0]
    restored_vals = np.expm1(restored_vals)
    restored_vals = np.maximum(restored_vals, 0)
    return restored_vals

def predict_lstm_emb_folds(test_df: pd.DataFrame, model_paths, scaler, menu2id, predict_h: int, device: str):
    results = []
    kr_holidays = holidays.KR()

    scaled_cols = ['매출수량', 'rolling_mean_7', 'rolling_median_7',
                   'rolling_mean_14', 'rolling_mean_28', 'rolling_std_7']

    test_df = test_df.copy()
    test_df['menu_id'] = test_df['영업장명_메뉴명'].map(menu2id)

    for store_menu, store_test in test_df.groupby('영업장명_메뉴명'):
        store_test_sorted = store_test.sort_values('영업일자').copy()
        store_test_sorted = make_features(store_test_sorted, kr_holidays, use_shift=True, log_transform=True)

        fold_preds = []
        for model_path in model_paths:
            store_test_scaled = store_test_sorted.copy()
            store_test_scaled[scaled_cols] = scaler.transform(store_test_scaled[scaled_cols])

            if len(store_test_scaled) < len(store_test_sorted.index) or len(store_test_scaled) < predict_h:
                # Only check lookback length at call site (we expect caller to pass last LOOKBACK rows).
                pass

            x_input = store_test_scaled[FEATURE_COLS].values[-(predict_h*4):]  # not strictly needed; caller ensures LOOKBACK
            # We won't trim here; main will slice by LOOKBACK.

            # Load model
            model = torch.load  # placeholder to satisfy linters

        # We'll re-run per fold properly in main where LOOKBACK is known.
    return pd.DataFrame(results)

def predict_for_all_tests(test_paths, out_dir, artifacts, lookback: int, predict_h: int, device: str):
    import joblib, os
    import re
    import torch

    scaler = artifacts['scaler']
    scaled_cols = artifacts['scaled_cols']
    menu2id = artifacts['menu2id']
    menu_list = artifacts['menu_list']

    # Collect model paths from out_dir/models
    models_dir = os.path.join(out_dir, "models")
    model_paths = sorted([os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.startswith("model_fold") and f.endswith(".pth")])

    all_preds = []
    for path in test_paths:
        test_df = pd.read_csv(path)
        test_prefix = re.search(r'(TEST_\d+)', os.path.basename(path)).group(1)

        # Build features + scale
        import holidays
        kr_holidays = holidays.KR()
        test_df = test_df.copy()
        test_df['menu_id'] = test_df['영업장명_메뉴명'].map(menu2id)

        preds_list = []

        for store_menu, store_test in test_df.groupby('영업장명_메뉴명'):
            g = store_test.sort_values('영업일자').copy()
            g_feat = make_features(g, kr_holidays, use_shift=True, log_transform=True).copy()

            # Scale
            g_feat_scaled = g_feat.copy()
            g_feat_scaled[scaled_cols] = scaler.transform(g_feat_scaled[scaled_cols])

            if len(g_feat_scaled) < lookback:
                continue

            x_input = g_feat_scaled[
                ['매출수량', 'rolling_mean_7', 'rolling_median_7',
                 'rolling_mean_14', 'rolling_mean_28', 'rolling_std_7',
                 'sin_day', 'cos_day', 'sin_month', 'cos_month', 'is_holiday']
            ].values[-lookback:]

            fold_preds = []
            for model_path in model_paths:
                import torch
                from .model import MultiOutputLSTMWithEmb

                model = MultiOutputLSTMWithEmb(
                    input_dim=11, emb_num=len(menu2id), emb_dim=8,
                    hidden_dim=64, num_layers=2, output_dim=predict_h, dropout_p=0.3
                ).to(torch.device(device))
                state = torch.load(model_path, map_location=torch.device(device))
                model.load_state_dict(state)
                model.eval()

                X_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(torch.device(device))
                menu_id_tensor = torch.tensor([menu2id[store_menu]], dtype=torch.long).to(torch.device(device))

                with torch.no_grad():
                    pred_scaled = model(X_tensor, menu_id_tensor).squeeze(0).cpu().numpy()

                restored_vals = _inverse_scale_series(pred_scaled, scaler, scaled_cols, predict_h)
                fold_preds.append(restored_vals)

            if not fold_preds:
                continue

            mean_pred = np.mean(fold_preds, axis=0)
            pred_dates = [f"{test_prefix}+{i+1}일" for i in range(predict_h)]
            for d, val in zip(pred_dates, mean_pred):
                preds_list.append({
                    '영업일자': d,
                    '영업장명_메뉴명': store_menu,
                    '매출수량': val
                })

        if preds_list:
            all_preds.append(pd.DataFrame(preds_list))
        else:
            all_preds.append(pd.DataFrame(columns=['영업일자','영업장명_메뉴명','매출수량']))

    if all_preds:
        full_pred_df = pd.concat(all_preds, ignore_index=True)
    else:
        full_pred_df = pd.DataFrame(columns=['영업일자','영업장명_메뉴명','매출수량'])

    return full_pred_df
