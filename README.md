
# 2025_Food_and_Beverage_Outlet_Menu_Demand_Forecasting_(LG_Aimers_7th)

## Competition result
- **13th place** out of 817 teams, *Team Bellissimo*
- **Top 1.5%**
- [Competition link](https://dacon.io/competitions/official/236559/overview/description)

---

## Overview
**Task**: Forecast daily menu-level sales demand for food & beverage outlets and generate a submission file
**Key ideas**:
   - Periodic features (day of week, month) + holiday flag
   - Log-transformed sales and rolling statistics
   - Multi-output LSTM with menu ID embeddings
   - Date-based K-Fold cross-validation + fold ensembling
**Outputs**:
   - outputs/models/model_fold{K}.pth (fold-level models)
   - outputs/scaler.pkl (scaler and mapping info)
   - outputs/submission_10Fold_15Epoch.csv (final submission file)
   - Reference: Modularized from the original notebook into a reproducible pipeline

---

## Repository structure
```
.
├─ data/                        # train/test/sample_submission
├─ outputs/                     # saved models, scaler, predictions
├─ src/
│  ├─ config.py                 # configs
│  ├─ seed.py                   # reproducible seed
│  ├─ features.py               # feature engineering
│  ├─ data.py                   # data loading & menu mapping
│  ├─ dataset.py                # PyTorch Dataset
│  ├─ model.py                  # LSTM with embedding
│  ├─ losses.py                 # Scaled SMAPE loss
│  ├─ kfold.py                  # date-based K-Fold splits
│  ├─ optimizers.py             # RAdam + Lookahead
│  ├─ train.py                  # training loop
│  ├─ predict.py                # fold-ensemble inference
│  ├─ submission.py             # convert predictions → submission
│  └─ main.py                   # CLI entrypoint
├─ scripts/
│  ├─ train_folds.sh            # training script
│  └─ predict_and_submit.sh     # inference + submission script
├─ requirements.txt
└─ README.md
```
---

## Requirements
- Python 3.10+
- NVIDIA GPU recommended (PyTorch CUDA)
- Installation:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data
- The following files are required under data/:
   train/train.csv
   test/TEST_*.csv
   sample_submission.csv
- On Colab: automatically mounts at /content/drive/MyDrive/Colab Notebooks
- Locally: defaults to ./data/

---

## Configuration
src/config.py:
```bash
lookback = 28
predict = 7
batch_size = 32
epochs = 15
folds = 10
device = "cuda"
zero_weight = 0.001
```

---

## How to run

### End-to-end pipeline

Training:
```bash
python -m src.main train --data_dir ./data --out_dir ./outputs
```

Inference + submission::
```bash
python -m src.main predict --data_dir ./data --out_dir ./outputs
```

---

## Modeling
- **Group-wise training**: sequence per menu → Multi-output LSTM + embedding
- **Loss/metrics**: Scaled SMAPE
- **K-Fold**: date-based, no overlap between train/val periods
- **Ensembling**: average of fold predictions

---

## Features
- Date-based: weekday, month, sine/cosine seasonal encodings
- Holiday flag: based on holidays.KR()
- Log transform: log1p applied to sales
- Rolling statistics: mean, median, std (7/14/28 days)

---

## Outputs
- Intermediate: outputs/models/model_fold{K}.pth
- Scaler + mapping: outputs/scaler.pkl
- Final submission: outputs/submission_10Fold_15Epoch.csv

## Reproducibility
- Fixed seeds: Python random, NumPy, Torch
- Fold split: deterministic, date-based

# Acknowledgements
- Modularized from the original notebook into a reproducible pipeline
- Data and task definition based on the official competition