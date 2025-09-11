
import os
import argparse
import joblib
import torch
import holidays

from .config import TrainConfig
from .data import load_train, load_sample_submission, load_all_test_paths
from .train import train_folds
from .predict import predict_for_all_tests
from .submission import convert_to_submission_format

def cmd_train(args):
    cfg = TrainConfig(
        lookback=args.lookback,
        predict=args.predict,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        emb_dim=args.emb_dim,
        dropout_p=args.dropout,
        folds=args.folds,
        device=("cuda" if args.device == "auto" else args.device),
        zero_weight=args.zero_weight
    )
    train_df = load_train(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    train_folds(train_df, args.out_dir, cfg)

def cmd_predict(args):
    artifacts_path = os.path.join(args.out_dir, "scaler.pkl")
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Artifacts not found: {artifacts_path}. Run train first.")

    artifacts = joblib.load(artifacts_path)

    test_paths = load_all_test_paths(args.data_dir)
    if not test_paths:
        print(f"Warning: No test files found in {os.path.join(args.data_dir, 'test')}.")
    full_pred_df = predict_for_all_tests(
        test_paths=test_paths,
        out_dir=args.out_dir,
        artifacts=artifacts,
        lookback=args.lookback,
        predict_h=args.predict,
        device=("cuda" if args.device == "auto" else args.device),
    )

    sample_submission = load_sample_submission(args.data_dir)
    submission = convert_to_submission_format(full_pred_df, sample_submission)

    out_csv = os.path.join(args.out_dir, f"submission_{args.folds}Fold_{args.epochs}Epoch.csv")
    submission.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {out_csv}")

def build_parser():
    p = argparse.ArgumentParser(description="Sales forecasting LSTM (modular)")
    sub = p.add_subparsers(dest="cmd")

    # train
    t = sub.add_parser("train", help="Train K folds")
    t.add_argument("--data_dir", required=True, type=str)
    t.add_argument("--out_dir", required=True, type=str)
    t.add_argument("--lookback", type=int, default=28)
    t.add_argument("--predict", type=int, default=7)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden_dim", type=int, default=64)
    t.add_argument("--num_layers", type=int, default=2)
    t.add_argument("--emb_dim", type=int, default=8)
    t.add_argument("--dropout", type=float, default=0.3)
    t.add_argument("--folds", type=int, default=10)
    t.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    t.add_argument("--zero_weight", type=float, default=0.001)
    t.set_defaults(func=cmd_train)

    # predict
    pr = sub.add_parser("predict", help="Run inference on TEST_* and create submission")
    pr.add_argument("--data_dir", required=True, type=str)
    pr.add_argument("--out_dir", required=True, type=str)
    pr.add_argument("--lookback", type=int, default=28)
    pr.add_argument("--predict", type=int, default=7)
    pr.add_argument("--folds", type=int, default=10)
    pr.add_argument("--epochs", type=int, default=15)
    pr.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    pr.set_defaults(func=cmd_predict)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
