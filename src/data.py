
import os
import pandas as pd

def load_train(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "train", "train.csv")
    return pd.read_csv(path)

def load_sample_submission(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "sample_submission.csv")
    return pd.read_csv(path)

def load_all_test_paths(data_dir: str):
    import glob
    return sorted(glob.glob(os.path.join(data_dir, "test", "TEST_*.csv")))

def load_test_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def map_menu_ids(df: pd.DataFrame):
    menu_list = sorted(df['영업장명_메뉴명'].unique())
    menu2id = {m: i for i, m in enumerate(menu_list)}
    df = df.copy()
    df['menu_id'] = df['영업장명_메뉴명'].map(menu2id)
    return df, menu2id, menu_list
