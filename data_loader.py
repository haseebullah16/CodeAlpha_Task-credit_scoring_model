# utils/data_loader.py
import os
import pandas as pd
from sklearn.datasets import make_classification

# Default public dataset URL (used if local file not present)
DEFAULT_URL = "https://raw.githubusercontent.com/dphi-official/Datasets/master/credit_risk_dataset.csv"

def download_dataset(url, save_path):
    try:
        print(f"Attempting to download dataset from:\n  {url}")
        df = pd.read_csv(url)
        df.to_csv(save_path, index=False)
        print(f"Downloaded and saved to: {save_path}")
        return df
    except Exception as e:
        print("Download failed:", e)
        return None

def load_data(local_path="data/credit_data.csv"):
    """
    Try local CSV -> try download from DEFAULT_URL -> fallback to synthetic data.
    Returns: X (DataFrame), y (Series)
    """
    # 1) If local file exists, use it
    if os.path.exists(local_path):
        print("Loading local dataset:", local_path)
        df = pd.read_csv(local_path)
    else:
        # 2) Try to download a default dataset and save it locally
        df = download_dataset(DEFAULT_URL, local_path)
        if df is None:
            # 3) Fallback: create a small synthetic dataset so pipeline continues
            print("Creating a small synthetic dataset (offline fallback).")
            Xs, ys = make_classification(n_samples=500, n_features=10, n_informative=6,
                                         n_redundant=2, random_state=42)
            df = pd.DataFrame(Xs, columns=[f"feat_{i}" for i in range(Xs.shape[1])])
            df["target"] = ys
            df.to_csv(local_path, index=False)
            print(f"Synthetic dataset saved to: {local_path}")

    # quick info
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns)[:20])

    # Decide target column:
    possible_targets = ["target", "Creditability", "credit_score", "default", "loan_status", "class"]
    target_col = None
    for cand in possible_targets:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        # if nothing found, use the last column as target
        target_col = df.columns[-1]
        print(f"No common target column found. Using last column as target: '{target_col}'")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
