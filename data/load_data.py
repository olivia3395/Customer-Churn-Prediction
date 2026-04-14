"""
Step 1 — Download & Prepare Data
=================================
Downloads the IBM Telco Customer Churn dataset directly from GitHub,
cleans it, engineers new features, and splits into train/test/production sets.

Usage:
    python data/load_data.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)
SAVE_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RAW_PATH   = os.path.join(SAVE_DIR, "telco_raw.csv")
TRAIN_PATH = os.path.join(SAVE_DIR, "train.csv")
TEST_PATH  = os.path.join(SAVE_DIR, "test.csv")
PROD_PATH  = os.path.join(SAVE_DIR, "production_sample.csv")


# ── Download ──────────────────────────────────────────────────────────────────
def download(url: str, save_path: str) -> pd.DataFrame:
    if os.path.exists(save_path):
        print(f"[CACHE] Found {save_path}, skipping download.")
        return pd.read_csv(save_path)

    print(f"[DOWNLOAD] Fetching dataset from GitHub...")
    df = pd.read_csv(url)
    df.to_csv(save_path, index=False)
    print(f"[DOWNLOAD] Saved {len(df)} rows → {save_path}")
    return df


# ── Clean ─────────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is stored as string and has spaces for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # New customers (tenure=0) have NaN TotalCharges → fill with MonthlyCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Target: Yes/No → 1/0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Drop ID column
    df.drop(columns=["customerID"], inplace=True)

    return df


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Monthly cost efficiency — high charge relative to tenure suggests churn risk
    df["charges_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    # How many add-on services does the customer use? (stickiness indicator)
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

    # Long-term customer flag
    df["is_long_term"] = (df["tenure"] >= 24).astype(int)

    return df


# ── Split ─────────────────────────────────────────────────────────────────────
def split_and_save(df: pd.DataFrame) -> None:
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Churn"]
    )

    # Simulate production data: copy test set with slight drift on MonthlyCharges
    np.random.seed(0)
    prod_df = test_df.copy()
    prod_df["MonthlyCharges"] += np.random.normal(6, 3, len(prod_df))
    prod_df["MonthlyCharges"] = prod_df["MonthlyCharges"].clip(lower=10)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH,   index=False)
    prod_df.to_csv(PROD_PATH,   index=False)

    print(f"[SPLIT] Train: {len(train_df)} | Test: {len(test_df)} | Prod: {len(prod_df)}")
    print(f"[SPLIT] Churn rate — Train: {train_df['Churn'].mean():.2%} | Test: {test_df['Churn'].mean():.2%}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    df = download(DATASET_URL, RAW_PATH)

    print(f"\n[INFO] Raw shape:    {df.shape}")
    print(f"[INFO] Columns:      {list(df.columns)}")
    print(f"[INFO] Null counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    df = clean(df)
    df = engineer_features(df)

    print(f"\n[INFO] Cleaned shape: {df.shape}")
    print(f"[INFO] Overall Churn rate: {df['Churn'].mean():.2%}")

    split_and_save(df)
    print("\n[DONE] Data ready. Run: python src/train.py")
