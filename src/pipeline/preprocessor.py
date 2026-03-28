"""
src/pipeline/preprocessor.py

Loads engineered features, applies StandardScaler, and creates
stratified 70/15/15 train/val/test splits.

Inputs  : data/processed/io_features.parquet
Outputs : data/features/X_train.parquet  → X_val.parquet  → X_test.parquet
          data/features/y_train.parquet  → y_val.parquet  → y_test.parquet
          models/scaler.pkl
"""

import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "io_features.parquet"
FEAT_DIR   = ROOT / "data" / "features"
MODEL_DIR  = ROOT / "models"

# ── Config ───────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
TEST_SIZE    = 0.30   # 30% for val+test combined
VAL_FRACTION = 0.50   # 50% of that 30% → 15% val, 15% test

# Columns to drop before training (non-numeric / redundant)
DROP_COLS = ["workload_type"]
LABEL_COL = "label"


def load_features(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load engineered feature DataFrame from Parquet."""
    print(f"Loading features from: {path}")
    df = pd.read_parquet(path)
    print(f"  Shape : {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X from target vector y.

    Returns
    -------
    X : pd.DataFrame  — all numeric feature columns
    y : pd.Series     — integer class labels (0–4)
    """
    X = df.drop(columns=DROP_COLS + [LABEL_COL], errors="ignore")
    y = df[LABEL_COL]
    print(f"  Features : {X.shape[1]} columns | Samples : {len(X):,}")
    print(f"  Label distribution:\n{y.value_counts().sort_index().to_string()}")
    return X, y


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    """
    Create stratified 70 / 15 / 15 train / val / test splits.

    Stratification ensures each split has the same class proportions
    as the original dataset — critical for 5-class classification.
    """
    # Step 1: split off 30% (val + test) from 70% (train)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Step 2: split the 30% equally into val (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(X_train):>7,}  ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val   : {len(X_val):>7,}  ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test  : {len(X_test):>7,}  ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on training data only, then transform all splits.

    ⚠️  NEVER fit on val/test — that would leak future information.
    """
    feature_names = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val),
        columns=feature_names,
        index=X_val.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )

    print(f"\nScaling done — mean≈0, std≈1 on training set.")
    return X_train_s, X_val_s, X_test_s, scaler


def save_splits(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series,    y_val: pd.Series,    y_test: pd.Series,
) -> None:
    """Persist all six split files as Parquet for fast loading later."""
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train.to_frame(), "y_val": y_val.to_frame(),
        "y_test" : y_test.to_frame(),
    }
    for name, data in splits.items():
        path = FEAT_DIR / f"{name}.parquet"
        data.to_parquet(path, index=False)
        print(f"  Saved {name:>8} → {path.relative_to(ROOT)}")


def save_scaler(scaler: StandardScaler) -> None:
    """Persist fitted scaler so inference never needs to refit."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "scaler.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler  → {path.relative_to(ROOT)}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(" Preprocessing Pipeline")
    print("=" * 55)

    df = load_features()

    print("\nExtracting features and labels...")
    X, y = split_features_labels(df)

    print("\nCreating stratified 70/15/15 splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    print("\nScaling features (fit on train only)...")
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    print("\nSaving splits to data/features/...")
    save_splits(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)

    print("\nSaving scaler to models/...")
    save_scaler(scaler)

    print("\n✅ Preprocessing complete!")
    print(f"   Feature columns ({X_train_s.shape[1]}): {list(X_train_s.columns)}")
