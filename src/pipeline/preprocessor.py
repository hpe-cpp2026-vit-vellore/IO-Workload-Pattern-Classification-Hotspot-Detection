"""
src/pipeline/preprocessor.py

Loads engineered features, applies robust scaling, and creates
chronological train/val/test splits.

CPP3 Objectives:
- CHRONOLOGICAL split (not random!) for time-series data
- Scaler fitted on train only (no data leakage)

Inputs  : data/processed/io_features.parquet
Outputs : data/features/X_train.parquet  → X_val.parquet  → X_test.parquet
          data/features/y_train.parquet  → y_val.parquet  → y_test.parquet
          models/scaler.pkl
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "io_features.parquet"
FEAT_DIR   = ROOT / "data" / "features"
MODEL_DIR  = ROOT / "models"

# ── Config ───────────────────────────────────────────────────────────────────
TRAIN_DAYS  = 21
VAL_DAYS    = 4

# Columns to drop before training (non-numeric / identifiers / leaky features)
DROP_COLS = [
    # ── Identifiers (not features) ──
    "workload_type",          # Text label (we use 'label' instead)
    "volume_id",              # Volume identifier
    "node_id",                # Node identifier
    "pool_id",                # Pool identifier
    "tier",                   # Storage tier (categorical)
    "timestamp",              # Time identifier
    
    # ── LEAKY: Capacity features (volume-specific, not workload-specific) ──
    "capacity_total_gb",      # Fixed per volume → memorization
    "capacity_headroom_gb",   # Derived from total → memorization
    "capacity_used_gb",       # Volume-specific usage → memorization
    "capacity_used_pct",      # Volume-specific percentage → memorization
    "capacity_burn_rate",     # Derived from capacity_used_gb → memorization
    
    # ── LEAKY: Capacity lag/delta features (derived from capacity_used_pct) ──
    "capacity_used_pct_lag_5m",
    "capacity_used_pct_lag_15m",
    "capacity_used_pct_lag_30m",
    "capacity_used_pct_lag_60m",
    "capacity_used_pct_delta",
    "capacity_used_pct_pct_change",
    
    # ── LEAKY: Time features that reveal train/val/test split ──
    "day_of_month",           # Chronological split → reveals which split
]
LABEL_COL = "label"


def load_features(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load engineered feature DataFrame from Parquet."""
    print(f"Loading features from: {path}")
    df = pd.read_parquet(path)
    print(f"  Shape : {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


def split_chronological(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically into 21/4/5 days based on the dataset start date.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    start_date = df["timestamp"].min().normalize()
    train_end = start_date + pd.Timedelta(days=TRAIN_DAYS)
    val_end = train_end + pd.Timedelta(days=VAL_DAYS)

    train_df = df[df["timestamp"] < train_end]
    val_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end)]
    test_df = df[df["timestamp"] >= val_end]

    print("\nChronological split:")
    print(f"  Train: {train_df['timestamp'].min()} → {train_df['timestamp'].max()} ({len(train_df):,})")
    print(f"  Val  : {val_df['timestamp'].min()} → {val_df['timestamp'].max()} ({len(val_df):,})")
    print(f"  Test : {test_df['timestamp'].min()} → {test_df['timestamp'].max()} ({len(test_df):,})")

    return train_df, val_df, test_df


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


def compute_iqr_bounds(
    df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Compute global IQR bounds for numeric columns (train-only)."""
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def clip_outliers(
    df: pd.DataFrame,
    cols: list[str],
    bounds: tuple[pd.Series, pd.Series],
) -> pd.DataFrame:
    """Clip numeric columns using global IQR bounds (train-derived)."""
    df = df.copy()
    low, high = bounds
    df[cols] = df[cols].clip(lower=low, upper=high, axis=1)
    return df


def signed_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """Signed log1p transform for numeric features (handles negatives)."""
    values = np.sign(df.to_numpy()) * np.log1p(np.abs(df.to_numpy()))
    return pd.DataFrame(values, columns=df.columns, index=df.index)


def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Apply signed log1p, then fit RobustScaler on training data only.

    ⚠️  NEVER fit on val/test — that would leak future information.
    """
    feature_names = X_train.columns.tolist()

    X_train_t = signed_log1p(X_train)
    X_val_t = signed_log1p(X_val)
    X_test_t = signed_log1p(X_test)

    scaler = RobustScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train_t),
        columns=feature_names,
        index=X_train.index,
    ).astype("float32")
    X_val_s = pd.DataFrame(
        scaler.transform(X_val_t),
        columns=feature_names,
        index=X_val.index,
    ).astype("float32")
    X_test_s = pd.DataFrame(
        scaler.transform(X_test_t),
        columns=feature_names,
        index=X_test.index,
    ).astype("float32")

    print("\nScaling done — robust scaling on training set.")
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


def save_scaler(scaler: RobustScaler) -> None:
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

    df = load_features().drop_duplicates()

    print("\nCreating chronological 21/4/5 day splits...")
    train_df, val_df, test_df = split_chronological(df)

    # Train-derived outlier clipping (per workload)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if LABEL_COL in numeric_cols:
        numeric_cols.remove(LABEL_COL)

    bounds = compute_iqr_bounds(train_df, numeric_cols)
    train_df = clip_outliers(train_df, numeric_cols, bounds)
    val_df = clip_outliers(val_df, numeric_cols, bounds)
    test_df = clip_outliers(test_df, numeric_cols, bounds)

    print("\nExtracting features and labels...")
    X_train, y_train = split_features_labels(train_df)
    X_val, y_val = split_features_labels(val_df)
    X_test, y_test = split_features_labels(test_df)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    print("\nScaling features (fit on train only)...")
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    print("\nSaving splits to data/features/...")
    save_splits(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)

    print("\nSaving scaler to models/...")
    save_scaler(scaler)

    print("\n✅ Preprocessing complete!")
    print(f"   Feature columns ({X_train_s.shape[1]}): {list(X_train_s.columns)}")
