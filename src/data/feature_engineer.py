"""
src/data/feature_engineer.py
Reads raw synthetic IO data and engineers ML-ready features.

Blueprint Requirements (Page 7, Section 1.3-1.4):
- Rolling window features (1min, 5min, 15min, 1hr)
- Rate of change features (delta and percentage change)
- IO size entropy, latency jitter, lag features
- Capacity burn rate
- Cyclical time encodings (sin/cos for hour and day_of_week)

Input  : data/synthetic/io_workload_data.parquet
Output : data/processed/io_features.parquet
         data/processed/io_features.csv
"""

import numpy as np
import pandas as pd
import os

# ── Label map: workload type text → integer for ML models ──────────────────
LABEL_MAP = {
    "DB_OLTP"     : 0,
    "VM"          : 1,
    "Backup"      : 2,
    "AI_Training" : 3,
    "AI_Inference": 4,
}


def load_raw_data(path: str = "data/synthetic/io_workload_data.parquet") -> pd.DataFrame:
    """
    Load raw synthetic data from parquet.
    Handles both single-file and partitioned parquet formats.
    """
    print(f"Loading raw data from {path} ...")
    df = pd.read_parquet(path)  # PyArrow automatically handles partitioned directories
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning before feature engineering.
    - Remove duplicates
    - Ensure timestamp is datetime
    - Forward-fill numeric gaps per volume (no backfill to avoid leakage)
    """
    df = df.drop_duplicates().copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Sort once for all time-series operations
    df = df.sort_values(["volume_id", "timestamp"]).reset_index(drop=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df.groupby("volume_id", sort=False, observed=True)[num_cols].ffill()
    df[num_cols] = df[num_cols].fillna(0)

    return df


def engineer_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic derived features from raw IO metrics.
    """
    # 1. IOPS per queue depth (efficiency)
    df["iops_per_queue"] = (df["total_iops"] / df["queue_depth"]).round(4)

    # 2. Total throughput
    df["total_throughput_mbps"] = (df["read_throughput_mbps"] + df["write_throughput_mbps"]).round(2)

    # 3. Write pressure (1 - read_write_ratio)
    df["write_pressure"] = (1 - df["read_write_ratio"]).round(4)

    # 4. Latency jitter (p99 - p50) for reads and writes
    df["read_latency_jitter"] = (df["read_latency_p99_us"] - df["read_latency_p50_us"]).round(2)
    df["write_latency_jitter"] = (df["write_latency_p99_us"] - df["write_latency_p50_us"]).round(2)

    # 5. Average latency (mean of p50)
    df["avg_latency_us"] = ((df["read_latency_p50_us"] + df["write_latency_p50_us"]) / 2).round(2)

    # 6. IOPS to latency performance score
    df["iops_latency_score"] = (df["total_iops"] / (df["avg_latency_us"] + 1)).round(4)

    # 7. Capacity burn rate (GB per day)
    df["capacity_burn_rate"] = (
        df.groupby("volume_id", sort=False, observed=True)["capacity_used_gb"]
        .diff()            # GB gained since last minute
        .fillna(0)
        * 1440            # scale to GB/day (1440 min/day)
    )

    # 8. Capacity headroom (remaining capacity)
    df["capacity_headroom_gb"] = (df["capacity_total_gb"] - df["capacity_used_gb"]).round(2)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features with cyclical encoding.
    Blueprint requirement: hour, day_of_week, is_weekend, sin/cos encodings
    """
    # Extract time components
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = df["timestamp"].dt.day
    
    # Cyclical encoding for hour (0-23)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).round(4)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).round(4)
    
    # Cyclical encoding for day_of_week (0-6)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).round(4)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).round(4)
    
    return df


def add_io_size_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IO size entropy per volume per hour to capture randomness signature.
    """
    if "io_size_avg_kb" not in df.columns:
        return df

    df["hour_bucket"] = df["timestamp"].dt.floor("h")

    counts = (
        df.groupby(["volume_id", "hour_bucket", "io_size_avg_kb"], sort=False, observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["prob"] = counts["count"] / counts.groupby(
        ["volume_id", "hour_bucket"], sort=False, observed=True
    )["count"].transform("sum")
    counts["entropy_component"] = np.where(
        counts["prob"] > 0,
        -counts["prob"] * np.log2(counts["prob"]),
        0.0,
    )

    entropy_df = (
        counts.groupby(["volume_id", "hour_bucket"], sort=False, observed=True)["entropy_component"]
        .sum()
        .rename("io_size_entropy")
        .reset_index()
    )

    df = df.merge(entropy_df, on=["volume_id", "hour_bucket"], how="left", sort=False)
    df = df.drop(columns=["hour_bucket"])

    return df


def add_rolling_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window features.
    With 5-minute intervals: 3 rows = 15min, 6 rows = 30min, 12 rows = 1hr.
    Rolling mean and std for key metrics over windows.
    """
    print("  Computing rolling window features (this may take a while)...")
    
    # Define windows (in rows; each row = 5 minutes)
    windows = {3: "15m", 6: "30m", 12: "1h"}  # rows → label
    
    # Metrics to compute rolling features for
    metrics = ["total_iops", "avg_latency_us", "total_throughput_mbps"]
    
    for window_rows, label in windows.items():
        for metric in metrics:
            # Rolling mean
            df[f"{metric}_roll_{label}_mean"] = (
                df.groupby("volume_id", sort=False, observed=True)[metric]
                .rolling(window=window_rows, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
                .round(2)
            )
            
            # Rolling std
            df[f"{metric}_roll_{label}_std"] = (
                df.groupby("volume_id", sort=False, observed=True)[metric]
                .rolling(window=window_rows, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0)
                .round(2)
            )
    
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features.
    With 5-minute intervals: 1 row = 5min, 3 rows = 15min, 12 rows = 60min.
    Blueprint requirement: lag features for time-series patterns.
    """
    print("  Computing lag features...")
    
    # Lag periods (in rows; each row = 5 minutes)
    lags = {1: "5m", 3: "15m", 6: "30m", 12: "60m"}
    
    # Metrics to lag
    metrics = ["total_iops", "avg_latency_us", "capacity_used_pct"]
    
    for lag_rows, label in lags.items():
        for metric in metrics:
            df[f"{metric}_lag_{label}"] = (
                df.groupby("volume_id", sort=False, observed=True)[metric]
                .shift(lag_rows)
                .fillna(0)
                .round(2)
            )
    
    return df


def add_rate_of_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rate of change features (delta and percentage change).
    Blueprint requirement: delta (current - previous) and % change per metric.
    """
    print("  Computing rate of change features...")
    
    # Metrics to compute rate of change
    metrics = ["total_iops", "avg_latency_us", "capacity_used_pct"]
    
    for metric in metrics:
        # Delta (current - previous)
        df[f"{metric}_delta"] = (
            df.groupby("volume_id", sort=False, observed=True)[metric]
            .diff()
            .fillna(0)
            .round(2)
        )
        
        # Percentage change
        df[f"{metric}_pct_change"] = (
            df.groupby("volume_id", sort=False, observed=True)[metric]
            .pct_change()
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
            .round(4)
        )
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline.
    Combines all feature engineering steps.
    """
    print("\nEngineering features...")
    
    # 0. Clean data and ensure sorted order
    print("  Step 0/7: Cleaning raw data...")
    df = clean_raw_data(df)

    # 1. Basic derived features
    print("  Step 1/7: Basic derived features...")
    df = engineer_basic_features(df)
    
    # 2. Time-based features
    print("  Step 2/7: Time-based features...")
    df = add_time_features(df)

    # 3. IO size entropy
    print("  Step 3/7: IO size entropy...")
    df = add_io_size_entropy(df)
    
    # 4. Rolling window features (expensive!)
    print("  Step 4/7: Rolling window features...")
    df = add_rolling_window_features(df)
    
    # 5. Lag features
    print("  Step 5/7: Lag features...")
    df = add_lag_features(df)
    
    # 6. Rate of change features
    print("  Step 6/7: Rate of change features...")
    df = add_rate_of_change_features(df)
    
    # 7. Encode workload_type text → integer label for ML
    print("  Step 7/7: Label encoding...")
    df["label"] = df["workload_type"].map(LABEL_MAP)

    # Downcast numeric columns to reduce memory usage
    int_cols = df.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype("int32")
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    
    return df


def validate_features(df: pd.DataFrame) -> None:
    """Quick sanity checks on engineered features."""
    print("\n── Validation ──────────────────────────────────")

    nan_count = df.isnull().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
    print(f"NaN values  : {nan_count}  ({'✅' if nan_count == 0 else '❌ FIX NEEDED'})")
    print(f"Inf values  : {inf_count}  ({'✅' if inf_count == 0 else '❌ FIX NEEDED'})")

    labels_found = sorted(df["label"].unique())
    print(f"Labels found: {labels_found}  ({'✅' if labels_found == [0,1,2,3,4] else '❌'})")

    expected_rows = 50 * 30 * 24 * (60 // 5)  # 432,000 with 5-min intervals
    print(f"Total rows  : {len(df):,}  ({'✅' if len(df) == expected_rows else '❌'})")
    print("────────────────────────────────────────────────\n")


def save_features(df: pd.DataFrame) -> None:
    """Save engineered features to data/processed/."""
    os.makedirs("data/processed", exist_ok=True)

    parquet_path = "data/processed/io_features.parquet"
    csv_path     = "data/processed/io_features.csv"

    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet → {parquet_path}")

    df.to_csv(csv_path, index=False)
    print(f"Saved CSV     → {csv_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print feature summary per workload type."""
    print("\n── Feature Means per Workload Type ─────────────")
    summary_cols = [
        "iops_per_queue",
        "write_pressure",
        "iops_latency_score",
        "capacity_burn_rate",
        "total_iops",
        "avg_latency_us",
    ]
    available_cols = [col for col in summary_cols if col in df.columns]
    summary = df.groupby("workload_type", observed=True)[available_cols].mean().round(2)
    print(summary.to_string())
    print("────────────────────────────────────────────────")
    print(f"\nFinal columns ({df.shape[1]}):")
    print(list(df.columns))


if __name__ == "__main__":
    print("=" * 70)
    print(" Feature Engineering Pipeline")
    print(" Blueprint-Compliant: Time-series + Rolling + Lag + Rate-of-Change")
    print("=" * 70)

    df_raw = load_raw_data()

    df_featured = engineer_features(df_raw)

    validate_features(df_featured)

    save_features(df_featured)

    print_summary(df_featured)

    print("\n✅ Feature engineering complete!")
    print(f"   Output: {df_featured.shape[0]:,} rows × {df_featured.shape[1]} columns")