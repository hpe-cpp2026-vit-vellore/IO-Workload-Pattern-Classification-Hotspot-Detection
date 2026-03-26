"""
src/data/feature_engineer.py
Reads raw synthetic IO data and engineers ML-ready features.
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
    """Load raw synthetic data from parquet."""
    print(f"Loading raw data from {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw IO metrics.

    Feature logic:
    ─────────────────────────────────────────────────────────────────────────
    iops_per_queue          → IOPS efficiency per queue slot.
                              High = DB_OLTP (many IOs, few queues)
                              Low  = Backup  (few IOs, small queues)

    bytes_per_io_kb         → Average size of each IO operation in KB.
                              throughput(MB/s) × 1024 / iops = KB per IO
                              High = Backup/AI_Training (large sequential IOs)
                              Low  = DB_OLTP (tiny 4KB random IOs)

    write_pressure          → How write-heavy is this workload.
                              1 - read_ratio
                              High = Backup (90% writes)
                              Low  = AI_Training (92% reads)

    latency_bandwidth_ratio → Latency cost per MB/s of throughput.
                              latency_us / throughput_mbps
                              High = Backup (high latency, high throughput)
                              Low  = DB_OLTP (low latency, moderate throughput)

    disk_pressure_score     → Combined disk stress indicator.
                              (disk_util_pct × latency_us) / 1000
                              High = AI_Training + Backup
                              Low  = VM + AI_Inference

    throughput_efficiency   → How much throughput per % disk utilization.
                              throughput_mbps / disk_util_pct
                              High = AI_Training (massive throughput, high util)
                              Low  = DB_OLTP (modest throughput, moderate util)

    iops_latency_score      → Performance score: more IOPS, less latency = better.
                              iops / latency_us
                              High = DB_OLTP (high IOPS, low latency)
                              Low  = Backup  (low IOPS, high latency)
    ─────────────────────────────────────────────────────────────────────────
    """
    df = df.copy()

    # 1. IOPS per queue depth
    df["iops_per_queue"] = (df["iops"] / df["queue_depth"]).round(4)

    # 2. Average IO size in KB (throughput MB/s → KB/s ÷ IOPS = KB per IO)
    df["bytes_per_io_kb"] = (
        (df["throughput_mbps"] * 1024) / df["iops"]
    ).round(4)

    # 3. Write pressure (inverse of read ratio)
    df["write_pressure"] = (1 - df["read_ratio"]).round(4)

    # 4. Latency cost per unit of throughput
    df["latency_bandwidth_ratio"] = (
        df["latency_us"] / df["throughput_mbps"]
    ).round(4)

    # 5. Combined disk stress score
    df["disk_pressure_score"] = (
        (df["disk_util_pct"] * df["latency_us"]) / 1000
    ).round(4)

    # 6. Throughput efficiency per % disk utilization
    df["throughput_efficiency"] = (
        df["throughput_mbps"] / df["disk_util_pct"]
    ).round(4)

    # 7. IOPS to latency performance score
    df["iops_latency_score"] = (
        df["iops"] / df["latency_us"]
    ).round(4)

    # 8. Encode workload_type text → integer label for ML
    df["label"] = df["workload_type"].map(LABEL_MAP)

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

    print(f"Total rows  : {len(df):,}  ({'✅' if len(df) == 500_000 else '❌'})")
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
    summary = df.groupby("workload_type")[[
        "iops_per_queue",
        "bytes_per_io_kb",
        "write_pressure",
        "disk_pressure_score",
        "iops_latency_score",
    ]].mean().round(2)
    print(summary.to_string())
    print("────────────────────────────────────────────────")
    print(f"\nFinal columns ({df.shape[1]}):")
    print(list(df.columns))


if __name__ == "__main__":
    print("=" * 55)
    print(" Feature Engineering Pipeline")
    print("=" * 55)

    df_raw      = load_raw_data()

    print("\nEngineering features...")
    df_featured = engineer_features(df_raw)

    validate_features(df_featured)

    save_features(df_featured)

    print_summary(df_featured)

    print("\n✅ Feature engineering complete!")