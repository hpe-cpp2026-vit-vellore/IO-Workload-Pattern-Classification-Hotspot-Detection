"""
src/data/data_generator.py
Generates 500K rows of realistic synthetic IO workload data
for 5 workload types: DB_OLTP, VM, Backup, AI_Training, AI_Inference
"""

import numpy as np
import pandas as pd
import os

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Total rows per workload type
ROWS_PER_WORKLOAD = 100_000  # 5 types × 100K = 500K total


def generate_db_oltp(n: int) -> pd.DataFrame:
    """
    Database OLTP — high IOPS, low latency, small random reads/writes
    """
    return pd.DataFrame({
        "workload_type"  : ["DB_OLTP"] * n,
        "iops"           : np.random.normal(45000, 5000, n).clip(10000, 80000),
        "throughput_mbps": np.random.normal(180, 30, n).clip(50, 400),
        "latency_us"     : np.random.normal(500, 100, n).clip(100, 2000),
        "read_ratio"     : np.random.normal(0.70, 0.08, n).clip(0.3, 0.95),
        "block_size_kb"  : np.random.choice([4, 8, 16], n, p=[0.6, 0.3, 0.1]),
        "queue_depth"    : np.random.randint(8, 32, n),
        "disk_util_pct"  : np.random.normal(65, 10, n).clip(20, 95),
    })


def generate_vm(n: int) -> pd.DataFrame:
    """
    Virtual Machines — mixed read/write, bursty patterns
    """
    return pd.DataFrame({
        "workload_type"  : ["VM"] * n,
        "iops"           : np.random.normal(12000, 4000, n).clip(1000, 40000),
        "throughput_mbps": np.random.normal(120, 40, n).clip(20, 300),
        "latency_us"     : np.random.normal(1500, 400, n).clip(200, 5000),
        "read_ratio"     : np.random.normal(0.55, 0.15, n).clip(0.2, 0.9),
        "block_size_kb"  : np.random.choice([8, 16, 64], n, p=[0.3, 0.5, 0.2]),
        "queue_depth"    : np.random.randint(4, 24, n),
        "disk_util_pct"  : np.random.normal(50, 15, n).clip(10, 90),
    })


def generate_backup(n: int) -> pd.DataFrame:
    """
    Backup/Archive — large sequential writes, low IOPS, high throughput
    """
    return pd.DataFrame({
        "workload_type"  : ["Backup"] * n,
        "iops"           : np.random.normal(800, 200, n).clip(100, 2000),
        "throughput_mbps": np.random.normal(900, 100, n).clip(400, 1400),
        "latency_us"     : np.random.normal(8000, 2000, n).clip(2000, 20000),
        "read_ratio"     : np.random.normal(0.10, 0.05, n).clip(0.01, 0.3),
        "block_size_kb"  : np.random.choice([256, 512, 1024], n, p=[0.3, 0.5, 0.2]),
        "queue_depth"    : np.random.randint(1, 8, n),
        "disk_util_pct"  : np.random.normal(80, 8, n).clip(40, 98),
    })


def generate_ai_training(n: int) -> pd.DataFrame:
    """
    AI Training — massive sequential reads, very high bandwidth
    """
    return pd.DataFrame({
        "workload_type"  : ["AI_Training"] * n,
        "iops"           : np.random.normal(5000, 1000, n).clip(1000, 15000),
        "throughput_mbps": np.random.normal(2500, 300, n).clip(800, 4000),
        "latency_us"     : np.random.normal(3000, 800, n).clip(500, 10000),
        "read_ratio"     : np.random.normal(0.92, 0.04, n).clip(0.75, 1.0),
        "block_size_kb"  : np.random.choice([256, 512, 1024], n, p=[0.2, 0.4, 0.4]),
        "queue_depth"    : np.random.randint(16, 64, n),
        "disk_util_pct"  : np.random.normal(88, 6, n).clip(60, 99),
    })


def generate_ai_inference(n: int) -> pd.DataFrame:
    """
    AI Inference — spiky low-latency reads, unpredictable bursts
    """
    return pd.DataFrame({
        "workload_type"  : ["AI_Inference"] * n,
        "iops"           : np.random.normal(22000, 8000, n).clip(2000, 60000),
        "throughput_mbps": np.random.normal(350, 100, n).clip(50, 800),
        "latency_us"     : np.random.normal(800, 300, n).clip(100, 3000),
        "read_ratio"     : np.random.normal(0.85, 0.07, n).clip(0.6, 1.0),
        "block_size_kb"  : np.random.choice([4, 8, 16, 32], n, p=[0.4, 0.3, 0.2, 0.1]),
        "queue_depth"    : np.random.randint(2, 16, n),
        "disk_util_pct"  : np.random.normal(55, 18, n).clip(10, 95),
    })


def generate_all(n_per_workload: int = ROWS_PER_WORKLOAD) -> pd.DataFrame:
    """
    Combines all 5 workload types into one shuffled DataFrame
    """
    generators = [
        generate_db_oltp,
        generate_vm,
        generate_backup,
        generate_ai_training,
        generate_ai_inference,
    ]

    frames = [gen(n_per_workload) for gen in generators]
    df = pd.concat(frames, ignore_index=True)

    # Shuffle rows so workloads are mixed
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Round for cleanliness
    df["iops"]            = df["iops"].round(0).astype(int)
    df["throughput_mbps"] = df["throughput_mbps"].round(2)
    df["latency_us"]      = df["latency_us"].round(2)
    df["read_ratio"]      = df["read_ratio"].round(4)
    df["disk_util_pct"]   = df["disk_util_pct"].round(2)

    return df


if __name__ == "__main__":
    print("Generating 500K rows of synthetic IO data...")

    df = generate_all()

    # Save as CSV
    os.makedirs("data/synthetic", exist_ok=True)
    csv_path = "data/synthetic/io_workload_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV  → {csv_path}")

    # Save as Parquet (faster for ML pipelines)
    parquet_path = "data/synthetic/io_workload_data.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet → {parquet_path}")

    # Quick sanity check
    print(f"\nTotal rows : {len(df):,}")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nRows per workload:")
    print(df["workload_type"].value_counts())
    print(f"\nSample stats:")
    print(df.describe())