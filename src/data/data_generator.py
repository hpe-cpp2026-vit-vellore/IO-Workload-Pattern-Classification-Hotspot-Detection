"""
src/data/data_generator.py
Generates 2.16M rows of realistic synthetic IO workload data
for 5 workload types: DB_OLTP, VM, Backup, AI_Training, AI_Inference

Blueprint Requirements (Page 7, Section 1.1):
- 50 volumes across 5 nodes over 30 days at 1-minute intervals
- Includes: volume_id, node_id, pool_id, tier, timestamp
- Separate read_iops/write_iops, latency percentiles (p50/p95/p99)
- Capacity metrics: capacity_used_gb, capacity_total_gb, capacity_used_pct
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

# Reproducibility seed (set inside function to ensure reproducibility)
SEED = 42

# ── Configuration per Blueprint ──────────────────────────────────────────────
NUM_VOLUMES = 50          # 50 volumes total
NUM_NODES = 5             # 5 storage nodes
NUM_DAYS = 30             # 30 days of data
INTERVAL_MINUTES = 1      # 1-minute intervals

# Total rows = 50 volumes × 30 days × 24 hours × 60 minutes = 2,160,000 rows
TOTAL_ROWS = NUM_VOLUMES * NUM_DAYS * 24 * 60

# Storage tiers
TIERS = ["NVMe", "SSD", "HDD"]

# Pool configuration (2 pools per node)
POOLS_PER_NODE = 2

# Workload types
WORKLOAD_TYPES = ["DB_OLTP", "VM", "Backup", "AI_Training", "AI_Inference"]


def generate_db_oltp(n: int) -> pd.DataFrame:
    """
    Database OLTP — high IOPS, low latency, small random reads/writes
    Blueprint: 5K-15K OPS, 4-16KB, 70/30 R/W (70% reads, 30% writes), Random, 9AM-6PM peak
    """
    total_iops = np.random.normal(10000, 2000, n).clip(5000, 15000)
    read_ratio = np.random.normal(0.70, 0.08, n).clip(0.3, 0.95)
    
    return pd.DataFrame({
        "workload_type"       : ["DB_OLTP"] * n,
        "read_iops"           : (total_iops * read_ratio).round(0).astype(int),
        "write_iops"          : (total_iops * (1 - read_ratio)).round(0).astype(int),
        "total_iops"          : total_iops.round(0).astype(int),
        "read_throughput_mbps": np.random.normal(120, 20, n).clip(30, 250),
        "write_throughput_mbps": np.random.normal(60, 15, n).clip(10, 150),
        "read_latency_p50_us" : np.random.normal(400, 80, n).clip(100, 1000),
        "read_latency_p95_us" : np.random.normal(800, 150, n).clip(300, 2000),
        "read_latency_p99_us" : np.random.normal(1200, 200, n).clip(500, 3000),
        "write_latency_p50_us": np.random.normal(500, 100, n).clip(150, 1200),
        "write_latency_p95_us": np.random.normal(1000, 180, n).clip(400, 2500),
        "write_latency_p99_us": np.random.normal(1500, 250, n).clip(600, 3500),
        "io_size_avg_kb"      : np.random.choice([4, 8, 16], n, p=[0.6, 0.3, 0.1]),
        "queue_depth"         : np.random.randint(8, 32, n),
        "sequential_ratio"    : np.random.normal(0.15, 0.05, n).clip(0.05, 0.35),  # Mostly random
        "read_write_ratio"    : read_ratio.round(4),
    })


def generate_vm(n: int) -> pd.DataFrame:
    """
    Virtual Machines — mixed read/write, bursty patterns
    Blueprint: 1K-5K OPS, 8-64KB, 60/40 R/W, Mixed, Business hours
    """
    total_iops = np.random.normal(3000, 1000, n).clip(1000, 5000)
    read_ratio = np.random.normal(0.60, 0.12, n).clip(0.3, 0.85)
    
    return pd.DataFrame({
        "workload_type"       : ["VM"] * n,
        "read_iops"           : (total_iops * read_ratio).round(0).astype(int),
        "write_iops"          : (total_iops * (1 - read_ratio)).round(0).astype(int),
        "total_iops"          : total_iops.round(0).astype(int),
        "read_throughput_mbps": np.random.normal(80, 25, n).clip(15, 200),
        "write_throughput_mbps": np.random.normal(50, 20, n).clip(10, 150),
        "read_latency_p50_us" : np.random.normal(1200, 300, n).clip(300, 3000),
        "read_latency_p95_us" : np.random.normal(2500, 500, n).clip(800, 5000),
        "read_latency_p99_us" : np.random.normal(3500, 700, n).clip(1200, 7000),
        "write_latency_p50_us": np.random.normal(1500, 350, n).clip(400, 3500),
        "write_latency_p95_us": np.random.normal(3000, 600, n).clip(1000, 6000),
        "write_latency_p99_us": np.random.normal(4200, 800, n).clip(1500, 8000),
        "io_size_avg_kb"      : np.random.choice([8, 16, 64], n, p=[0.3, 0.5, 0.2]),
        "queue_depth"         : np.random.randint(4, 24, n),
        "sequential_ratio"    : np.random.normal(0.40, 0.10, n).clip(0.2, 0.7),  # Mixed
        "read_write_ratio"    : read_ratio.round(4),
    })


def generate_backup(n: int) -> pd.DataFrame:
    """
    Backup/Archive — large sequential writes, low IOPS, high throughput
    Blueprint: 100-500 OPS, 256KB-1MB, 5/95 R/W, Sequential, 1AM-4AM burst
    """
    total_iops = np.random.normal(300, 100, n).clip(100, 500)
    read_ratio = np.random.normal(0.05, 0.03, n).clip(0.01, 0.15)
    
    return pd.DataFrame({
        "workload_type"       : ["Backup"] * n,
        "read_iops"           : (total_iops * read_ratio).round(0).astype(int),
        "write_iops"          : (total_iops * (1 - read_ratio)).round(0).astype(int),
        "total_iops"          : total_iops.round(0).astype(int),
        "read_throughput_mbps": np.random.normal(50, 20, n).clip(5, 150),
        "write_throughput_mbps": np.random.normal(850, 150, n).clip(400, 1400),
        "read_latency_p50_us" : np.random.normal(5000, 1000, n).clip(1000, 12000),
        "read_latency_p95_us" : np.random.normal(10000, 2000, n).clip(3000, 20000),
        "read_latency_p99_us" : np.random.normal(15000, 3000, n).clip(5000, 30000),
        "write_latency_p50_us": np.random.normal(6000, 1200, n).clip(1500, 15000),
        "write_latency_p95_us": np.random.normal(12000, 2500, n).clip(4000, 25000),
        "write_latency_p99_us": np.random.normal(18000, 3500, n).clip(6000, 35000),
        "io_size_avg_kb"      : np.random.choice([256, 512, 1024], n, p=[0.3, 0.5, 0.2]),
        "queue_depth"         : np.random.randint(1, 8, n),
        "sequential_ratio"    : np.random.normal(0.92, 0.04, n).clip(0.8, 0.99),  # Highly sequential
        "read_write_ratio"    : read_ratio.round(4),
    })


def generate_ai_training(n: int) -> pd.DataFrame:
    """
    AI Training — massive sequential reads, very high bandwidth
    Blueprint: 2K-8K OPS, 64-256KB, 70/30 R/W (70% reads, 30% writes), Sequential, Long sustained
    """
    total_iops = np.random.normal(5000, 1500, n).clip(2000, 8000)
    read_ratio = np.random.normal(0.70, 0.08, n).clip(0.5, 0.95)
    
    return pd.DataFrame({
        "workload_type"       : ["AI_Training"] * n,
        "read_iops"           : (total_iops * read_ratio).round(0).astype(int),
        "write_iops"          : (total_iops * (1 - read_ratio)).round(0).astype(int),
        "total_iops"          : total_iops.round(0).astype(int),
        "read_throughput_mbps": np.random.normal(1800, 300, n).clip(600, 3000),
        "write_throughput_mbps": np.random.normal(700, 150, n).clip(200, 1200),
        "read_latency_p50_us" : np.random.normal(2500, 500, n).clip(500, 6000),
        "read_latency_p95_us" : np.random.normal(5000, 1000, n).clip(1500, 10000),
        "read_latency_p99_us" : np.random.normal(7500, 1500, n).clip(2500, 15000),
        "write_latency_p50_us": np.random.normal(3000, 600, n).clip(800, 7000),
        "write_latency_p95_us": np.random.normal(6000, 1200, n).clip(2000, 12000),
        "write_latency_p99_us": np.random.normal(9000, 1800, n).clip(3000, 18000),
        "io_size_avg_kb"      : np.random.choice([64, 128, 256], n, p=[0.3, 0.4, 0.3]),
        "queue_depth"         : np.random.randint(16, 64, n),
        "sequential_ratio"    : np.random.normal(0.88, 0.05, n).clip(0.7, 0.98),  # Highly sequential
        "read_write_ratio"    : read_ratio.round(4),
    })


def generate_ai_inference(n: int) -> pd.DataFrame:
    """
    AI Inference — spiky low-latency reads, unpredictable bursts
    Blueprint: 8K-25K OPS, 8-32KB, 90/10 R/W, Random, Spiky latency-critical
    """
    total_iops = np.random.normal(16000, 5000, n).clip(8000, 25000)
    read_ratio = np.random.normal(0.90, 0.05, n).clip(0.75, 0.98)
    
    return pd.DataFrame({
        "workload_type"       : ["AI_Inference"] * n,
        "read_iops"           : (total_iops * read_ratio).round(0).astype(int),
        "write_iops"          : (total_iops * (1 - read_ratio)).round(0).astype(int),
        "total_iops"          : total_iops.round(0).astype(int),
        "read_throughput_mbps": np.random.normal(280, 80, n).clip(80, 600),
        "write_throughput_mbps": np.random.normal(35, 15, n).clip(5, 100),
        "read_latency_p50_us" : np.random.normal(600, 150, n).clip(100, 1500),
        "read_latency_p95_us" : np.random.normal(1200, 300, n).clip(300, 2500),
        "read_latency_p99_us" : np.random.normal(1800, 450, n).clip(500, 3500),
        "write_latency_p50_us": np.random.normal(800, 200, n).clip(150, 2000),
        "write_latency_p95_us": np.random.normal(1600, 400, n).clip(400, 3500),
        "write_latency_p99_us": np.random.normal(2400, 600, n).clip(600, 5000),
        "io_size_avg_kb"      : np.random.choice([8, 16, 32], n, p=[0.4, 0.4, 0.2]),
        "queue_depth"         : np.random.randint(2, 16, n),
        "sequential_ratio"    : np.random.normal(0.20, 0.08, n).clip(0.05, 0.45),  # Mostly random
        "read_write_ratio"    : read_ratio.round(4),
    })


def assign_topology_and_capacity(df: pd.DataFrame, volumes: list, volume_to_node: dict, 
                                  volume_to_pool: dict, volume_to_tier: dict) -> pd.DataFrame:
    """
    Assign topology (node_id, pool_id, tier) and capacity metrics to each volume.
    Capacity grows over time to simulate realistic storage usage.
    """
    # No df.copy() needed - we return a modified DataFrame and caller doesn't keep old reference
    
    # Map volume_id to topology
    df["node_id"] = df["volume_id"].map(volume_to_node)
    df["pool_id"] = df["volume_id"].map(volume_to_pool)
    df["tier"] = df["volume_id"].map(volume_to_tier)
    
    # Capacity metrics (varies by tier and workload)
    # Generate capacity values per tier based on actual volume distribution
    tier_volumes = {tier: [v for v in volumes if volume_to_tier[v] == tier] for tier in TIERS}
    
    capacity_ranges = {
        "NVMe": (500, 2001),    # 500GB - 2000GB (inclusive)
        "SSD": (1000, 3001),    # 1TB - 3000GB (inclusive)
        "HDD": (2000, 5001),    # 2TB - 5000GB (inclusive)
    }
    
    # Assign total capacity based on tier (independent random values per volume)
    volume_capacity = {}
    for tier, vols in tier_volumes.items():
        lo, hi = capacity_ranges[tier]
        caps = np.random.randint(lo, hi, len(vols))
        for vol, cap in zip(vols, caps):
            volume_capacity[vol] = cap
    
    df["capacity_total_gb"] = df["volume_id"].map(volume_capacity)
    
    # Simulate capacity growth over time (starts at 30-60%, workload-specific logistic growth)
    df["days_elapsed"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 86400
    
    # Initial usage: 30-60% of total capacity
    initial_usage_pct = np.random.uniform(0.30, 0.60, NUM_VOLUMES)
    volume_initial_usage = dict(zip(volumes, initial_usage_pct))
    
    # Growth rate: 0.5-3% per day (varies by workload)
    # Using dampened exponential growth to avoid clipping at 98%
    growth_rate_map = {
        "DB_OLTP": 0.012,      # 1.2% per day
        "VM": 0.008,           # 0.8% per day
        "Backup": 0.018,       # 1.8% per day (fastest growth)
        "AI_Training": 0.015,  # 1.5% per day
        "AI_Inference": 0.006, # 0.6% per day (slowest)
    }
    
    df["growth_rate"] = df["workload_type"].map(growth_rate_map)
    df["initial_usage_pct"] = df["volume_id"].map(volume_initial_usage)
    
    # Calculate current usage percentage with dampened growth (logistic curve)
    # Growth rate shapes the curve per workload type
    k = df["growth_rate"]
    max_capacity = 0.95
    df["capacity_used_pct"] = (
        df["initial_usage_pct"] + 
        (max_capacity - df["initial_usage_pct"]) * 
        (1 - np.exp(-k * df["days_elapsed"]))
    ).clip(0.10, 0.98)
    
    # Calculate used capacity in GB
    df["capacity_used_gb"] = (df["capacity_total_gb"] * df["capacity_used_pct"]).round(2)
    
    # Drop temporary columns
    df = df.drop(columns=["days_elapsed", "growth_rate", "initial_usage_pct"])
    
    return df


def add_time_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add realistic time-of-day patterns to IO metrics (VECTORIZED for performance).
    - DB_OLTP: peaks 9AM-6PM
    - Backup: bursts 1AM-4AM
    - VM: business hours
    - AI_Training: sustained 24/7
    - AI_Inference: spiky throughout day
    """
    # No df.copy() needed - we return a modified DataFrame and caller doesn't keep old reference
    df["hour"] = df["timestamp"].dt.hour
    
    # Initialize time_multiplier with default value
    df["time_multiplier"] = 1.0
    
    # Vectorized time-of-day multipliers per workload type
    # DB_OLTP: Peak 9AM-6PM (hours 9-18)
    mask_db_peak = (df["workload_type"] == "DB_OLTP") & (df["hour"] >= 9) & (df["hour"] <= 18)
    mask_db_off = (df["workload_type"] == "DB_OLTP") & ((df["hour"] < 9) | (df["hour"] > 18))
    df.loc[mask_db_peak, "time_multiplier"] = np.random.uniform(1.3, 1.6, mask_db_peak.sum())
    df.loc[mask_db_off, "time_multiplier"] = np.random.uniform(0.4, 0.7, mask_db_off.sum())
    
    # Backup: Burst 1AM-4AM (hours 1-4)
    mask_backup_burst = (df["workload_type"] == "Backup") & (df["hour"] >= 1) & (df["hour"] <= 4)
    mask_backup_off = (df["workload_type"] == "Backup") & ((df["hour"] < 1) | (df["hour"] > 4))
    df.loc[mask_backup_burst, "time_multiplier"] = np.random.uniform(2.0, 3.0, mask_backup_burst.sum())
    df.loc[mask_backup_off, "time_multiplier"] = np.random.uniform(0.1, 0.3, mask_backup_off.sum())
    
    # VM: Business hours 8AM-7PM
    mask_vm_peak = (df["workload_type"] == "VM") & (df["hour"] >= 8) & (df["hour"] <= 19)
    mask_vm_off = (df["workload_type"] == "VM") & ((df["hour"] < 8) | (df["hour"] > 19))
    df.loc[mask_vm_peak, "time_multiplier"] = np.random.uniform(1.2, 1.5, mask_vm_peak.sum())
    df.loc[mask_vm_off, "time_multiplier"] = np.random.uniform(0.5, 0.8, mask_vm_off.sum())
    
    # AI_Training: Sustained 24/7 with minor variance
    mask_ai_train = df["workload_type"] == "AI_Training"
    df.loc[mask_ai_train, "time_multiplier"] = np.random.uniform(0.9, 1.1, mask_ai_train.sum())
    
    # AI_Inference: Spiky throughout day
    mask_ai_infer = df["workload_type"] == "AI_Inference"
    df.loc[mask_ai_infer, "time_multiplier"] = np.random.uniform(0.7, 1.8, mask_ai_infer.sum())
    
    # Apply multiplier to IOPS and throughput
    df["read_iops"] = (df["read_iops"] * df["time_multiplier"]).round(0).astype(int)
    df["write_iops"] = (df["write_iops"] * df["time_multiplier"]).round(0).astype(int)
    df["total_iops"] = df["read_iops"] + df["write_iops"]
    df["read_throughput_mbps"] = (df["read_throughput_mbps"] * df["time_multiplier"]).round(2)
    df["write_throughput_mbps"] = (df["write_throughput_mbps"] * df["time_multiplier"]).round(2)
    
    # Drop temporary columns
    df = df.drop(columns=["time_multiplier", "hour"])
    
    return df


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage."""
    int_cols = df.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns

    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype("int32")
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")

    return df


def generate_time_series_data() -> pd.DataFrame:
    """
    Generate complete time-series dataset with topology and capacity.
    
    Structure:
    - 50 volumes across 5 nodes
    - 30 days of data at 1-minute intervals
    - Each volume assigned a workload type
    - Realistic time-of-day patterns
    - Capacity growth over time
    """
    # Set seed at function start to ensure reproducibility
    np.random.seed(SEED)
    
    print(f"Generating time-series data...")
    print(f"  Volumes: {NUM_VOLUMES}")
    print(f"  Nodes: {NUM_NODES}")
    print(f"  Days: {NUM_DAYS}")
    print(f"  Interval: {INTERVAL_MINUTES} minute(s)")
    print(f"  Expected rows: {TOTAL_ROWS:,}")
    
    # Create volume IDs
    volumes = [f"vol_{i:03d}" for i in range(NUM_VOLUMES)]
    
    # Assign volumes to nodes (evenly distributed)
    volume_to_node = {}
    for i, vol in enumerate(volumes):
        node_id = f"node_{(i % NUM_NODES):02d}"
        volume_to_node[vol] = node_id
    
    # Assign volumes to pools (2 pools per node)
    volume_to_pool = {}
    for i, vol in enumerate(volumes):
        node_num = i % NUM_NODES
        pool_num = (i // NUM_NODES) % POOLS_PER_NODE
        pool_id = f"pool_{node_num:02d}_{pool_num:02d}"
        volume_to_pool[vol] = pool_id
    
    # Assign volumes to tiers (distributed across NVMe/SSD/HDD)
    volume_to_tier = {}
    tier_distribution = [0.2, 0.5, 0.3]  # 20% NVMe, 50% SSD, 30% HDD
    tier_counts = [int(NUM_VOLUMES * p) for p in tier_distribution]
    tier_counts[-1] = NUM_VOLUMES - sum(tier_counts[:-1])  # Adjust for rounding
    
    tier_list = []
    for tier, count in zip(TIERS, tier_counts):
        tier_list.extend([tier] * count)
    np.random.shuffle(tier_list)
    volume_to_tier = dict(zip(volumes, tier_list))
    
    # Assign workload types to volumes (balanced distribution)
    # Distribute remainder round-robin instead of always to first workload
    workload_per_volume = {}
    base_count = NUM_VOLUMES // len(WORKLOAD_TYPES)
    remainder = NUM_VOLUMES % len(WORKLOAD_TYPES)
    
    workload_list = []
    for i, wl in enumerate(WORKLOAD_TYPES):
        count = base_count + (1 if i < remainder else 0)  # Round-robin remainder
        workload_list.extend([wl] * count)
    
    np.random.shuffle(workload_list)
    workload_per_volume = dict(zip(volumes, workload_list))
    
    # Generate timestamps (30 days, using INTERVAL_MINUTES)
    start_time = datetime(2026, 4, 1, 0, 0, 0)  # Start: April 1, 2026
    num_timestamps = NUM_DAYS * 24 * (60 // INTERVAL_MINUTES)
    
    # Use pd.date_range for 27x faster timestamp generation
    timestamps = pd.date_range(start=start_time, periods=num_timestamps, freq=f'{INTERVAL_MINUTES}min')
    
    # Create base DataFrame with volume_id and timestamp (OPTIMIZED: correct tile/repeat order)
    print("\nCreating base time-series structure...")
    # Correct order: tile volumes, repeat timestamps → produces interleaved layout
    # Layout: [t0:vol0, t0:vol1, ..., t0:vol49, t1:vol0, t1:vol1, ..., t1:vol49, ...]
    # This matches natural (timestamp, volume_id) sort order → no sort needed
    volume_ids = np.tile(volumes, len(timestamps))
    timestamps_repeated = np.repeat(timestamps, NUM_VOLUMES)
    
    df_base = pd.DataFrame({
        "volume_id": volume_ids,
        "timestamp": timestamps_repeated
    })
    print(f"  Base rows created: {len(df_base):,}")
    
    # Assign workload type to each volume
    df_base["workload_type"] = df_base["volume_id"].map(workload_per_volume)
    
    # Generate workload-specific metrics for each row
    print("\nGenerating workload-specific IO metrics...")
    workload_dfs = []
    for workload in WORKLOAD_TYPES:
        mask = df_base["workload_type"] == workload
        n = mask.sum()
        print(f"  {workload}: {n:,} rows")
        
        if workload == "DB_OLTP":
            metrics = generate_db_oltp(n)
        elif workload == "VM":
            metrics = generate_vm(n)
        elif workload == "Backup":
            metrics = generate_backup(n)
        elif workload == "AI_Training":
            metrics = generate_ai_training(n)
        elif workload == "AI_Inference":
            metrics = generate_ai_inference(n)
        
        # Drop workload_type from metrics (already in df_base)
        metrics = metrics.drop(columns=["workload_type"])
        
        # Combine with base data
        df_workload = df_base[mask].reset_index(drop=True)
        df_workload = pd.concat([df_workload, metrics], axis=1)
        workload_dfs.append(df_workload)

    # Combine all workloads and sort chronologically
    df = pd.concat(workload_dfs, ignore_index=True)
    df = df.sort_values(["timestamp", "volume_id"]).reset_index(drop=True)
    
    # Add topology and capacity
    print("\nAdding topology and capacity metrics...")
    df = assign_topology_and_capacity(df, volumes, volume_to_node, volume_to_pool, volume_to_tier)
    
    # Add time-of-day patterns
    print("Applying time-of-day patterns...")
    df = add_time_patterns(df)
    
    # Downcast numeric columns to reduce memory usage
    df = downcast_numeric(df)

    # Convert string columns to category dtype for 94% memory savings
    print("Optimizing memory usage (converting to category dtype)...")
    for col in ['workload_type', 'tier', 'node_id', 'pool_id', 'volume_id']:
        df[col] = df[col].astype('category')
    
    print(f"\n✅ Generated {len(df):,} rows with {df.shape[1]} columns")
    
    return df


if __name__ == "__main__":
    print("=" * 70)
    print(" IO Workload Time-Series Data Generator")
    print(" Blueprint-Compliant: 50 volumes, 5 nodes, 30 days, 1-min intervals")
    print("=" * 70)

    df = generate_time_series_data()

    # Save as CSV
    os.makedirs("data/synthetic", exist_ok=True)
    csv_path = "data/synthetic/io_workload_data.csv"
    print(f"\nSaving CSV → {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved ({os.path.getsize(csv_path) / 1024 / 1024:.1f} MB)")

    # Save as Parquet with partitioning for efficient querying
    parquet_path = "data/synthetic/io_workload_data.parquet"
    print(f"\nSaving Parquet (partitioned by workload_type) → {parquet_path}")
    if os.path.exists(parquet_path) and os.path.isfile(parquet_path):
        print(f"  ! Existing file at {parquet_path} will be removed to create a partitioned directory")
        os.remove(parquet_path)
    df.to_parquet(parquet_path, partition_cols=["workload_type"], index=False, engine="pyarrow")
    print(f"  ✅ Saved (partitioned into 5 workload subdirectories)")

    # Summary statistics
    print("\n" + "=" * 70)
    print(" DATA SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    
    print(f"\n── Workload Distribution ──")
    print(df["workload_type"].value_counts().sort_index())
    
    print(f"\n── Topology Distribution ──")
    print(f"Unique volumes: {df['volume_id'].nunique()}")
    print(f"Unique nodes: {df['node_id'].nunique()}")
    print(f"Unique pools: {df['pool_id'].nunique()}")
    print(f"\nTier distribution:")
    print(df.groupby("tier")["volume_id"].nunique())
    
    print(f"\n── Sample Data (first 3 rows) ──")
    print(df.head(3).to_string())
    
    print("\n" + "=" * 70)
    print(" ✅ DATA GENERATION COMPLETE")
    print("=" * 70)