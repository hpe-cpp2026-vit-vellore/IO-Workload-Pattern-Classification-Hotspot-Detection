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
import shutil
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


def build_workload_schedule(
    volumes: list[str],
    num_timestamps: int,
    seed: int = SEED + 3,
) -> tuple[np.ndarray, dict[str, str]]:
    """
    Build a per-volume workload schedule with a dominant workload and time windows
    that switch to other workloads.
    """
    rng = np.random.default_rng(seed)

    base_count = len(volumes) // len(WORKLOAD_TYPES)
    remainder = len(volumes) % len(WORKLOAD_TYPES)

    workload_list: list[str] = []
    for i, wl in enumerate(WORKLOAD_TYPES):
        count = base_count + (1 if i < remainder else 0)
        workload_list.extend([wl] * count)

    rng.shuffle(workload_list)
    dominant_map = dict(zip(volumes, workload_list))

    mixed_count = max(1, int(len(volumes) * 0.2))
    mixed_vols = set(rng.choice(volumes, size=mixed_count, replace=False))

    workload_rows = np.empty(num_timestamps * len(volumes), dtype=object)
    time_index = np.arange(num_timestamps)

    for i, vol in enumerate(volumes):
        dominant = dominant_map[vol]
        schedule = np.full(num_timestamps, dominant, dtype=object)

        switch_ratio = rng.uniform(0.4, 0.6) if vol in mixed_vols else rng.uniform(0.1, 0.3)
        target = int(num_timestamps * switch_ratio)
        used = 0

        while used < target:
            window_len = int(rng.integers(30, 240))
            start = int(rng.integers(0, num_timestamps - window_len))
            alt = rng.choice([w for w in WORKLOAD_TYPES if w != dominant])
            schedule[start:start + window_len] = alt
            used += window_len

        idx = i + time_index * len(volumes)
        workload_rows[idx] = schedule

    return workload_rows, dominant_map


def generate_metrics_from_latent(
    df: pd.DataFrame,
    volumes: list[str],
    seed: int = SEED + 5,
) -> pd.DataFrame:
    """
    Generate IO metrics from shared latent factors.
    Workload type only nudges the latent factors; it does not fully determine them.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    volume_index = pd.Index(volumes)
    vol_idx = volume_index.get_indexer(df["volume_id"])

    nudge_intensity = {
        "DB_OLTP": 0.6,
        "VM": 0.3,
        "Backup": -0.2,
        "AI_Training": 0.5,
        "AI_Inference": 0.7,
    }
    nudge_burstiness = {
        "DB_OLTP": 0.4,
        "VM": 0.3,
        "Backup": 0.7,
        "AI_Training": 0.2,
        "AI_Inference": 0.8,
    }
    nudge_sequentiality = {
        "DB_OLTP": -0.4,
        "VM": -0.1,
        "Backup": 0.8,
        "AI_Training": 0.7,
        "AI_Inference": -0.2,
    }
    nudge_read_bias = {
        "DB_OLTP": 0.2,
        "VM": 0.1,
        "Backup": -0.5,
        "AI_Training": 0.4,
        "AI_Inference": 0.6,
    }
    nudge_pressure = {
        "DB_OLTP": 0.2,
        "VM": 0.1,
        "Backup": 0.4,
        "AI_Training": 0.3,
        "AI_Inference": 0.4,
    }

    base_intensity = df["workload_type"].map(nudge_intensity).to_numpy()
    base_burstiness = df["workload_type"].map(nudge_burstiness).to_numpy()
    base_sequentiality = df["workload_type"].map(nudge_sequentiality).to_numpy()
    base_read_bias = df["workload_type"].map(nudge_read_bias).to_numpy()
    base_pressure = df["workload_type"].map(nudge_pressure).to_numpy()

    vol_offsets = rng.normal(0, 0.25, size=(len(volumes), 5))

    intensity = base_intensity + vol_offsets[vol_idx, 0] + rng.normal(0, 0.35, n)
    burstiness = base_burstiness + vol_offsets[vol_idx, 1] + rng.normal(0, 0.35, n)
    sequentiality = base_sequentiality + vol_offsets[vol_idx, 2] + rng.normal(0, 0.25, n)
    read_bias = base_read_bias + vol_offsets[vol_idx, 3] + rng.normal(0, 0.25, n)
    capacity_pressure = base_pressure + vol_offsets[vol_idx, 4] + rng.normal(0, 0.25, n)

    intensity = np.clip(intensity, -1.5, 1.5)
    burstiness = np.clip(burstiness, -1.5, 1.5)
    sequentiality = np.clip(sequentiality, -1.5, 1.5)
    read_bias = np.clip(read_bias, -1.5, 1.5)
    capacity_pressure = np.clip(capacity_pressure, -1.5, 1.5)

    total_iops = (
        500
        + (intensity + 1.2) * 6000
        + (burstiness + 1.0) * 800
        + rng.normal(0, 900, n)
    )
    total_iops = np.clip(total_iops, 100, 25000)

    read_ratio = 1 / (1 + np.exp(-read_bias * 2))
    read_ratio = np.clip(read_ratio, 0.05, 0.95)

    read_iops = (total_iops * read_ratio).round(0).astype(int)
    write_iops = (total_iops * (1 - read_ratio)).round(0).astype(int)

    sequential_ratio = 1 / (1 + np.exp(-sequentiality * 1.5))
    sequential_ratio = np.clip(sequential_ratio, 0.05, 0.99)

    queue_depth = (
        2 + (intensity + 1.2) * 8 + (burstiness + 1.0) * 3 + rng.normal(0, 2, n)
    )
    queue_depth = np.clip(queue_depth, 1, 64).round(0).astype(int)

    seq_bins = np.array([0.1, 0.22, 0.35, 0.5, 0.65, 0.78, 0.88, 0.95])
    size_buckets = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    bucket_idx = np.digitize(sequential_ratio, seq_bins)
    io_size_avg_kb = size_buckets[bucket_idx]

    total_throughput_mbps = (total_iops * io_size_avg_kb) / 1024.0
    eff = np.clip(rng.normal(1.0, 0.15, n) + 0.1 * intensity, 0.5, 1.6)
    total_throughput_mbps = (total_throughput_mbps * eff).round(2)

    read_throughput_mbps = (total_throughput_mbps * read_ratio).round(2)
    write_throughput_mbps = (total_throughput_mbps * (1 - read_ratio)).round(2)

    latency_base = (
        300
        + 800 * (capacity_pressure + 1.0)
        + 400 * (1 - sequential_ratio)
        + 200 * (burstiness + 1.0)
        + rng.normal(0, 150, n)
    )
    latency_base = np.clip(latency_base, 100, 20000)

    read_latency_p50 = (latency_base * rng.normal(1.0, 0.06, n)).clip(80, 30000)
    write_latency_p50 = (latency_base * rng.normal(1.1, 0.08, n)).clip(100, 35000)

    p95_mult = 1.6 + 0.5 * np.abs(burstiness)
    p99_mult = 2.2 + 0.7 * np.abs(burstiness)

    read_latency_p95 = (read_latency_p50 * p95_mult).clip(150, 50000)
    read_latency_p99 = (read_latency_p50 * p99_mult).clip(200, 70000)
    write_latency_p95 = (write_latency_p50 * p95_mult).clip(200, 60000)
    write_latency_p99 = (write_latency_p50 * p99_mult).clip(250, 80000)

    df["read_iops"] = read_iops
    df["write_iops"] = write_iops
    df["total_iops"] = read_iops + write_iops
    df["read_throughput_mbps"] = read_throughput_mbps
    df["write_throughput_mbps"] = write_throughput_mbps
    df["read_latency_p50_us"] = read_latency_p50.round(2)
    df["read_latency_p95_us"] = read_latency_p95.round(2)
    df["read_latency_p99_us"] = read_latency_p99.round(2)
    df["write_latency_p50_us"] = write_latency_p50.round(2)
    df["write_latency_p95_us"] = write_latency_p95.round(2)
    df["write_latency_p99_us"] = write_latency_p99.round(2)
    df["io_size_avg_kb"] = io_size_avg_kb
    df["queue_depth"] = queue_depth
    df["sequential_ratio"] = sequential_ratio.round(4)
    df["read_write_ratio"] = read_ratio.round(4)

    return df


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
    
    # Growth rate: mostly volume/tier-driven with weak workload influence
    rng = np.random.default_rng(SEED + 21)
    tier_base_growth = {
        "NVMe": 0.010,
        "SSD": 0.008,
        "HDD": 0.006,
    }
    workload_nudge = {
        "DB_OLTP": 0.0005,
        "VM": 0.0002,
        "Backup": 0.0015,
        "AI_Training": 0.0010,
        "AI_Inference": 0.0004,
    }

    volume_growth = {}
    volume_wave_amp = {}
    volume_wave_phase = {}
    for vol in volumes:
        tier = volume_to_tier[vol]
        base = tier_base_growth[tier] + rng.normal(0.0, 0.002)
        volume_growth[vol] = float(np.clip(base, 0.002, 0.020))
        volume_wave_amp[vol] = float(rng.uniform(0.0, 0.002))
        volume_wave_phase[vol] = float(rng.uniform(0, 2 * np.pi))

    df["growth_rate"] = df["volume_id"].map(volume_growth)
    df["growth_rate"] = df["growth_rate"] + df["workload_type"].map(workload_nudge)
    df["growth_wave"] = (
        df["volume_id"].map(volume_wave_amp)
        * np.sin((2 * np.pi * df["days_elapsed"] / 7) + df["volume_id"].map(volume_wave_phase))
    )
    df["initial_usage_pct"] = df["volume_id"].map(volume_initial_usage)
    
    # Calculate current usage percentage with dampened growth (logistic curve)
    # Growth rate shapes the curve per workload type
    k = (df["growth_rate"] + df["growth_wave"]).clip(0.001, 0.03)
    max_capacity = 0.95
    df["capacity_used_pct"] = (
        df["initial_usage_pct"] + 
        (max_capacity - df["initial_usage_pct"]) * 
        (1 - np.exp(-k * df["days_elapsed"]))
    ).clip(0.10, 0.98)
    
    # Calculate used capacity in GB
    df["capacity_used_gb"] = (df["capacity_total_gb"] * df["capacity_used_pct"]).round(2)
    
    # Drop temporary columns
    df = df.drop(columns=["days_elapsed", "growth_rate", "growth_wave", "initial_usage_pct"])
    
    return df


def add_time_patterns(
    df: pd.DataFrame,
    volume_dominant: dict[str, str],
    seed: int = SEED + 11,
) -> pd.DataFrame:
    """
    Add probabilistic time-of-day patterns with day-to-day shifts.
    Time is a weak signal, not a hard label.
    """
    rng = np.random.default_rng(seed)

    hour = df["timestamp"].dt.hour.to_numpy()
    day_of_year = df["timestamp"].dt.dayofyear.to_numpy()

    phase_mean = {
        "DB_OLTP": 13.0,
        "VM": 12.0,
        "Backup": 2.0,
        "AI_Training": 15.0,
        "AI_Inference": 14.0,
    }
    amp_mean = {
        "DB_OLTP": 0.45,
        "VM": 0.35,
        "Backup": 0.55,
        "AI_Training": 0.15,
        "AI_Inference": 0.25,
    }

    volume_phase = {}
    volume_amp = {}
    for vol, dom in volume_dominant.items():
        phase = rng.normal(phase_mean[dom], 3.0) % 24
        amp = np.clip(rng.normal(amp_mean[dom], 0.12), 0.05, 0.8)
        volume_phase[vol] = phase
        volume_amp[vol] = amp

    phase = df["volume_id"].map(volume_phase).to_numpy()
    amp = df["volume_id"].map(volume_amp).to_numpy()

    day_shift = 0.6 * np.sin(2 * np.pi * (day_of_year / 7))
    noise = rng.normal(0, 0.05, len(df))

    time_multiplier = 1 + amp * np.sin(2 * np.pi * (hour - phase + day_shift) / 24) + noise
    anomaly_mask = rng.random(len(df)) < 0.02
    time_multiplier[anomaly_mask] *= rng.uniform(1.2, 1.6, anomaly_mask.sum())
    time_multiplier = np.clip(time_multiplier, 0.6, 1.6)

    df["read_iops"] = (df["read_iops"] * time_multiplier).round(0).astype(int)
    df["write_iops"] = (df["write_iops"] * time_multiplier).round(0).astype(int)
    df["total_iops"] = df["read_iops"] + df["write_iops"]
    df["read_throughput_mbps"] = (df["read_throughput_mbps"] * time_multiplier).round(2)
    df["write_throughput_mbps"] = (df["write_throughput_mbps"] * time_multiplier).round(2)

    latency_scale = 1 + (time_multiplier - 1) * 0.15
    latency_cols = [
        "read_latency_p50_us",
        "read_latency_p95_us",
        "read_latency_p99_us",
        "write_latency_p50_us",
        "write_latency_p95_us",
        "write_latency_p99_us",
    ]
    for col in latency_cols:
        df[col] = (df[col] * latency_scale).round(2)
    
    return df


def apply_concept_bleed(
    df: pd.DataFrame,
    volumes: list[str],
    bleed_ratio: float = 0.03,
    seed: int = SEED + 7,
) -> pd.DataFrame:
    """Swap a small fraction of feature rows across workloads to blur boundaries."""
    n_bleed = int(len(df) * bleed_ratio)
    if n_bleed <= 0:
        return df

    rng = np.random.default_rng(seed)
    workloads = df["workload_type"].to_numpy()
    indices_by_workload = {w: np.where(workloads == w)[0] for w in WORKLOAD_TYPES}

    target_idx = rng.choice(len(df), size=n_bleed, replace=False)
    target_workloads = workloads[target_idx]
    donor_idx = np.empty(n_bleed, dtype=int)

    for w in WORKLOAD_TYPES:
        mask = target_workloads == w
        if not mask.any():
            continue
        other_indices = np.concatenate(
            [indices_by_workload[ow] for ow in WORKLOAD_TYPES if ow != w]
        )
        donor_idx[mask] = rng.choice(other_indices, size=mask.sum(), replace=True)

    bleed_cols = [
        "read_iops",
        "write_iops",
        "read_throughput_mbps",
        "write_throughput_mbps",
        "read_latency_p50_us",
        "read_latency_p95_us",
        "read_latency_p99_us",
        "write_latency_p50_us",
        "write_latency_p95_us",
        "write_latency_p99_us",
        "io_size_avg_kb",
        "queue_depth",
        "sequential_ratio",
        "read_write_ratio",
    ]

    df.loc[target_idx, bleed_cols] = df.loc[donor_idx, bleed_cols].to_numpy()
    df.loc[target_idx, "total_iops"] = (
        df.loc[target_idx, "read_iops"] + df.loc[target_idx, "write_iops"]
    )

    return df


def inject_noisy_neighbor_events(
    df: pd.DataFrame,
    volumes: list[str],
    volume_to_node: dict[str, str],
    n_events: int = 500,
    duration_minutes: int = 15,
    seed: int = SEED + 13,
) -> pd.DataFrame:
    """
    Inject noisy neighbor events: one volume spikes IOPS and neighbors see latency spikes.
    Assumes df is sorted by timestamp then volume_id with fixed 1-minute intervals.
    """
    rng = np.random.default_rng(seed)
    num_timestamps = NUM_DAYS * 24 * (60 // INTERVAL_MINUTES)
    if num_timestamps <= duration_minutes:
        return df

    volume_pos = {vol: i for i, vol in enumerate(volumes)}
    node_to_vols: dict[str, list[str]] = {}
    for vol in volumes:
        node_to_vols.setdefault(volume_to_node[vol], []).append(vol)

    start_indices = rng.integers(0, num_timestamps - duration_minutes, size=n_events)
    noisy_vols = rng.choice(volumes, size=n_events, replace=True)

    latency_cols = [
        "read_latency_p50_us",
        "read_latency_p95_us",
        "read_latency_p99_us",
        "write_latency_p50_us",
        "write_latency_p95_us",
        "write_latency_p99_us",
    ]

    for start_idx, noisy_vol in zip(start_indices, noisy_vols):
        node_id = volume_to_node[noisy_vol]
        neighbor_vols = [v for v in node_to_vols[node_id] if v != noisy_vol]
        if not neighbor_vols:
            continue

        t_range = np.arange(start_idx, start_idx + duration_minutes)
        base_rows = t_range * NUM_VOLUMES

        noisy_pos = volume_pos[noisy_vol]
        noisy_rows = base_rows + noisy_pos

        neighbor_pos = np.array([volume_pos[v] for v in neighbor_vols], dtype=int)
        neighbor_rows = (base_rows[:, None] + neighbor_pos[None, :]).reshape(-1)

        iops_mult = rng.uniform(1.4, 1.9)
        thr_mult = rng.uniform(1.2, 1.6)
        df.loc[noisy_rows, "read_iops"] = (
            df.loc[noisy_rows, "read_iops"] * iops_mult
        ).round(0).astype(int)
        df.loc[noisy_rows, "write_iops"] = (
            df.loc[noisy_rows, "write_iops"] * iops_mult
        ).round(0).astype(int)
        df.loc[noisy_rows, "total_iops"] = (
            df.loc[noisy_rows, "read_iops"] + df.loc[noisy_rows, "write_iops"]
        )
        df.loc[noisy_rows, "read_throughput_mbps"] = (
            df.loc[noisy_rows, "read_throughput_mbps"] * thr_mult
        ).round(2)
        df.loc[noisy_rows, "write_throughput_mbps"] = (
            df.loc[noisy_rows, "write_throughput_mbps"] * thr_mult
        ).round(2)

        latency_mult = rng.uniform(1.4, 2.2)
        for col in latency_cols:
            df.loc[neighbor_rows, col] = (
                df.loc[neighbor_rows, col] * latency_mult
            ).round(2)

        thr_down = rng.uniform(0.7, 0.9)
        df.loc[neighbor_rows, "read_throughput_mbps"] = (
            df.loc[neighbor_rows, "read_throughput_mbps"] * thr_down
        ).round(2)
        df.loc[neighbor_rows, "write_throughput_mbps"] = (
            df.loc[neighbor_rows, "write_throughput_mbps"] * thr_down
        ).round(2)

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
    
    # Assign time-varying workload schedule per volume
    print("\nAssigning time-varying workload schedules...")
    workload_rows, dominant_map = build_workload_schedule(volumes, num_timestamps)
    df_base["workload_type"] = workload_rows

    # Generate metrics from shared latent factors
    print("Generating IO metrics from latent factors...")
    df = generate_metrics_from_latent(df_base, volumes)
    
    # Add topology and capacity
    print("\nAdding topology and capacity metrics...")
    df = assign_topology_and_capacity(df, volumes, volume_to_node, volume_to_pool, volume_to_tier)
    
    # Add probabilistic time-of-day patterns
    print("Applying time-of-day patterns...")
    df = add_time_patterns(df, dominant_map)

    # Apply light concept bleed before injecting anomalies
    print("Applying concept bleed...")
    df = apply_concept_bleed(df, volumes, bleed_ratio=0.03)

    # Inject noisy neighbor events for hotspot/anomaly detection
    print("Injecting noisy neighbor events...")
    df = inject_noisy_neighbor_events(
        df,
        volumes,
        volume_to_node,
        n_events=500,
        duration_minutes=15,
    )
    
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
    if os.path.exists(parquet_path):
        if os.path.isdir(parquet_path):
            print(f"  ! Existing partitioned directory at {parquet_path} will be removed")
            shutil.rmtree(parquet_path)
        elif os.path.isfile(parquet_path):
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