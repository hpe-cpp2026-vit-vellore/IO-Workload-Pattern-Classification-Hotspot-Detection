"""
src/models/forecasting/tft_forecaster.py

Temporal Fusion Transformer (TFT) Pipeline for Performance & Tail Latency Risk Forecasting.
HPE Blueprint Phase 4.2.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.models.forecasting.tft_model import TemporalFusionTransformer, QuantileLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
INPUT_SIZE = 24           # Lookback: 24 hours of hourly history
FORECAST_SIZE = 6         # Forecast horizon: 6 hours ahead
D_MODEL = 32
N_HEADS = 2
DROPOUT = 0.1
N_EPOCHS = 10
BATCH_SIZE = 256
LR = 5e-4
PATIENCE = 3
QUANTILES = [0.5, 0.9, 0.95] # Median, 90th percentile, 95th percentile (Tail Risk)

# Latency risk threshold (e.g., 10,000 microseconds / 10ms is flagged as high-risk tail event)
LATENCY_RISK_THRESHOLD_US = 8000.0 

# Features to feed into TFT Variable Selection Network
FEATURE_COLS = [
    "read_iops",
    "write_iops",
    "read_throughput_mbps",
    "write_throughput_mbps",
    "queue_depth",
    "sequential_ratio",
    "latency_p95_us" # Target is the last feature
]

def load_features(root: Path) -> pd.DataFrame:
    """Load preprocessed IO features."""
    pq = root / "data" / "processed" / "io_features.parquet"
    if pq.exists():
        logger.info("Loading features from parquet: %s", pq)
        return pd.read_parquet(pq)
    
    csv = root / "data" / "processed" / "io_features.csv"
    if csv.exists():
        logger.info("Loading features from CSV: %s", csv)
        return pd.read_csv(csv)
        
    raise FileNotFoundError(f"No features file at {pq} or {csv}")

def prepare_hourly_latency(features: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Resample high-frequency metrics to hourly resolution for all volumes,
    retaining peak p95 tail latencies and averaging physical performance metrics.
    
    Returns:
        - volume_features: {volume_id: np.ndarray of shape [720, num_features]}
        - volume_targets: {volume_id: np.ndarray of shape [720]} representing tail latency target
    """
    # Create worst-case tail latency column
    df = features.copy()
    df["latency_p95_us"] = df[["read_latency_p95_us", "write_latency_p95_us"]].max(axis=1)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_ts"] = pd.to_datetime(df["timestamp"].dt.round("h"))

    volume_features: Dict[str, np.ndarray] = {}
    volume_targets: Dict[str, np.ndarray] = {}

    # Define aggregations: mean for throughput/iops/queue, max for tail latency
    agg_rules = {
        "read_iops": "mean",
        "write_iops": "mean",
        "read_throughput_mbps": "mean",
        "write_throughput_mbps": "mean",
        "queue_depth": "mean",
        "sequential_ratio": "mean",
        "latency_p95_us": "max" # We track peak tail latency risk
    }

    for vol_id, grp in df.groupby("volume_id", observed=False):
        grp = grp.set_index("hour_ts").sort_index()
        
        # Resample to hourly steps to establish a uniform timeline
        grp_resampled = grp.resample("h").agg(agg_rules)
        
        # Gap-filling (Forward-fill and fallback backward-fill)
        grp_resampled = grp_resampled.ffill().bfill()
        
        # Scale protection: ensure tail latencies are positive
        grp_resampled["latency_p95_us"] = grp_resampled["latency_p95_us"].clip(lower=0.0)

        # Standard normalization: scale each feature to prevent gradient explosion
        # Note: we retain absolute metrics for forecasting targets later
        features_matrix = grp_resampled[FEATURE_COLS].values.astype(np.float32)
        target_series = grp_resampled["latency_p95_us"].values.astype(np.float32)

        volume_features[str(vol_id)] = features_matrix
        volume_targets[str(vol_id)] = target_series

    return volume_features, volume_targets

def split_tft_series(
    volume_features: Dict[str, np.ndarray],
    volume_targets: Dict[str, np.ndarray],
    val_hours: int = 72, # Validation set: last 3 days (72 hours)
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Split time series into training and validation sets chronologically without information leaks.
    """
    train_feats, train_targets = [], []
    val_feats, val_targets = [], []
    
    window = INPUT_SIZE + FORECAST_SIZE

    for vol_id in volume_features.keys():
        feats = volume_features[vol_id]
        targets = volume_targets[vol_id]

        if len(feats) < window:
            continue

        # Chronological holdout limit
        max_possible_val = len(feats) - window
        actual_val_hours = min(val_hours, max_possible_val)

        if actual_val_hours >= FORECAST_SIZE:
            # Leak-free split
            train_feats.append(feats[:-actual_val_hours])
            train_targets.append(targets[:-actual_val_hours])
            
            val_feats.append(feats[-(actual_val_hours + INPUT_SIZE):])
            val_targets.append(targets[-(actual_val_hours + INPUT_SIZE):])
        else:
            # Fallback to train-only on short series
            train_feats.append(feats)
            train_targets.append(targets)

    return train_feats, train_targets, val_feats, val_targets

class TFTDataset(Dataset):
    """
    PyTorch Dataset caching time-series sliding windows for zero-overhead batch loading.
    """
    def __init__(
        self,
        features_list: List[np.ndarray],
        targets_list: List[np.ndarray],
        input_size: int,
        forecast_size: int,
    ) -> None:
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        window = input_size + forecast_size

        for feats, targets in zip(features_list, targets_list):
            if len(feats) < window:
                continue
            for i in range(len(feats) - window + 1):
                # VSN Input: historical features
                x = feats[i : i + input_size].astype(np.float32)
                # Quantile target: future tail latency values
                y = targets[i + input_size : i + window].astype(np.float32)
                
                self.samples.append((torch.from_numpy(x), torch.from_numpy(y)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

def train_tft(
    model: TemporalFusionTransformer,
    train_dataset: TFTDataset,
    val_dataset: Optional[TFTDataset],
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    patience: int = PATIENCE,
    device: str = "cpu"
) -> Dict:
    """Train the TFT model using Quantile Loss with Early Stopping."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = QuantileLoss(quantiles=model.quantiles)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch) # [batch, forecast_size, num_quantiles]
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)

        # Validation step
        val_loss = epoch_loss
        if val_loader:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_pred = model(x_batch)
                    val_total += criterion(y_pred, y_batch).item() * x_batch.size(0)
            val_loss = val_total / len(val_dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
                        epoch, n_epochs, epoch_loss, val_loss, optimizer.param_groups[0]["lr"])

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    
    logger.info("TFT Training Complete. Best Epoch: %d, Best Val Loss: %.4f", best_epoch, best_val_loss)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }

def main() -> int:
    out_dir = PROJECT_ROOT / "models" / "forecasting"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & prepare data ─────────────────────────────────────────────
    logger.info("Phase 4.2: TFT Tail Latency Risk Forecasting")
    features = load_features(PROJECT_ROOT)
    logger.info("Features loaded: %d rows", len(features))

    volume_features, volume_targets = prepare_hourly_latency(features)
    n_volumes = len(volume_features)
    logger.info("Hourly series prepared: %d volumes, %d hours each",
                n_volumes, next(iter(volume_features.values())).shape[0])

    # ── 2. Build Datasets ──────────────────────────────────────────────────
    train_feats, train_targets, val_feats, val_targets = split_tft_series(
        volume_features, volume_targets, val_hours=72
    )

    if not train_feats:
        logger.error("No valid training series found. Exiting.")
        return 1

    train_dataset = TFTDataset(train_feats, train_targets, INPUT_SIZE, FORECAST_SIZE)
    val_dataset = TFTDataset(val_feats, val_targets, INPUT_SIZE, FORECAST_SIZE) if val_feats else None

    if len(train_dataset) == 0:
        logger.error("Train dataset is empty (no sliding windows could be extracted). Exiting.")
        return 1

    logger.info("TFT Train samples: %d | Val samples: %d",
                len(train_dataset), len(val_dataset) if val_dataset else 0)

    # ── 3. Initialize & Train TFT Model ────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training TFT on device=%s", device)

    model = TemporalFusionTransformer(
        input_size=INPUT_SIZE,
        num_features=len(FEATURE_COLS),
        forecast_size=FORECAST_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        quantiles=QUANTILES
    )
    logger.info("TFT Model parameters: %d", model.n_parameters)

    t0 = time.time()
    history = train_tft(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=PATIENCE,
        device=device
    )
    train_time = time.time() - t0

    # Save model weights
    model_path = out_dir / "tft_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved: %s", model_path)

    # ── 4. Generate Multi-Quantile Forecasts & Tail Latency Risk Alerts ────
    logger.info("Generating multi-quantile latency predictions...")
    model.eval()
    
    results = []
    high_risk_count = 0

    with torch.no_grad():
        for vol_id in volume_features.keys():
            feats = volume_features[vol_id]
            
            # Use last lookback inputs
            current_input = feats[-INPUT_SIZE:]
            x = torch.from_numpy(current_input).unsqueeze(0).to(device)
            
            pred = model(x).cpu().numpy().squeeze(0) # [forecast_size, num_quantiles]

            # Unpack quantile forecasts
            p50_forecast = pred[:, 0]
            p90_forecast = pred[:, 1]
            p95_forecast = pred[:, 2]

            max_p95_forecast = float(np.max(p95_forecast))
            current_p95 = float(feats[-1, -1]) # last element of target feature column

            # Identify if there is a tail latency risk breach forecasted
            is_risk = max_p95_forecast >= LATENCY_RISK_THRESHOLD_US
            if is_risk:
                high_risk_count += 1
                
            severity = "critical" if max_p95_forecast >= (LATENCY_RISK_THRESHOLD_US * 1.5) else ("warning" if is_risk else "normal")

            results.append({
                "volume_id": vol_id,
                "current_p95_latency_us": round(current_p95, 2),
                "forecasted_p50_latency_us": [round(float(v), 2) for v in p50_forecast],
                "forecasted_p90_latency_us": [round(float(v), 2) for v in p90_forecast],
                "forecasted_p95_latency_us": [round(float(v), 2) for v in p95_forecast],
                "peak_forecasted_p95_us": round(max_p95_forecast, 2),
                "severity": severity,
                "action_recommended": "SLA Alert: Relocate high-IOPS VMs" if severity == "critical" else ("Balance workload" if severity == "warning" else "Optimal performance")
            })

    # Sort results by peak predicted tail latency (worst first)
    results = sorted(results, key=lambda x: x["peak_forecasted_p95_us"], reverse=True)

    # Save details to JSON
    details_path = out_dir / "tft_latency_forecast.json"
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote details JSON: %s", details_path)

    # Save summaries to CSV
    summary_df = pd.DataFrame([
        {
            "volume_id": r["volume_id"],
            "current_p95_us": r["current_p95_latency_us"],
            "pred_p50_peak_us": max(r["forecasted_p50_latency_us"]),
            "pred_p90_peak_us": max(r["forecasted_p90_latency_us"]),
            "pred_p95_peak_us": r["peak_forecasted_p95_us"],
            "severity": r["severity"],
            "action": r["action_recommended"]
        }
        for r in results
    ])
    csv_path = out_dir / "tft_latency_forecast.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info("Wrote summary CSV: %s (%d rows)", csv_path, len(summary_df))

    # Save training stats
    stats = {
        "model": {
            "input_size": INPUT_SIZE,
            "forecast_size": FORECAST_SIZE,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "dropout": DROPOUT,
            "n_parameters": model.n_parameters
        },
        "training": {
            "n_epochs": len(history["train_losses"]),
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "train_time_seconds": round(train_time, 1),
            "device": device
        },
        "statistics": {
            "total_volumes": n_volumes,
            "high_risk_volumes": high_risk_count,
            "warning_volumes": sum(1 for r in results if r["severity"] == "warning"),
            "critical_volumes": sum(1 for r in results if r["severity"] == "critical"),
            "normal_volumes": sum(1 for r in results if r["severity"] == "normal")
        }
    }
    stats_path = out_dir / "tft_training_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Wrote stats JSON: %s", stats_path)

    # ── Console Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Temporal Fusion Transformer (TFT) - Latency Risk Summary")
    print("=" * 65)
    print(f"  Model Params    : {model.n_parameters:,}")
    print(f"  Training Time   : {train_time:.1f}s ({device})")
    print(f"  Best Val Loss   : {history['best_val_loss']:.6f}")
    print(f"  Volumes Total   : {n_volumes}")
    print(f"  --- Severity Breakdown ---")
    print(f"  CRITICAL (Peak p95 >= 12ms) : {stats['statistics']['critical_volumes']}")
    print(f"  WARNING (Peak p95 >= 8ms)   : {stats['statistics']['warning_volumes']}")
    print(f"  NORMAL (p95 latency healthy) : {stats['statistics']['normal_volumes']}")
    print()

    # Print top 10 worst performing volumes (highest forecasted tail latency risk)
    print("  Top 10 High Latency Risk Volumes:")
    print(f"  {'Volume':<10} {'Cur-p95':<10} {'Fcast-p50':<11} {'Fcast-p95':<11} {'Severity':<10} {'Action'}")
    print(f"  {'-'*10} {'-'*10} {'-'*11} {'-'*11} {'-'*10} {'-'*20}")
    for r in results[:10]:
        cur = f"{r['current_p95_latency_us']/1000:.1f}ms"
        p50 = f"{max(r['forecasted_p50_latency_us'])/1000:.1f}ms"
        p95 = f"{r['peak_forecasted_p95_us']/1000:.1f}ms"
        print(f"  {r['volume_id']:<10} {cur:<10} {p50:<11} {p95:<11} {r['severity']:<10} {r['action_recommended']}")
    print("=" * 65)

    return 0

if __name__ == "__main__":
    sys.exit(main())
