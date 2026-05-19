"""
src/models/forecasting/dtf_forecaster.py

Days-to-Fill (DTF) Capacity Forecaster — HPE Blueprint Phase 4.1
=================================================================

Pipeline:
    1. Load processed features (io_features.parquet/csv)
    2. Resample capacity_used_pct to daily resolution per volume
    3. Train a global N-BEATS model across all 50 volumes
    4. Forecast 30 days ahead per volume (autoregressive roll-forward)
    5. Compute DTF: days until capacity crosses 85% (warning) and 95% (critical)
    6. Produce prioritized DTF report with recommendations

Outputs (in models/forecasting/):
    - nbeats_model.pth          : Trained model weights
    - dtf_forecast.json         : Per-volume DTF results with details
    - dtf_forecast.csv          : Flat table sorted by urgency
    - nbeats_training_stats.json: Training curve and hyperparameters

HPE Success Criteria:
    - Accurate capacity exhaustion prediction (DTF)
    - Output feeds into the automated recommendations engine
    - Supports what-if simulation (simulate_add_capacity)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# ── path bootstrap ─────────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.forecasting.nbeats_model import (  # noqa: E402
    CapacityDataset,
    NBeatsModel,
    forecast_volume,
    train_nbeats,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
INPUT_SIZE = 20          # Lookback: 20 days of daily capacity history
FORECAST_SIZE = 7        # Native forecast horizon: 7 days
DTF_FORECAST_DAYS = 60   # Forecast up to 60 days ahead for DTF computation
N_STACKS = 3
N_BLOCKS = 3
HIDDEN_SIZE = 128
N_LAYERS = 4
DROPOUT = 0.1
N_EPOCHS = 150
BATCH_SIZE = 64
LR = 5e-4
PATIENCE = 20

# Capacity thresholds (from blueprint)
WARNING_THRESHOLD = 0.85
CRITICAL_THRESHOLD = 0.95


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────
def load_features(project_root: Path) -> pd.DataFrame:
    """Load processed features from parquet or CSV."""
    pq = project_root / "data" / "processed" / "io_features.parquet"
    csv = project_root / "data" / "processed" / "io_features.csv"
    if pq.exists():
        logger.info("Loading features from parquet: %s", pq)
        return pd.read_parquet(pq)
    if csv.exists():
        logger.info("Loading features from CSV: %s", csv)
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No features file at {pq} or {csv}")


def prepare_daily_capacity(features: pd.DataFrame) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Resample capacity_used_pct to daily resolution (end-of-day value per volume),
    ensuring all missing calendar days are filled and scale is verified.

    Returns tuple:
        - volume_series: {volume_id: np.array of daily capacity_used_pct}
        - volume_capacities: {volume_id: total capacity in GB}
    """
    df = features[["volume_id", "timestamp", "capacity_used_pct", "capacity_total_gb"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = pd.to_datetime(df["timestamp"].dt.date)

    # Robust scale check: convert [0, 100] values to [0, 1] fractions if needed
    max_val = df["capacity_used_pct"].max()
    if max_val > 1.0:
        logger.warning("Detected capacity_used_pct values > 1.0 (max = %s). "
                       "Automatically scaling to [0, 1] fraction.", max_val)
        df["capacity_used_pct"] = df["capacity_used_pct"] / 100.0

    # Take end-of-day (last) value for each volume each day
    daily = (
        df.groupby(["volume_id", "date"], observed=False)[["capacity_used_pct", "capacity_total_gb"]]
        .last()
        .reset_index()
    )

    volume_series: Dict[str, np.ndarray] = {}
    volume_capacities: Dict[str, float] = {}
    for vol_id, grp in daily.groupby("volume_id", observed=False):
        # Set date as index and sort it
        grp = grp.set_index("date").sort_index()

        # True daily resampling to fill in any missing calendar days
        grp_resampled = grp.resample("D").last()

        # Forward-fill missing capacity values (capacity retains its value)
        grp_resampled = grp_resampled.ffill()

        # Backward-fill as a fallback for leading missing days
        grp_resampled = grp_resampled.bfill()

        volume_series[str(vol_id)] = grp_resampled["capacity_used_pct"].values.astype(np.float32)
        volume_capacities[str(vol_id)] = float(grp_resampled["capacity_total_gb"].iloc[-1])

    return volume_series, volume_capacities


def split_series(
    volume_series: Dict[str, np.ndarray],
    val_days: int = 7,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Split each volume's daily series for training and validation without leaks.

    If the series is too short to support the requested val_days, we dynamically
    reduce val_days or fall back to training on the full series and disabling validation
    for that volume.
    """
    train_list = []
    val_list = []
    window = INPUT_SIZE + FORECAST_SIZE

    for vol_id, series in volume_series.items():
        if len(series) < window:
            continue

        # We need len(series) - val_days >= window => val_days <= len(series) - window
        max_possible_val_days = len(series) - window
        actual_val_days = min(val_days, max_possible_val_days)

        if actual_val_days >= FORECAST_SIZE:
            # We can support a leak-free validation set
            train_portion = series[:-actual_val_days]
            val_portion = series[-(actual_val_days + INPUT_SIZE):]
            train_list.append(train_portion)
            val_list.append(val_portion)
        else:
            # Fall back to training on full series and skip validation
            train_list.append(series)

    return train_list, val_list


# ─────────────────────────────────────────────────────────────────────────────
# DTF computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_dtf(
    current_pct: float,
    forecast: np.ndarray,
    threshold: float,
) -> Optional[float]:
    """
    Compute days-to-fill from a forecast curve.

    Returns None if the threshold is never crossed within the forecast horizon.
    If already above threshold, returns 0.
    """
    if current_pct >= threshold:
        return 0.0

    # Find first day where forecast crosses threshold
    for i, val in enumerate(forecast):
        if val >= threshold:
            # Linear interpolation for sub-day precision
            prev_val = forecast[i - 1] if i > 0 else current_pct
            frac = (threshold - prev_val) / (val - prev_val + 1e-9)
            return float(i + frac)

    # Threshold not crossed within forecast — extrapolate linearly
    if len(forecast) >= 2:
        daily_rate = float(np.median(np.diff(forecast[-7:])))  # last week's rate
        if daily_rate > 0:
            remaining = threshold - forecast[-1]
            extra_days = remaining / daily_rate
            return float(len(forecast)) + extra_days

    return None  # Cannot determine (flat or declining)


def simulate_add_capacity(
    model: NBeatsModel,
    history: np.ndarray,
    current_total_gb: float,
    added_gb: float,
    n_steps_ahead: int = DTF_FORECAST_DAYS,
    device: str = "cpu",
) -> dict:
    """
    Simulate adding capacity to a volume and calculate new forecasted DTF.
    This assumes the absolute capacity growth (in GB) remains the same,
    meaning the forecasted capacity percentage is scaled by:
    scaling_factor = current_total_gb / (current_total_gb + added_gb)

    Parameters
    ----------
    model : trained NBeatsModel
    history : np.ndarray of historical daily capacity_used_pct [0, 1]
    current_total_gb : current total capacity of the volume in GB
    added_gb : amount of capacity to add in GB
    n_steps_ahead : forecast horizon

    Returns
    -------
    dict with before/after comparison
    """
    # 1. Scaling factor
    scaling_factor = current_total_gb / (current_total_gb + added_gb)

    # 2. Run forecasting on original history
    forecast_orig = forecast_volume(model, history, n_steps_ahead, device)
    forecast_orig = np.clip(forecast_orig, 0.0, 1.0)

    # 3. Scale forecast and current value directly
    forecast_sim = forecast_orig * scaling_factor
    forecast_sim = np.clip(forecast_sim, 0.0, 1.0)

    current_pct_orig = float(history[-1])
    current_pct_sim = current_pct_orig * scaling_factor

    # 4. Compute DTFs
    dtf_w_orig = compute_dtf(current_pct_orig, forecast_orig, WARNING_THRESHOLD)
    dtf_c_orig = compute_dtf(current_pct_orig, forecast_orig, CRITICAL_THRESHOLD)

    dtf_w_sim = compute_dtf(current_pct_sim, forecast_sim, WARNING_THRESHOLD)
    dtf_c_sim = compute_dtf(current_pct_sim, forecast_sim, CRITICAL_THRESHOLD)

    return {
        "added_capacity_gb": added_gb,
        "new_total_capacity_gb": current_total_gb + added_gb,
        "scaling_factor": round(scaling_factor, 4),
        "current_capacity_pct_before": round(current_pct_orig * 100, 2),
        "current_capacity_pct_after": round(current_pct_sim * 100, 2),
        "dtf_warning_before_days": round(dtf_w_orig, 1) if dtf_w_orig is not None else None,
        "dtf_warning_after_days": round(dtf_w_sim, 1) if dtf_w_sim is not None else None,
        "dtf_critical_before_days": round(dtf_c_orig, 1) if dtf_c_orig is not None else None,
        "dtf_critical_after_days": round(dtf_c_sim, 1) if dtf_c_sim is not None else None,
    }


def get_severity(dtf: Optional[float]) -> str:
    """Map DTF to severity level per blueprint."""
    if dtf is None:
        return "safe"
    if dtf <= 0:
        return "breached"
    if dtf < 7:
        return "critical"
    if dtf < 30:
        return "warning"
    return "normal"


def get_recommendation(
    volume_id: str,
    dtf_warning: Optional[float],
    dtf_critical: Optional[float],
    severity: str,
) -> str:
    """Generate human-readable recommendation per blueprint."""
    if severity == "breached":
        return (f"{volume_id} has BREACHED capacity threshold! "
                f"Immediate action: expand capacity or migrate cold data to tier-2.")
    if severity == "critical":
        days = int(dtf_critical) if dtf_critical else "?"
        return (f"{volume_id} will breach capacity in {days} days. "
                f"Add capacity immediately or migrate cold data to tier-2.")
    if severity == "warning":
        days = int(dtf_warning) if dtf_warning else "?"
        return (f"{volume_id} will need capacity expansion in ~{days} days. "
                f"Plan accordingly.")
    return f"{volume_id} capacity is healthy. No action needed."


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = PROJECT_ROOT / "models" / "forecasting"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & prepare data ─────────────────────────────────────────────
    logger.info("Phase 4.1: N-BEATS DTF Forecasting")
    features = load_features(PROJECT_ROOT)
    logger.info("Features loaded: %d rows, %d volumes",
                len(features), features["volume_id"].nunique())

    volume_series, volume_capacities = prepare_daily_capacity(features)
    if not volume_series:
        logger.error("No daily capacity series could be prepared from features. Exiting.")
        return 1

    n_volumes = len(volume_series)
    series_lengths = [len(s) for s in volume_series.values()]
    logger.info("Daily series prepared: %d volumes, %d-%d days each",
                n_volumes, min(series_lengths), max(series_lengths))

    # ── 2. Build datasets ──────────────────────────────────────────────────
    train_list, val_list = split_series(volume_series, val_days=7)
    if not train_list:
        logger.error("No valid training series found (all series are too short). Exiting.")
        return 1

    train_dataset = CapacityDataset(train_list, INPUT_SIZE, FORECAST_SIZE)
    if len(train_dataset) == 0:
        logger.error("Train dataset is empty (no sliding windows could be extracted). Ensure series are long enough. Exiting.")
        return 1

    val_dataset = CapacityDataset(val_list, INPUT_SIZE, FORECAST_SIZE) if val_list else None
    logger.info("Train samples: %d | Val samples: %d",
                len(train_dataset), len(val_dataset) if val_dataset else 0)

    # ── 3. Train N-BEATS model ─────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training N-BEATS on device=%s", device)

    model = NBeatsModel(
        input_size=INPUT_SIZE,
        forecast_size=FORECAST_SIZE,
        n_stacks=N_STACKS,
        n_blocks=N_BLOCKS,
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    )
    logger.info("Model parameters: %d", model.n_parameters)

    t0 = time.time()
    history = train_nbeats(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=PATIENCE,
        device=device,
    )
    train_time = time.time() - t0
    logger.info("Training took %.1f seconds", train_time)

    # Save model
    model_path = out_dir / "nbeats_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved: %s", model_path)

    # ── 4. Forecast per volume ─────────────────────────────────────────────
    logger.info("Forecasting %d days ahead for each volume...", DTF_FORECAST_DAYS)
    results: List[dict] = []

    for vol_id, series in volume_series.items():
        current_pct = float(series[-1])
        forecast = forecast_volume(
            model=model,
            history=series,
            n_steps_ahead=DTF_FORECAST_DAYS,
            device=device,
        )
        # Clip forecast to [0, 1] range
        forecast = np.clip(forecast, 0.0, 1.0)

        dtf_warning = compute_dtf(current_pct, forecast, WARNING_THRESHOLD)
        dtf_critical = compute_dtf(current_pct, forecast, CRITICAL_THRESHOLD)

        severity = get_severity(dtf_critical)
        if severity == "normal" or severity == "safe":
            severity = get_severity(dtf_warning)
            if severity == "normal":
                severity = "normal"

        recommendation = get_recommendation(vol_id, dtf_warning, dtf_critical, severity)

        results.append({
            "volume_id": vol_id,
            "current_capacity_pct": round(current_pct * 100, 2),
            "dtf_warning_85pct_days": round(dtf_warning, 1) if dtf_warning is not None else None,
            "dtf_critical_95pct_days": round(dtf_critical, 1) if dtf_critical is not None else None,
            "severity": severity,
            "recommendation": recommendation,
            "forecast_7d": [round(float(v) * 100, 2) for v in forecast[:7]],
            "forecast_30d": [round(float(v) * 100, 2) for v in forecast[:30]],
        })

    # ── 5. Sort by urgency and persist ────────────────────────────────────
    # Sort: critical first, then warning, then by DTF ascending
    severity_order = {"breached": 0, "critical": 1, "warning": 2, "normal": 3, "safe": 4}
    results.sort(key=lambda r: (
        severity_order.get(r["severity"], 5),
        r["dtf_critical_95pct_days"] if r["dtf_critical_95pct_days"] is not None else 9999,
    ))

    # JSON output
    json_path = out_dir / "dtf_forecast.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    logger.info("Wrote DTF forecast JSON: %s", json_path)

    # CSV output
    csv_data = [{k: v for k, v in r.items() if k not in ("forecast_7d", "forecast_30d")}
                for r in results]
    csv_df = pd.DataFrame(csv_data)
    csv_path = out_dir / "dtf_forecast.csv"
    csv_df.to_csv(csv_path, index=False)
    logger.info("Wrote DTF forecast CSV: %s (%d rows)", csv_path, len(csv_df))

    # Training stats
    stats = {
        "model_architecture": {
            "type": "N-BEATS (Generic)",
            "input_size": INPUT_SIZE,
            "forecast_size": FORECAST_SIZE,
            "n_stacks": N_STACKS,
            "n_blocks": N_BLOCKS,
            "hidden_size": HIDDEN_SIZE,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
            "n_parameters": model.n_parameters,
        },
        "training": {
            "n_epochs_run": len(history["train_losses"]),
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "final_train_loss": history["train_losses"][-1],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0,
            "training_time_seconds": round(train_time, 1),
            "device": device,
        },
        "data": {
            "n_volumes": n_volumes,
            "daily_series_length": int(np.mean(series_lengths)),
            "dtf_forecast_days": DTF_FORECAST_DAYS,
        },
        "thresholds": {
            "warning_pct": WARNING_THRESHOLD * 100,
            "critical_pct": CRITICAL_THRESHOLD * 100,
        },
        "results_summary": {
            "volumes_breached": sum(1 for r in results if r["severity"] == "breached"),
            "volumes_critical": sum(1 for r in results if r["severity"] == "critical"),
            "volumes_warning": sum(1 for r in results if r["severity"] == "warning"),
            "volumes_normal": sum(1 for r in results if r["severity"] in ("normal", "safe")),
        },
    }
    stats_path = out_dir / "nbeats_training_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Wrote training stats: %s", stats_path)

    # -- Console summary ----------------------------------------------------
    print("\n" + "=" * 60)
    print("  N-BEATS DTF Forecasting - Results Summary")
    print("=" * 60)
    print(f"  Model params   : {model.n_parameters:,}")
    print(f"  Training time  : {train_time:.1f}s ({device})")
    print(f"  Best val loss  : {history['best_val_loss']:.6f}")
    print(f"  Volumes total  : {n_volumes}")
    print(f"  --- Severity Breakdown ---")
    print(f"  BREACHED       : {stats['results_summary']['volumes_breached']}")
    print(f"  CRITICAL (<7d) : {stats['results_summary']['volumes_critical']}")
    print(f"  WARNING (<30d) : {stats['results_summary']['volumes_warning']}")
    print(f"  NORMAL/SAFE    : {stats['results_summary']['volumes_normal']}")
    print()

    # Print top 10 most urgent
    print("  Top 10 Most Urgent Volumes:")
    print(f"  {'Volume':<10} {'Current%':<10} {'DTF-85%':<10} {'DTF-95%':<10} {'Severity'}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in results[:10]:
        dtf_w = f"{r['dtf_warning_85pct_days']:.1f}d" if r['dtf_warning_85pct_days'] is not None else "N/A"
        dtf_c = f"{r['dtf_critical_95pct_days']:.1f}d" if r['dtf_critical_95pct_days'] is not None else "N/A"
        print(f"  {r['volume_id']:<10} {r['current_capacity_pct']:<10.1f} "
              f"{dtf_w:<10} {dtf_c:<10} {r['severity']}")
    print("=" * 60)

    # What-if simulation showcase for top 3 most urgent volumes
    urgent_candidates = results[:3]
    if urgent_candidates:
        print("\n" + "=" * 60)
        print("  Capacity Expansion Simulation (What-If Analysis)")
        print("  Adding 200 GB capacity to top volumes:")
        print("=" * 60)
        print(f"  {'Volume':<10} {'Cur% (Pre)':<11} {'Cur% (Post)':<12} {'DTF-95%(Pre)':<14} {'DTF-95%(Post)'}")
        print(f"  {'-'*10} {'-'*11} {'-'*12} {'-'*14} {'-'*14}")
        for r in urgent_candidates:
            vol_id = r["volume_id"]
            history_series = volume_series[vol_id]
            current_total_gb = volume_capacities[vol_id]
            sim_res = simulate_add_capacity(
                model=model,
                history=history_series,
                current_total_gb=current_total_gb,
                added_gb=200.0,
                device=device,
            )
            pre_dtf = f"{r['dtf_critical_95pct_days']:.1f}d" if r['dtf_critical_95pct_days'] is not None else "N/A"
            post_dtf = f"{sim_res['dtf_critical_after_days']:.1f}d" if sim_res['dtf_critical_after_days'] is not None else "N/A"
            print(f"  {vol_id:<10} {sim_res['current_capacity_pct_before']:<11.1f} "
                  f"{sim_res['current_capacity_pct_after']:<12.1f} {pre_dtf:<14} {post_dtf}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
