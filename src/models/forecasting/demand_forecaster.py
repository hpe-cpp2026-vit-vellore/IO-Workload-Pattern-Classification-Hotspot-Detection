"""Demand forecaster using linear quantile regression for IOPS and throughput.

Produces per-volume hourly 24h forecasts for p50, p95 and p99 quantiles.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

logger = logging.getLogger("demand_forecaster")


class DemandForecaster:
    """Per-volume quantile demand forecaster for IOPS and throughput.

    Models are stored in `self.models` keyed by (volume_id, metric, quantile).
    metric is one of: 'iops', 'throughput'
    quantile is a float like 0.5, 0.95, 0.99
    """

    def __init__(self) -> None:
        self.models: Dict[Tuple[str, str, float], Any] = {}
        self.trained_volumes = set()
        self.fit_status: Dict[str, Any] = {
            "state": "idle",
            "completed": 0,
            "total": 0,
            "percent": 0.0,
            "current_volume": None,
            "started_at": None,
            "finished_at": None,
        }

    def get_fit_status(self) -> Dict[str, Any]:
        """Return a snapshot of the latest fit progress state."""
        return dict(self.fit_status)

    def fit(self, df: pd.DataFrame) -> None:
        required = {
            "volume_id",
            "timestamp",
            "total_iops",
            "total_throughput_mbps",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        }
        if not required.issubset(set(df.columns)):
            raise ValueError(f"DataFrame missing required columns: {required - set(df.columns)}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        quantiles = [0.5, 0.95, 0.99]
        feature_cols = ["hour_sin", "hour_cos", "day_sin", "day_cos"]

        n_vols = int(df["volume_id"].nunique())
        started_at = time.perf_counter()
        self.fit_status.update({
            "state": "running",
            "completed": 0,
            "total": n_vols,
            "percent": 0.0,
            "current_volume": None,
            "started_at": started_at,
            "finished_at": None,
        })

        for idx, (vol_id, g) in enumerate(df.groupby("volume_id"), start=1):
            try:
                g = g.dropna(subset=feature_cols)
                sample_count = len(g)
                percent = (idx - 1) * 100.0 / n_vols if n_vols else 100.0
                self.fit_status.update({
                    "completed": idx - 1,
                    "percent": percent,
                    "current_volume": vol_id,
                })
                if sample_count < 10:
                    logger.info(
                        "Progress %.1f%% (%d/%d) skipping volume %s: insufficient samples (%d)",
                        percent,
                        idx,
                        n_vols,
                        vol_id,
                        sample_count,
                    )
                    continue

                logger.info(
                    "Progress %.1f%% (%d/%d) fitting volume %s samples=%d",
                    percent,
                    idx,
                    n_vols,
                    vol_id,
                    sample_count,
                )

                X = g[feature_cols].values.astype(float)
                y_iops = g["total_iops"].astype(float).values
                y_th = g["total_throughput_mbps"].astype(float).values

                for q in quantiles:
                    try:
                        m_iops = QuantileRegressor(quantile=q, solver="highs")
                        m_iops.fit(X, y_iops)
                        self.models[(vol_id, "iops", q)] = m_iops
                    except Exception as e:
                        logger.debug("Failed to fit iops quantile %s for %s: %s", q, vol_id, e)

                    try:
                        m_th = QuantileRegressor(quantile=q, solver="highs")
                        m_th.fit(X, y_th)
                        self.models[(vol_id, "throughput", q)] = m_th
                    except Exception as e:
                        logger.debug("Failed to fit throughput quantile %s for %s: %s", q, vol_id, e)

                self.trained_volumes.add(vol_id)
                elapsed = time.perf_counter() - started_at
                percent_done = idx * 100.0 / n_vols if n_vols else 100.0
                self.fit_status.update({
                    "completed": idx,
                    "percent": percent_done,
                    "current_volume": vol_id,
                })
                logger.info(
                    "Completed %.1f%% (%d/%d) volume %s in %.1fs",
                    percent_done,
                    idx,
                    n_vols,
                    vol_id,
                    elapsed,
                )
            except Exception as e:
                logger.warning("Unexpected error while fitting volume %s (%d/%d): %s", vol_id, idx, n_vols, e)
                continue

        total_elapsed = time.perf_counter() - started_at
        self.fit_status.update({
            "state": "done",
            "completed": n_vols,
            "total": n_vols,
            "percent": 100.0 if n_vols else 100.0,
            "current_volume": None,
            "finished_at": time.perf_counter(),
        })
        logger.info("DemandForecaster fit complete: 100.0%% (%d/%d) in %.1fs", n_vols, n_vols, total_elapsed)

    def predict_next_24h(self, volume_id: str) -> Dict[str, Any]:
        now = pd.Timestamp.now().floor("H")
        hours = pd.date_range(start=now, periods=24, freq="H")

        hours_list = []
        X = []
        for ts in hours:
            h = ts.hour
            dow = ts.dayofweek
            hour_sin = np.sin(2 * np.pi * h / 24.0)
            hour_cos = np.cos(2 * np.pi * h / 24.0)
            day_sin = np.sin(2 * np.pi * dow / 7.0)
            day_cos = np.cos(2 * np.pi * dow / 7.0)
            hours_list.append(ts.isoformat())
            X.append([hour_sin, hour_cos, day_sin, day_cos])

        X = np.asarray(X, dtype=float)

        def _pred(metric: str, q: float):
            key = (volume_id, metric, q)
            model = self.models.get(key)
            if model is None:
                return [float("nan")] * 24
            try:
                vals = model.predict(X)
                return [float(v) for v in np.maximum(vals, 0.0)]
            except Exception as e:
                logger.debug("Prediction failed for %s %s q=%s: %s", volume_id, metric, q, e)
                return [float("nan")] * 24

        iops_p50 = _pred("iops", 0.5)
        iops_p95 = _pred("iops", 0.95)
        iops_p99 = _pred("iops", 0.99)

        th_p50 = _pred("throughput", 0.5)
        th_p95 = _pred("throughput", 0.95)
        th_p99 = _pred("throughput", 0.99)

        peak_iops = float(np.nanmax(iops_p99)) if not all(np.isnan(iops_p99)) else float("nan")
        peak_th = float(np.nanmax(th_p99)) if not all(np.isnan(th_p99)) else float("nan")

        return {
            "iops_p50": iops_p50,
            "iops_p95": iops_p95,
            "iops_p99": iops_p99,
            "throughput_p50_mbps": th_p50,
            "throughput_p95_mbps": th_p95,
            "throughput_p99_mbps": th_p99,
            "peak_iops": peak_iops,
            "peak_throughput_mbps": peak_th,
            "forecast_hours": hours_list,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "demand_forecaster.pkl"
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: Path) -> "DemandForecaster":
        target = Path(path) / "demand_forecaster.pkl"
        return joblib.load(target)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    features_pq = project_root / "data" / "processed" / "io_features.parquet"
    if not features_pq.exists():
        features_pq = project_root / "data" / "processed" / "io_features.csv"

    if features_pq.suffix == ".parquet":
        df = pd.read_parquet(features_pq)
    else:
        df = pd.read_csv(features_pq)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Ensure cyclical features exist or compute them
    if "hour_sin" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    forecaster = DemandForecaster()
    forecaster.fit(df)

    out_dir = project_root / "models" / "forecasting"
    forecaster.save(out_dir)
    logger.info("Saved DemandForecaster to %s", out_dir)

    # Sanity checks on 3 sample volumes
    sample_vols = list(df["volume_id"].unique()[:3])
    for v in sample_vols:
        try:
            pred = forecaster.predict_next_24h(v)
            print(f"{v}: peak_iops={pred['peak_iops']}, peak_throughput_mbps={pred['peak_throughput_mbps']}")
        except Exception as e:
            logger.warning("Sanity check failed for %s: %s", v, e)


if __name__ == "__main__":
    main()
