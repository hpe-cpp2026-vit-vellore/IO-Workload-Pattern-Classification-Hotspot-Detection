"""
src/control_plane/inference_hub.py

Real-Time Model Inference Hub (HPE Blueprint Phase 5.2)
=========================================================
Coordinates all trained models and returns a comprehensive analysis per volume.
"""

import sys
import yaml
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Bootstrap path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Add anomaly model directory to sys.path for internal imports
ANOMALY_DIR = PROJECT_ROOT / "src" / "models" / "anomaly"
if str(ANOMALY_DIR) not in sys.path:
    sys.path.insert(0, str(ANOMALY_DIR))

# Sibling model imports
from src.models.anomaly.ensemble_detector import EnsembleDetector, VolumeMetrics
from src.pipeline.topology_graph import TopologyGraph
from src.models.anomaly.noisy_neighbor import NoisyNeighborDetector, NoisyNeighborEvent
from src.models.forecasting.nbeats_model import NBeatsModel, forecast_volume
from src.models.forecasting.tft_model import TemporalFusionTransformer
from src.models.forecasting.tft_forecaster import prepare_hourly_latency
from src.models.forecasting.demand_forecaster import DemandForecaster

logger = logging.getLogger(__name__)

LABEL_NAMES: Dict[int, str] = {
    0: "DB_OLTP",
    1: "VM",
    2: "Backup",
    3: "AI_Training",
    4: "AI_Inference",
}

class InferenceHub:
    """Coordinates workload classification, anomalies, capacity, and latency forecasting."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        
        # Load config
        policy_path = self.project_root / "configs" / "policy.yaml"
        with open(policy_path, "r") as f:
            self.policy = yaml.safe_load(f)

        # Load raw data once to calibrate scalers and fit baselines
        features_pq = self.project_root / "data" / "processed" / "io_features.parquet"
        if not features_pq.exists():
            features_pq = self.project_root / "data" / "processed" / "io_features.csv"
        
        if features_pq.suffix == ".parquet":
            self.features_df = pd.read_parquet(features_pq)
        else:
            self.features_df = pd.read_csv(features_pq)

        self.features_df["timestamp"] = pd.to_datetime(self.features_df["timestamp"])

        # Live features buffer: populated by API's _append_feature_rows.
        # Kept separate from the immutable historical snapshot.
        self.live_features_df = pd.DataFrame(columns=self.features_df.columns)
        
        # Topology
        self.topology = TopologyGraph.from_dataframe(self.features_df)

        # Classifier & its Scaler
        classifier_path = self.project_root / "models" / "classifier" / "lightgbm_tuned_model.pkl"
        scaler_path = self.project_root / "models" / "scaler.pkl"
        self.classifier = joblib.load(classifier_path)
        self.classifier_scaler = joblib.load(scaler_path)

        # ARF+ADWIN streaming classifier (optional — loaded if artifact exists)
        arf_path = self.project_root / "models" / "classifier" / "arf_model.pkl"
        self.arf_classifier = None
        self._arf_feature_cols = None
        if arf_path.exists():
            try:
                self.arf_classifier = joblib.load(arf_path)
                # ARF uses the same scaled feature columns as LightGBM
                self._arf_feature_cols = self.classifier_scaler.feature_names_in_.tolist()
                logger.info("ARF+ADWIN classifier loaded from %s", arf_path)
            except Exception as e:
                logger.warning("Failed to load ARF classifier: %s. Proceeding without it.", e)

        # Anomaly Ensemble
        ensemble_path = self.project_root / "models" / "anomaly" / "ensemble" / "models"
        self.ensemble = EnsembleDetector.load(ensemble_path)

        # Noisy Neighbor Detector
        self.noisy_neighbor = NoisyNeighborDetector(
            topology=self.topology,
            aggressor_threshold=self.policy.get("rebalance_policy", {}).get("min_hotspot_score_to_trigger", 75.0),
            latency_z_threshold=2.0,
            iops_z_threshold=1.0,
        )
        self.noisy_neighbor.fit_baselines(self.features_df)
        self.noisy_neighbor.index_features(self.features_df)

        # N-BEATS Capacity Forecaster
        nbeats_path = self.project_root / "models" / "forecasting" / "nbeats_model.pth"
        self.nbeats = NBeatsModel(
            input_size=20,
            forecast_size=7,
            n_stacks=3,
            n_blocks=3,
            hidden_size=128,
            n_layers=4,
            dropout=0.1
        )
        self.nbeats.load_state_dict(torch.load(nbeats_path, map_location="cpu"))
        self.nbeats.eval()

        # TFT Data Preparation (returns scaled features/targets and fitted scaler)
        self.tft_features, self.tft_targets, self.tft_scaler = prepare_hourly_latency(
            self.features_df, val_hours=72
        )

        # TFT Tail Latency Forecaster
        tft_path = self.project_root / "models" / "forecasting" / "tft_model.pth"
        self.tft = TemporalFusionTransformer(
            input_size=24,
            num_features=7,
            forecast_size=6,
            d_model=32,
            n_heads=2,
            dropout=0.1,
            quantiles=[0.5, 0.9, 0.95]
        )
        self.tft.load_state_dict(torch.load(tft_path, map_location="cpu"))
        self.tft.eval()

        # Demand forecaster (Quantile regressors per-volume) — optional
        demand_path = self.project_root / "models" / "forecasting" / "demand_forecaster.pkl"
        self.demand_forecaster = None
        if demand_path.exists():
            try:
                self.demand_forecaster = DemandForecaster.load(self.project_root / "models" / "forecasting")
                logger.info("DemandForecaster loaded.")
            except Exception as e:
                logger.warning("Failed to load DemandForecaster: %s", e)

    def combined_features(self) -> pd.DataFrame:
        """Return historical + live features merged for model queries.

        Live rows take priority when the same (volume_id, timestamp) exists
        in both frames, but in practice the timestamps are disjoint.
        """
        if self.live_features_df.empty:
            return self.features_df
        return pd.concat([self.features_df, self.live_features_df], ignore_index=True)

    def get_raw_feature_row(self, volume_id: str, timestamp: pd.Timestamp) -> pd.Series:
        """Extract the exact feature row at timestamp for volume_id."""
        cdf = self.combined_features()
        df_vol = cdf[(cdf["volume_id"] == volume_id) & (cdf["timestamp"] == timestamp)]
        if df_vol.empty:
            df_vol = cdf[cdf["volume_id"] == volume_id].sort_values("timestamp")
            return df_vol.iloc[-1]
        return df_vol.iloc[0]

    def get_lstm_sequence(self, volume_id: str, timestamp: pd.Timestamp) -> np.ndarray:
        """Extract 12-step sequence of the 10 features for the volume up to timestamp."""
        cdf = self.combined_features()
        df_vol = cdf[(cdf["volume_id"] == volume_id) & (cdf["timestamp"] <= timestamp)]
        df_vol = df_vol.sort_values("timestamp").tail(12)
        
        # Extract features
        seq_data = df_vol[EnsembleDetector.FEATURE_COLS].values.astype(np.float32)
        if len(seq_data) < 12:
            # Pad with the oldest sample if insufficient sequence length
            seq_data = np.pad(seq_data, ((12 - len(seq_data), 0), (0, 0)), mode="edge")
        return seq_data

    def get_nbeats_input(self, volume_id: str, timestamp: pd.Timestamp) -> np.ndarray:
        """Prepare 20 days of daily capacity history."""
        cdf = self.combined_features()
        df_vol = cdf[(cdf["volume_id"] == volume_id) & (cdf["timestamp"] <= timestamp)].copy()
        df_vol["date"] = pd.to_datetime(df_vol["timestamp"].dt.date)
        daily = df_vol.groupby("date")["capacity_used_pct"].last()
        
        if daily.max() > 1.0:
            daily = daily / 100.0
            
        vals = daily.values.astype(np.float32)
        if len(vals) < 20:
            vals = np.pad(vals, (20 - len(vals), 0), mode="edge")
        else:
            vals = vals[-20:]
        return vals

    def get_tft_input(self, volume_id: str, timestamp: pd.Timestamp) -> np.ndarray:
        """Prepare 24 hours of hourly feature history."""
        cdf = self.combined_features()
        df_vol = cdf[(cdf["volume_id"] == volume_id) & (cdf["timestamp"] <= timestamp)].copy()
        df_vol["latency_p95_us"] = df_vol[["read_latency_p95_us", "write_latency_p95_us"]].max(axis=1)
        df_vol["hour_ts"] = pd.to_datetime(df_vol["timestamp"].dt.round("h"))
        
        agg_rules = {
            "read_iops": "mean",
            "write_iops": "mean",
            "read_throughput_mbps": "mean",
            "write_throughput_mbps": "mean",
            "queue_depth": "mean",
            "sequential_ratio": "mean",
            "latency_p95_us": "mean"
        }
        hourly = df_vol.groupby("hour_ts").agg(agg_rules)
        hourly = hourly.ffill().fillna(0.0)
        hourly["latency_p95_us"] = hourly["latency_p95_us"].clip(lower=0.0)
        
        cols = ["read_iops", "write_iops", "read_throughput_mbps", "write_throughput_mbps", "queue_depth", "sequential_ratio", "latency_p95_us"]
        vals = hourly[cols].values.astype(np.float32)
        
        if len(vals) < 24:
            vals = np.pad(vals, ((24 - len(vals), 0), (0, 0)), mode="edge")
        else:
            vals = vals[-24:]
        return vals

    def analyze_volume(self, volume_id: str, timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """Runs full coordinated inference for a volume at the given timestamp."""
        if timestamp is None:
            # Use the latest timestamp across historical + live features
            cdf = self.combined_features()
            timestamp = cdf[cdf["volume_id"] == volume_id]["timestamp"].max()
            
        timestamp = pd.to_datetime(timestamp)
        
        # 1. Fetch raw features and scale for classifier
        row = self.get_raw_feature_row(volume_id, timestamp)
        
        classifier_feature_cols = self.classifier_scaler.feature_names_in_.tolist()
        features_arr = row[classifier_feature_cols].to_numpy(dtype=np.float64).reshape(1, -1)
        features_log = np.sign(features_arr) * np.log1p(np.abs(features_arr))
        features_log_df = pd.DataFrame(features_log, columns=classifier_feature_cols)
        features_scaled = self.classifier_scaler.transform(features_log_df)
        features_scaled_df = pd.DataFrame(features_scaled, columns=classifier_feature_cols)
        
        # Predict workload
        pred_class = int(self.classifier.predict(features_scaled_df)[0])
        pred_probs = self.classifier.predict_proba(features_scaled_df)[0].tolist()
        workload_type = LABEL_NAMES[pred_class]

        # ARF+ADWIN secondary classification (online streaming model)
        arf_pred_class = None
        arf_workload_type = None
        if self.arf_classifier is not None:
            try:
                x_dict = dict(zip(self._arf_feature_cols, features_scaled_df[self._arf_feature_cols].iloc[0].tolist()))
                arf_pred_raw = self.arf_classifier.predict_one(x_dict)
                if arf_pred_raw is not None:
                    arf_pred_class = int(arf_pred_raw)
                    arf_workload_type = LABEL_NAMES.get(arf_pred_class, "Unknown")
                # Online learning: update ARF with LightGBM's prediction as pseudo-label
                # Only update if LightGBM confidence is high (max prob > 0.85)
                if max(pred_probs) > 0.85:
                    self.arf_classifier.learn_one(x_dict, pred_class)
            except Exception as e:
                logger.debug("ARF inference failed for %s: %s", volume_id, e)
        
        # 2. Ensemble Anomaly Score
        metrics = VolumeMetrics(
            timestamp=timestamp,
            total_iops=float(row["total_iops"]),
            avg_latency_us=float(row["avg_latency_us"]),
            total_throughput_mbps=float(row["total_throughput_mbps"]),
            read_latency_p99_us=float(row["read_latency_p99_us"]),
            write_latency_p99_us=float(row["write_latency_p99_us"]),
            capacity_used_pct=float(row.get("capacity_used_pct", 0.0))
        )
        
        raw_features_10 = row[EnsembleDetector.FEATURE_COLS].values.astype(np.float32)
        sequence_12 = self.get_lstm_sequence(volume_id, timestamp)
        
        hotspot_score, ensemble_alert = self.ensemble.detect(
            volume_id=volume_id,
            metrics=metrics,
            raw_features=raw_features_10,
            sequence=sequence_12
        )
        
        # 3. Noisy Neighbor victims detection
        # If hotspot score is above threshold, check if this volume is an aggressor
        noisy_neighbor_victims = {}
        if hotspot_score >= self.policy.get("rebalance_policy", {}).get("min_hotspot_score_to_trigger", 75.0):
            ev = self.noisy_neighbor.detect_event(
                aggressor_id=volume_id,
                timestamp=timestamp,
                aggressor_score=hotspot_score
            )
            if ev:
                for victim in ev.victims:
                    noisy_neighbor_victims[victim.volume_id] = victim.impact_score

        # 4. Capacity forecaster (N-BEATS)
        nbeats_in = self.get_nbeats_input(volume_id, timestamp)
        nbeats_forecast = forecast_volume(self.nbeats, nbeats_in, n_steps_ahead=60, device="cpu")
        nbeats_forecast = np.clip(nbeats_forecast, 0.0, 1.0)
        
        # Find Days To Fill (DTF)
        current_cap = float(row.get("capacity_used_pct", 0.0))
        if current_cap > 1.0:
            current_cap = current_cap / 100.0
            
        dtf_85 = None
        dtf_95 = None
        for i, val in enumerate(nbeats_forecast):
            if val >= 0.85 and dtf_85 is None:
                dtf_85 = float(i)
            if val >= 0.95 and dtf_95 is None:
                dtf_95 = float(i)
                break
                
        # 5. Latency tail risk forecasting (TFT)
        tft_in = self.get_tft_input(volume_id, timestamp)
        tft_scaled = self.tft_scaler.transform(tft_in)
        
        x = torch.from_numpy(tft_scaled).unsqueeze(0)
        with torch.no_grad():
            pred = self.tft(x).numpy().squeeze(0)  # [forecast_size, num_quantiles]
            
        target_mean = float(self.tft_scaler.mean_[-1])
        target_scale = float(self.tft_scaler.scale_[-1])
        
        p50 = pred[:, 0] * target_scale + target_mean
        p90 = pred[:, 1] * target_scale + target_mean
        p95 = pred[:, 2] * target_scale + target_mean
        
        # Quantile crossing protection
        forecasts = np.stack([p50, p90, p95], axis=1)
        forecasts = np.sort(forecasts, axis=1)
        p50 = np.clip(forecasts[:, 0], 0.0, None)
        p90 = np.clip(forecasts[:, 1], 0.0, None)
        p95 = np.clip(forecasts[:, 2], 0.0, None)
        
        max_p95 = float(np.max(p95))
        # Latency risk score: probability of SLO breach (approximated here by severity)
        latency_risk_score = 1.0 if max_p95 >= 8000.0 else (max_p95 / 8000.0)

        # IOPS and throughput quantile demand forecast
        demand_forecast_24h = None
        if self.demand_forecaster is not None:
            try:
                demand_forecast_24h = self.demand_forecaster.predict_next_24h(volume_id)
            except Exception as e:
                logger.debug("Demand forecast failed for %s: %s", volume_id, e)

        return {
            "volume_id": volume_id,
            "timestamp": timestamp.isoformat(),
            "workload_type": workload_type,
            "workload_confidence": pred_probs,
            "arf_workload_type": arf_workload_type,
            "arf_agrees_with_lgbm": (arf_pred_class == pred_class) if arf_pred_class is not None else None,
            "hotspot_score": round(float(hotspot_score), 2),
            "noisy_neighbor_victims": noisy_neighbor_victims,
            "days_to_fill": {
                "warning_85pct_days": round(dtf_85, 1) if dtf_85 is not None else None,
                "critical_95pct_days": round(dtf_95, 1) if dtf_95 is not None else None
            },
            "bandwidth_forecast_24h": {
                "p50_latency_us": [round(float(v), 2) for v in p50],
                "p90_latency_us": [round(float(v), 2) for v in p90],
                "p95_latency_us": [round(float(v), 2) for v in p95]
            },
            "demand_forecast_24h": demand_forecast_24h,
            "latency_risk_score": round(latency_risk_score, 4)
        }

if __name__ == "__main__":
    hub = InferenceHub()
    res = hub.analyze_volume("vol_003")
    import pprint
    pprint.pprint(res)
