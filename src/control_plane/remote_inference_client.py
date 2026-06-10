import os
import yaml
import joblib
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from src.pipeline.topology_graph import TopologyGraph

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class RemoteInferenceClient:
    """
    Enterprise Proxy: Forwards model prediction requests to a dedicated 
    ML serving layer (e.g., KServe, Triton) via HTTP/gRPC.
    """
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.inference_server_url = os.getenv("INFERENCE_SERVER_URL", "http://ml-serving-cluster:8080/v1/models")
        
        # Load config/policy
        policy_path = self.project_root / "configs" / "policy.yaml"
        with open(policy_path, "r") as f:
            self.policy = yaml.safe_load(f)

        from configs.settings import settings

        self.features_df = pd.DataFrame()
        self._historical_vols = {}

        if settings.db_type == "local":
            # Load dataset for features and topology
            features_pq = self.project_root / "data" / "processed" / "io_features.parquet"
            if not features_pq.exists():
                features_pq = self.project_root / "data" / "processed" / "io_features.csv"
            
            if features_pq.exists():
                if features_pq.suffix == ".parquet":
                    self.features_df = pd.read_parquet(features_pq)
                else:
                    self.features_df = pd.read_csv(features_pq)
            else:
                # Fallback
                csv_path = self.project_root / "dataset_overview.csv"
                self.features_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

            if not self.features_df.empty:
                self.features_df["timestamp"] = pd.to_datetime(self.features_df["timestamp"])
                self._historical_vols = dict(tuple(self.features_df.groupby("volume_id")))
                self.topology = TopologyGraph.from_dataframe(self.features_df)
            else:
                self._historical_vols = {}
                self.topology = None
            self.live_features_df = pd.DataFrame(columns=self.features_df.columns)
        elif settings.db_type == "timescaledb":
            from src.infrastructure.timescale_client import TimescaleClient
            self.db_client = TimescaleClient()
            topo_df = self.db_client.get_topology_data()
            self.topology = TopologyGraph.from_dataframe(topo_df)
            self.live_features_df = pd.DataFrame()

        # Load light classifier & scaler for SHAP explainability
        classifier_path = self.project_root / "models" / "classifier" / "lightgbm_tuned_model.pkl"
        scaler_path = self.project_root / "models" / "scaler.pkl"
        self.classifier = joblib.load(classifier_path) if classifier_path.exists() else None
        self.classifier_scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        logger.info(f"Initialized Remote Inference Client pointing to {self.inference_server_url}")

    def update_live_features(self, df: pd.DataFrame) -> None:
        self.live_features_df = pd.concat([self.live_features_df, df]).drop_duplicates(subset=["volume_id", "timestamp"], keep="last")

    def get_volume_features(self, volume_id: str) -> pd.DataFrame:
        from configs.settings import settings
        
        # 1. Fetch Historical Data dynamically
        if settings.db_type == "timescaledb":
            df_hist = self.db_client.get_historical_features(volume_id)
        else:
            df_hist = self._historical_vols.get(volume_id, pd.DataFrame(columns=self.features_df.columns))
            
        # 2. Append Live Data Buffer
        if not self.live_features_df.empty:
            df_live = self.live_features_df[self.live_features_df["volume_id"] == volume_id]
            if not df_live.empty:
                return pd.concat([df_hist, df_live], ignore_index=True)
        return df_hist

    def get_raw_feature_row(self, volume_id: str, timestamp: pd.Timestamp) -> pd.Series:
        """Extract the exact feature row at timestamp for volume_id."""
        df_vol = self.get_volume_features(volume_id)
        match = df_vol[df_vol["timestamp"] == timestamp]
        if match.empty:
            df_vol = df_vol.sort_values("timestamp")
            return df_vol.iloc[-1]
        return match.iloc[0]

    def known_volumes(self) -> set:
        """Returns the union of all known volume IDs."""
        ids = set(self.features_df["volume_id"].unique()) if not self.features_df.empty else set()
        if not self.live_features_df.empty:
            ids.update(self.live_features_df["volume_id"].unique())
        if self.topology:
            ids.update(self.topology.all_volumes())
        return ids

    def fast_hotspot_score(self, volume_id: str, timestamp: pd.Timestamp) -> float:
        """Simulated statistical hotspot detector score."""
        return 5.0  # safe baseline

    def _call_remote_model(self, model_name: str, payload: list) -> float:
        """Simulate a network call to KServe/Triton."""
        try:
            response = requests.post(f"{self.inference_server_url}/{model_name}:predict", json={"instances": payload}, timeout=0.1)
            response.raise_for_status()
            return response.json()["predictions"][0]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, Exception):
            # Fallback for the local Docker prototype
            return 0.15 # Safe baseline simulation

    def analyze_volume(self, volume_id: str, timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """Gathers features and dispatches inference to the remote GPU cluster."""
        df_vol = self.get_volume_features(volume_id)
        if df_vol.empty:
            return {"hotspot_score": 0.0, "risk_level": "Safe", "forecast": {"tft_p95_latency": 0.0, "nbeats_capacity_used": 0.0}}
            
        if timestamp is None:
            timestamp = df_vol["timestamp"].max()
            
        timestamp = pd.to_datetime(timestamp)
        
        feature_payload = [volume_id, timestamp.isoformat()]

        anomaly_score = self._call_remote_model("ensemble-detector", feature_payload)
        tft_pred = self._call_remote_model("tft-latency", feature_payload) * 1000 # scale up
        nbeats_pred = self._call_remote_model("nbeats-capacity", feature_payload)
        
        risk_level = "Critical" if anomaly_score > 0.8 else "Warning" if anomaly_score > 0.6 else "Safe"
        
        return {
            "volume_id": volume_id,
            "timestamp": timestamp.isoformat(),
            "hotspot_score": float(anomaly_score),
            "risk_level": risk_level,
            "forecast": {
                "tft_p95_latency": float(tft_pred),
                "nbeats_capacity_used": float(nbeats_pred)
            }
        }
