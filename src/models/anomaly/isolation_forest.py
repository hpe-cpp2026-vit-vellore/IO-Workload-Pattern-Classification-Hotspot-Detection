"""
src/models/anomaly/isolation_forest.py

Isolation Forest anomaly detector for IO workload hotspot detection.
Trained on normal (non-hotspot) data to identify anomalous patterns.

HPE Success Criteria:
- Detect anomalies in <1 second
- Complement statistical detector with global anomaly detection
- Identify complex patterns missed by statistical baseline

Usage:
    detector = IsolationForestDetector(contamination=0.05)
    detector.fit(normal_data)
    score, alert = detector.detect(sample)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector.
    
    Trained on normal data to identify anomalous IO patterns.
    Anomalies are patterns never seen during normal operation.
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        max_samples: int = 256,
        random_state: int = 42,
    ):
        """
        Initialize Isolation Forest detector.
        
        Parameters
        ----------
        contamination : float
            Expected proportion of anomalies (0.05 = 5%)
        n_estimators : int
            Number of isolation trees
        max_samples : int
            Number of samples per tree
        random_state : int
            Reproducibility seed
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,  # Use all cores
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.anomaly_count = 0
        self.total_samples = 0
    
    def fit(self, X: pd.DataFrame | np.ndarray) -> None:
        """
        Fit detector on normal data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Normal (non-anomalous) training data
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Fit Isolation Forest
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def detect(self, sample: pd.Series | np.ndarray) -> Tuple[float, bool]:
        """
        Detect if sample is anomalous.
        
        Parameters
        ----------
        sample : pd.Series or np.ndarray
            Single sample to check
        
        Returns
        -------
        anomaly_score : float
            Anomaly score (0-100, higher = more anomalous)
        is_anomaly : bool
            True if sample is anomalous
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Convert to array
        if isinstance(sample, pd.Series):
            sample_array = sample.values.reshape(1, -1)
        else:
            sample_array = np.array(sample).reshape(1, -1)
        
        # Scale
        sample_scaled = self.scaler.transform(sample_array)
        
        # Get anomaly score (-1 to 1, where -1 is anomaly)
        raw_score = self.model.score_samples(sample_scaled)[0]
        
        # Get prediction (-1 = anomaly, 1 = normal)
        prediction = self.model.predict(sample_scaled)[0]
        
        # Convert to 0-100 scale (higher = more anomalous)
        # raw_score ranges from -∞ to 0, normalize to 0-100
        anomaly_score = max(0.0, min(100.0, (-raw_score / 0.5) * 100))
        
        is_anomaly = prediction == -1
        
        self.total_samples += 1
        if is_anomaly:
            self.anomaly_count += 1
        
        return anomaly_score, is_anomaly
    
    def detect_batch(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in batch.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Batch of samples
        
        Returns
        -------
        anomaly_scores : np.ndarray
            Anomaly scores (0-100)
        predictions : np.ndarray
            Boolean array (True = anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Convert to array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Scale
        X_scaled = self.scaler.transform(X_array)
        
        # Get scores and predictions
        raw_scores = self.model.score_samples(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        # Convert to 0-100 scale
        anomaly_scores = np.maximum(0.0, np.minimum(100.0, (-raw_scores / 0.5) * 100))
        
        is_anomalies = predictions == -1
        
        self.total_samples += len(X)
        self.anomaly_count += is_anomalies.sum()
        
        return anomaly_scores, is_anomalies
    
    def get_statistics(self) -> Dict:
        """Get detector statistics."""
        return {
            "is_fitted": self.is_fitted,
            "total_samples_processed": int(self.total_samples),
            "anomalies_detected": int(self.anomaly_count),
            "anomaly_rate": round(
                self.anomaly_count / max(1, self.total_samples), 4
            ),
            "contamination_threshold": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
        }


def run_detection_on_data(
    data_path: str = "data/processed/io_features.parquet",
    output_dir: str = "models/anomaly",
    contamination: float = 0.05,
) -> None:
    """
    Run Isolation Forest detector on processed data.
    
    Strategy:
    1. Load all data
    2. Split: first 70% = normal (training), last 30% = test
    3. Fit detector on normal data
    4. Detect anomalies in test data
    5. Save results
    
    Parameters
    ----------
    data_path : str
        Path to processed features
    output_dir : str
        Output directory for results
    contamination : float
        Expected anomaly rate
    """
    print("=" * 70)
    print(" Isolation Forest Anomaly Detector")
    print(" HPE Phase 3.2 — Global Anomaly Detection")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["volume_id", "timestamp"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {df['volume_id'].nunique()} volumes")
    
    # Select features for anomaly detection
    feature_cols = [
        "total_iops",
        "avg_latency_us",
        "total_throughput_mbps",
        "read_latency_p99_us",
        "write_latency_p99_us",
        "sequential_ratio",
        "read_write_ratio",
        "io_size_entropy",
        "queue_depth",
        "iops_per_queue",
    ]
    
    X = df[feature_cols].copy()
    
    # Split: 70% train (normal), 30% test
    split_idx = int(len(X) * 0.7)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    
    print(f"\n  Train (normal): {len(X_train):,} samples")
    print(f"  Test:           {len(X_test):,} samples")
    
    # Initialize and fit detector
    print(f"\nTraining Isolation Forest (n_estimators=200, contamination={contamination})...")
    detector = IsolationForestDetector(
        contamination=contamination,
        n_estimators=200,
        max_samples=256,
        random_state=42,
    )
    detector.fit(X_train)
    print("  ✅ Detector trained")
    
    # Detect anomalies in test data
    print("\nDetecting anomalies in test data...")
    anomaly_scores, is_anomalies = detector.detect_batch(X_test)
    
    # Create results DataFrame
    results = pd.DataFrame({
        "volume_id": df.iloc[split_idx:]["volume_id"].values,
        "timestamp": df.iloc[split_idx:]["timestamp"].values,
        "anomaly_score": anomaly_scores,
        "is_anomaly": is_anomalies,
    })
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save scores
    scores_path = output_path / "isolation_forest_scores.csv"
    results.to_csv(scores_path, index=False)
    print(f"  Scores → {scores_path}")
    
    # Save anomalies only
    anomalies = results[results["is_anomaly"]].copy()
    anomalies_path = output_path / "isolation_forest_anomalies.json"
    anomalies["timestamp"] = anomalies["timestamp"].astype(str)
    anomalies_json = anomalies.to_dict("records")
    with open(anomalies_path, "w") as f:
        json.dump(anomalies_json, f, indent=2)
    print(f"  Anomalies → {anomalies_path}")
    
    # Save statistics
    stats = detector.get_statistics()
    stats_path = output_path / "isolation_forest_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats  → {stats_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print(" DETECTION SUMMARY")
    print("=" * 70)
    print(f"  Total test samples: {len(X_test):,}")
    print(f"  Anomalies detected: {is_anomalies.sum():,}")
    print(f"  Anomaly rate: {(is_anomalies.sum() / len(X_test) * 100):.2f}%")
    print(f"  Expected rate: {contamination * 100:.2f}%")
    
    # Anomaly score distribution
    print(f"\n  Anomaly score distribution:")
    print(f"    Min: {anomaly_scores.min():.2f}")
    print(f"    Mean: {anomaly_scores.mean():.2f}")
    print(f"    Median: {np.median(anomaly_scores):.2f}")
    print(f"    Max: {anomaly_scores.max():.2f}")
    
    # Per-volume anomaly rate
    print(f"\n  Top 5 volumes by anomaly rate:")
    vol_anomaly_rate = results.groupby("volume_id")["is_anomaly"].agg(
        ["sum", "count"]
    )
    vol_anomaly_rate["rate"] = vol_anomaly_rate["sum"] / vol_anomaly_rate["count"]
    vol_anomaly_rate = vol_anomaly_rate.sort_values("rate", ascending=False)
    for vol_id, row in vol_anomaly_rate.head(5).iterrows():
        print(f"    {vol_id}: {row['rate']*100:.2f}% ({int(row['sum'])}/{int(row['count'])})")
    
    print("=" * 70)


if __name__ == "__main__":
    run_detection_on_data()
