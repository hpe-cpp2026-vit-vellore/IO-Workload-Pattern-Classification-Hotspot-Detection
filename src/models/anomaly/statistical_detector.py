"""
src/models/anomaly/statistical_detector.py

Statistical baseline hotspot detector using rolling 24-hour baselines.
Detects anomalies when current metrics exceed baseline_mean + 3*std.

HPE Success Criteria:
- Detect hotspots within seconds (<1 second response time)
- Flag volumes with abnormal latency/IOPS/throughput patterns

Usage:
    detector = StatisticalHotspotDetector(window_hours=24)
    
    # Update baseline with historical data
    for volume_id, metrics in historical_data:
        detector.update_baseline(volume_id, metrics)
    
    # Detect hotspots on new data
    score, alerts = detector.detect_hotspot(volume_id, current_metrics)
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class VolumeMetrics:
    """Container for volume IO metrics."""
    timestamp: pd.Timestamp
    total_iops: float
    avg_latency_us: float
    total_throughput_mbps: float
    read_latency_p99_us: float
    write_latency_p99_us: float
    capacity_used_pct: float


@dataclass
class HotspotAlert:
    """Hotspot detection alert."""
    volume_id: str
    timestamp: pd.Timestamp
    hotspot_score: float
    severity: str
    triggered_metrics: List[str]
    details: Dict[str, float]


class StatisticalHotspotDetector:
    """
    Statistical baseline detector for hotspot identification.
    
    Maintains rolling 24-hour baselines per volume and flags anomalies
    when current metrics exceed baseline_mean + threshold*std.
    """
    
    def __init__(
        self,
        window_hours: int = 24,
        threshold_sigma: float = 3.0,
        min_samples: int = 12,
    ):
        """
        Initialize statistical detector.
        
        Parameters
        ----------
        window_hours : int
            Rolling window size in hours for baseline calculation
        threshold_sigma : float
            Number of standard deviations for anomaly threshold
        min_samples : int
            Minimum samples required before detection (default: 12 = 1 hour at 5-min intervals)
        """
        self.window_hours = window_hours
        self.threshold_sigma = threshold_sigma
        self.min_samples = min_samples
        
        # Rolling window size (5-minute intervals)
        self.window_size = window_hours * 12  # 12 intervals per hour
        
        # Per-volume metric history (deque for efficient rolling window)
        self.history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                "total_iops": deque(maxlen=self.window_size),
                "avg_latency_us": deque(maxlen=self.window_size),
                "total_throughput_mbps": deque(maxlen=self.window_size),
                "read_latency_p99_us": deque(maxlen=self.window_size),
                "write_latency_p99_us": deque(maxlen=self.window_size),
            }
        )
        
        # Per-volume baseline statistics (cached for performance)
        self.baselines: Dict[str, Dict[str, Tuple[float, float]]] = {}
        
        # Detection counters
        self.total_detections = 0
        self.detections_per_volume: Dict[str, int] = defaultdict(int)
    
    def update_baseline(self, volume_id: str, metrics: VolumeMetrics) -> None:
        """
        Update rolling baseline with new metrics.
        
        Parameters
        ----------
        volume_id : str
            Volume identifier
        metrics : VolumeMetrics
            Current volume metrics
        """
        hist = self.history[volume_id]
        
        # Append new metrics to rolling window
        hist["total_iops"].append(metrics.total_iops)
        hist["avg_latency_us"].append(metrics.avg_latency_us)
        hist["total_throughput_mbps"].append(metrics.total_throughput_mbps)
        hist["read_latency_p99_us"].append(metrics.read_latency_p99_us)
        hist["write_latency_p99_us"].append(metrics.write_latency_p99_us)
        
        # Recompute baseline statistics if enough samples
        if len(hist["total_iops"]) >= self.min_samples:
            self.baselines[volume_id] = {
                metric: (np.mean(values), np.std(values))
                for metric, values in hist.items()
            }
    
    def detect_hotspot(
        self,
        volume_id: str,
        metrics: VolumeMetrics,
    ) -> Tuple[float, HotspotAlert | None]:
        """
        Detect if current metrics indicate a hotspot.
        
        Parameters
        ----------
        volume_id : str
            Volume identifier
        metrics : VolumeMetrics
            Current volume metrics
        
        Returns
        -------
        hotspot_score : float
            Hotspot score (0-100)
        alert : HotspotAlert | None
            Alert object if hotspot detected, None otherwise
        """
        # Check if baseline exists
        if volume_id not in self.baselines:
            return 0.0, None
        
        baseline = self.baselines[volume_id]
        
        # Calculate z-scores for each metric
        z_scores = {}
        triggered = []
        
        current = {
            "total_iops": metrics.total_iops,
            "avg_latency_us": metrics.avg_latency_us,
            "total_throughput_mbps": metrics.total_throughput_mbps,
            "read_latency_p99_us": metrics.read_latency_p99_us,
            "write_latency_p99_us": metrics.write_latency_p99_us,
        }
        
        for metric, value in current.items():
            mean, std = baseline[metric]
            
            # Avoid division by zero
            if std < 1e-6:
                z_scores[metric] = 0.0
                continue
            
            z_score = (value - mean) / std
            z_scores[metric] = z_score
            
            # Flag if exceeds threshold
            if z_score > self.threshold_sigma:
                triggered.append(metric)
        
        # Compute hotspot score (0-100)
        # Weighted combination: latency (50%), IOPS (30%), throughput (20%)
        latency_z = max(
            z_scores.get("avg_latency_us", 0),
            z_scores.get("read_latency_p99_us", 0),
            z_scores.get("write_latency_p99_us", 0),
        )
        iops_z = z_scores.get("total_iops", 0)
        throughput_z = z_scores.get("total_throughput_mbps", 0)
        
        weighted_z = (
            0.5 * latency_z +
            0.3 * iops_z +
            0.2 * throughput_z
        )
        
        # Map z-score to 0-100 scale (z=3 → 60, z=5 → 80, z=7 → 100)
        hotspot_score = min(100.0, max(0.0, (weighted_z / 7.0) * 100))
        
        # Create alert if score exceeds threshold
        alert = None
        if hotspot_score >= 40.0:  # Minimum warning threshold
            severity = self.get_alert_severity(hotspot_score)
            
            alert = HotspotAlert(
                volume_id=volume_id,
                timestamp=metrics.timestamp,
                hotspot_score=round(hotspot_score, 2),
                severity=severity,
                triggered_metrics=triggered,
                details={
                    "latency_z_score": round(latency_z, 2),
                    "iops_z_score": round(iops_z, 2),
                    "throughput_z_score": round(throughput_z, 2),
                    "current_latency_us": round(metrics.avg_latency_us, 2),
                    "baseline_latency_us": round(baseline["avg_latency_us"][0], 2),
                },
            )
            
            self.total_detections += 1
            self.detections_per_volume[volume_id] += 1
        
        return hotspot_score, alert
    
    def get_alert_severity(self, score: float) -> str:
        """
        Map hotspot score to severity level.
        
        Parameters
        ----------
        score : float
            Hotspot score (0-100)
        
        Returns
        -------
        severity : str
            One of: "normal", "warning", "high", "critical"
        """
        if score < 40:
            return "normal"
        elif score < 60:
            return "warning"
        elif score < 80:
            return "high"
        else:
            return "critical"
    
    def get_statistics(self) -> Dict:
        """
        Get detector statistics.
        
        Returns
        -------
        stats : dict
            Detection statistics
        """
        return {
            "total_detections": self.total_detections,
            "volumes_monitored": len(self.history),
            "volumes_with_baselines": len(self.baselines),
            "detections_per_volume": dict(self.detections_per_volume),
        }
    
    def save_statistics(self, path: Path) -> None:
        """Save detector statistics to JSON."""
        stats = self.get_statistics()
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)


def run_detection_on_data(
    data_path: str = "data/processed/io_features.parquet",
    output_dir: str = "models/anomaly",
) -> None:
    """
    Run statistical detector on processed data and save results.
    
    Parameters
    ----------
    data_path : str
        Path to processed features
    output_dir : str
        Output directory for results
    """
    print("=" * 70)
    print(" Statistical Hotspot Detector")
    print(" HPE Phase 3.1 — Baseline Anomaly Detection")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["volume_id", "timestamp"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {df['volume_id'].nunique()} volumes")
    
    # Initialize detector
    detector = StatisticalHotspotDetector(
        window_hours=24,
        threshold_sigma=3.0,
        min_samples=12,
    )
    
    # Process data chronologically
    print("\nProcessing data chronologically...")
    alerts = []
    hotspot_scores = []
    
    for idx, row in df.iterrows():
        volume_id = row["volume_id"]
        
        metrics = VolumeMetrics(
            timestamp=row["timestamp"],
            total_iops=row["total_iops"],
            avg_latency_us=row["avg_latency_us"],
            total_throughput_mbps=row["total_throughput_mbps"],
            read_latency_p99_us=row["read_latency_p99_us"],
            write_latency_p99_us=row["write_latency_p99_us"],
            capacity_used_pct=row.get("capacity_used_pct", 0.0),
        )
        
        # Detect hotspot
        score, alert = detector.detect_hotspot(volume_id, metrics)
        
        # Update baseline for next iteration
        detector.update_baseline(volume_id, metrics)
        
        # Store results
        hotspot_scores.append({
            "volume_id": volume_id,
            "timestamp": metrics.timestamp,
            "hotspot_score": score,
        })
        
        if alert:
            alerts.append({
                "volume_id": alert.volume_id,
                "timestamp": str(alert.timestamp),
                "hotspot_score": alert.hotspot_score,
                "severity": alert.severity,
                "triggered_metrics": alert.triggered_metrics,
                "details": alert.details,
            })
        
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx + 1:,} / {len(df):,} rows...")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save alerts
    alerts_path = output_path / "statistical_detector_alerts.json"
    with open(alerts_path, "w") as f:
        json.dump(alerts, f, indent=2)
    print(f"\n  Alerts → {alerts_path}")
    print(f"  Total alerts: {len(alerts)}")
    
    # Save hotspot scores
    scores_df = pd.DataFrame(hotspot_scores)
    scores_path = output_path / "statistical_detector_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"  Scores → {scores_path}")
    
    # Save statistics
    stats_path = output_path / "statistical_detector_stats.json"
    detector.save_statistics(stats_path)
    print(f"  Stats  → {stats_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print(" DETECTION SUMMARY")
    print("=" * 70)
    stats = detector.get_statistics()
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Volumes monitored: {stats['volumes_monitored']}")
    print(f"  Volumes with baselines: {stats['volumes_with_baselines']}")
    
    if alerts:
        severity_counts = pd.Series([a["severity"] for a in alerts]).value_counts()
        print(f"\n  Alerts by severity:")
        for severity, count in severity_counts.items():
            print(f"    {severity}: {count}")
    
    print("=" * 70)


if __name__ == "__main__":
    run_detection_on_data()
