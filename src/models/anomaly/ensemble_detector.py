"""
src/models/anomaly/ensemble_detector.py

Ensemble Anomaly Detector — HPE Phase 3.4
==========================================
Combines three complementary detectors into a single, unified pipeline:

  ┌──────────────────────────────────────────────────────────────┐
  │  1. StatisticalHotspotDetector  (online, per-volume)         │
  │     Rolling 24h z-score baseline → hotspot_score  [0-100]   │
  │                                                              │
  │  2. IsolationForestDetector     (offline, global)            │
  │     Tree ensemble trained on normal data → anomaly_score     │
  │     [0-100]                                                  │
  │                                                              │
  │  3. LSTMAutoencoder             (offline, sequential)        │
  │     Reconstruction-error on 12-step sequences →              │
  │     normalized_error [0-100]                                 │
  │                                                              │
  │       Weighted Average / Meta-Learner Fusion                 │
  │              ↓                                               │
  │         ensemble_score [0-100]  +  severity label           │
  └──────────────────────────────────────────────────────────────┘

Fusion strategy
---------------
- Each model outputs a score normalised to [0, 100].
- A configurable weight vector (w_stat, w_if, w_lstm) produces a weighted
  average ensemble score.  Weights are automatically redistributed when a
  model is unavailable (e.g. LSTM not yet trained).
- Optional meta-learner (LogisticRegression) can be fitted on a labelled
  held-out set to learn optimal fusion weights automatically.
- Consensus gate: an alert is only raised when the ensemble score exceeds
  the alarm threshold AND at least `min_agreement` of the individual
  detectors independently flag the sample.  This eliminates single-model
  false positives.

HPE Success Criteria
--------------------
- Sub-second inference per sample (all models share a single forward pass)
- Higher precision / recall than any single model in isolation
- Graceful degradation: system keeps running when LSTM or IF is not ready
- Volume-level and global anomalies both detected

Usage
-----
    # --- offline batch mode ---
    ensemble = EnsembleDetector()
    ensemble.fit_isolation_forest(X_train)
    ensemble.fit_lstm(X_train, volume_ids_train)

    score, alert = ensemble.detect(volume_id, metrics, raw_features, sequence)
    results      = ensemble.detect_batch(df)

    # --- optional meta-learner calibration ---
    ensemble.fit_meta_learner(X_val, y_val_labels)

    # --- save / load ---
    ensemble.save("models/anomaly/ensemble/")
    ensemble = EnsembleDetector.load("models/anomaly/ensemble/")
"""

from __future__ import annotations

import io
import json
import logging
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── sibling detector imports ──────────────────────────────────────────────────
# Adjust the import paths to match your project layout.
from statistical_detector import (  # noqa: E402
    HotspotAlert,
    StatisticalHotspotDetector,
    VolumeMetrics,
)
from isolation_forest import IsolationForestDetector  # noqa: E402
from lstm_autoencoder import LSTMAutoencoder          # noqa: E402

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleAlert:
    """Unified alert produced by the ensemble."""

    volume_id: str
    timestamp: pd.Timestamp
    ensemble_score: float          # weighted aggregate [0-100]
    severity: str                  # normal / warning / high / critical
    is_anomaly: bool

    # Per-model contributions
    stat_score: float              # statistical detector score [0-100]
    if_score: float                # isolation forest score    [0-100]
    lstm_score: float              # LSTM normalised score     [0-100]

    # Which individual detectors flagged this sample
    stat_flagged: bool
    if_flagged: bool
    lstm_flagged: bool
    n_agreeing: int                # number of detectors that flagged

    # Pass-through from statistical detector for root-cause context
    stat_alert: Optional[HotspotAlert] = None
    details: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble detector
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleDetector:
    """
    Weighted ensemble of StatisticalHotspotDetector, IsolationForestDetector,
    and LSTMAutoencoder for IO workload hotspot detection.

    Parameters
    ----------
    w_stat : float
        Weight for the statistical detector score (default 0.35).
        Higher values emphasise fast, per-volume z-score anomalies.
    w_if : float
        Weight for the Isolation Forest score (default 0.35).
        Higher values emphasise global cross-volume pattern anomalies.
    w_lstm : float
        Weight for the LSTM autoencoder score (default 0.30).
        Higher values emphasise sequential temporal pattern anomalies.
        Weights are renormalised automatically when a model is unavailable.
    alarm_threshold : float
        Ensemble score (0–100) above which an EnsembleAlert is emitted
        (default 40.0 — same as the statistical detector warning level).
    min_agreement : int
        Minimum number of individual detectors that must independently flag
        a sample before an alert is raised (default 1).  Set to 2 or 3 for
        higher precision at the cost of recall.
    window_hours : int
        Rolling window for the statistical detector (default 24 h).
    stat_threshold_sigma : float
        Z-score threshold for the statistical detector (default 3.0).
    stat_min_samples : int
        Minimum samples before the statistical detector produces a baseline
        (default 12 = 1 hour at 5-min intervals).
    lstm_error_percentile_cap : float
        Percentile of observed LSTM errors used to cap the normalisation
        denominator (default 99.0).  Prevents a single extreme outlier from
        collapsing all other scores toward zero.
    """

    # ── Feature columns shared by IF and LSTM ─────────────────────────────────
    FEATURE_COLS: List[str] = [
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

    def __init__(
        self,
        w_stat: float = 0.35,
        w_if: float = 0.35,
        w_lstm: float = 0.30,
        alarm_threshold: float = 40.0,
        min_agreement: int = 1,
        # Statistical detector settings
        window_hours: int = 24,
        stat_threshold_sigma: float = 3.0,
        stat_min_samples: int = 12,
        # LSTM normalisation
        lstm_error_percentile_cap: float = 99.0,
        # IF / LSTM settings (passed through to sub-detectors)
        if_contamination: float = 0.05,
        if_n_estimators: int = 200,
        lstm_input_dim: int = 10,
        lstm_sequence_length: int = 12,
        lstm_hidden_dim: int = 64,
        lstm_latent_dim: int = 8,
        lstm_threshold_percentile: float = 95.0,
    ) -> None:
        # Validate weights
        total_w = w_stat + w_if + w_lstm
        if total_w <= 0:
            raise ValueError("Detector weights must sum to a positive value.")
        self.w_stat = w_stat / total_w
        self.w_if   = w_if   / total_w
        self.w_lstm = w_lstm / total_w

        self.alarm_threshold = alarm_threshold
        self.min_agreement   = min_agreement

        # ── Sub-detectors ──────────────────────────────────────────────────────
        self.stat_detector = StatisticalHotspotDetector(
            window_hours=window_hours,
            threshold_sigma=stat_threshold_sigma,
            min_samples=stat_min_samples,
        )

        self.if_detector = IsolationForestDetector(
            contamination=if_contamination,
            n_estimators=if_n_estimators,
        )

        self.lstm_detector = LSTMAutoencoder(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden_dim,
            latent_dim=lstm_latent_dim,
            sequence_length=lstm_sequence_length,
            threshold_percentile=lstm_threshold_percentile,
        )

        # ── LSTM score normalisation ───────────────────────────────────────────
        # Populated by _calibrate_lstm_normalisation() or during batch inference.
        self._lstm_error_cap: Optional[float] = None
        self._lstm_error_percentile_cap = lstm_error_percentile_cap

        # ── Optional meta-learner ──────────────────────────────────────────────
        self._meta_learner: Optional[LogisticRegression] = None
        self._meta_scaler:  Optional[StandardScaler]     = None
        self._use_meta = False

        # ── Runtime tracking ──────────────────────────────────────────────────
        self._total_detections = 0
        self._detections_per_volume: Dict[str, int] = defaultdict(int)
        self._score_history: deque = deque(maxlen=10_000)  # rolling window for stats

    # ─────────────────────────────────────────────────────────────────────────
    # Fitting helpers
    # ─────────────────────────────────────────────────────────────────────────

    def fit_isolation_forest(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
    ) -> "EnsembleDetector":
        """
        Train the Isolation Forest on normal data.

        Parameters
        ----------
        X_train : DataFrame or ndarray, shape (n_samples, n_features)
            Normal (non-anomalous) training data.  Must contain exactly the
            columns listed in ``EnsembleDetector.FEATURE_COLS`` (or the
            same number of columns in that order).
        """
        logger.info("Fitting Isolation Forest on %d samples…", len(X_train))
        self.if_detector.fit(X_train)
        logger.info("Isolation Forest fitted.")
        return self

    def fit_lstm(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        volume_ids: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> "EnsembleDetector":
        """
        Train the LSTM autoencoder on normal data.

        Parameters
        ----------
        X_train : DataFrame or ndarray, shape (n_samples, n_features)
        volume_ids : ndarray of shape (n_samples,), optional
            Volume identifier per row for per-volume sequence grouping.
        epochs, batch_size, verbose : LSTM training hyper-parameters.
        """
        logger.info(
            "Fitting LSTM Autoencoder on %d samples (epochs=%d)…",
            len(X_train), epochs,
        )
        self.lstm_detector.fit(
            X_train,
            volume_ids=volume_ids,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        logger.info("LSTM Autoencoder fitted. Threshold=%.6f", self.lstm_detector.threshold)
        return self

    def calibrate_lstm_normalisation(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        volume_ids: Optional[np.ndarray] = None,
    ) -> "EnsembleDetector":
        """
        Compute the LSTM error cap from a validation set so LSTM reconstruction
        errors can be faithfully normalised to [0, 100].

        Call this after ``fit_lstm`` using held-out (normal + some anomalous)
        data for the best calibration.  If not called, the cap is estimated
        lazily from the first batch seen at inference time.

        Parameters
        ----------
        X_val : DataFrame or ndarray, shape (n_samples, n_features)
        volume_ids : ndarray, optional — needed for per-volume sequence grouping.
        """
        if not self.lstm_detector.is_fitted:
            raise RuntimeError("Call fit_lstm() before calibrate_lstm_normalisation().")

        if isinstance(X_val, pd.DataFrame):
            X_array = X_val.values.astype(np.float32)
        else:
            X_array = np.asarray(X_val, dtype=np.float32)

        if volume_ids is None:
            volume_ids = np.zeros(len(X_array), dtype=object)

        X_scaled = self.lstm_detector.scaler.transform(X_array)
        sequences, _ = self.lstm_detector._create_sequences(X_scaled, volume_ids)
        errors, _ = self.lstm_detector.detect_batch(sequences, _already_scaled=True)

        self._lstm_error_cap = float(
            np.percentile(errors, self._lstm_error_percentile_cap)
        )
        logger.info(
            "LSTM normalisation cap (p%.0f) = %.6f",
            self._lstm_error_percentile_cap, self._lstm_error_cap,
        )
        return self

    def fit_meta_learner(
        self,
        X_scores: np.ndarray,
        y_labels: np.ndarray,
    ) -> "EnsembleDetector":
        """
        Fit an optional logistic-regression meta-learner on top of the three
        detector scores to learn optimal fusion weights from labelled data.

        Parameters
        ----------
        X_scores : ndarray, shape (n_samples, 3)
            Columns: [stat_score, if_score, lstm_score] each in [0, 100].
            Collect these by running ``detect_batch`` on a labelled validation
            set and extracting the per-model scores from the returned alerts.
        y_labels : ndarray, shape (n_samples,)
            Binary ground-truth labels: 1 = anomaly, 0 = normal.
        """
        self._meta_scaler = StandardScaler()
        X_scaled = self._meta_scaler.fit_transform(X_scores)

        self._meta_learner = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self._meta_learner.fit(X_scaled, y_labels)
        self._use_meta = True

        # Log learned coefficients as relative weights
        coef = self._meta_learner.coef_[0]
        logger.info(
            "Meta-learner fitted. Relative weights — "
            "stat: %.3f  IF: %.3f  LSTM: %.3f",
            *np.abs(coef) / np.abs(coef).sum(),
        )
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Core detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect(
        self,
        volume_id: str,
        metrics: VolumeMetrics,
        raw_features: Optional[np.ndarray] = None,
        sequence: Optional[np.ndarray] = None,
    ) -> Tuple[float, Optional[EnsembleAlert]]:
        """
        Detect anomaly for a single sample (online / streaming mode).

        The statistical detector is always updated and queried.
        The Isolation Forest and LSTM are queried when fitted and when their
        respective inputs are supplied.

        Parameters
        ----------
        volume_id : str
        metrics : VolumeMetrics
            Current volume IO metrics (used by statistical detector).
        raw_features : ndarray, shape (n_features,), optional
            Feature vector for the Isolation Forest.  If None, IF is skipped.
        sequence : ndarray, shape (seq_len, n_features), optional
            Pre-built sequence for the LSTM.  If None, LSTM is skipped.

        Returns
        -------
        ensemble_score : float — weighted aggregate score [0-100]
        alert : EnsembleAlert | None — alert if score >= alarm_threshold
        """
        # ── 1. Statistical detector (always runs; updates baseline) ───────────
        stat_score, stat_alert = self.stat_detector.detect_hotspot(volume_id, metrics)
        self.stat_detector.update_baseline(volume_id, metrics)
        stat_flagged = stat_score >= self.alarm_threshold

        # ── 2. Isolation Forest ───────────────────────────────────────────────
        if_score, if_flagged = self._run_if(raw_features)

        # ── 3. LSTM ───────────────────────────────────────────────────────────
        lstm_score, lstm_flagged = self._run_lstm_single(sequence)

        # ── 4. Fusion ─────────────────────────────────────────────────────────
        ensemble_score = self._fuse_scores(stat_score, if_score, lstm_score)

        # ── 5. Consensus gate ─────────────────────────────────────────────────
        n_agreeing = sum([stat_flagged, if_flagged, lstm_flagged])
        is_anomaly = (ensemble_score >= self.alarm_threshold) and (n_agreeing >= self.min_agreement)

        self._score_history.append(ensemble_score)

        if not is_anomaly:
            return ensemble_score, None

        severity = _score_to_severity(ensemble_score)
        self._total_detections += 1
        self._detections_per_volume[volume_id] += 1

        alert = EnsembleAlert(
            volume_id=volume_id,
            timestamp=metrics.timestamp,
            ensemble_score=round(ensemble_score, 2),
            severity=severity,
            is_anomaly=True,
            stat_score=round(stat_score, 2),
            if_score=round(if_score, 2),
            lstm_score=round(lstm_score, 2),
            stat_flagged=stat_flagged,
            if_flagged=if_flagged,
            lstm_flagged=lstm_flagged,
            n_agreeing=n_agreeing,
            stat_alert=stat_alert,
            details=self._build_details(stat_alert, raw_features, sequence),
        )
        return ensemble_score, alert

    def detect_batch(
        self,
        df: pd.DataFrame,
        update_stat_baseline: bool = True,
    ) -> pd.DataFrame:
        """
        Detect anomalies across a full DataFrame (batch mode).

        Runs the statistical detector row-by-row (chronological order per
        volume) while running the IF and LSTM vectorised over the whole set.

        Parameters
        ----------
        df : DataFrame
            Must contain columns:
              - volume_id
              - timestamp
              - All columns in ``EnsembleDetector.FEATURE_COLS``
              - The VolumeMetrics columns:
                  total_iops, avg_latency_us, total_throughput_mbps,
                  read_latency_p99_us, write_latency_p99_us, capacity_used_pct
        update_stat_baseline : bool
            Whether to feed rows into the statistical baseline as the batch
            is processed (True = realistic online-simulation mode).

        Returns
        -------
        results : DataFrame with columns:
            volume_id, timestamp, ensemble_score, severity, is_anomaly,
            stat_score, if_score, lstm_score,
            stat_flagged, if_flagged, lstm_flagged, n_agreeing
        """
        df = df.sort_values(["volume_id", "timestamp"]).reset_index(drop=True)

        # ── Validate required columns ──────────────────────────────────────────
        required = set(self.FEATURE_COLS) | {
            "volume_id", "timestamp",
            "total_iops", "avg_latency_us", "total_throughput_mbps",
            "read_latency_p99_us", "write_latency_p99_us",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")

        # ── Pre-compute IF scores (vectorised) ────────────────────────────────
        if_scores_all, if_flags_all = self._batch_if_scores(df)

        # ── Pre-compute LSTM scores (vectorised, sequence-aligned) ────────────
        lstm_scores_map, lstm_flags_map = self._batch_lstm_scores(df)

        # ── Statistical detector + fusion (row-by-row to respect ordering) ────
        records = []
        for idx, row in df.iterrows():
            vol_id = row["volume_id"]
            ts     = row["timestamp"]

            metrics = VolumeMetrics(
                timestamp=pd.Timestamp(ts),
                total_iops=float(row["total_iops"]),
                avg_latency_us=float(row["avg_latency_us"]),
                total_throughput_mbps=float(row["total_throughput_mbps"]),
                read_latency_p99_us=float(row["read_latency_p99_us"]),
                write_latency_p99_us=float(row["write_latency_p99_us"]),
                capacity_used_pct=float(row.get("capacity_used_pct", 0.0)),
            )

            stat_score, _ = self.stat_detector.detect_hotspot(vol_id, metrics)
            if update_stat_baseline:
                self.stat_detector.update_baseline(vol_id, metrics)

            if_score   = float(if_scores_all[idx])
            if_flag    = bool(if_flags_all[idx])
            stat_flag  = stat_score >= self.alarm_threshold

            lstm_key   = (vol_id, pd.Timestamp(ts))
            lstm_score = lstm_scores_map.get(lstm_key, 0.0)
            lstm_flag  = lstm_flags_map.get(lstm_key, False)

            ensemble_score = self._fuse_scores(stat_score, if_score, lstm_score)
            n_agreeing     = sum([stat_flag, if_flag, lstm_flag])
            is_anomaly     = (ensemble_score >= self.alarm_threshold) and (n_agreeing >= self.min_agreement)

            records.append({
                "volume_id":     vol_id,
                "timestamp":     ts,
                "ensemble_score": round(ensemble_score, 4),
                "severity":       _score_to_severity(ensemble_score) if is_anomaly else "normal",
                "is_anomaly":    is_anomaly,
                "stat_score":    round(stat_score, 4),
                "if_score":      round(if_score, 4),
                "lstm_score":    round(lstm_score, 4),
                "stat_flagged":  stat_flag,
                "if_flagged":    if_flag,
                "lstm_flagged":  lstm_flag,
                "n_agreeing":    n_agreeing,
            })

        results = pd.DataFrame(records)

        # Update tracking
        anomaly_rows = results[results["is_anomaly"]]
        self._total_detections += len(anomaly_rows)
        for vol_id, cnt in anomaly_rows["volume_id"].value_counts().items():
            self._detections_per_volume[vol_id] += cnt

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_if(
        self, raw_features: Optional[np.ndarray]
    ) -> Tuple[float, bool]:
        """Query the IF detector for a single feature vector."""
        if raw_features is None or not self.if_detector.is_fitted:
            return 0.0, False
        try:
            score, is_anom = self.if_detector.detect(raw_features)
            return float(score), bool(is_anom)
        except Exception as exc:
            logger.warning("IF detect() failed: %s", exc)
            return 0.0, False

    def _run_lstm_single(
        self, sequence: Optional[np.ndarray]
    ) -> Tuple[float, bool]:
        """Query the LSTM detector for a single pre-built sequence."""
        if sequence is None or not self.lstm_detector.is_fitted:
            return 0.0, False
        try:
            error, is_anom = self.lstm_detector.detect(sequence)
            norm_score = self._normalise_lstm_error(error)
            return norm_score, bool(is_anom)
        except Exception as exc:
            logger.warning("LSTM detect() failed: %s", exc)
            return 0.0, False

    def _batch_if_scores(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run IF on the full DataFrame; return score and flag arrays indexed by df.index."""
        n = len(df)
        if not self.if_detector.is_fitted:
            return np.zeros(n), np.zeros(n, dtype=bool)

        # Guard missing feature columns silently (rare in practice after validation above)
        missing_feat = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing_feat:
            logger.warning("IF: missing feature columns %s — skipping IF.", missing_feat)
            return np.zeros(n), np.zeros(n, dtype=bool)

        X = df[self.FEATURE_COLS].values
        try:
            scores, flags = self.if_detector.detect_batch(X)
        except Exception as exc:
            logger.warning("IF detect_batch() failed: %s", exc)
            scores, flags = np.zeros(n), np.zeros(n, dtype=bool)

        return scores, flags

    def _batch_lstm_scores(
        self, df: pd.DataFrame
    ) -> Tuple[Dict, Dict]:
        """
        Run LSTM on all per-volume sequences derived from df.

        Returns two dicts keyed by (volume_id, timestamp):
            lstm_scores_map : float score [0-100]
            lstm_flags_map  : bool flag
        """
        score_map: Dict = {}
        flag_map:  Dict = {}

        if not self.lstm_detector.is_fitted:
            return score_map, flag_map

        volume_ids = df["volume_id"].values
        X = df[self.FEATURE_COLS].values.astype(np.float32)

        try:
            X_scaled = self.lstm_detector.scaler.transform(X)
            sequences, seq_vol_ids = self.lstm_detector._create_sequences(
                X_scaled, volume_ids
            )
            errors, flags = self.lstm_detector.detect_batch(
                sequences, _already_scaled=True
            )
        except Exception as exc:
            logger.warning("LSTM batch detection failed: %s — returning zero scores.", exc)
            return score_map, flag_map

        # Lazy calibration: estimate error cap from this batch if not yet set
        if self._lstm_error_cap is None:
            self._lstm_error_cap = float(
                np.percentile(errors, self._lstm_error_percentile_cap)
            )
            logger.info(
                "LSTM error cap auto-calibrated from batch: %.6f",
                self._lstm_error_cap,
            )

        norm_scores = self._normalise_lstm_error_batch(errors)

        # Map sequences back to their terminal timestamp (same logic as lstm runner)
        seq_idx = 0
        seq_len = self.lstm_detector.sequence_length

        for vol_id in dict.fromkeys(volume_ids):
            vol_df = df[df["volume_id"] == vol_id]
            n_rows = len(vol_df)
            n_seqs = n_rows - seq_len + 1
            if n_seqs <= 0:
                continue

            terminal_ts = vol_df["timestamp"].iloc[seq_len - 1:].values

            for i, ts in enumerate(terminal_ts):
                key = (vol_id, pd.Timestamp(ts))
                score_map[key] = float(norm_scores[seq_idx + i])
                flag_map[key]  = bool(flags[seq_idx + i])

            seq_idx += n_seqs

        return score_map, flag_map

    def _fuse_scores(
        self,
        stat_score: float,
        if_score:   float,
        lstm_score: float,
    ) -> float:
        """
        Weighted fusion with dynamic weight redistribution.

        When a detector is unavailable (score == 0 due to not being fitted),
        its weight is redistributed proportionally among the available models.
        """
        if self._use_meta and self._meta_learner is not None:
            return self._meta_fuse(stat_score, if_score, lstm_score)

        stat_avail = self.if_detector.is_fitted or True   # stat always runs
        if_avail   = self.if_detector.is_fitted
        lstm_avail = self.lstm_detector.is_fitted

        active_weights = {
            "stat": self.w_stat if stat_avail else 0.0,
            "if":   self.w_if   if if_avail   else 0.0,
            "lstm": self.w_lstm if lstm_avail  else 0.0,
        }
        total = sum(active_weights.values())
        if total == 0:
            return 0.0

        scores = {"stat": stat_score, "if": if_score, "lstm": lstm_score}
        ensemble = sum(
            (active_weights[k] / total) * scores[k]
            for k in active_weights
        )
        return float(np.clip(ensemble, 0.0, 100.0))

    def _meta_fuse(
        self,
        stat_score: float,
        if_score:   float,
        lstm_score: float,
    ) -> float:
        """Use the trained meta-learner to produce a calibrated ensemble score."""
        X = np.array([[stat_score, if_score, lstm_score]])
        X_scaled = self._meta_scaler.transform(X)
        prob = self._meta_learner.predict_proba(X_scaled)[0, 1]  # P(anomaly)
        return float(np.clip(prob * 100.0, 0.0, 100.0))

    def _normalise_lstm_error(self, error: float) -> float:
        """Map a single LSTM reconstruction error to [0, 100]."""
        cap = self._lstm_error_cap or max(error, 1e-9)
        return float(np.clip((error / cap) * 100.0, 0.0, 100.0))

    def _normalise_lstm_error_batch(self, errors: np.ndarray) -> np.ndarray:
        """Vectorised version of _normalise_lstm_error."""
        cap = self._lstm_error_cap or float(np.percentile(errors, self._lstm_error_percentile_cap))
        return np.clip((errors / cap) * 100.0, 0.0, 100.0)

    @staticmethod
    def _build_details(
        stat_alert: Optional[HotspotAlert],
        raw_features: Optional[np.ndarray],
        sequence: Optional[np.ndarray],
    ) -> Dict:
        """Assemble a human-readable details dict for an alert."""
        details: Dict = {}
        if stat_alert:
            details["triggered_metrics"] = stat_alert.triggered_metrics
            details.update(stat_alert.details)
        if raw_features is not None:
            details["n_features_supplied"] = len(raw_features)
        if sequence is not None:
            details["sequence_shape"] = list(sequence.shape)
        return details

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, directory: Union[str, Path]) -> None:
        """
        Save the full ensemble to ``directory``.

        Saves
        -----
        - ``ensemble_config.json``      — weights, thresholds, LSTM error cap
        - ``isolation_forest_model.pkl``— joblib-serialised IF detector
        - ``lstm_ae_model.pth``         — PyTorch checkpoint
        - ``meta_learner.pkl``          — optional LogisticRegression + scaler
        - ``stat_detector_stats.json``  — statistical detector statistics
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        # Config
        config = {
            "w_stat": self.w_stat,
            "w_if":   self.w_if,
            "w_lstm": self.w_lstm,
            "alarm_threshold":          self.alarm_threshold,
            "min_agreement":            self.min_agreement,
            "lstm_error_cap":           self._lstm_error_cap,
            "lstm_error_percentile_cap": self._lstm_error_percentile_cap,
            "use_meta":                 self._use_meta,
            "total_detections":         self._total_detections,
        }
        with open(out / "ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Sub-model: statistical detector baselines + history
        stat_stats = self.stat_detector.get_statistics()
        with open(out / "stat_detector_stats.json", "w") as f:
            json.dump(stat_stats, f, indent=2)

        # Sub-model: Isolation Forest
        if self.if_detector.is_fitted:
            self.if_detector.save(out / "isolation_forest_model.pkl")

        # Sub-model: LSTM
        if self.lstm_detector.is_fitted:
            self.lstm_detector.save(out / "lstm_ae_model.pth")

        # Optional: meta-learner
        if self._use_meta and self._meta_learner is not None:
            joblib.dump(
                {"meta_learner": self._meta_learner, "meta_scaler": self._meta_scaler},
                out / "meta_learner.pkl",
            )

        logger.info("Ensemble saved to %s", out)

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        device: Optional[str] = None,
    ) -> "EnsembleDetector":
        """
        Load a previously saved ensemble from ``directory``.

        Parameters
        ----------
        directory : str or Path
        device : str, optional
            Override LSTM device ('cuda' / 'cpu').  None = auto-detect.
        """
        out = Path(directory)

        with open(out / "ensemble_config.json") as f:
            config = json.load(f)

        # Reconstruct with the saved (already-normalised) weights
        ensemble = cls(
            w_stat=config["w_stat"],
            w_if=config["w_if"],
            w_lstm=config["w_lstm"],
            alarm_threshold=config["alarm_threshold"],
            min_agreement=config["min_agreement"],
            lstm_error_percentile_cap=config.get("lstm_error_percentile_cap", 99.0),
        )
        # Bypass re-normalisation: weights are already stored normalised
        ensemble._lstm_error_cap = config.get("lstm_error_cap")
        ensemble._use_meta       = config.get("use_meta", False)
        ensemble._total_detections = config.get("total_detections", 0)

        # Load sub-models if present
        if_path   = out / "isolation_forest_model.pkl"
        lstm_path = out / "lstm_ae_model.pth"

        if if_path.exists():
            ensemble.if_detector = IsolationForestDetector.load(if_path)
            logger.info("Isolation Forest loaded from %s", if_path)

        if lstm_path.exists():
            ensemble.lstm_detector = LSTMAutoencoder.load(lstm_path, device=device)
            logger.info("LSTM Autoencoder loaded from %s", lstm_path)

        meta_path = out / "meta_learner.pkl"
        if meta_path.exists() and ensemble._use_meta:
            meta_bundle = joblib.load(meta_path)
            ensemble._meta_learner = meta_bundle["meta_learner"]
            ensemble._meta_scaler  = meta_bundle["meta_scaler"]
            logger.info("Meta-learner loaded from %s", meta_path)

        logger.info("Ensemble loaded from %s", out)
        return ensemble

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict:
        """Return a JSON-serialisable statistics summary."""
        scores = np.array(self._score_history) if self._score_history else np.array([])
        stat_stats = self.stat_detector.get_statistics()

        return {
            "ensemble": {
                "total_detections":       self._total_detections,
                "detections_per_volume":  dict(self._detections_per_volume),
                "weights": {
                    "statistical":  round(self.w_stat, 4),
                    "isolation_forest": round(self.w_if, 4),
                    "lstm":         round(self.w_lstm, 4),
                },
                "alarm_threshold": self.alarm_threshold,
                "min_agreement":   self.min_agreement,
                "meta_learner_active": self._use_meta,
                "score_distribution": {
                    "count":  len(scores),
                    "mean":   round(float(scores.mean()), 4)  if len(scores) else None,
                    "std":    round(float(scores.std()),  4)  if len(scores) else None,
                    "p50":    round(float(np.median(scores)), 4) if len(scores) else None,
                    "p95":    round(float(np.percentile(scores, 95)), 4) if len(scores) else None,
                    "p99":    round(float(np.percentile(scores, 99)), 4) if len(scores) else None,
                },
            },
            "statistical_detector": stat_stats,
            "isolation_forest": self.if_detector.get_statistics(),
            "lstm_autoencoder": self.lstm_detector.get_statistics(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_severity(score: float) -> str:
    """Map an ensemble score to a severity label."""
    if   score < 40:  return "normal"
    elif score < 60:  return "warning"
    elif score < 80:  return "high"
    else:             return "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble_on_data(
    data_path:      str   = "data/processed/io_features.parquet",
    output_dir:     str   = "models/anomaly/ensemble",
    train_fraction: float = 0.70,
    lstm_epochs:    int   = 50,
    batch_size:     int   = 256,
    w_stat:         float = 0.35,
    w_if:           float = 0.35,
    w_lstm:         float = 0.30,
    alarm_threshold: float = 40.0,
    min_agreement:  int   = 1,
) -> None:
    """
    End-to-end ensemble pipeline.

    Strategy
    --------
    1. Load and sort data chronologically per volume.
    2. Split: first ``train_fraction`` → training (assumed normal).
    3. Fit Isolation Forest + LSTM on training data.
    4. Calibrate LSTM normalisation on a 10 % held-out slice of training data.
    5. Run ensemble detection on the test split.
    6. Save results, model artefacts, and statistics.

    Parameters
    ----------
    data_path      : Path to processed features parquet.
    output_dir     : Where results and models are written.
    train_fraction : Fraction of data used for training (e.g. 0.70).
    lstm_epochs    : LSTM training epochs.
    batch_size     : Mini-batch size for LSTM training.
    w_stat / w_if / w_lstm : Ensemble weights (will be normalised).
    alarm_threshold : Ensemble score above which alerts are raised.
    min_agreement   : Minimum detectors that must agree to raise alert.
    """
    _banner("Ensemble Anomaly Detector — HPE Phase 3.4")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading data from {data_path}…")
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["volume_id", "timestamp"]).reset_index(drop=True)
    df["capacity_used_pct"] = df.get("capacity_used_pct", pd.Series(0.0, index=df.index))
    print(f"      {len(df):,} rows | {df['volume_id'].nunique()} volumes")

    # Validate feature columns
    missing_cols = [c for c in EnsembleDetector.FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required feature columns missing: {missing_cols}\n"
            f"Available: {df.columns.tolist()}"
        )

    # ── Split ──────────────────────────────────────────────────────────────────
    split_idx = int(len(df) * train_fraction)
    df_train  = df.iloc[:split_idx].reset_index(drop=True)
    df_test   = df.iloc[split_idx:].reset_index(drop=True)
    print(f"\n[2/6] Split — Train: {len(df_train):,}  |  Test: {len(df_test):,}")

    X_train      = df_train[EnsembleDetector.FEATURE_COLS].values
    vol_ids_train = df_train["volume_id"].values

    # Hold out 10 % of training data for LSTM error cap calibration
    cal_idx   = int(len(df_train) * 0.90)
    X_cal     = df_train.iloc[cal_idx:][EnsembleDetector.FEATURE_COLS].values
    vol_cal   = df_train.iloc[cal_idx:]["volume_id"].values
    X_fit     = X_train[:cal_idx]
    vol_fit   = vol_ids_train[:cal_idx]

    # ── Initialise ────────────────────────────────────────────────────────────
    print("\n[3/6] Initialising ensemble…")
    ensemble = EnsembleDetector(
        w_stat=w_stat,
        w_if=w_if,
        w_lstm=w_lstm,
        alarm_threshold=alarm_threshold,
        min_agreement=min_agreement,
        lstm_input_dim=len(EnsembleDetector.FEATURE_COLS),
    )

    # ── Train sub-models ──────────────────────────────────────────────────────
    print("\n[4/6] Training sub-models…")
    print("      → Isolation Forest…")
    ensemble.fit_isolation_forest(X_fit)

    print("      → LSTM Autoencoder…")
    ensemble.fit_lstm(
        X_fit,
        volume_ids=vol_fit,
        epochs=lstm_epochs,
        batch_size=batch_size,
        verbose=True,
    )

    print("      → Calibrating LSTM normalisation…")
    ensemble.calibrate_lstm_normalisation(X_cal, vol_cal)

    # ── Detect on test set ────────────────────────────────────────────────────
    print("\n[5/6] Running ensemble detection on test set…")
    results = ensemble.detect_batch(df_test, update_stat_baseline=True)
    print(f"      Done. {len(results):,} rows processed.")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[6/6] Saving results…")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Full scores
    scores_path = out / "ensemble_scores.csv"
    results.to_csv(scores_path, index=False)
    print(f"      Scores  → {scores_path}")

    # Alerts only
    alerts = results[results["is_anomaly"]].copy()
    alerts["timestamp"] = alerts["timestamp"].astype(str)
    alerts_path = out / "ensemble_alerts.json"
    with open(alerts_path, "w") as f:
        json.dump(alerts.to_dict("records"), f, indent=2)
    print(f"      Alerts  → {alerts_path} ({len(alerts):,} alerts)")

    # Statistics
    stats = ensemble.get_statistics()
    stats_path = out / "ensemble_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"      Stats   → {stats_path}")

    # Model artefacts
    ensemble.save(out / "model")
    print(f"      Models  → {out / 'model'}/")

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("DETECTION SUMMARY")
    n_anomalies = int(results["is_anomaly"].sum())
    print(f"  Test samples:      {len(results):,}")
    print(f"  Anomalies flagged: {n_anomalies:,}  ({n_anomalies / len(results) * 100:.2f} %)")
    print()

    # Severity breakdown
    sev_counts = results[results["is_anomaly"]]["severity"].value_counts()
    print("  Alerts by severity:")
    for sev, cnt in sev_counts.items():
        print(f"    {sev:<10}: {cnt:,}")

    # Agreement breakdown
    print("\n  Alerts by agreement level:")
    for n in [1, 2, 3]:
        cnt = int((results["n_agreeing"] == n).sum())
        print(f"    {n} detector{'s' if n > 1 else ''} agree: {cnt:,}")

    # Score statistics
    es = results["ensemble_score"]
    print(f"\n  Ensemble score — min: {es.min():.2f}  mean: {es.mean():.2f}  "
          f"p95: {es.quantile(0.95):.2f}  max: {es.max():.2f}")

    # Top anomalous volumes
    print("\n  Top 5 volumes by anomaly rate:")
    vol_stats = (
        results.groupby("volume_id")["is_anomaly"]
        .agg(anomalies="sum", total="count")
        .assign(rate=lambda d: d["anomalies"] / d["total"])
        .sort_values("rate", ascending=False)
        .head(5)
    )
    for vol_id, row in vol_stats.iterrows():
        print(
            f"    {vol_id}: {row['rate'] * 100:.1f} %"
            f"  ({int(row['anomalies'])}/{int(row['total'])})"
        )

    # Model contribution
    print("\n  Per-model anomaly rates in test set:")
    for col, label in [
        ("stat_flagged",  "Statistical"),
        ("if_flagged",    "Isolation Forest"),
        ("lstm_flagged",  "LSTM AE"),
    ]:
        pct = results[col].mean() * 100
        print(f"    {label:<20}: {pct:.2f} %")

    _banner("DONE")


def _banner(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    run_ensemble_on_data()