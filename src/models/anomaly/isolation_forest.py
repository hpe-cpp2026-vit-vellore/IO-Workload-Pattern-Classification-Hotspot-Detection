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
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector.

    Trained on normal data to identify anomalous IO patterns.
    Anomalies are patterns never seen during normal operation.

    The anomaly score is normalized to [0, 100] using percentile bounds
    computed from the training data, so scores are always interpretable
    relative to what was seen at fit time.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        max_samples: int = 256,
        random_state: int = 42,
        score_percentile_low: float = 2.0,
        score_percentile_high: float = 98.0,
    ) -> None:
        """
        Initialize Isolation Forest detector.

        Parameters
        ----------
        contamination : float
            Expected proportion of anomalies in training data (0 < value < 0.5).
        n_estimators : int
            Number of isolation trees.
        max_samples : int
            Number of samples per tree.
        random_state : int
            Reproducibility seed.
        score_percentile_low : float
            Lower percentile of training scores used to anchor the 0-100 scale.
        score_percentile_high : float
            Upper percentile of training scores used to anchor the 0-100 scale.
        """
        if not (0.0 < contamination < 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5), got {contamination}"
            )

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.score_percentile_low = score_percentile_low
        self.score_percentile_high = score_percentile_high

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )

        self.scaler = StandardScaler()
        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None
        self.n_features_: Optional[int] = None

        # Percentile anchors for score normalization (set during fit).
        # BUG FIX: original formula (-raw / 0.5) * 100 loses all resolution
        # for scores below -0.5 — they all saturate to 100. Instead, we
        # calibrate the scale from the training-data score distribution so
        # the full [0, 100] range is used meaningfully.
        self._score_norm_low: Optional[float] = None   # high percentile (near-normal)
        self._score_norm_high: Optional[float] = None  # low percentile (most anomalous seen in train)

        self._anomaly_count: int = 0
        self._total_samples: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "IsolationForestDetector":
        """
        Fit detector on normal data.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Normal (non-anomalous) training data.

        Returns
        -------
        self
        """
        # BUG FIX: reset counters on re-fit so statistics stay consistent
        # with the current model, not a previous one.
        self._anomaly_count = 0
        self._total_samples = 0

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values.astype(float)
        else:
            X_array = np.array(X, dtype=float)

        self.n_features_ = X_array.shape[1]

        # BUG FIX: check for NaN/Inf before scaling; StandardScaler silently
        # propagates NaN, producing scores of NaN for every sample.
        X_array = self._sanitize(X_array, context="fit")

        X_scaled = self.scaler.fit_transform(X_array)
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Calibrate score normalisation from training-data score distribution.
        train_raw_scores = self.model.score_samples(X_scaled)
        self._score_norm_high = float(
            np.percentile(train_raw_scores, self.score_percentile_low)
        )   # most anomalous end of *normal* data
        self._score_norm_low = float(
            np.percentile(train_raw_scores, self.score_percentile_high)
        )   # most normal end of normal data

        logger.info(
            "IsolationForestDetector fitted on %d samples (%d features). "
            "Score normalization anchors: [%.4f, %.4f]",
            len(X_array), self.n_features_,
            self._score_norm_high, self._score_norm_low,
        )
        return self

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self, sample: Union[pd.Series, np.ndarray]
    ) -> Tuple[float, bool]:
        """
        Detect if a single sample is anomalous.

        Parameters
        ----------
        sample : pd.Series or 1-D ndarray
            Single observation to evaluate.

        Returns
        -------
        anomaly_score : float
            Score in [0, 100]; higher means more anomalous.
        is_anomaly : bool
            True when the model classifies the sample as an anomaly.
        """
        self._check_is_fitted()

        if isinstance(sample, pd.Series):
            arr = sample.values.astype(float)
        else:
            arr = np.array(sample, dtype=float).ravel()

        # BUG FIX: original code called reshape(1, -1) on whatever shape came
        # in; if the array was already (1, n) this would silently produce the
        # wrong shape for a 2-D input with a single feature.
        if arr.ndim != 1:
            raise ValueError(
                f"detect() expects a 1-D sample, got shape {arr.shape}. "
                "Use detect_batch() for multiple samples."
            )

        self._validate_feature_count(arr, "detect")

        sample_2d = arr.reshape(1, -1)
        sample_2d = self._sanitize(sample_2d, context="detect")
        sample_scaled = self.scaler.transform(sample_2d)

        raw_score = float(self.model.score_samples(sample_scaled)[0])
        prediction = int(self.model.predict(sample_scaled)[0])

        anomaly_score = self._normalize_score(raw_score)
        is_anomaly = prediction == -1

        self._total_samples += 1
        if is_anomaly:
            self._anomaly_count += 1

        return anomaly_score, is_anomaly

    def detect_batch(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in a batch of samples.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Batch to evaluate.

        Returns
        -------
        anomaly_scores : ndarray of float, shape (n_samples,)
            Scores in [0, 100]; higher means more anomalous.
        is_anomalies : ndarray of bool, shape (n_samples,)
            True where the model classifies a row as an anomaly.
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(float)
        else:
            X_array = np.array(X, dtype=float)

        self._validate_feature_count(X_array, "detect_batch")
        X_array = self._sanitize(X_array, context="detect_batch")
        X_scaled = self.scaler.transform(X_array)

        raw_scores = self.model.score_samples(X_scaled)
        predictions = self.model.predict(X_scaled)

        anomaly_scores = self._normalize_score_batch(raw_scores)
        is_anomalies = predictions == -1

        # BUG FIX: cast numpy integer to plain int to keep the counter type
        # stable; mixing int + numpy.int64 can cause JSON serialization issues.
        self._total_samples += len(X_array)
        self._anomaly_count += int(is_anomalies.sum())

        return anomaly_scores, is_anomalies

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Persist the fitted detector to disk.

        Parameters
        ----------
        path : str or Path
            File path (e.g. 'models/iso_forest.pkl').
        """
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Detector saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IsolationForestDetector":
        """
        Load a previously saved detector.

        Parameters
        ----------
        path : str or Path
            File path written by :meth:`save`.

        Returns
        -------
        IsolationForestDetector
        """
        detector = joblib.load(path)
        if not isinstance(detector, cls):
            raise TypeError(
                f"Loaded object is {type(detector)}, expected {cls}"
            )
        logger.info("Detector loaded from %s", path)
        return detector

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Return a JSON-serializable summary of detector state and counters."""
        return {
            "is_fitted": self.is_fitted,
            "n_features": self.n_features_,
            "feature_names": self.feature_names,
            "total_samples_processed": self._total_samples,
            "anomalies_detected": self._anomaly_count,
            "anomaly_rate": round(
                self._anomaly_count / max(1, self._total_samples), 4
            ),
            "contamination_threshold": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "score_norm_anchors": {
                "low (most-normal pct)": self._score_norm_low,
                "high (least-normal pct)": self._score_norm_high,
            },
        }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"IsolationForestDetector("
            f"contamination={self.contamination}, "
            f"n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, "
            f"status={status})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Detector is not fitted yet. Call fit() before detect()."
            )

    def _validate_feature_count(
        self, X: np.ndarray, context: str
    ) -> None:
        """Raise if the number of features doesn't match what was used at fit time."""
        n_cols = X.shape[1] if X.ndim == 2 else X.shape[0]
        if n_cols != self.n_features_:
            raise ValueError(
                f"{context}: expected {self.n_features_} features "
                f"(from fit), got {n_cols}."
            )

    @staticmethod
    def _sanitize(X: np.ndarray, context: str) -> np.ndarray:
        """
        Replace NaN and Inf with column medians / zeros so the model never
        receives invalid values. Logs a warning when imputation is triggered.

        BUG FIX: StandardScaler propagates NaN silently, producing NaN scores
        that compare False to every threshold — anomalies would be missed
        entirely without this guard.
        """
        if not np.isfinite(X).all():
            bad_mask = ~np.isfinite(X)
            n_bad = int(bad_mask.sum())
            logger.warning(
                "%s: found %d NaN/Inf value(s); replacing with column median.",
                context, n_bad,
            )
            col_medians = np.where(
                np.isfinite(X).any(axis=0),
                np.nanmedian(X, axis=0),
                0.0,
            )
            for col in range(X.shape[1]):
                col_bad = bad_mask[:, col]
                if col_bad.any():
                    X = X.copy()
                    X[col_bad, col] = col_medians[col]
        return X

    def _normalize_score(self, raw_score: float) -> float:
        """
        Map a single raw IsolationForest score to [0, 100].

        BUG FIX: the original formula ``(-raw_score / 0.5) * 100`` saturates
        at 100 for any score below -0.5 — highly anomalous samples all
        collapse to the same value, destroying score resolution.

        Instead we linearly interpolate between the percentile anchors
        calibrated from training data:
            0   ≡ score at the high (normal) percentile
            100 ≡ score at the low (anomalous) percentile, or below
        """
        if self._score_norm_low is None or self._score_norm_high is None:
            # Fallback only; should never reach here after fit().
            return float(np.clip((-raw_score / 0.5) * 100, 0.0, 100.0))

        denom = self._score_norm_low - self._score_norm_high
        if denom == 0:
            return 0.0
        normalized = (self._score_norm_low - raw_score) / denom * 100.0
        return float(np.clip(normalized, 0.0, 100.0))

    def _normalize_score_batch(self, raw_scores: np.ndarray) -> np.ndarray:
        """Vectorized version of :meth:`_normalize_score`."""
        if self._score_norm_low is None or self._score_norm_high is None:
            return np.clip((-raw_scores / 0.5) * 100, 0.0, 100.0)

        denom = self._score_norm_low - self._score_norm_high
        if denom == 0:
            return np.zeros(len(raw_scores), dtype=float)
        normalized = (self._score_norm_low - raw_scores) / denom * 100.0
        return np.clip(normalized, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_detection_on_data(
    data_path: str = "data/processed/io_features.parquet",
    output_dir: str = "models/anomaly",
    contamination: float = 0.05,
    train_fraction: float = 0.7,
) -> None:
    """
    Run Isolation Forest detector on processed IO feature data.

    Strategy
    --------
    1. Load and sort data chronologically.
    2. Validate that all required feature columns are present.
    3. Split: first ``train_fraction`` = normal training set,
       remainder = test set.
    4. Fit detector on training data.
    5. Detect anomalies in the test set.
    6. Persist scores, anomaly records, model, and statistics.

    Parameters
    ----------
    data_path : str
        Path to processed features parquet file.
    output_dir : str
        Directory where results and the model will be saved.
    contamination : float
        Expected anomaly rate (used by IsolationForest threshold).
    train_fraction : float
        Proportion of data used for training (must be in (0, 1)).
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(
            f"train_fraction must be in (0, 1), got {train_fraction}"
        )

    print("=" * 70)
    print(" Isolation Forest Anomaly Detector")
    print(" HPE Phase 3.2 — Global Anomaly Detection")
    print("=" * 70)

    # ------------------------------------------------------------------ Load
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["volume_id", "timestamp"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {df['volume_id'].nunique()} volumes")

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

    # BUG FIX: validate feature columns exist before indexing to provide a
    # clear error message instead of a cryptic KeyError.
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required feature columns are missing from the "
            f"data: {missing}. Available columns: {df.columns.tolist()}"
        )

    X = df[feature_cols].copy()

    # ----------------------------------------------------------------- Split
    split_idx = int(len(X) * train_fraction)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    print(f"\n  Train (normal): {len(X_train):,} samples")
    print(f"  Test:           {len(X_test):,} samples")

    # ------------------------------------------------------------------- Fit
    print(
        f"\nTraining Isolation Forest "
        f"(n_estimators=200, contamination={contamination})..."
    )
    detector = IsolationForestDetector(
        contamination=contamination,
        n_estimators=200,
        max_samples=256,
        random_state=42,
    )
    detector.fit(X_train)
    print("  Detector trained")

    # --------------------------------------------------------------- Detect
    print("\nDetecting anomalies in test data...")
    anomaly_scores, is_anomalies = detector.detect_batch(X_test)

    results = pd.DataFrame(
        {
            "volume_id": df.iloc[split_idx:]["volume_id"].values,
            "timestamp": df.iloc[split_idx:]["timestamp"].values,
            "anomaly_score": anomaly_scores,
            "is_anomaly": is_anomalies,
        }
    )

    # ------------------------------------------------------------------ Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scores_path = output_path / "isolation_forest_scores.csv"
    results.to_csv(scores_path, index=False)
    print(f"  Scores    -> {scores_path}")

    anomalies = results[results["is_anomaly"]].copy()
    anomalies_path = output_path / "isolation_forest_anomalies.json"
    anomalies["timestamp"] = anomalies["timestamp"].astype(str)
    with open(anomalies_path, "w") as f:
        json.dump(anomalies.to_dict("records"), f, indent=2)
    print(f"  Anomalies -> {anomalies_path}")

    stats = detector.get_statistics()
    stats_path = output_path / "isolation_forest_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats     -> {stats_path}")

    model_path = output_path / "isolation_forest_model.pkl"
    detector.save(model_path)
    print(f"  Model     -> {model_path}")

    # --------------------------------------------------------------- Summary
    n_anomalies = int(is_anomalies.sum())
    print("\n" + "=" * 70)
    print(" DETECTION SUMMARY")
    print("=" * 70)
    print(f"  Total test samples : {len(X_test):,}")
    print(f"  Anomalies detected : {n_anomalies:,}")
    print(f"  Actual anomaly rate: {n_anomalies / len(X_test) * 100:.2f}%")
    print(f"  Expected rate      : {contamination * 100:.2f}%")

    print(f"\n  Anomaly score distribution (0-100):")
    print(f"    Min    : {anomaly_scores.min():.2f}")
    print(f"    Mean   : {anomaly_scores.mean():.2f}")
    print(f"    Median : {float(np.median(anomaly_scores)):.2f}")
    print(f"    Max    : {anomaly_scores.max():.2f}")

    print(f"\n  Top 5 volumes by anomaly rate:")
    vol_stats = (
        results.groupby("volume_id")["is_anomaly"]
        .agg(anomalies="sum", total="count")
        .assign(rate=lambda d: d["anomalies"] / d["total"])
        .sort_values("rate", ascending=False)
    )
    for vol_id, row in vol_stats.head(5).iterrows():
        print(
            f"    {vol_id}: {row['rate']*100:.2f}%"
            f" ({int(row['anomalies'])}/{int(row['total'])})"
        )

    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_detection_on_data()