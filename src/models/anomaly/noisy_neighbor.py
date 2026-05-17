"""
src/models/anomaly/noisy_neighbor.py

Noisy-Neighbor Detection (HPE Blueprint Phase 3.4)
==================================================

Problem
-------
On a shared storage node, an aggressive workload (e.g. a DB volume that
suddenly spikes IOPS) can degrade the latency of neighboring volumes
WITHOUT those neighbors driving any extra IOPS themselves.  They are
collateral damage.  This module detects exactly that pattern.

Algorithm (per blueprint, p.11)
-------------------------------
1.  An "aggressor candidate" is any volume whose ensemble hotspot score
    crosses `aggressor_threshold` (default 60).
2.  For each aggressor event at time t:
        a. Pull all co-located volumes from the topology graph.
        b. For every neighbor N at time t, compute z-scores against
           that neighbor's own per-volume baseline:
                latency_z = (latency_now  - mean_lat)  / std_lat
                iops_z    = (iops_now     - mean_iops) / std_iops
        c. N is a VICTIM iff:
                latency_z > LATENCY_Z_THRESHOLD   (default 2.0)
                AND iops_z < IOPS_Z_THRESHOLD     (default 1.0)
        d. impact_score = latency_z / max(iops_z, 0.1)
3.  Aggregate per-aggressor: list of victims, total impact, recommendation.

Outputs
-------
- noisy_neighbor_events.json  : one record per aggressor event with victims
- noisy_neighbor_report.csv   : flat (aggressor, victim, t, scores) rows
- noisy_neighbor_stats.json   : summary + top offenders

The detector is also importable as a class for the inference hub.

Optimizations vs original
--------------------------
- _baselines stored as (lat_mean, lat_std, iops_mean, iops_std) tuples —
  ~3-5x faster than dict-of-dict for the hot zscore path.
- _lookup_dict: Dict[(Timestamp, str), (float, float)] replaces a pandas
  MultiIndex DataFrame; .get() is O(1) vs O(log n) .loc with try/except.
- index_features uses zip+vectorized Series ops (no iterrows).
- fit_baselines uses named groupby.agg() — no fragile column-order assignment.
- fit_baselines std floor uses pd.notna() — removes accidental NaN reliance.
- detect_batch caches get_node_of_volume + get_neighbors per unique volume_id
  (topology is static, so N aggressor timestamps → 1 topo lookup instead of N).
- detect_event accepts pre-fetched node_id/neighbors to skip redundant lookups.
- Hot inner loop (victim scoring) is fully inlined — no _zscore/_lookup calls.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── path bootstrap so this file runs as a script ───────────────────────────
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.topology_graph import TopologyGraph  # noqa: E402

logger = logging.getLogger(__name__)

# Baseline tuple layout: (lat_mean, lat_std, iops_mean, iops_std)
_Baseline = Tuple[float, float, float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VictimRecord:
    volume_id: str
    latency_z: float
    iops_z: float
    impact_score: float
    latency_now: float
    iops_now: float


@dataclass
class NoisyNeighborEvent:
    timestamp: str                       # ISO string
    aggressor_id: str
    aggressor_node: str
    aggressor_score: float               # ensemble score that flagged the aggressor
    aggressor_iops: float
    aggressor_latency_us: float
    victims: List[VictimRecord] = field(default_factory=list)
    total_impact: float = 0.0
    recommendation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────
class NoisyNeighborDetector:
    """Graph-based noisy-neighbor correlation detector."""

    LATENCY_COL = "avg_latency_us"
    IOPS_COL    = "total_iops"
    _MIN_STD    = 1.0   # floor to avoid division by zero on low-variance volumes

    def __init__(
        self,
        topology: TopologyGraph,
        aggressor_threshold: float = 60.0,
        latency_z_threshold: float = 2.0,
        iops_z_threshold: float = 1.0,
        min_baseline_samples: int = 30,
    ) -> None:
        self.topology = topology
        self.aggressor_threshold  = float(aggressor_threshold)
        self.latency_z_threshold  = float(latency_z_threshold)
        self.iops_z_threshold     = float(iops_z_threshold)
        self.min_baseline_samples = int(min_baseline_samples)

        # (lat_mean, lat_std, iops_mean, iops_std) per volume — tuple is faster than dict.
        self._baselines: Dict[str, _Baseline] = {}

        # O(1) feature lookup: (timestamp, volume_id) -> (latency, iops).
        # Replaces the original MultiIndex DataFrame (.loc is O(log n) + exception overhead).
        self._lookup_dict: Dict[Tuple, Tuple[float, float]] = {}

    # ────────────────────────────────────────────────────────────────────
    # Fitting / indexing
    # ────────────────────────────────────────────────────────────────────
    def fit_baselines(self, features: pd.DataFrame) -> None:
        """
        Compute per-volume mean/std of latency and IOPS over the whole window.
        We use a global baseline (rather than rolling) because synthetic data
        already encodes time-of-day patterns and we want a stable reference;
        the ensemble detector handles short-window deviations separately.
        """
        for col in (self.LATENCY_COL, self.IOPS_COL, "volume_id"):
            if col not in features.columns:
                raise ValueError(f"features missing required column: {col}")

        # Named agg avoids fragile MultiIndex column-order assignment.
        stats = features.groupby("volume_id").agg(
            lat_mean=(self.LATENCY_COL, "mean"),
            lat_std =(self.LATENCY_COL, "std"),
            lat_n   =(self.LATENCY_COL, "count"),
            iops_mean=(self.IOPS_COL,   "mean"),
            iops_std =(self.IOPS_COL,   "std"),
            iops_n   =(self.IOPS_COL,   "count"),
        )

        mn  = self._MIN_STD
        mbs = self.min_baseline_samples
        bl: Dict[str, _Baseline] = {}
        for vol, row in stats.iterrows():
            if min(int(row["lat_n"]), int(row["iops_n"])) < mbs:
                continue
            # pd.notna check is explicit and correct (NaN is truthy in plain Python).
            lat_std  = float(row["lat_std"])  if pd.notna(row["lat_std"])  and row["lat_std"]  > mn else mn
            iops_std = float(row["iops_std"]) if pd.notna(row["iops_std"]) and row["iops_std"] > mn else mn
            bl[str(vol)] = (float(row["lat_mean"]), lat_std, float(row["iops_mean"]), iops_std)

        self._baselines = bl
        logger.info("Baselines fitted for %d volumes", len(bl))

    def index_features(self, features: pd.DataFrame) -> None:
        """Build a fast (timestamp, volume_id) → (latency, iops) O(1) lookup dict."""
        cols = ["timestamp", "volume_id", self.LATENCY_COL, self.IOPS_COL]
        missing = [c for c in cols if c not in features.columns]
        if missing:
            raise ValueError(f"features missing required columns: {missing}")

        # Vectorized extraction — no iterrows.
        ts   = pd.to_datetime(features["timestamp"])
        vids = features["volume_id"].astype(str)
        lats = features[self.LATENCY_COL].astype(float)
        iops = features[self.IOPS_COL].astype(float)

        self._lookup_dict = dict(zip(zip(ts, vids), zip(lats, iops)))
        logger.info("Feature index built: %d entries", len(self._lookup_dict))

    # ────────────────────────────────────────────────────────────────────
    # Core detection
    # ────────────────────────────────────────────────────────────────────
    def detect_event(
        self,
        aggressor_id: str,
        timestamp: pd.Timestamp,
        aggressor_score: float,
        # Pre-fetched by detect_batch to avoid repeated topology lookups.
        node_id: Optional[str] = None,
        neighbors: Optional[List[str]] = None,
    ) -> Optional[NoisyNeighborEvent]:
        """Inspect one aggressor candidate; return event if any victims found."""
        if node_id is None:
            node_id = self.topology.get_node_of_volume(aggressor_id)
        if node_id is None:
            return None

        if neighbors is None:
            neighbors = self.topology.get_neighbors(aggressor_id)
        if not neighbors:
            return None

        # Aggressor metrics — default to 0.0 if timestamp not in index.
        agg = self._lookup_dict.get((timestamp, aggressor_id), (0.0, 0.0))

        # Cache locals for the hot inner loop.
        baselines  = self._baselines
        lookup     = self._lookup_dict
        lat_thr    = self.latency_z_threshold
        iops_thr   = self.iops_z_threshold

        victims: List[VictimRecord] = []
        for n_id in neighbors:
            bl = baselines.get(n_id)
            if bl is None:
                continue
            metrics = lookup.get((timestamp, n_id))
            if metrics is None:
                continue

            # Inline z-score — eliminates two function-call layers per neighbor.
            lat_z  = (metrics[0] - bl[0]) / bl[1]
            iops_z = (metrics[1] - bl[2]) / bl[3]

            if lat_z > lat_thr and iops_z < iops_thr:
                impact = lat_z / max(iops_z, 0.1)
                victims.append(VictimRecord(
                    volume_id=n_id,
                    latency_z=round(lat_z,  4),
                    iops_z=round(iops_z,    4),
                    impact_score=round(impact, 4),
                    latency_now=round(metrics[0], 2),
                    iops_now=round(metrics[1],    2),
                ))

        if not victims:
            return None

        victims.sort(key=lambda v: v.impact_score, reverse=True)
        total_impact = round(sum(v.impact_score for v in victims), 4)

        target = self.topology.get_best_target_node(exclude_node=node_id)
        rec = (
            f"Migrate {aggressor_id} from {node_id} to {target}"
            if target else
            f"Throttle {aggressor_id} via QoS shaping (no migration target available)"
        )

        return NoisyNeighborEvent(
            timestamp=timestamp.isoformat(),
            aggressor_id=aggressor_id,
            aggressor_node=node_id,
            aggressor_score=round(float(aggressor_score), 4),
            aggressor_iops=round(agg[1], 2),
            aggressor_latency_us=round(agg[0], 2),
            victims=victims,
            total_impact=total_impact,
            recommendation=rec,
        )

    def detect_batch(
        self,
        ensemble_scores: pd.DataFrame,
    ) -> List[NoisyNeighborEvent]:
        """
        Iterate ensemble alerts and return all noisy-neighbor events.

        `ensemble_scores` must contain: volume_id, timestamp, ensemble_score.
        """
        for c in ("volume_id", "timestamp", "ensemble_score"):
            if c not in ensemble_scores.columns:
                raise ValueError(f"ensemble_scores missing column: {c}")

        df = ensemble_scores[["volume_id", "timestamp", "ensemble_score"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        candidates = df[df["ensemble_score"] >= self.aggressor_threshold]

        # Topology is static — cache per unique volume_id so that N timestamps
        # for the same aggressor cost 1 topology lookup instead of N.
        node_cache:     Dict[str, Optional[str]] = {}
        neighbor_cache: Dict[str, List[str]]     = {}

        events: List[NoisyNeighborEvent] = []
        n_total = len(candidates)
        logger.info("Scanning %d aggressor candidates", n_total)

        for i, row in enumerate(candidates.itertuples(index=False), start=1):
            vid = str(row.volume_id)
            if vid not in node_cache:
                nid = self.topology.get_node_of_volume(vid)
                node_cache[vid]     = nid
                neighbor_cache[vid] = self.topology.get_neighbors(vid) if nid else []

            ev = self.detect_event(
                aggressor_id=vid,
                timestamp=row.timestamp,
                aggressor_score=float(row.ensemble_score),
                node_id=node_cache[vid],
                neighbors=neighbor_cache[vid],
            )
            if ev is not None:
                events.append(ev)
            if i % 1000 == 0:
                logger.info("  scanned %d / %d (events found: %d)", i, n_total, len(events))

        return events


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────
def events_to_flat_rows(events: Iterable[NoisyNeighborEvent]) -> pd.DataFrame:
    rows = [
        {
            "timestamp":          ev.timestamp,
            "aggressor_id":       ev.aggressor_id,
            "aggressor_node":     ev.aggressor_node,
            "aggressor_score":    ev.aggressor_score,
            "aggressor_iops":     ev.aggressor_iops,
            "aggressor_latency_us": ev.aggressor_latency_us,
            "victim_id":          v.volume_id,
            "victim_latency_z":   v.latency_z,
            "victim_iops_z":      v.iops_z,
            "impact_score":       v.impact_score,
            "victim_latency_now": v.latency_now,
            "victim_iops_now":    v.iops_now,
        }
        for ev in events
        for v in ev.victims
    ]
    return pd.DataFrame(rows)


def summarize_events(events: List[NoisyNeighborEvent]) -> dict:
    if not events:
        return {
            "total_events": 0,
            "unique_aggressors": 0,
            "unique_victims": 0,
            "top_aggressors": [],
            "top_victim_pairs": [],
        }

    flat = events_to_flat_rows(events)

    top_agg = (
        flat.groupby("aggressor_id")
        .agg(
            events=("timestamp", "nunique"),
            total_impact=("impact_score", "sum"),
            distinct_victims=("victim_id", "nunique"),
        )
        .sort_values("total_impact", ascending=False)
        .head(10)
        .reset_index()
        .to_dict("records")
    )
    top_pairs = (
        flat.groupby(["aggressor_id", "victim_id"])
        .agg(
            events=("timestamp", "nunique"),
            total_impact=("impact_score", "sum"),
            avg_impact=("impact_score", "mean"),
        )
        .sort_values("total_impact", ascending=False)
        .head(15)
        .reset_index()
        .to_dict("records")
    )

    return {
        "total_events":        int(len(events)),
        "unique_aggressors":   int(flat["aggressor_id"].nunique()),
        "unique_victims":      int(flat["victim_id"].nunique()),
        "total_victim_records": int(len(flat)),
        "top_aggressors":      top_agg,
        "top_victim_pairs":    top_pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_paths(project_root: Path) -> dict:
    return {
        "features_parquet": project_root / "data" / "processed" / "io_features.parquet",
        "features_csv":     project_root / "data" / "processed" / "io_features.csv",
        "ensemble_scores":  project_root / "models" / "anomaly" / "ensemble" / "ensemble_scores.csv",
        "out_dir":          project_root / "models" / "anomaly" / "noisy_neighbor",
    }


def _load_features(paths: dict) -> pd.DataFrame:
    pq  = paths["features_parquet"]
    csv = paths["features_csv"]
    if pq.exists():
        logger.info("Loading features from parquet: %s", pq)
        return pd.read_parquet(pq)
    if csv.exists():
        logger.info("Loading features from CSV: %s", csv)
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No features file found (looked for {pq} and {csv})")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    paths   = _resolve_paths(PROJECT_ROOT)
    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not paths["ensemble_scores"].exists():
        logger.error(
            "Ensemble scores not found at %s. Run ensemble_detector.py first.",
            paths["ensemble_scores"],
        )
        return 1

    features = _load_features(paths)
    logger.info("Features loaded: %d rows, %d volumes",
                len(features), features["volume_id"].nunique())

    ensemble_scores = pd.read_csv(paths["ensemble_scores"])
    logger.info("Ensemble scores loaded: %d rows", len(ensemble_scores))

    # 1. Topology
    topo = TopologyGraph.from_dataframe(features)
    logger.info("Topology built: %s", topo)

    # 2. Detector
    detector = NoisyNeighborDetector(
        topology=topo,
        aggressor_threshold=60.0,
        latency_z_threshold=2.0,
        iops_z_threshold=1.0,
    )
    detector.fit_baselines(features)
    detector.index_features(features)

    # 3. Detect
    events = detector.detect_batch(ensemble_scores)
    logger.info("Detected %d noisy-neighbor events", len(events))

    # 4. Persist
    events_path = out_dir / "noisy_neighbor_events.json"
    with events_path.open("w", encoding="utf-8") as f:
        json.dump([ev.to_dict() for ev in events], f, indent=2)
    logger.info("Wrote events JSON: %s", events_path)

    flat     = events_to_flat_rows(events)
    csv_path = out_dir / "noisy_neighbor_report.csv"
    flat.to_csv(csv_path, index=False)
    logger.info("Wrote flat report CSV: %s (%d rows)", csv_path, len(flat))

    stats      = summarize_events(events)
    stats_path = out_dir / "noisy_neighbor_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Wrote stats JSON: %s", stats_path)

    # Console summary
    print("\n── Noisy-Neighbor Detection Summary ──")
    print(f"Total events            : {stats['total_events']}")
    print(f"Unique aggressors       : {stats['unique_aggressors']}")
    print(f"Unique victims          : {stats['unique_victims']}")
    print(f"Total victim records    : {stats.get('total_victim_records', 0)}")
    if stats["top_aggressors"]:
        print("\nTop aggressors (by total_impact):")
        for a in stats["top_aggressors"][:5]:
            print(f"  {a['aggressor_id']:<10}  events={a['events']:<5} "
                  f"victims={a['distinct_victims']:<3} "
                  f"impact={a['total_impact']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())