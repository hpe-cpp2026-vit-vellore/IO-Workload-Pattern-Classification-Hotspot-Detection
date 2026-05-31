"""
src/pipeline/topology_graph.py

Topology & Capacity Graph (HPE Blueprint Phase 1.5).

A NetworkX-backed bipartite graph that knows:
  - Which storage NODES exist and their capacity / tier
  - Which VOLUMES live on which node / pool / tier
  - Live per-volume metrics (IOPS, latency, throughput) for utilization queries

Used by the noisy-neighbor detector and (later) by the rebalance engine to
find migration targets.

Public API mirrors the blueprint:
    add_storage_node(node_id, capacity_gb, tier)
    add_volume(volume_id, node_id, pool_id, tier, capacity_gb)
    get_volumes_on_node(node_id)
    get_node_utilization(node_id)
    get_best_target_node(exclude_node)
    get_neighbors(volume_id)            # co-located volumes
    update_volume_metrics(volume_id, metrics)
    visualize(save_path=None)           # plotly network graph
    from_dataframe(df)                  # build directly from feature parquet/csv

Optimizations vs original
--------------------------
- _volume_to_node / _node_volumes dicts make get_node_of_volume, get_volumes_on_node,
  get_neighbors and all_nodes/all_volumes O(1) instead of O(graph_nodes).
- add_volume is idempotent (duplicate calls are no-ops).
- from_dataframe uses itertuples (10-100x faster than iterrows).
- get_node_utilization avoids float() casting on every metric key.
- visualize builds edge lists with list extension instead of per-item append.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd


class TopologyGraph:
    """Bipartite graph of storage nodes and volumes with O(1) adjacency lookups."""

    KIND_NODE = "node"
    KIND_VOLUME = "volume"

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self._volume_metrics: Dict[str, dict] = {}
        # Auxiliary indices — keep these in sync with the graph at all times.
        self._volume_to_node: Dict[str, str] = {}          # volume_id -> node_id  O(1)
        self._node_volumes:   Dict[str, List[str]] = {}    # node_id   -> [volume_id, ...]  O(1)
        # Replica relationships
        # _replica_of maps replica_volume_id -> primary_volume_id
        self._replica_of: Dict[str, str] = {}
        # _replicas maps primary_volume_id -> [replica_volume_id, ...]
        self._replicas: Dict[str, List[str]] = {}

    # ────────────────────────────────────────────────────────────────────
    # Construction
    # ────────────────────────────────────────────────────────────────────
    def add_storage_node(
        self,
        node_id: str,
        capacity_gb: Optional[float] = None,
        tier: Optional[str] = None,
    ) -> None:
        if self.graph.has_node(node_id):
            data = self.graph.nodes[node_id]
            if capacity_gb is not None:
                data["capacity_gb"] = capacity_gb
            if tier is not None:
                data["tier"] = tier
            return
        self.graph.add_node(node_id, kind=self.KIND_NODE, capacity_gb=capacity_gb, tier=tier)
        self._node_volumes.setdefault(node_id, [])

    def add_volume(
        self,
        volume_id: str,
        node_id: str,
        pool_id: Optional[str] = None,
        tier: Optional[str] = None,
        capacity_gb: Optional[float] = None,
    ) -> None:
        # Idempotent: skip if already registered (from_dataframe deduplicates upstream,
        # but defensive guard prevents _node_volumes duplicate entries).
        if volume_id in self._volume_to_node:
            return

        if node_id not in self._node_volumes:
            self.add_storage_node(node_id, tier=tier)

        self.graph.add_node(
            volume_id,
            kind=self.KIND_VOLUME,
            node_id=node_id,
            pool_id=pool_id,
            tier=tier,
            capacity_gb=capacity_gb,
        )
        self.graph.add_edge(volume_id, node_id, relation="resides_on")

        # Ensure replica indices are present for this volume (may be populated later)
        self._replica_of.setdefault(volume_id, None)
        self._replicas.setdefault(volume_id, [])

        # Keep auxiliary indices in sync.
        self._volume_to_node[volume_id] = node_id
        self._node_volumes[node_id].append(volume_id)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TopologyGraph":
        """Build topology from any feature/raw frame containing volume/node columns."""
        if df.empty:
            return cls()

        missing = {"volume_id", "node_id"} - set(df.columns)
        if missing:
            raise ValueError(f"from_dataframe missing columns: {missing}")

        # One row per volume; most-recent state wins.
        if "timestamp" in df.columns:
            latest = df.sort_values("timestamp").drop_duplicates("volume_id", keep="last")
        else:
            latest = df.drop_duplicates("volume_id", keep="last")

        has_pool = "pool_id" in latest.columns
        has_tier = "tier" in latest.columns
        has_cap  = "capacity_total_gb" in latest.columns

        topo = cls()
        # itertuples is 10-100x faster than iterrows for row-by-row work.
        for row in latest.itertuples(index=False):
            pool = str(row.pool_id) if has_pool and pd.notna(row.pool_id) else None
            tier = str(row.tier)    if has_tier and pd.notna(row.tier)    else None
            cap  = float(row.capacity_total_gb) if has_cap and pd.notna(row.capacity_total_gb) else None
            topo.add_volume(
                volume_id=str(row.volume_id),
                node_id=str(row.node_id),
                pool_id=pool,
                tier=tier,
                capacity_gb=cap,
            )
        return topo

    # ────────────────────────────────────────────────────────────────────
    # Queries — all O(1) via auxiliary dicts
    # ────────────────────────────────────────────────────────────────────
    def all_nodes(self) -> List[str]:
        return list(self._node_volumes.keys())

    def all_volumes(self) -> List[str]:
        return list(self._volume_to_node.keys())

    def get_volumes_on_node(self, node_id: str) -> List[str]:
        return self._node_volumes.get(node_id, [])

    def get_neighbors(self, volume_id: str) -> List[str]:
        """All other volumes that share the same storage node as `volume_id`."""
        node_id = self._volume_to_node.get(volume_id)
        if node_id is None:
            return []
        return [v for v in self._node_volumes[node_id] if v != volume_id]

    def get_node_of_volume(self, volume_id: str) -> Optional[str]:
        return self._volume_to_node.get(volume_id)

    def get_node_utilization(self, node_id: str) -> dict:
        """Aggregate live per-volume metrics for a node."""
        vols = self._node_volumes.get(node_id, [])
        total_iops = total_tp = lat_sum = 0.0
        lat_count = 0
        for v in vols:
            m = self._volume_metrics.get(v)
            if not m:
                continue
            total_iops += m.get("total_iops", 0.0)
            total_tp   += m.get("total_throughput_mbps", 0.0)
            lat = m.get("avg_latency_us")
            if lat is not None:
                lat_sum += lat
                lat_count += 1
        return {
            "node_id": node_id,
            "n_volumes": len(vols),
            "total_iops": total_iops,
            "total_throughput_mbps": total_tp,
            "avg_latency_us": lat_sum / lat_count if lat_count else 0.0,
        }

    def get_best_target_node(self, exclude_node: str) -> Optional[str]:
        """Return the least-loaded node (lowest total IOPS) other than `exclude_node`."""
        candidates = [n for n in self._node_volumes if n != exclude_node]
        if not candidates:
            return None
        return min(candidates, key=lambda n: self.get_node_utilization(n)["total_iops"])

    def _normalize_capacity_used_fraction(self, raw_value: Optional[float]) -> Optional[float]:
        """Normalize capacity-used value into [0.0, 1.0] fraction.

        Accepts either fraction-scale (0.0-1.0) or percent-scale (0-100).
        """
        if raw_value is None:
            return None
        try:
            val = float(raw_value)
        except (TypeError, ValueError):
            return None

        if val < 0:
            return 0.0
        if val <= 1.0:
            return val
        if val <= 100.0:
            return val / 100.0
        return 1.0

    def _get_volume_capacity_inputs(self, volume_id: str) -> tuple[Optional[float], Optional[float]]:
        """Return (capacity_total_gb, used_fraction) for one volume.

        Priority:
        1) live metrics in self._volume_metrics
        2) topology node attributes fallback (capacity_gb)
        """
        metrics = self._volume_metrics.get(volume_id, {})
        node_attr = self.graph.nodes.get(volume_id, {})

        total_gb = metrics.get("capacity_total_gb")
        if total_gb is None:
            total_gb = node_attr.get("capacity_gb")

        try:
            total_gb = float(total_gb) if total_gb is not None else None
        except (TypeError, ValueError):
            total_gb = None

        used_fraction = self._normalize_capacity_used_fraction(metrics.get("capacity_used_pct"))
        if used_fraction is None and total_gb not in (None, 0.0):
            used_gb = metrics.get("capacity_used_gb")
            try:
                if used_gb is not None:
                    used_fraction = max(0.0, min(1.0, float(used_gb) / float(total_gb)))
            except (TypeError, ValueError, ZeroDivisionError):
                used_fraction = None

        return total_gb, used_fraction

    def _aggregate_headroom_by(self, key_attr: str, default_key: str) -> Dict[str, Dict[str, float]]:
        """Aggregate capacity headroom by a volume attribute key (tier/pool)."""
        buckets: Dict[str, Dict[str, object]] = {}

        for volume_id in self.all_volumes():
            node_attr = self.graph.nodes.get(volume_id, {})
            key = node_attr.get(key_attr) or default_key
            key = str(key)

            if key not in buckets:
                buckets[key] = {
                    "total_capacity_gb": 0.0,
                    "used_capacity_gb": 0.0,
                    "headroom_gb": 0.0,
                    "volume_count": 0,
                    "critical_volumes": [],
                }

            bucket = buckets[key]
            bucket["volume_count"] = int(bucket["volume_count"]) + 1

            total_gb, used_fraction = self._get_volume_capacity_inputs(volume_id)
            if total_gb is not None:
                used_gb = (used_fraction * total_gb) if used_fraction is not None else 0.0
                headroom_gb = max(total_gb - used_gb, 0.0)

                bucket["total_capacity_gb"] = float(bucket["total_capacity_gb"]) + float(total_gb)
                bucket["used_capacity_gb"] = float(bucket["used_capacity_gb"]) + float(used_gb)
                bucket["headroom_gb"] = float(bucket["headroom_gb"]) + float(headroom_gb)

            if used_fraction is not None and used_fraction >= 0.90:
                bucket["critical_volumes"].append(volume_id)

        result: Dict[str, Dict[str, float]] = {}
        for key, info in buckets.items():
            total_capacity = float(info["total_capacity_gb"])
            used_capacity = float(info["used_capacity_gb"])
            used_pct = (used_capacity / total_capacity * 100.0) if total_capacity > 0.0 else 0.0

            result[key] = {
                "total_capacity_gb": round(total_capacity, 2),
                "used_capacity_gb": round(used_capacity, 2),
                "used_pct": round(used_pct, 2),
                "headroom_gb": round(float(info["headroom_gb"]), 2),
                "volume_count": int(info["volume_count"]),
                "critical_volumes": sorted(info["critical_volumes"]),
            }

        return result

    def get_tier_headroom(self) -> Dict[str, Dict[str, float]]:
        """Return per-tier capacity summary aggregated from all volumes."""
        return self._aggregate_headroom_by(key_attr="tier", default_key="Unknown")

    def get_pool_headroom(self) -> Dict[str, Dict[str, float]]:
        """Return per-pool capacity summary aggregated from all volumes."""
        return self._aggregate_headroom_by(key_attr="pool_id", default_key="Unknown")

    # ────────────────────────────────────────────────────────────────────
    # Live updates
    # ────────────────────────────────────────────────────────────────────
    def update_volume_metrics(self, volume_id: str, metrics: dict) -> None:
        """Store the latest metric snapshot for a volume."""
        self._volume_metrics[volume_id] = dict(metrics)

    # ────────────────────────────────────────────────────────────────────
    # Serialisation
    # ────────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n, **attr} for n, attr in self.graph.nodes(data=True)],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ],
        }

    # ────────────────────────────────────────────────────────────────────
    # Replica relationships and placement validation
    # ────────────────────────────────────────────────────────────────────
    def set_replica(self, replica_volume_id: str, primary_volume_id: str) -> None:
        """Register that `replica_volume_id` is a replica of `primary_volume_id`.

        This keeps both an index and a graph edge to make placements queryable.
        """
        # Ensure both volumes exist in the graph
        if not self.graph.has_node(replica_volume_id) or not self.graph.has_node(primary_volume_id):
            raise ValueError("Both primary and replica volumes must already be added to the topology.")

        # Store mapping
        self._replica_of[replica_volume_id] = primary_volume_id
        self._replicas.setdefault(primary_volume_id, [])
        if replica_volume_id not in self._replicas[primary_volume_id]:
            self._replicas[primary_volume_id].append(replica_volume_id)

        # Add a lightweight graph edge for visualization/inspection
        if not self.graph.has_edge(replica_volume_id, primary_volume_id):
            self.graph.add_edge(replica_volume_id, primary_volume_id, relation="replica_of")

    def get_primary(self, volume_id: str) -> Optional[str]:
        """Return the primary volume id for a replica, or None if the volume is primary or unknown."""
        return self._replica_of.get(volume_id)

    def get_replicas(self, primary_volume_id: str) -> List[str]:
        """Return replicas for a given primary (may be empty)."""
        return list(self._replicas.get(primary_volume_id, []))

    def get_replica_group(self, volume_id: str) -> List[str]:
        """Return all volumes in the replica group for `volume_id` (primary + replicas).

        If the volume has a primary, the group is [primary] + replicas. If the volume is a primary,
        the group is [primary] + replicas. If unknown, returns [volume_id].
        """
        primary = self.get_primary(volume_id)
        if primary:
            group = [primary] + self.get_replicas(primary)
            return group
        # If this volume is recorded as a primary
        if volume_id in self._replicas and self._replicas.get(volume_id):
            return [volume_id] + self.get_replicas(volume_id)
        return [volume_id]

    def validate_migration(self, volume_id: str, target_node: str) -> bool:
        """Validate whether migrating `volume_id` to `target_node` would violate replica placement.

        Current policy: anti-affinity at node granularity — no two replicas from the same
        replica group (primary + replicas) may reside on the same storage node.
        Returns True if migration is allowed, False otherwise.
        """
        if not self.graph.has_node(volume_id):
            return False

        # Build the replica group for the volume
        group = self.get_replica_group(volume_id)

        # For each volume in the group, check its current node. If any already resides on target_node,
        # the migration would colocate replicas and must be disallowed.
        for v in group:
            node = self._volume_to_node.get(v)
            if node == target_node:
                return False
        return True

    # ────────────────────────────────────────────────────────────────────
    # Visualisation (Plotly)
    # ────────────────────────────────────────────────────────────────────
    def visualize(self, save_path: Optional[str] = None):
        """Render a simple bipartite Plotly graph. Returns the figure."""
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise RuntimeError("plotly is required for visualize()") from exc

        if self.graph.number_of_nodes() == 0:
            raise RuntimeError("Topology graph is empty.")

        pos = nx.spring_layout(self.graph, seed=42, k=0.6)

        # Build edge coords with extend instead of per-item append.
        edge_x: list = []
        edge_y: list = []
        for u, v in self.graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x:  list = []
        node_y:  list = []
        text:    list = []
        color:   list = []
        size:    list = []
        for n, attr in self.graph.nodes(data=True):
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            if attr.get("kind") == self.KIND_NODE:
                util = self.get_node_utilization(n)
                text.append(
                    f"{n}<br>kind=node<br>volumes={util['n_volumes']}"
                    f"<br>total_iops={util['total_iops']:.0f}"
                )
                color.append("#FF7043")
                size.append(22)
            else:
                text.append(
                    f"{n}<br>kind=volume<br>node={attr.get('node_id')}"
                    f"<br>tier={attr.get('tier')}"
                )
                color.append("#42A5F5")
                size.append(10)

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=0.6, color="#888"), hoverinfo="none",
                ),
                go.Scatter(
                    x=node_x, y=node_y, mode="markers",
                    marker=dict(size=size, color=color, line=dict(width=1, color="#222")),
                    text=text, hoverinfo="text",
                ),
            ],
            layout=go.Layout(
                title="Storage Topology & Capacity Graph",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
        return fig

    # ────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"TopologyGraph(nodes={len(self._node_volumes)}, "
            f"volumes={len(self._volume_to_node)}, "
            f"edges={self.graph.number_of_edges()})"
        )