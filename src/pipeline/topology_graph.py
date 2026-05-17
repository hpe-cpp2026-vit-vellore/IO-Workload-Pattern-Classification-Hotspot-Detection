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