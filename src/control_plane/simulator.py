"""
src/control_plane/simulator.py

What-If Simulation Engine (HPE Blueprint Phase 4.3)
=====================================================
Supports capacity, migration, QoS, and tier change simulation scenarios.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.pipeline.topology_graph import TopologyGraph
from src.models.forecasting.dtf_forecaster import simulate_add_capacity
from src.control_plane.inference_hub import InferenceHub

logger = logging.getLogger(__name__)

class WhatIfSimulator:
    """Simulates performance and capacity changes across storage nodes and volumes."""

    def __init__(self, hub: InferenceHub) -> None:
        self.hub = hub
        self.topology = hub.topology
        self.features_df = hub.features_df

    def simulate_add_capacity_scenario(self, volume_id: str, added_gb: float) -> Dict[str, Any]:
        """Simulate adding storage capacity and compute new Days-to-Fill (DTF)."""
        if volume_id not in self.features_df["volume_id"].unique():
            raise ValueError(f"Invalid volume ID: {volume_id}")

        # Get latest record to retrieve total capacity
        df_vol = self.features_df[self.features_df["volume_id"] == volume_id].sort_values("timestamp")
        last_row = df_vol.iloc[-1]
        
        current_total_gb = float(last_row.get("capacity_total_gb", 1000.0))
        
        # Get history array (daily)
        history = self.hub.get_nbeats_input(volume_id, last_row["timestamp"])
        
        sim_res = simulate_add_capacity(
            model=self.hub.nbeats,
            history=history,
            current_total_gb=current_total_gb,
            added_gb=added_gb,
            device="cpu"
        )

        before_days = float(sim_res.get("dtf_warning_before_days")) if sim_res.get("dtf_warning_before_days") is not None else None
        after_days = float(sim_res.get("dtf_warning_after_days")) if sim_res.get("dtf_warning_after_days") is not None else None
        improvement = (after_days - before_days) if (before_days is not None and after_days is not None) else 0.0

        recommendation = (
            f"Adding {added_gb}GB extends time-to-full by {round(improvement, 1)} days."
            if improvement > 0 else "Capacity increase did not shift the warning threshold date significantly."
        )

        return {
            "volume_id": volume_id,
            "added_gb": added_gb,
            "current_total_gb": current_total_gb,
            "new_total_gb": current_total_gb + added_gb,
            "original_dtf_days": before_days,
            "simulated_dtf_days": after_days,
            "improvement_days": round(improvement, 1),
            "recommendation": recommendation
        }

    def simulate_migration_scenario(self, volume_id: str, target_node: str) -> Dict[str, Any]:
        """Simulate volume migration to a new storage node."""
        if volume_id not in self.features_df["volume_id"].unique():
            raise ValueError(f"Invalid volume ID: {volume_id}")
        if target_node not in self.topology.all_nodes():
            raise ValueError(f"Invalid target node ID: {target_node}")

        source_node = self.topology.get_node_of_volume(volume_id)
        if not source_node:
            raise ValueError(f"Volume {volume_id} is not mapped to any node.")

        if source_node == target_node:
            return {
                "migration_safe": False,
                "reason": "Volume is already residing on the target node."
            }

        # Calculate metrics of the migrating volume
        vol_metrics = self.topology._volume_metrics.get(volume_id, {})
        vol_iops = vol_metrics.get("total_iops", 1000.0)
        vol_latency = vol_metrics.get("avg_latency_us", 1000.0)
        vol_throughput = vol_metrics.get("total_throughput_mbps", 50.0)
        vol_capacity = vol_metrics.get("capacity_used_pct", 50.0)

        # Source node details
        source_util = self.topology.get_node_utilization(source_node)
        target_util = self.topology.get_node_utilization(target_node)

        # Source node improvement: latency decreases by reducing queueing delay
        predicted_source_latency_reduction_pct = 15.0  # Estimated standard queue relief

        # Target node impact
        new_target_iops = target_util["total_iops"] + vol_iops
        new_target_throughput = target_util["total_throughput_mbps"] + vol_throughput

        # Safety Check: check if target node overflows (e.g. > 30000 IOPS)
        max_iops_capacity = 30000.0
        migration_safe = new_target_iops < max_iops_capacity

        # Estimated migration time: assume 50MB/s link rate
        # Let's read capacity_used_gb or estimate from capacity_total_gb
        df_vol = self.features_df[self.features_df["volume_id"] == volume_id].sort_values("timestamp")
        last_row = df_vol.iloc[-1]
        used_gb = float(last_row.get("capacity_used_gb", 500.0))
        
        migration_speed_mbps = 50.0
        estimated_time_seconds = int((used_gb * 1024) / migration_speed_mbps)

        return {
            "volume_id": volume_id,
            "source_node": source_node,
            "target_node": target_node,
            "predicted_source_latency_improvement_pct": predicted_source_latency_reduction_pct,
            "target_utilization_increase_pct": round((vol_iops / max_iops_capacity) * 100.0, 2),
            "migration_safe": migration_safe,
            "estimated_time_seconds": estimated_time_seconds,
            "recommendation": (
                f"Migration of {volume_id} to {target_node} is SAFE. Estimated transfer time: {estimated_time_seconds}s."
                if migration_safe else f"Migration UNSAFE. Target node {target_node} will overflow IOPS capacity."
            )
        }

    def simulate_qos_shaping_scenario(self, volume_id: str, iops_limit: float) -> Dict[str, Any]:
        """Simulate applying an IOPS QoS limit to check relief on neighbors."""
        if volume_id not in self.features_df["volume_id"].unique():
            raise ValueError(f"Invalid volume ID: {volume_id}")

        vol_metrics = self.topology._volume_metrics.get(volume_id, {})
        current_iops = vol_metrics.get("total_iops", 1000.0)

        # Neighbors
        neighbors = self.topology.get_neighbors(volume_id)
        
        # Estimate relief: relief is proportional to the fraction of IOPS throttled
        throttling_pct = max(0.0, (current_iops - iops_limit) / current_iops) if current_iops > 0 else 0.0
        expected_neighbor_relief_pct = round(throttling_pct * 30.0, 2)  # Max 30% latency variance reduction

        return {
            "volume_id": volume_id,
            "iops_limit": iops_limit,
            "current_iops": current_iops,
            "throttling_pct": round(throttling_pct * 100.0, 2),
            "impact_on_self_throughput_reduction_pct": round(throttling_pct * 100.0, 2),
            "expected_noisy_neighbor_relief_pct": expected_neighbor_relief_pct,
            "recommendation": (
                f"Throttling {volume_id} to {iops_limit} IOPS will reduce neighboring latency noise by {expected_neighbor_relief_pct}%."
                if throttling_pct > 0 else "QoS limit is above current IOPS. No active throttling will occur."
            )
        }

    def simulate_tier_change_scenario(self, volume_id: str, new_tier: str) -> Dict[str, Any]:
        """Simulate tier upgrade/downgrade (e.g. SSD, HDD, NVMe)."""
        if volume_id not in self.features_df["volume_id"].unique():
            raise ValueError(f"Invalid volume ID: {volume_id}")

        current_tier = "HDD"
        if self.topology.graph.has_node(volume_id):
            current_tier = self.topology.graph.nodes[volume_id].get("tier", "HDD")

        # Map tier transitions
        # HDD -> SSD: 80% latency drop, 5x throughput capacity
        # HDD -> NVMe: 95% latency drop, 15x throughput capacity
        # SSD -> NVMe: 50% latency drop, 3x throughput capacity
        # Else: 0% change
        lat_imp = 0.0
        tp_gain = 1.0

        if current_tier == "HDD" and new_tier == "SSD":
            lat_imp = 80.0
            tp_gain = 5.0
        elif current_tier == "HDD" and new_tier == "NVMe":
            lat_imp = 95.0
            tp_gain = 15.0
        elif current_tier == "SSD" and new_tier == "NVMe":
            lat_imp = 50.0
            tp_gain = 3.0
        elif current_tier == new_tier:
            lat_imp = 0.0
            tp_gain = 1.0
        else:
            # Downgrades
            lat_imp = -100.0  # Latency doubles
            tp_gain = 0.2

        return {
            "volume_id": volume_id,
            "current_tier": current_tier,
            "new_tier": new_tier,
            "expected_latency_improvement_pct": lat_imp,
            "expected_throughput_gain_factor": tp_gain,
            "recommendation": (
                f"Upgrading {volume_id} to {new_tier} yields a predicted {lat_imp}% latency reduction and {tp_gain}x throughput bandwidth."
                if lat_imp > 0 else f"Changing {volume_id} to {new_tier} is a downgrade or equivalent tier."
            )
        }
