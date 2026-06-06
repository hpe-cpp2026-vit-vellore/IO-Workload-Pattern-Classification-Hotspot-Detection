"""
src/control_plane/capacity_planner.py

Cluster-wide capacity and performance planner.
Generates prioritized recommendations for storage expansion, re-tiering,
QoS shaping, and job rescheduling by evaluating capacity and tail-latency risks.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class RecommendationType(str, Enum):
    ADD_NODES           = "ADD_NODES"
    EXPAND_POOL         = "EXPAND_POOL"
    TIER_COLD_DATA      = "TIER_COLD_DATA"
    QOS_SHAPE           = "QOS_SHAPE"
    RESCHEDULE_JOBS     = "RESCHEDULE_JOBS"
    NO_ACTION           = "NO_ACTION"

class Urgency(str, Enum):
    INFO     = "INFO"       # Low urgency
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"   # High urgency

@dataclass
class Recommendation:
    rec_id: str
    rec_type: str
    urgency: str
    title: str
    description: str
    estimated_headroom_gained_days: float
    estimated_cost_impact: str          # "low" | "medium" | "high"
    auto_actionable: bool               # can be done without human approval
    payload: dict = field(default_factory=dict)

@dataclass
class CapacityPlan:
    plan_id: str
    timestamp_utc: str
    recommendations: List[Recommendation]
    overall_urgency: str
    summary: str
    auto_scale_triggered: bool = False

class CapacityPlanner:
    """Analyzes topology and capacity trends to generate a CapacityPlan."""

    def __init__(
        self,
        warning_util_pct: float = 80.0,
        critical_util_pct: float = 90.0,
        warning_dtf_days: float = 30.0,
        critical_dtf_days: float = 7.0,
        auto_scale_enabled: bool = False,
    ) -> None:
        self.warning_util = warning_util_pct
        self.critical_util = critical_util_pct
        self.warning_dtf = warning_dtf_days
        self.critical_dtf = critical_dtf_days
        self.auto_scale_enabled = auto_scale_enabled
        self._rec_counter = 0

    def _next_id(self) -> str:
        self._rec_counter += 1
        return f"rec-{self._rec_counter:04d}"

    def plan(
        self,
        topology,
        cached_analysis: Dict[str, Dict[str, Any]]
    ) -> CapacityPlan:
        """Evaluate topology nodes and volume analysis caches to build a CapacityPlan."""
        recs: List[Recommendation] = []
        critical_node_count = 0

        # Group volumes by node to check node capacity state
        nodes = topology.all_storage_nodes() if hasattr(topology, "all_storage_nodes") else topology.all_nodes()
        
        for node_id in nodes:
            node_vols = topology.get_volumes_on_node(node_id) if hasattr(topology, "get_volumes_on_node") else []
            node_util = topology.get_node_utilization(node_id) if hasattr(topology, "get_node_utilization") else {"used_capacity_gb": 0.0, "total_capacity_gb": 1.0, "util_pct": 0.0}
            
            util_pct = node_util.get("util_pct", 0.0)
            
            # Check volumes on this node for capacity / dtf issues
            min_dtf = None
            workload_mix: Dict[str, float] = {}
            
            for vol_id in node_vols:
                analysis = cached_analysis.get(vol_id, {})
                if not analysis:
                    continue
                
                # DTF check
                dtf_dict = analysis.get("days_to_fill", {})
                crit_days = dtf_dict.get("critical_95pct_days")
                if crit_days is not None:
                    if min_dtf is None or crit_days < min_dtf:
                        min_dtf = crit_days
                
                # Workload mix tracking
                wl = analysis.get("workload_type", "Unknown")
                workload_mix[wl] = workload_mix.get(wl, 0.0) + 1.0
                
                # Latency/SLO risk checks
                ttv_dict = analysis.get("latency_ttv", {})
                risk = ttv_dict.get("risk_level", "none")
                if risk in ("critical", "high"):
                    recs.append(Recommendation(
                        rec_id=self._next_id(),
                        rec_type=RecommendationType.QOS_SHAPE.value,
                        urgency=Urgency.HIGH.value if risk == "high" else Urgency.CRITICAL.value,
                        title=f"Node {node_id}: latency risk on volume {vol_id}",
                        description=f"Tail latency risk is {risk.upper()}. Recommend applying QoS shaping or throttling noisy neighbors.",
                        estimated_headroom_gained_days=0.0,
                        estimated_cost_impact="low",
                        auto_actionable=True,
                        payload={"volume_id": vol_id, "node_id": node_id, "action": "qos"}
                    ))
                    
                    # If it's a batch workload noisy neighbor, also recommend rescheduling
                    if wl in ("Backup", "AI_Training"):
                        recs.append(Recommendation(
                            rec_id=self._next_id(),
                            rec_type=RecommendationType.RESCHEDULE_JOBS.value,
                            urgency=Urgency.HIGH.value,
                            title=f"Node {node_id}: reschedule noisy batch job on {vol_id}",
                            description=f"Volume {vol_id} running batch workload {wl} is causing contention. Reschedule to off-peak hours.",
                            estimated_headroom_gained_days=0.0,
                            estimated_cost_impact="low",
                            auto_actionable=self.auto_scale_enabled,
                            payload={"volume_id": vol_id, "node_id": node_id, "workload_type": wl}
                        ))

            # Normalize workload mix
            total_vols = len(node_vols)
            if total_vols > 0:
                workload_mix = {k: v / total_vols for k, v in workload_mix.items()}

            # Evaluate Node Capacity & DTF warnings
            if min_dtf is not None:
                if min_dtf < self.critical_dtf:
                    urgency = Urgency.CRITICAL.value
                    critical_node_count += 1
                elif min_dtf < self.warning_dtf:
                    urgency = Urgency.HIGH.value if min_dtf < 14.0 else Urgency.MEDIUM.value
                else:
                    urgency = Urgency.LOW.value

                if min_dtf < self.warning_dtf:
                    # Check if Backup data can be re-tiered to object storage
                    backup_fraction = workload_mix.get("Backup", 0.0)
                    if backup_fraction > 0.15:
                        recs.append(Recommendation(
                            rec_id=self._next_id(),
                            rec_type=RecommendationType.TIER_COLD_DATA.value,
                            urgency=urgency,
                            title=f"Node {node_id}: re-tier Backup data to colder storage",
                            description=f"Node has {backup_fraction:.0%} backup workload footprint. Re-tiering would extend headroom.",
                            estimated_headroom_gained_days=min_dtf * (1.0 + backup_fraction),
                            estimated_cost_impact="low",
                            auto_actionable=True,
                            payload={"node_id": node_id, "workload_type": "Backup"}
                        ))
                    else:
                        recs.append(Recommendation(
                            rec_id=self._next_id(),
                            rec_type=RecommendationType.EXPAND_POOL.value,
                            urgency=urgency,
                            title=f"Node {node_id}: expand capacity/storage pool",
                            description=f"Node util is {util_pct:.1f}% with estimated {min_dtf:.1f} days to fill. Plan pool expansion.",
                            estimated_headroom_gained_days=min_dtf * 2,
                            estimated_cost_impact="medium",
                            auto_actionable=False,
                            payload={"node_id": node_id}
                        ))

        # Cluster level: if multiple nodes are critical, recommend adding nodes
        if critical_node_count >= 2:
            recs.append(Recommendation(
                rec_id=self._next_id(),
                rec_type=RecommendationType.ADD_NODES.value,
                urgency=Urgency.CRITICAL.value,
                title="Add new storage nodes to cluster",
                description=f"{critical_node_count} nodes are at critical days-to-fill capacity. Scale out the cluster.",
                estimated_headroom_gained_days=90.0,
                estimated_cost_impact="high",
                auto_actionable=self.auto_scale_enabled,
                payload={"nodes_needed": 1}
            ))

        # Sort recommendations by urgency level
        urgency_order = {
            Urgency.CRITICAL.value: 0,
            Urgency.HIGH.value: 1,
            Urgency.MEDIUM.value: 2,
            Urgency.LOW.value: 3,
            Urgency.INFO.value: 4,
        }
        recs.sort(key=lambda r: urgency_order.get(r.urgency, 9))

        overall_urgency = recs[0].urgency if recs else Urgency.INFO.value
        
        # Determine summary
        if overall_urgency == Urgency.CRITICAL.value:
            summary = "CRITICAL: Immediate actions required to expand capacity or mitigate tail latency risks."
        elif overall_urgency in (Urgency.HIGH.value, Urgency.MEDIUM.value):
            summary = "WARNING: Capacity planning warnings active. Review migration and re-tiering opportunities."
        else:
            summary = "Cluster healthy. No immediate rebalancing or capacity scaling required."

        auto_scale_triggered = False
        if self.auto_scale_enabled and overall_urgency == Urgency.CRITICAL.value:
            auto_scale_triggered = True

        return CapacityPlan(
            plan_id=str(uuid.uuid4()),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            recommendations=recs,
            overall_urgency=overall_urgency,
            summary=summary,
            auto_scale_triggered=auto_scale_triggered
        )
