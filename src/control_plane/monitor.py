"""
src/control_plane/monitor.py

Execution Monitor & Rollback (HPE Blueprint Phase 5.5)
======================================================
Monitors post-action metrics, triggers rollbacks if latency increases by >20%,
and tracks rollback rate to ensure it remains below the 1.0% target.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from src.control_plane.rebalancer import Rebalancer
from src.pipeline.topology_graph import TopologyGraph

logger = logging.getLogger(__name__)

class ActionMonitor:
    """Tracks executed actions, evaluates post-action latency, and handles rollbacks."""

    def __init__(self, rollback_threshold_pct: float = 20.0, rollback_timeout_minutes: float = 5.0) -> None:
        self.rollback_threshold_pct = rollback_threshold_pct
        self.rollback_timeout_minutes = rollback_timeout_minutes
        
        self.actions: Dict[str, Dict[str, Any]] = {}
        self.total_actions = 0
        self.rolled_back_count = 0

    def register_action(
        self,
        action_id: str,
        action_state: Dict[str, Any],
        pre_latency: float,
        timestamp: pd.Timestamp
    ) -> None:
        """Register a newly executed rebalance action for monitoring."""
        self.actions[action_id] = {
            "action_id": action_id,
            "action_state": action_state,
            "pre_latency": max(1.0, pre_latency),  # avoid division by zero
            "timestamp": timestamp,
            "status": "monitoring",  # "monitoring", "success", "rolled_back"
            "elapsed_minutes": 0.0,
            "current_latency": pre_latency
        }
        self.total_actions += 1
        logger.info("Registered action %s for monitoring. Pre-latency: %.2f us", action_id, pre_latency)

    def register_event(
        self,
        action_id: str,
        action_state: Dict[str, Any],
        timestamp: pd.Timestamp,
        status: str = "success"
    ) -> None:
        """Register a non-latency action (e.g., autoscale) as an immediate event."""
        self.actions[action_id] = {
            "action_id": action_id,
            "action_state": action_state,
            "pre_latency": 0.0,
            "timestamp": timestamp,
            "status": status,
            "elapsed_minutes": 0.0,
            "current_latency": 0.0
        }
        self.total_actions += 1
        logger.info("Registered event %s with status %s.", action_id, status)

    def update_metrics(
        self,
        action_id: str,
        current_latency: float,
        elapsed_minutes: float,
        rebalancer: Rebalancer,
        topology: TopologyGraph
    ) -> str:
        """
        Update the current metrics for an action.
        Triggers rollback if latency worsens by > threshold.
        """
        action = self.actions.get(action_id)
        if not action or action["status"] != "monitoring":
            return action["status"] if action else "unknown"

        action["current_latency"] = current_latency
        action["elapsed_minutes"] = elapsed_minutes

        pre_latency = action["pre_latency"]
        latency_increase_pct = ((current_latency - pre_latency) / pre_latency) * 100.0

        logger.info(
            "Action %s monitoring: current_latency=%.2f us, increase=%.2f%%, elapsed=%.2f min",
            action_id, current_latency, latency_increase_pct, elapsed_minutes
        )

        # Evaluate rollback condition
        if latency_increase_pct > self.rollback_threshold_pct:
            logger.warning(
                "Rollback triggered for action %s: latency increased by %.2f%% (threshold %.2f%%)",
                action_id, latency_increase_pct, self.rollback_threshold_pct
            )
            action["status"] = "rolled_back"
            self.rolled_back_count += 1
            
            # Revert action using rebalancer
            rebalancer.rollback_action(action["action_state"], topology)
            return "rolled_back"

        # Evaluate success condition
        if elapsed_minutes >= self.rollback_timeout_minutes:
            logger.info("Action %s completed monitoring successfully.", action_id)
            action["status"] = "success"
            return "success"

        return "monitoring"

    def get_rollback_rate(self) -> float:
        """Compute the rollback rate percentage."""
        if self.total_actions == 0:
            return 0.0
        return (self.rolled_back_count / self.total_actions) * 100.0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics of all rebalance operations."""
        return {
            "total_actions": self.total_actions,
            "rolled_back_count": self.rolled_back_count,
            "rollback_rate_pct": round(self.get_rollback_rate(), 2),
            "active_monitors": sum(1 for a in self.actions.values() if a["status"] == "monitoring")
        }
