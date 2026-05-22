"""
src/control_plane/decision_engine.py

Decision & Policy Engine (HPE Blueprint Phase 5.3)
===================================================
Evaluates model inference results, checks hotspot persistence, respects rate limits,
simulates action alternatives, and executes the optimal rebalance action.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from src.control_plane.inference_hub import InferenceHub
from src.control_plane.rebalancer import Rebalancer
from src.control_plane.monitor import ActionMonitor

logger = logging.getLogger(__name__)

class DecisionEngine:
    """Evaluates telemetry policies and decides corrective control-plane actions."""

    def __init__(
        self,
        inference_hub: InferenceHub,
        rebalancer: Rebalancer,
        monitor: ActionMonitor
    ) -> None:
        self.hub = inference_hub
        self.rebalancer = rebalancer
        self.monitor = monitor

        # Read config thresholds
        self.policy = self.hub.policy
        self.rebalance_policy = self.policy.get("rebalance_policy", {})
        self.qos_policy = self.policy.get("qos_policy", {})
        
        self.enabled = self.rebalance_policy.get("enabled", True)
        self.dry_run_mode = self.rebalance_policy.get("dry_run_mode", False)
        self.min_hotspot_score = self.rebalance_policy.get("min_hotspot_score_to_trigger", 75.0)
        self.min_hotspot_duration = self.rebalance_policy.get("min_hotspot_duration_minutes", 2.0)
        self.max_moves_per_hour = self.rebalance_policy.get("max_volumes_moved_per_hour", 3)
        self.max_concurrent_migrations = self.rebalance_policy.get("max_concurrent_migrations", 1)

        # State tracking
        self.hotspot_start_times: Dict[str, pd.Timestamp] = {}
        self.action_history: List[Dict[str, Any]] = []
        self.action_queue: List[Dict[str, Any]] = []
        # Autoscale guard
        self.last_autoscale_time: Optional[pd.Timestamp] = None

    def simulate_actions(self, volume_id: str, inference_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate migrate, QoS, and tier-change to calculate expected improvement."""
        topology = self.hub.topology
        source_node = topology.get_node_of_volume(volume_id)
        
        # 1. Migrate option
        target_node = topology.get_best_target_node(exclude_node=source_node) if source_node else None
        if target_node and source_node:
            source_util = topology.get_node_utilization(source_node)
            target_util = topology.get_node_utilization(target_node)
            # Improvement is proportional to the IOPS imbalance reduction
            migrate_improvement = max(0.0, source_util["total_iops"] - target_util["total_iops"])
            # Validate placement against replica/anti-affinity constraints
            try:
                migrate_safe = topology.validate_migration(volume_id, target_node)
            except Exception:
                migrate_safe = False
        else:
            migrate_improvement = 0.0
            migrate_safe = False

        # 2. QoS option
        # Capping IOPS reduces aggressor load and frees bandwidth/IOPS for victims
        # Improvement = sum of impact scores on co-located victim volumes
        qos_improvement = float(sum(inference_results.get("noisy_neighbor_victims", {}).values()))
        
        # 3. Tier change option
        # Moving from tier-2 to tier-1 expected improvement is set to a constant booster if current is slow tier
        current_tier = topology.graph.nodes[volume_id].get("tier") if topology.graph.has_node(volume_id) else "tier-2"
        if current_tier == "tier-2":
            tier_improvement = 50.0
        else:
            tier_improvement = 0.0

        return [
            {
                "action": "migrate",
                "target_node": target_node,
                "expected_improvement": migrate_improvement,
                "safe": migrate_safe
            },
            {
                "action": "qos",
                "iops_limit": self.qos_policy.get("backup_iops_cap", 3000.0),
                "expected_improvement": qos_improvement,
                "safe": True
            },
            {
                "action": "tier_change",
                "new_tier": "tier-1",
                "expected_improvement": tier_improvement,
                "safe": True
            }
        ]

    def evaluate_volume(self, volume_id: str, timestamp: pd.Timestamp, inference_results: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Evaluate a single volume's metrics and decide if any action is triggered."""
        if not self.enabled:
            logger.info("Decision engine disabled.")
            return None

        # 1. Perform model inference
        if inference_results is None:
            res = self.hub.analyze_volume(volume_id, timestamp)
        else:
            res = inference_results
        hotspot_score = res["hotspot_score"]
        
        # Get current pre-action latency for registration
        topology = self.hub.topology
        metrics_dict = topology._volume_metrics.get(volume_id, {})
        pre_latency = metrics_dict.get("avg_latency_us", 1000.0)

        # 2. Evaluate hotspot threshold
        if hotspot_score >= self.min_hotspot_score:
            if volume_id not in self.hotspot_start_times:
                self.hotspot_start_times[volume_id] = timestamp
                logger.info(
                    "Hotspot detected on volume %s (score %.2f >= %.2f). Persistence tracking started.",
                    volume_id, hotspot_score, self.min_hotspot_score
                )
            
            elapsed_min = (timestamp - self.hotspot_start_times[volume_id]).total_seconds() / 60.0
            
            if elapsed_min < self.min_hotspot_duration:
                logger.info(
                    "Hotspot on volume %s persisting for %.2f min (needs %.2f min). Waiting.",
                    volume_id, elapsed_min, self.min_hotspot_duration
                )
                return {"status": "waiting_persistence", "elapsed_minutes": elapsed_min}
            
            # Hotspot is persistent! Decide action.
            logger.warning(
                "Persistent hotspot verified on volume %s (duration %.2f min >= %.2f min). Evaluating actions.",
                volume_id, elapsed_min, self.min_hotspot_duration
            )
        else:
            # Hotspot cleared or was not present
            if volume_id in self.hotspot_start_times:
                logger.info("Hotspot cleared on volume %s.", volume_id)
                del self.hotspot_start_times[volume_id]
            return None

        # 3. Simulate and pick the best action
        simulations = self.simulate_actions(volume_id, res)
        # Filter safe actions
        safe_actions = [s for s in simulations if s["safe"]]
        if not safe_actions:
            logger.warning("No safe rebalancing actions simulated for volume %s.", volume_id)
            return {"status": "no_safe_actions"}

        # Pick action with highest expected improvement
        best_choice = max(safe_actions, key=lambda s: s["expected_improvement"])
        
        # If improvement is zero, skip action
        if best_choice["expected_improvement"] <= 0.0:
            logger.info("No action provides a positive expected improvement. Skipping.")
            return {"status": "no_improvement"}

        action_type = best_choice["action"]

        # 4. Check rate limits and concurrency for migrations
        if action_type == "migrate":
            # Concurrent migrations limit check
            active_migrations = sum(
                1 for act in self.monitor.actions.values()
                if act["status"] == "monitoring" and act["action_state"]["action"] == "migrate"
            )
            if active_migrations >= self.max_concurrent_migrations:
                logger.warning(
                    "Migration limit reached. Active: %d (max: %d). Queuing volume %s migration.",
                    active_migrations, self.max_concurrent_migrations, volume_id
                )
                self.action_queue.append({
                    "volume_id": volume_id,
                    "action_choice": best_choice,
                    "timestamp": timestamp
                })
                return {"status": "queued", "reason": "concurrent_limit"}

            # Rate limit (per hour) check
            one_hour_ago = timestamp - pd.Timedelta(hours=1)
            recent_moves = sum(
                1 for act in self.action_history
                if act["timestamp"] >= one_hour_ago and act["action"] == "migrate"
            )
            if recent_moves >= self.max_moves_per_hour:
                logger.warning(
                    "Hourly migration rate limit reached. Recent moves: %d (max: %d). Queuing volume %s migration.",
                    recent_moves, self.max_moves_per_hour, volume_id
                )
                self.action_queue.append({
                    "volume_id": volume_id,
                    "action_choice": best_choice,
                    "timestamp": timestamp
                })
                return {"status": "queued", "reason": "rate_limit"}

        # 5. Execute Action (or dry run)
        action_id = str(uuid.uuid4())
        
        if self.dry_run_mode:
            logger.info("[DRY RUN] Would execute %s action on volume %s", action_type, volume_id)
            return {
                "status": "dry_run",
                "action_id": action_id,
                "action": action_type,
                "choice": best_choice
            }

        # Execute live change
        action_state = {}
        if action_type == "migrate":
            action_state = self.rebalancer.execute_migration(volume_id, best_choice["target_node"], self.hub.topology)
        elif action_type == "qos":
            action_state = self.rebalancer.execute_qos_shaping(volume_id, best_choice["iops_limit"], self.hub.topology)
        elif action_type == "tier_change":
            action_state = self.rebalancer.execute_tier_change(volume_id, best_choice["new_tier"], self.hub.topology)

        # Log to action history
        exec_record = {
            "action_id": action_id,
            "volume_id": volume_id,
            "action": action_type,
            "choice": best_choice,
            "timestamp": timestamp,
            "action_state": action_state,
            "status": "executed"
        }
        self.action_history.append(exec_record)

        # Register with ActionMonitor
        self.monitor.register_action(
            action_id=action_id,
            action_state=action_state,
            pre_latency=pre_latency,
            timestamp=timestamp
        )

        return exec_record

    def process_queued_actions(self, timestamp: pd.Timestamp) -> Dict[str, List[Dict[str, Any]]]:
        """Try to execute deferred migration actions in the queue.

        Returns a dict with executed migration actions and autoscale events.
        """
        # Check autoscale triggers first (may add nodes/pools to topology)
        autoscale_actions: List[Dict[str, Any]] = []
        try:
            autoscale_actions = self._check_and_trigger_autoscale(timestamp)
        except Exception:
            logger.exception("Autoscale trigger failed")

        executed: List[Dict[str, Any]] = []
        if not self.action_queue:
            return {"migrations": executed, "autoscale": autoscale_actions}

        # Filter active migrations
        active_migrations = sum(
            1 for act in self.monitor.actions.values()
            if act["status"] == "monitoring" and act["action_state"]["action"] == "migrate"
        )
        
        one_hour_ago = timestamp - pd.Timedelta(hours=1)
        recent_moves = sum(
            1 for act in self.action_history
            if act["timestamp"] >= one_hour_ago and act["action"] == "migrate"
        )

        still_queued = []
        for task in self.action_queue:
            vol_id = task["volume_id"]
            best_choice = task["action_choice"]
            
            # Check if limits permit now
            if active_migrations < self.max_concurrent_migrations and recent_moves < self.max_moves_per_hour:
                logger.info("Executing queued migration for volume %s", vol_id)
                
                # Fetch pre-latency
                metrics_dict = self.hub.topology._volume_metrics.get(vol_id, {})
                pre_latency = metrics_dict.get("avg_latency_us", 1000.0)
                
                action_id = str(uuid.uuid4())
                action_state = self.rebalancer.execute_migration(vol_id, best_choice["target_node"], self.hub.topology)
                
                exec_record = {
                    "action_id": action_id,
                    "volume_id": vol_id,
                    "action": "migrate",
                    "choice": best_choice,
                    "timestamp": timestamp,
                    "action_state": action_state,
                    "status": "executed"
                }
                self.action_history.append(exec_record)
                self.monitor.register_action(
                    action_id=action_id,
                    action_state=action_state,
                    pre_latency=pre_latency,
                    timestamp=timestamp
                )
                executed.append(exec_record)
                
                # Update counters
                active_migrations += 1
                recent_moves += 1
            else:
                still_queued.append(task)

        self.action_queue = still_queued
        return {"migrations": executed, "autoscale": autoscale_actions}

    def _check_and_trigger_autoscale(self, timestamp: pd.Timestamp) -> List[Dict[str, Any]]:
        """Scan forecasts (DTF) and cluster thresholds to trigger autoscaling actions.

        Simple heuristic implemented here:
          - If any volume forecasts indicate `warning_85pct_days` <= policy threshold,
            create one new virtual node and tag it for the volume's pool.
        This is intentionally conservative for a prototype.
        """
        autos = self.rebalance_policy.get("autoscale", {})
        if not autos.get("enabled", True):
            return []

        min_interval_hours = autos.get("min_interval_hours", 24)
        if self.last_autoscale_time is not None:
            if (timestamp - self.last_autoscale_time).total_seconds() < min_interval_hours * 3600:
                logger.debug("Autoscale skipped: min interval not passed.")
                return []

        max_new = autos.get("max_new_nodes_per_run", 1)
        created: List[Dict[str, Any]] = []

        def _create_autoscale_node(pool_id: str, reason: str, source_volume: Optional[str]) -> Optional[Dict[str, Any]]:
            new_node_id = f"autoscale-{uuid.uuid4().hex[:8]}"
            try:
                add_res = self.rebalancer.add_virtual_node(
                    new_node_id,
                    capacity_gb=autos.get("new_node_capacity_gb"),
                    tier=None,
                    topology=self.hub.topology
                )
                expand_res = self.rebalancer.expand_logical_pool(pool_id, new_node_id, self.hub.topology)
            except Exception:
                logger.exception("Failed to perform autoscale actions for volume %s", source_volume)
                return None

            action_id = str(uuid.uuid4())
            record = {
                "action_id": action_id,
                "action": "autoscale_add_node",
                "node_id": new_node_id,
                "pool_id": pool_id,
                "reason": reason,
                "volume_id": source_volume,
                "timestamp": timestamp,
                "status": "executed",
                "add_res": add_res,
                "expand_res": expand_res
            }
            self.action_history.append(record)
            if self.monitor is not None:
                self.monitor.register_event(
                    action_id=action_id,
                    action_state={
                        "action": "autoscale_add_node",
                        "node_id": new_node_id,
                        "pool_id": pool_id,
                        "reason": reason,
                        "volume_id": source_volume
                    },
                    timestamp=timestamp,
                    status="success"
                )
            created.append(record)
            return record

        # Cluster-level threshold trigger
        cluster_threshold = autos.get("cluster_capacity_threshold_pct")
        cluster_pool = None
        cluster_volume = None
        if cluster_threshold is not None:
            samples = []
            for vol_id, metrics in self.hub.topology._volume_metrics.items():
                cap = metrics.get("capacity_used_pct")
                if cap is None:
                    continue
                try:
                    cap_val = float(cap)
                except (ValueError, TypeError):
                    continue
                if cap_val <= 1.0:
                    cap_val *= 100.0
                samples.append((vol_id, cap_val))

            if samples:
                avg_capacity = sum(v for _, v in samples) / len(samples)
                if avg_capacity >= float(cluster_threshold):
                    cluster_volume = max(samples, key=lambda x: x[1])[0]
                    cluster_pool = self.hub.topology.graph.nodes.get(cluster_volume, {}).get("pool_id") or "default_pool"
                    if len(created) < max_new:
                        _create_autoscale_node(cluster_pool, "cluster_capacity", cluster_volume)

        # Per-volume DTF trigger
        for vol in self.hub.topology.all_volumes():
            if len(created) >= max_new:
                break
            try:
                analysis = self.hub.analyze_volume(vol, timestamp)
            except Exception:
                continue
            dtf = analysis.get("days_to_fill", {}).get("warning_85pct_days")
            if dtf is None:
                continue
            if dtf <= autos.get("warning_days", 7):
                pool = self.hub.topology.graph.nodes.get(vol, {}).get("pool_id") or "default_pool"
                _create_autoscale_node(pool, "dtf_warning", vol)

        if created:
            self.last_autoscale_time = timestamp
        return created
