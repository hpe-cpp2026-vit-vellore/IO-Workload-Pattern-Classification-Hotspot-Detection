"""
src/control_plane/monitor.py

Execution Monitor & Rollback (HPE Blueprint Phase 5.5)
======================================================
Monitors post-action metrics, triggers rollbacks if latency increases by >20%,
and tracks rollback rate to ensure it remains below the 1.0% target.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
from src.control_plane.rebalancer import Rebalancer
from src.pipeline.topology_graph import TopologyGraph

logger = logging.getLogger(__name__)

class MovePhase(str, Enum):
    QUEUED      = "QUEUED"
    PRE_CHECK   = "PRE_CHECK"
    EXECUTING   = "EXECUTING"
    VERIFYING   = "VERIFYING"
    COMPLETED   = "COMPLETED"
    FAILED      = "FAILED"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"

@dataclass
class MoveRecord:
    op_id: str
    volume_id: str
    source_node: str
    target_node: str
    size_gb: float
    phase: str = MovePhase.QUEUED.value
    started_at: float = field(default_factory=time.time)
    last_progress_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    bytes_moved_gb: float = 0.0
    error: Optional[str] = None

    @property
    def progress_pct(self) -> float:
        if self.size_gb <= 0:
            return 100.0
        return min(100.0, (self.bytes_moved_gb / self.size_gb) * 100.0)

class ActionMonitor:
    """Tracks executed actions, evaluates post-action latency, handles rollbacks, and monitors active progress."""

    def __init__(
        self,
        rollback_threshold_pct: float = 20.0,
        rollback_timeout_minutes: float = 5.0,
        stall_timeout_seconds: float = 300.0,
        max_rollback_rate_pct: float = 1.0
    ) -> None:
        self.rollback_threshold_pct = rollback_threshold_pct
        self.rollback_timeout_minutes = rollback_timeout_minutes
        self.stall_timeout_seconds = stall_timeout_seconds
        self.max_rollback_rate_pct = max_rollback_rate_pct
        
        self.actions: Dict[str, Dict[str, Any]] = {}
        self.total_actions = 0
        self.rolled_back_count = 0

        # Execution watchdog additions
        self.active_moves: Dict[str, MoveRecord] = {}
        self.move_history: List[MoveRecord] = []
        self._lock = threading.Lock()
        
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def register_action(
        self,
        action_id: str,
        action_state: Dict[str, Any],
        pre_latency: float,
        timestamp: pd.Timestamp
    ) -> None:
        """Register a newly executed rebalance action for monitoring."""
        with self._lock:
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
        with self._lock:
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
            self.rolled_back_count += 1
            action["status"] = "rolled_back"
            
            # Revert action using rebalancer
            rebalancer.rollback_action(action["action_state"], topology)
            return "rolled_back"

        # Evaluate success condition
        if elapsed_minutes >= self.rollback_timeout_minutes:
            logger.info("Action %s completed monitoring successfully.", action_id)
            action["status"] = "success"
            return "success"

        return "monitoring"

    # --- Execution watchdog additions ---

    def register_move(self, op_id: str, volume_id: str, source_node: str, target_node: str, size_gb: float) -> MoveRecord:
        """Register an active move operation under the safety watchdog."""
        with self._lock:
            record = MoveRecord(
                op_id=op_id,
                volume_id=volume_id,
                source_node=source_node,
                target_node=target_node,
                size_gb=size_gb,
                phase=MovePhase.QUEUED.value
            )
            self.active_moves[op_id] = record
        logger.info("Registered active move watchdog: op_id=%s vol=%s", op_id, volume_id)
        return record

    def update_phase(self, op_id: str, phase: str, bytes_moved_gb: Optional[float] = None) -> None:
        """Called by actuators to report phase transitions and progress."""
        with self._lock:
            record = self.active_moves.get(op_id)
            if not record:
                # Find in history if already finalized
                for r in self.move_history:
                    if r.op_id == op_id:
                        r.phase = phase
                        return
                logger.warning("update_phase: unknown op_id %s", op_id)
                return

            record.phase = phase
            record.last_progress_at = time.time()
            if bytes_moved_gb is not None:
                record.bytes_moved_gb = bytes_moved_gb

            if phase in (MovePhase.COMPLETED.value, MovePhase.ROLLED_BACK.value, MovePhase.FAILED.value):
                record.completed_at = time.time()
                self.move_history.append(record)
                self.active_moves.pop(op_id, None)

        logger.debug("op_id=%s phase=%s progress=%.1f%%", op_id, phase, record.progress_pct)

    def mark_failed(self, op_id: str, error: str) -> None:
        """Mark a move as failed."""
        with self._lock:
            record = self.active_moves.get(op_id)
            if record:
                record.phase = MovePhase.FAILED.value
                record.error = error
                record.completed_at = time.time()
                self.move_history.append(record)
                self.active_moves.pop(op_id, None)
        logger.error("Move %s failed: %s", op_id, error)

    def check_stalls(self, now: float) -> None:
        """Scan active moves for stalled execution and verify rollback safety bounds."""
        stalled_ops = []
        with self._lock:
            for op_id, record in self.active_moves.items():
                if record.phase == MovePhase.EXECUTING.value and (now - record.last_progress_at) > self.stall_timeout_seconds:
                    stalled_ops.append((op_id, f"Stalled: No progress for > {self.stall_timeout_seconds} seconds."))

        for op_id, err in stalled_ops:
            logger.warning("Safety watchdog detected stalled execution for move %s. Forcing fail.", op_id)
            self.mark_failed(op_id, err)

        # Evaluate SLO rollback rate
        rate = self.get_rollback_rate()
        if rate > self.max_rollback_rate_pct:
            logger.warning("SAFETY CRITICAL: Rollback rate %.2f%% exceeds SLO target limit %.2f%%", rate, self.max_rollback_rate_pct)

    def _watchdog_loop(self) -> None:
        """Watchdog thread that scans for stalled moves."""
        while True:
            time.sleep(5)
            self.check_stalls(time.time())

    def get_rollback_rate(self) -> float:
        """Compute the rollback rate percentage."""
        if self.total_actions == 0:
            return 0.0
        return (self.rolled_back_count / self.total_actions) * 100.0

    def check_rollback_rate_exceeded(self, threshold_pct: float) -> bool:
        """Returns True if the current rollback rate exceeds the threshold."""
        return self.get_rollback_rate() > threshold_pct

    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics of all rebalance operations."""
        with self._lock:
            total_active = len(self.active_moves)
            total_history = len(self.move_history)
        return {
            "total_actions": self.total_actions,
            "rolled_back_count": self.rolled_back_count,
            "rollback_rate_pct": round(self.get_rollback_rate(), 2),
            "active_monitors": sum(1 for a in self.actions.values() if a["status"] == "monitoring"),
            "active_moves_count": total_active,
            "completed_moves_count": total_history
        }
