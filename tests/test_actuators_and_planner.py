import unittest
import time
import pandas as pd
from src.pipeline.topology_graph import TopologyGraph
from src.control_plane.rebalancer import Rebalancer
from src.control_plane.monitor import ActionMonitor, MovePhase
from src.control_plane.actuators import get_actuator, StubActuator
from src.control_plane.capacity_planner import CapacityPlanner, RecommendationType, Urgency

class TestActuatorsAndPlanner(unittest.TestCase):
    def setUp(self):
        self.topo = TopologyGraph()
        self.topo.add_storage_node("nodeA")
        self.topo.add_storage_node("nodeB")
        self.topo.add_volume("vol1", "nodeA", pool_id="pool1", capacity_gb=100.0)
        self.topo.add_volume("vol2", "nodeA", pool_id="pool1", capacity_gb=100.0)

    def test_capacity_planner_recommendations(self):
        planner = CapacityPlanner(
            warning_dtf_days=30.0,
            critical_dtf_days=7.0,
            auto_scale_enabled=True
        )

        cached_analysis = {
            "vol1": {
                "workload_type": "Backup",
                "days_to_fill": {"warning_85pct_days": 15.0, "critical_95pct_days": 5.0},
                "latency_ttv": {"risk_level": "none"}
            },
            "vol2": {
                "workload_type": "AI_Training",
                "days_to_fill": {"warning_85pct_days": 40.0, "critical_95pct_days": None},
                "latency_ttv": {"risk_level": "critical", "hours_to_breach": 0.5}
            }
        }

        # Run planner
        plan = planner.plan(self.topo, cached_analysis)
        self.assertEqual(plan.overall_urgency, Urgency.CRITICAL.value)
        self.assertTrue(len(plan.recommendations) >= 3)

        rec_types = [r.rec_type for r in plan.recommendations]
        self.assertIn(RecommendationType.TIER_COLD_DATA.value, rec_types) # vol1 has >15% Backup footprint & low dtf
        self.assertIn(RecommendationType.QOS_SHAPE.value, rec_types)      # vol2 has critical latency risk
        self.assertIn(RecommendationType.RESCHEDULE_JOBS.value, rec_types) # vol2 is AI_Training (batch) causing contention

    def test_actuator_execution_phases(self):
        monitor = ActionMonitor(stall_timeout_seconds=600.0)
        actuator = get_actuator("stub", monitor, speed_up=1000.0)
        rebal = Rebalancer(actuator=actuator)

        # Execute migration
        res = rebal.execute_migration("vol1", "nodeB", self.topo)
        self.assertEqual(res["status"], "success")

        # Verify monitor states
        op_id = res["op_id"]
        # Stub actuator completes execution synchronously because it runs in-thread,
        # so it should be completed and moved to move_history
        self.assertEqual(len(monitor.move_history), 1)
        record = monitor.move_history[0]
        self.assertEqual(record.op_id, op_id)
        self.assertEqual(record.phase, MovePhase.COMPLETED.value)
        self.assertEqual(record.progress_pct, 100.0)

    def test_execution_watchdog_stalls(self):
        monitor = ActionMonitor(stall_timeout_seconds=100.0)
        op_id = "test-op-123"
        
        # Register a move and set to EXECUTING
        monitor.register_move(op_id, "vol1", "nodeA", "nodeB", 100.0)
        monitor.update_phase(op_id, MovePhase.EXECUTING.value, bytes_moved_gb=10.0)

        # Manually alter the last progress timestamp to be older than timeout
        record = monitor.active_moves[op_id]
        record.last_progress_at = time.time() - 200.0

        # Trigger stall check
        monitor.check_stalls(time.time())
        
        self.assertEqual(len(monitor.move_history), 1)
        record = monitor.move_history[0]
        self.assertEqual(record.phase, MovePhase.FAILED.value)
        self.assertIn("Stalled", record.error)

if __name__ == "__main__":
    unittest.main()
