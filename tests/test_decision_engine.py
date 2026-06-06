import unittest
import pandas as pd
from src.pipeline.topology_graph import TopologyGraph
from src.control_plane.rebalancer import Rebalancer
from src.control_plane.monitor import ActionMonitor
from src.control_plane.decision_engine import DecisionEngine

class FakeHub:
    def __init__(self, topology, policy):
        self.topology = topology
        self.policy = policy

class TestDecisionEngineReschedule(unittest.TestCase):
    def setUp(self):
        self.topo = TopologyGraph()
        self.topo.add_storage_node("node1")
        self.topo.add_storage_node("node2")
        self.topo.add_volume("volA", "node1", pool_id="poolA", tier="tier-1")
        
        self.policy = {
            "rebalance_policy": {
                "enabled": True,
                "dry_run_mode": False,
                "min_hotspot_score_to_trigger": 75.0,
                "min_hotspot_duration_minutes": 2.0,
                "max_volumes_moved_per_hour": 3,
                "max_concurrent_migrations": 1
            },
            "qos_policy": {
                "backup_iops_cap": 3000.0
            },
            "safety_guardrails": {
                "max_rollback_rate_pct": 1.0
            }
        }
        self.hub = FakeHub(self.topo, self.policy)
        self.rebal = Rebalancer()
        self.monitor = ActionMonitor()
        self.engine = DecisionEngine(self.hub, self.rebal, self.monitor)

    def test_reschedule_job_non_batch(self):
        # DB_OLTP is not a batch workload, so reschedule action should not be safe
        inference_results = {
            "hotspot_score": 80.0,
            "workload_type": "DB_OLTP",
            "noisy_neighbor_victims": {"volB": 20.0}
        }
        simulations = self.engine.simulate_actions("volA", inference_results)
        
        # Find reschedule_job simulation
        reschedule_action = next(s for s in simulations if s["action"] == "reschedule_job")
        self.assertFalse(reschedule_action["safe"])
        self.assertEqual(reschedule_action["expected_improvement"], 0.0)

    def test_reschedule_job_batch(self):
        # AI_Training is a batch workload, so reschedule should be safe and estimated
        # at 1.5 * qos_improvement
        inference_results = {
            "hotspot_score": 85.0,
            "workload_type": "AI_Training",
            "noisy_neighbor_victims": {"volB": 20.0}
        }
        simulations = self.engine.simulate_actions("volA", inference_results)
        
        reschedule_action = next(s for s in simulations if s["action"] == "reschedule_job")
        self.assertTrue(reschedule_action["safe"])
        # qos_improvement = sum of noisy_neighbor_victims = 20.0
        # reschedule_improvement = 20.0 * 1.5 = 30.0
        self.assertEqual(reschedule_action["expected_improvement"], 30.0)
        self.assertEqual(reschedule_action["recommendation_text"], "Reschedule AI_Training job to off-peak hours to relieve contention.")

    def test_evaluate_volume_triggers_reschedule(self):
        # Setup telemetry for volume so we don't throw KeyErrors
        self.topo._volume_metrics["volA"] = {"avg_latency_us": 1200.0}
        
        # Setup persistent hotspot to trigger action
        ts1 = pd.Timestamp.now()
        ts2 = ts1 + pd.Timedelta(minutes=3) # > 2.0 min duration
        
        inference_results = {
            "hotspot_score": 90.0,
            "workload_type": "AI_Training",
            "noisy_neighbor_victims": {"volB": 50.0}  # Large improvement (75.0 ROI)
        }
        
        # Initial evaluation should start persistence tracking
        res1 = self.engine.evaluate_volume("volA", ts1, inference_results)
        self.assertEqual(res1["status"], "waiting_persistence")
        
        # Second evaluation after 3 minutes should execute reschedule_job
        # because reschedule ROI (75.0) is higher than others (migrate is 0, QoS is 50, tier is 0)
        res2 = self.engine.evaluate_volume("volA", ts2, inference_results)
        self.assertEqual(res2["status"], "executed")
        self.assertEqual(res2["action_state"]["status"], "webhook_emitted")
        self.assertIn("Reschedule AI_Training job", res2["action_state"]["message"])

if __name__ == "__main__":
    unittest.main()
