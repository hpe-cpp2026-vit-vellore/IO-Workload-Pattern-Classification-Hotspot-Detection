import unittest
import pandas as pd
from pathlib import Path
from src.control_plane.remote_inference_client import RemoteInferenceClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class TestRemoteInference(unittest.TestCase):
    
    def setUp(self):
        self.client = RemoteInferenceClient(project_root=PROJECT_ROOT)
        
    def test_conformance(self):
        """Verify client initializes variables and matches contract requirements."""
        self.assertIsNotNone(self.client.policy)
        self.assertIsNotNone(self.client.features_df)
        self.assertIsNotNone(self.client.topology)
        self.assertIsNotNone(self.client.live_features_df)
        
    def test_known_volumes(self):
        """Verify known_volumes returns standard sets of volume IDs."""
        vols = self.client.known_volumes()
        self.assertTrue(len(vols) > 0)
        self.assertIn("vol_001", vols)
        
    def test_remote_simulation(self):
        """Verify analyze_volume triggers remote calls and receives fallback predictions."""
        res = self.client.analyze_volume("vol_001")
        self.assertEqual(res["volume_id"], "vol_001")
        self.assertIn("hotspot_score", res)
        self.assertIn("risk_level", res)
        self.assertIn("forecast", res)
        self.assertEqual(res["forecast"]["tft_p95_latency"], 150.0) # 0.15 * 1000
        
    def test_live_features_append(self):
        """Verify updating live features appends and updates records correctly."""
        new_row = pd.DataFrame([{
            "volume_id": "vol_test_99",
            "timestamp": pd.Timestamp("2026-06-10T12:00:00"),
            "total_iops": 5000.0,
            "avg_latency_us": 120.0
        }])
        self.client.update_live_features(new_row)
        vols = self.client.known_volumes()
        self.assertIn("vol_test_99", vols)

if __name__ == "__main__":
    unittest.main()
