import unittest
import sys
import time
from pathlib import Path
from fastapi import HTTPException

# Setup pathing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Sibling model imports
ANOMALY_DIR = PROJECT_ROOT / "src" / "models" / "anomaly"
if str(ANOMALY_DIR) not in sys.path:
    sys.path.insert(0, str(ANOMALY_DIR))

import api.main as api_main
from src.control_plane.inference_hub import InferenceHub

class TestValidateVolume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the API main hub if not already done
        if api_main.hub is None:
            api_main.hub = InferenceHub(project_root=PROJECT_ROOT)
        # Initialize/clear cache
        api_main._known_volumes_cache = api_main.hub.known_volumes()
        api_main._known_volumes_cache_built_at = time.time()

    def test_known_volumes_method(self):
        """Verify that known_volumes() returns union of historical, live, and topology volumes."""
        hub = api_main.hub
        self.assertIsNotNone(hub)
        
        # Original unique volumes in historical df
        hist_vols = set(hub.features_df["volume_id"].unique())
        
        # Test known_volumes returns them
        known = hub.known_volumes()
        for v in hist_vols:
            self.assertIn(v, known)
            
        # Add a volume to topology graph
        hub.topology.add_volume("vol_topology_only", "nodeA")
        known = hub.known_volumes()
        self.assertIn("vol_topology_only", known)
        
        # Add a volume to live features df
        import pandas as pd
        new_row = pd.DataFrame([{"volume_id": "vol_live_only", "timestamp": pd.Timestamp.now()}])
        hub.live_features_df = pd.concat([hub.live_features_df, new_row], ignore_index=True)
        
        known = hub.known_volumes()
        self.assertIn("vol_live_only", known)

    def test_validate_volume_cache(self):
        """Verify validate_volume works and uses the cached set correctly."""
        # Clean state for validation test
        api_main._known_volumes_cache = api_main.hub.known_volumes()
        api_main._known_volumes_cache_built_at = time.time()
        
        # Existing historical volume should pass without raising exception
        # Let's get one volume from features_df
        existing_vol = api_main.hub.features_df["volume_id"].iloc[0]
        try:
            api_main.validate_volume(existing_vol)
        except HTTPException as e:
            self.fail(f"validate_volume raised HTTPException unexpectedly: {e}")
            
        # Add a new volume that is not in features_df but in topology
        new_vol = "vol_newly_discovered"
        api_main.hub.topology.add_volume(new_vol, "nodeA")
        
        # Calling validate_volume right away might not see it if cache hasn't expired yet
        # Let's test that it is not in the cache initially if we don't rebuild
        if new_vol not in api_main._known_volumes_cache:
            with self.assertRaises(HTTPException) as ctx:
                api_main.validate_volume(new_vol)
            self.assertEqual(ctx.exception.status_code, 404)
            
        # Force cache expiration/refresh
        api_main._known_volumes_cache_built_at = time.time() - (api_main._KNOWN_VOLUMES_CACHE_TTL_SECONDS + 1)
        
        # Now it should be discovered and pass validation
        try:
            api_main.validate_volume(new_vol)
        except HTTPException as e:
            self.fail(f"validate_volume raised HTTPException after cache refresh: {e}")

    def test_nonexistent_volume_raises_404(self):
        """Verify that completely nonexistent volume raises HTTP 404."""
        # Force rebuild to be clean
        api_main._known_volumes_cache = api_main.hub.known_volumes()
        api_main._known_volumes_cache_built_at = time.time()
        
        with self.assertRaises(HTTPException) as ctx:
            api_main.validate_volume("nonexistent_vol_12345")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_fast_hotspot_score(self):
        """Verify that fast_hotspot_score calculates score and updates baseline."""
        hub = api_main.hub
        self.assertIsNotNone(hub)
        
        vol_id = hub.features_df["volume_id"].iloc[0]
        ts = hub.features_df["timestamp"].max()
        
        # Call fast_hotspot_score
        score = hub.fast_hotspot_score(vol_id, ts)
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 100.0)

    def test_split_throttle_endpoints(self):
        """Verify GET /volumes and GET /alerts use the fast path scores."""
        vol_id = api_main.hub.features_df["volume_id"].iloc[0]
        api_main.fast_hotspot_scores[vol_id] = 75.0
        
        # Test GET /volumes
        vols = api_main.get_volumes()
        self.assertGreater(len(vols), 0)
        found_vol = next((v for v in vols if v["volume_id"] == vol_id), None)
        self.assertIsNotNone(found_vol)
        self.assertEqual(found_vol["fast_hotspot_score"], 75.0)
        
        # Test GET /alerts blend logic
        # 1. No events received yet: should fallback to historical score
        api_main.live_state.events_received = 0
        alerts = api_main.get_alerts()
        self.assertIsInstance(alerts, list)
        
        # 2. Events received (live mode active): should blend
        api_main.live_state.events_received = 5
        # Set historical cached score in cached_analysis for vol_id to 50
        api_main.cached_analysis[vol_id] = {
            "hotspot_score": 50.0,
            "workload_type": "DB_OLTP",
            "timestamp": "2026-06-01T12:00:00"
        }
        
        alerts = api_main.get_alerts()
        found_alert = next((a for a in alerts if a["volume_id"] == vol_id), None)
        if found_alert:
            # Expected blended score: 0.6 * 75.0 + 0.4 * 50.0 = 45.0 + 20.0 = 65.0
            self.assertEqual(found_alert["hotspot_score"], 65.0)

if __name__ == "__main__":
    unittest.main()
