import os
import sys
import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path

# Setup pathing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Sibling model imports
ANOMALY_DIR = PROJECT_ROOT / "src" / "models" / "anomaly"
if str(ANOMALY_DIR) not in sys.path:
    sys.path.insert(0, str(ANOMALY_DIR))

# Ensure settings overrides are set before importing components
from configs.settings import settings

class TestProductionPipeline(unittest.TestCase):
    def setUp(self):
        # Backup settings
        self.original_env = settings.environment
        self.original_bus = settings.bus_type
        self.original_inference = settings.inference_mode
        self.original_db = settings.db_type

        # Production Overrides
        settings.environment = "production"
        settings.bus_type = "kafka"
        settings.inference_mode = "remote"
        settings.db_type = "timescaledb"

        # Prevent Redis reconnect loop delay during TestClient startup
        os.environ["REDIS_RETRY_ATTEMPTS"] = "1"
        os.environ["REDIS_RETRY_DELAY"] = "0"

    def tearDown(self):
        # Restore settings
        settings.environment = self.original_env
        settings.bus_type = self.original_bus
        settings.inference_mode = self.original_inference
        settings.db_type = self.original_db

        # Clean up global state in api.main to avoid leaking to other unit tests
        import api.main as api_main
        api_main.hub = None
        api_main.rebalancer = None
        api_main.monitor = None
        api_main.engine = None
        api_main.simulator = None
        api_main.capacity_planner = None

    @patch("src.infrastructure.kafka_bus.Producer")
    @patch("src.infrastructure.kafka_bus.Consumer")
    def test_kafka_bus_resolves_in_production(self, mock_consumer, mock_producer):
        """Assert that get_event_bus() resolves to KafkaBus under production settings."""
        from src.infrastructure.bus_factory import get_event_bus
        from src.infrastructure.kafka_bus import KafkaBus
        
        # get_event_bus() will internally instantiate KafkaBus and call connect() once
        bus = get_event_bus()
        self.assertIsInstance(bus, KafkaBus)
        mock_producer.assert_called_once()
        mock_consumer.assert_called_once()

    @patch("src.infrastructure.timescale_client.TimescaleClient")
    def test_remote_inference_client_init_and_conformance(self, mock_timescale_class):
        """Assert that the API hub factory instantiates RemoteInferenceClient with remote settings."""
        # Setup timescale mock to return valid topology structure and historical features
        mock_db_client = MagicMock()
        mock_db_client.get_topology_data.return_value = pd.DataFrame([
            {
                "volume_id": "vol_123",
                "node_id": "node_A",
                "pool_id": "pool_1",
                "tier": "ssd",
                "capacity_total_gb": 100.0
            }
        ])
        mock_db_client.get_historical_features.return_value = pd.DataFrame([
            {
                "volume_id": "vol_123",
                "timestamp": pd.Timestamp("2026-06-10T12:00:00")
            }
        ])
        mock_timescale_class.return_value = mock_db_client

        from src.control_plane.remote_inference_client import RemoteInferenceClient
        
        # Instantiate and assert class type
        client = RemoteInferenceClient(project_root=PROJECT_ROOT)
        self.assertIsInstance(client, RemoteInferenceClient)
        self.assertIsNotNone(client.topology)

        # Mock the requests.post call for remote serving proxy
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            # Remote Inference Client expect calls to respond with lists containing predictions
            mock_resp.json.return_value = {"predictions": [0.85]}
            mock_post.return_value = mock_resp

            # Verify that client conforms to analyze_volume interface without local PyTorch models
            analysis = client.analyze_volume("vol_123")
            self.assertEqual(analysis["volume_id"], "vol_123")
            self.assertEqual(analysis["hotspot_score"], 0.85)

    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.config.load_kube_config")
    def test_csi_actuator_production_mock_fallback(self, mock_load_kube, mock_load_incluster):
        """Assert CSIActuator initializes safely and falls back to mock status outside cluster."""
        from src.control_plane.actuators import CSIActuator
        
        # Cause K8s config loading to fail (simulating non-cluster runtime)
        mock_load_incluster.side_effect = Exception("No K8s cluster config")
        mock_load_kube.side_effect = Exception("No local kubeconfig")

        actuator = CSIActuator()
        # Ensure it didn't crash but gracefully resolved to mock fallback status
        self.assertFalse(actuator._k8s_client_initialized)
        self.assertIsNone(actuator._k8s_client)

    @patch("src.infrastructure.kafka_bus.Producer")
    @patch("src.infrastructure.kafka_bus.Consumer")
    @patch("src.infrastructure.timescale_client.TimescaleClient")
    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.config.load_kube_config")
    def test_fastapi_security_smoke_test(self, mock_load_kube, mock_load_incluster, mock_timescale_class, mock_consumer, mock_producer):
        """Integration smoke test using TestClient to verify authentication layers."""
        from fastapi.testclient import TestClient
        
        # Mock dependencies/environment configuration
        mock_load_incluster.side_effect = Exception("Fail")
        mock_load_kube.side_effect = Exception("Fail")
        
        mock_db_client = MagicMock()
        mock_db_client.get_topology_data.return_value = pd.DataFrame([
            {
                "volume_id": "vol_123",
                "node_id": "node_A",
                "pool_id": "pool_1",
                "tier": "ssd",
                "capacity_total_gb": 100.0
            }
        ])
        mock_timescale_class.return_value = mock_db_client

        # Import api.main app and spin up TestClient
        from api.main import app
        
        with TestClient(app) as client:
            # 1. Secured endpoint should reject requests without token
            resp = client.get("/alerts")
            self.assertEqual(resp.status_code, 401)

            # 2. Authenticate against /token
            login_resp = client.post("/token", data={"username": "admin", "password": "hpe_admin_2026"})
            self.assertEqual(login_resp.status_code, 200)
            token = login_resp.json().get("access_token")
            self.assertTrue(token)

            # 3. Secured endpoint should succeed with valid Bearer token
            headers = {"Authorization": f"Bearer {token}"}
            secured_resp = client.get("/alerts", headers=headers)
            self.assertEqual(secured_resp.status_code, 200)

if __name__ == "__main__":
    unittest.main()
