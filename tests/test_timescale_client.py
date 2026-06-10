import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
from src.infrastructure.timescale_client import TimescaleClient
from src.control_plane.inference_hub import InferenceHub
from src.control_plane.remote_inference_client import RemoteInferenceClient
from configs.settings import settings

class TestTimescaleClient(unittest.TestCase):
    
    @patch("src.infrastructure.timescale_client.create_engine")
    def setUp(self, mock_create_engine):
        self.mock_engine = MagicMock()
        mock_create_engine.return_value = self.mock_engine
        
        # Save original db_type to restore after tests
        self.original_db_type = settings.db_type
        
        # For setting up client
        self.client = TimescaleClient()
        
    def tearDown(self):
        settings.db_type = self.original_db_type

    @patch("src.infrastructure.timescale_client.pd.read_sql")
    def test_get_historical_features(self, mock_read_sql):
        # Setup mock return data
        mock_df = pd.DataFrame([{
            "volume_id": "vol_001",
            "timestamp": "2026-06-10 12:00:00",
            "avg_latency_us": 150.0,
            "total_iops": 200.0
        }])
        mock_read_sql.return_value = mock_df
        
        df = self.client.get_historical_features("vol_001", limit=10)
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["volume_id"], "vol_001")
        mock_read_sql.assert_called_once()
        
    @patch("src.infrastructure.timescale_client.pd.read_sql")
    def test_get_topology_data(self, mock_read_sql):
        mock_df = pd.DataFrame([{
            "volume_id": "vol_001",
            "node_id": "node_01",
            "pool_id": "pool_01",
            "tier": "SSD",
            "capacity_total_gb": 1000.0
        }])
        mock_read_sql.return_value = mock_df
        
        df = self.client.get_topology_data()
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["node_id"], "node_01")
        
    @patch("src.infrastructure.timescale_client.pd.read_sql")
    def test_get_noisy_neighbor_baselines(self, mock_read_sql):
        mock_df = pd.DataFrame([{
            "volume_id": "vol_001",
            "lat_mean": 100.0,
            "lat_std": 10.0,
            "lat_n": 50,
            "iops_mean": 500.0,
            "iops_std": 50.0,
            "iops_n": 50
        }])
        mock_read_sql.return_value = mock_df
        
        df = self.client.get_noisy_neighbor_baselines()
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["lat_mean"], 100.0)

    @patch("src.infrastructure.timescale_client.pd.read_sql")
    def test_get_neighbors_metrics(self, mock_read_sql):
        mock_df = pd.DataFrame([{
            "volume_id": "vol_002",
            "avg_latency_us": 120.0,
            "total_iops": 300.0
        }])
        mock_read_sql.return_value = mock_df
        
        df = self.client.get_neighbors_metrics(["vol_002"], pd.Timestamp("2026-06-10 12:00:00"))
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["volume_id"], "vol_002")

    @patch("src.infrastructure.timescale_client.TimescaleClient")
    def test_inference_hub_db_mode(self, mock_timescale_client):
        # Configure setting
        settings.db_type = "timescaledb"
        
        # Mock TimescaleClient instance & return values
        mock_client_inst = MagicMock()
        mock_timescale_client.return_value = mock_client_inst
        
        # Mock topology data (must return dataframe containing node_id and volume_id)
        mock_client_inst.get_topology_data.return_value = pd.DataFrame([{
            "volume_id": "vol_001",
            "node_id": "node_01",
            "pool_id": "pool_01",
            "tier": "SSD",
            "capacity_total_gb": 1000.0
        }])
        
        # Mock noisy neighbor baselines
        mock_client_inst.get_noisy_neighbor_baselines.return_value = pd.DataFrame([{
            "volume_id": "vol_001",
            "lat_mean": 100.0,
            "lat_std": 10.0,
            "lat_n": 50,
            "iops_mean": 500.0,
            "iops_std": 50.0,
            "iops_n": 50
        }])
        
        # Instantiate InferenceHub (should load from mock DB client and skip CSV files)
        hub = InferenceHub()
        self.assertIsNotNone(hub.topology)
        self.assertIn("vol_001", hub.known_volumes())
        self.assertEqual(hub.noisy_neighbor._baselines["vol_001"][0], 100.0)
        
    @patch("src.infrastructure.timescale_client.TimescaleClient")
    def test_remote_inference_client_db_mode(self, mock_timescale_client):
        settings.db_type = "timescaledb"
        
        mock_client_inst = MagicMock()
        mock_timescale_client.return_value = mock_client_inst
        
        mock_client_inst.get_topology_data.return_value = pd.DataFrame([{
            "volume_id": "vol_001",
            "node_id": "node_01",
            "pool_id": "pool_01",
            "tier": "SSD",
            "capacity_total_gb": 1000.0
        }])
        
        client = RemoteInferenceClient()
        self.assertIsNotNone(client.topology)
        self.assertIn("vol_001", client.known_volumes())

if __name__ == "__main__":
    unittest.main()
