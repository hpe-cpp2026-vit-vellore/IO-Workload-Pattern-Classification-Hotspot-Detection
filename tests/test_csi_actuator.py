import unittest
from unittest.mock import patch, MagicMock
from src.control_plane.actuators import CSIActuator, execute_action, get_actuator
from configs.settings import settings

class TestCSIActuator(unittest.TestCase):

    def setUp(self):
        self.original_env = settings.environment
        
    def tearDown(self):
        settings.environment = self.original_env

    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.config.load_kube_config")
    @patch("kubernetes.client.CoreV1Api")
    def test_csi_actuator_init_in_cluster(self, mock_core_v1, mock_load_kube_config, mock_load_incluster):
        # Test K8s client successfully bound to in-cluster config
        actuator = CSIActuator()
        self.assertTrue(actuator._k8s_client_initialized)
        mock_load_incluster.assert_called_once()
        mock_load_kube_config.assert_not_called()

    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.config.load_kube_config")
    @patch("kubernetes.client.CoreV1Api")
    def test_csi_actuator_init_kube_config(self, mock_core_v1, mock_load_kube_config, mock_load_incluster):
        # In-cluster config fails, but local kubeconfig succeeds
        mock_load_incluster.side_effect = Exception("Not in cluster")
        actuator = CSIActuator()
        self.assertTrue(actuator._k8s_client_initialized)
        mock_load_incluster.assert_called_once()
        mock_load_kube_config.assert_called_once()

    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.config.load_kube_config")
    def test_csi_actuator_mock_fallback(self, mock_load_kube_config, mock_load_incluster):
        # Both fail -> Mock/Fallback mode
        mock_load_incluster.side_effect = Exception("Fail")
        mock_load_kube_config.side_effect = Exception("Fail")
        actuator = CSIActuator()
        self.assertFalse(actuator._k8s_client_initialized)
        self.assertIsNone(actuator._k8s_client)

    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.client.CoreV1Api")
    def test_execute_move_real_patch(self, mock_core_v1, mock_load_incluster):
        # Setup mock client
        mock_client = MagicMock()
        mock_core_v1.return_value = mock_client
        
        actuator = CSIActuator()
        self.assertTrue(actuator._k8s_client_initialized)
        
        # Call execute_move
        success = actuator.execute_move("vol_123", "node_A", "node_B", 100.0, "op_001")
        self.assertTrue(success)
        
        # Check that PV patch was called
        mock_client.patch_persistent_volume.assert_called_once()
        args, kwargs = mock_client.patch_persistent_volume.call_args
        self.assertEqual(kwargs["name"], "pvc-vol_123")
        self.assertIn("nodeAffinity", kwargs["body"]["spec"])

    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.client.CoreV1Api")
    def test_execute_rebalance_success(self, mock_core_v1, mock_load_incluster):
        mock_client = MagicMock()
        mock_core_v1.return_value = mock_client
        
        actuator = CSIActuator()
        mock_event_bus = MagicMock()
        
        plan = {"volume_id": "vol_123", "target_node": "node_B"}
        success = actuator.execute_rebalance("act_001", plan, mock_event_bus)
        
        self.assertTrue(success)
        mock_client.patch_persistent_volume.assert_called_once()
        mock_event_bus.set_state.assert_called_with("action:act_001", {
            "status": "EXECUTING",
            "message": "CSI Driver triggered for vol_123 -> node_B"
        })

    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.client.CoreV1Api")
    def test_execute_rebalance_api_failure(self, mock_core_v1, mock_load_incluster):
        from kubernetes.client.rest import ApiException
        mock_client = MagicMock()
        mock_client.patch_persistent_volume.side_effect = ApiException(status=500, reason="API failure")
        mock_core_v1.return_value = mock_client
        
        actuator = CSIActuator()
        mock_event_bus = MagicMock()
        
        plan = {"volume_id": "vol_123", "target_node": "node_B"}
        success = actuator.execute_rebalance("act_001", plan, mock_event_bus)
        
        self.assertFalse(success)
        mock_event_bus.set_state.assert_any_call("action:act_001", {
            "status": "FAILED",
            "message": "(500)\nReason: API failure\n"
        })

    @patch("kubernetes.config.load_incluster_config")
    @patch("kubernetes.config.load_kube_config")
    def test_execute_action_helper_production(self, mock_load_kube, mock_load_incluster):
        settings.environment = "production"
        
        # Mute config loading exception so it runs mock mode but is tagged environment=production
        mock_load_incluster.side_effect = Exception("Fail")
        mock_load_kube.side_effect = Exception("Fail")
        
        mock_event_bus = MagicMock()
        plan = {"action": "rebalance_volume", "volume_id": "vol_123", "target_node": "node_B"}
        
        success = execute_action("act_001", plan, mock_event_bus)
        self.assertTrue(success)
        
        mock_event_bus.set_state.assert_any_call("action:act_001", {"status": "COMPLETED"})

if __name__ == "__main__":
    unittest.main()
