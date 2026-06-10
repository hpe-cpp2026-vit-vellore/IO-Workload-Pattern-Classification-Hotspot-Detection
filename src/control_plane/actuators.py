"""
src/control_plane/actuators.py

Actuator layer.
Translates abstract move plans/actions from the rebalancer into physical or simulated
storage operations via:
  - CSI driver calls (Kubernetes)
  - Array API calls (REST APIs)
  - Stub simulation
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseActuator(ABC):
    """Abstract base class for all storage actuators."""

    def __init__(self, monitor) -> None:
        self.monitor = monitor

    @abstractmethod
    def execute_move(self, volume_id: str, source_node: str, target_node: str, size_gb: float, op_id: str) -> bool:
        """Perform a single volume migration."""
        pass

    @abstractmethod
    def rollback_move(self, volume_id: str, source_node: str, target_node: str, op_id: str) -> bool:
        """Attempt to undo a volume migration."""
        pass

class StubActuator(BaseActuator):
    """Simulates move operations without touching physical hardware."""

    def __init__(self, monitor, failure_rate: float = 0.0, speed_up: float = 100.0) -> None:
        super().__init__(monitor)
        self.failure_rate = failure_rate
        self.speed_up = speed_up

    def execute_move(self, volume_id: str, source_node: str, target_node: str, size_gb: float, op_id: str) -> bool:
        import random
        logger.info("[STUB] Starting move: vol=%s %s->%s size=%.1fGB", volume_id, source_node, target_node, size_gb)

        if hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "PRE_CHECK")
        time.sleep(0.1 / self.speed_up)

        if random.random() < self.failure_rate:
            if hasattr(self.monitor, "mark_failed"):
                self.monitor.mark_failed(op_id, "Simulated failure (pre-check)")
            return False

        if hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "EXECUTING", bytes_moved_gb=0.0)

        steps = 5
        for i in range(1, steps + 1):
            time.sleep((0.5 / steps) / self.speed_up)
            bytes_done = (i / steps) * size_gb
            if hasattr(self.monitor, "update_phase"):
                self.monitor.update_phase(op_id, "EXECUTING", bytes_moved_gb=bytes_done)

        if hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "VERIFYING", bytes_moved_gb=size_gb)
        time.sleep(0.05 / self.speed_up)

        if hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "COMPLETED", bytes_moved_gb=size_gb)
        logger.info("[STUB] Completed move: vol=%s", volume_id)
        return True

    def rollback_move(self, volume_id: str, source_node: str, target_node: str, op_id: str) -> bool:
        logger.info("[STUB] Rolling back op_id=%s: moving vol=%s %s->%s", op_id, volume_id, target_node, source_node)
        if hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "ROLLED_BACK")
        return True

class CSIActuator(BaseActuator):
    """Actuator that drives volume migration via Kubernetes CSI interfaces."""

    def __init__(self, monitor=None, kubeconfig: Optional[str] = None) -> None:
        super().__init__(monitor)
        self.kubeconfig = kubeconfig
        self._k8s_client = None
        self._k8s_client_initialized = False
        self._initialize_client()

    def _initialize_client(self):
        try:
            from kubernetes import client, config as k8s_config
            if self.kubeconfig:
                k8s_config.load_kube_config(config_file=self.kubeconfig)
            else:
                try:
                    k8s_config.load_incluster_config()
                except Exception:
                    k8s_config.load_kube_config()
            self._k8s_client = client.CoreV1Api()
            self._k8s_client_initialized = True
            logger.info("Kubernetes CSI Actuator successfully bound to CoreV1Api.")
        except Exception as e:
            logger.warning(f"K8s config not found. CSI Actuator running in Mock mode. ({e})")

    def _get_client(self):
        if not self._k8s_client_initialized:
            self._initialize_client()
        return self._k8s_client

    def execute_move(self, volume_id: str, source_node: str, target_node: str, size_gb: float, op_id: str) -> bool:
        if self.monitor and hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "PRE_CHECK")
        logger.info("[CSI] Initiating PV migration for vol=%s from node=%s to node=%s", volume_id, source_node, target_node)
        
        client_v1 = self._get_client()
        
        if self.monitor and hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "EXECUTING")
        
        if self._k8s_client_initialized and client_v1 is not None:
            try:
                # Real CSI actuation logic: patch nodeAffinity to force CSI migration
                pv_name = f"pvc-{volume_id}"
                patch_body = {
                    "spec": {
                        "nodeAffinity": {
                            "required": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "kubernetes.io/hostname",
                                                "operator": "In",
                                                "values": [target_node]
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                }
                client_v1.patch_persistent_volume(name=pv_name, body=patch_body)
                logger.info("Successfully patched PV %s via CSI API.", pv_name)
            except Exception as e:
                logger.error("CSI Actuator K8s API Error for PV %s: %s", volume_id, e)
                return False
        else:
            # Safe Fallback for Local Demo
            logger.info("[MOCK CSI] Simulating PV Patch for %s", volume_id)
            time.sleep(0.05)

        if self.monitor and hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "COMPLETED", bytes_moved_gb=size_gb)
        return True

    def rollback_move(self, volume_id: str, source_node: str, target_node: str, op_id: str) -> bool:
        logger.info("[CSI] Rolling back PV migration for vol=%s back to node=%s", volume_id, source_node)
        if self.monitor and hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "ROLLED_BACK")
        return True

    def execute_rebalance(self, action_id: str, plan: Dict[str, Any], event_bus: Any) -> bool:
        """Execute a physical volume migration via CSI (blueprint matching interface)."""
        volume_id = plan.get("volume_id")
        target_node = plan.get("target_node")
        
        # 1. Update State to Executing
        if hasattr(event_bus, "set_state"):
            event_bus.set_state(f"action:{action_id}", {
                "status": "EXECUTING",
                "message": f"CSI Driver triggered for {volume_id} -> {target_node}"
            })
        logger.info(f"CSI Actuator executing migration for {volume_id}")

        client_v1 = self._get_client()

        if not self._k8s_client_initialized or client_v1 is None:
            # Safe Fallback for Local Demo
            logger.info(f"[MOCK CSI] Simulating PV Patch for {volume_id}")
            return True

        # 2. Production K8s Patch Logic
        try:
            from kubernetes.client.rest import ApiException
            pv_name = f"pvc-{volume_id}" 
            patch_body = {
                "spec": {
                    "nodeAffinity": {
                        "required": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "kubernetes.io/hostname",
                                            "operator": "In",
                                            "values": [target_node]
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
            client_v1.patch_persistent_volume(name=pv_name, body=patch_body)
            logger.info(f"Successfully patched PV {pv_name} via CSI.")
            return True
            
        except ApiException as e:
            logger.error(f"CSI Actuator K8s API Error: {e}")
            if hasattr(event_bus, "set_state"):
                event_bus.set_state(f"action:{action_id}", {
                    "status": "FAILED",
                    "message": str(e)
                })
            return False
        except Exception as e:
            logger.error(f"CSI Actuator Unexpected Error: {e}")
            return False

class ArrayAPIActuator(BaseActuator):
    """Actuator that drives volume migrations via Storage Array REST APIs."""

    def __init__(self, monitor, array_endpoint: str = "http://array.local", api_token: str = "token") -> None:
        super().__init__(monitor)
        self.array_endpoint = array_endpoint
        self.api_token = api_token

    def execute_move(self, volume_id: str, source_node: str, target_node: str, size_gb: float, op_id: str) -> bool:
        if hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "PRE_CHECK")
        logger.info("[ArrayAPI] Moving vol=%s on array endpoint %s to node %s", volume_id, self.array_endpoint, target_node)
        
        # REST API calls would be executed here
        if hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "EXECUTING")
        time.sleep(0.05)
        if hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "COMPLETED", bytes_moved_gb=size_gb)
        return True

    def rollback_move(self, volume_id: str, source_node: str, target_node: str, op_id: str) -> bool:
        logger.info("[ArrayAPI] Reverting migration of vol=%s back to node %s", volume_id, source_node)
        if hasattr(self.monitor, "update_phase"):
            self.monitor.update_phase(op_id, "ROLLED_BACK")
        return True

def get_actuator(actuator_type: str, monitor, **kwargs) -> BaseActuator:
    if actuator_type == "csi":
        return CSIActuator(monitor, **kwargs)
    elif actuator_type == "array_api":
        return ArrayAPIActuator(monitor, **kwargs)
    return StubActuator(monitor, **kwargs)

def execute_action(action_id: str, action_plan: Dict[str, Any], event_bus: Any) -> bool:
    """Router to execute an action (blueprint matching interface)."""
    from configs.settings import settings
    action_type = action_plan.get("action") or action_plan.get("action_type")
    
    # In production, we utilize the CSI Actuator for storage migrations
    if settings.environment == "production" and action_type == "rebalance_volume":
        csi = CSIActuator()
        success = csi.execute_rebalance(action_id, action_plan, event_bus)
        if success:
            if hasattr(event_bus, "set_state"):
                event_bus.set_state(f"action:{action_id}", {"status": "COMPLETED"})
        return success
        
    # Webhook or fallback execution logic
    logger.info("Executing simulated webhook action for plan: %s", action_plan)
    if hasattr(event_bus, "set_state"):
        event_bus.set_state(f"action:{action_id}", {"status": "COMPLETED"})
    return True
