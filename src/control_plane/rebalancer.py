"""
src/control_plane/rebalancer.py

Rebalance Execution Engine (HPE Blueprint Phase 5.4)
======================================================
Executes actions: migrate volume, QoS shape, tier change. Modifies topology graph.
Provides status for rollback tracking.
"""

import logging
from typing import Dict, Any, Optional
from src.pipeline.topology_graph import TopologyGraph

logger = logging.getLogger(__name__)

class Rebalancer:
    """Handles execution and rollback of control plane rebalancing actions."""

    def __init__(self) -> None:
        pass

    def execute_migration(self, volume_id: str, target_node: str, topology: TopologyGraph) -> Dict[str, Any]:
        """
        Migrate a volume to a target node.
        Updates topology graph structure.
        """
        source_node = topology.get_node_of_volume(volume_id)
        if not source_node:
            raise ValueError(f"Volume {volume_id} not found in topology.")

        # Placement validation: ensure move won't violate replica anti-affinity
        try:
            allowed = topology.validate_migration(volume_id, target_node)
        except Exception:
            allowed = False
        if not allowed:
            logger.warning("Migration for %s to %s blocked by placement constraints.", volume_id, target_node)
            return {"action": "migrate", "volume_id": volume_id, "source_node": source_node, "target_node": target_node, "status": "blocked_by_placement"}

        if source_node == target_node:
            logger.info("Volume %s is already on %s. Migration skipped.", volume_id, target_node)
            return {"action": "migrate", "volume_id": volume_id, "source_node": source_node, "target_node": target_node, "status": "no-op"}

        logger.info("Executing migration: %s from %s to %s", volume_id, source_node, target_node)

        # 1. Update graph edges
        if topology.graph.has_edge(volume_id, source_node):
            topology.graph.remove_edge(volume_id, source_node)
        topology.graph.add_edge(volume_id, target_node, relation="resides_on")

        # 2. Update auxiliary indices
        topology._volume_to_node[volume_id] = target_node
        if volume_id in topology._node_volumes[source_node]:
            topology._node_volumes[source_node].remove(volume_id)
        if target_node not in topology._node_volumes:
            topology._node_volumes[target_node] = []
        if volume_id not in topology._node_volumes[target_node]:
            topology._node_volumes[target_node].append(volume_id)

        # Update node attribute of the volume node
        topology.graph.nodes[volume_id]["node_id"] = target_node

        return {
            "action": "migrate",
            "volume_id": volume_id,
            "source_node": source_node,
            "target_node": target_node,
            "status": "success"
        }

    def execute_qos_shaping(self, volume_id: str, iops_limit: float, topology: TopologyGraph) -> Dict[str, Any]:
        """
        Apply QoS IOPS shaping/cap to a volume.
        Updates volume node attributes.
        """
        if not topology.graph.has_node(volume_id):
            raise ValueError(f"Volume {volume_id} not found in topology.")

        old_iops_limit = topology.graph.nodes[volume_id].get("iops_limit")
        logger.info("Executing QoS shaping: cap %s to %s IOPS (previous: %s)", volume_id, iops_limit, old_iops_limit)
        
        topology.graph.nodes[volume_id]["iops_limit"] = iops_limit

        return {
            "action": "qos",
            "volume_id": volume_id,
            "old_iops_limit": old_iops_limit,
            "new_iops_limit": iops_limit,
            "status": "success"
        }

    def execute_tier_change(self, volume_id: str, new_tier: str, topology: TopologyGraph) -> Dict[str, Any]:
        """
        Change storage tier for a volume.
        Updates volume node attributes.
        """
        if not topology.graph.has_node(volume_id):
            raise ValueError(f"Volume {volume_id} not found in topology.")

        old_tier = topology.graph.nodes[volume_id].get("tier")
        logger.info("Executing tier change: move %s from %s to %s", volume_id, old_tier, new_tier)
        
        topology.graph.nodes[volume_id]["tier"] = new_tier

        return {
            "action": "tier_change",
            "volume_id": volume_id,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "status": "success"
        }

    def rollback_action(self, action_state: Dict[str, Any], topology: TopologyGraph) -> None:
        """Rollback a previously executed action to restore previous state."""
        action = action_state.get("action")
        volume_id = action_state.get("volume_id")
        
        if not action or not volume_id:
            logger.warning("Invalid action state for rollback: %s", action_state)
            return

        logger.info("Rolling back action %s for volume %s", action, volume_id)

        if action == "migrate":
            source_node = action_state["source_node"]
            target_node = action_state["target_node"]
            # Restore to source_node
            if topology.graph.has_edge(volume_id, target_node):
                topology.graph.remove_edge(volume_id, target_node)
            topology.graph.add_edge(volume_id, source_node, relation="resides_on")
            
            topology._volume_to_node[volume_id] = source_node
            if volume_id in topology._node_volumes.get(target_node, []):
                topology._node_volumes[target_node].remove(volume_id)
            if volume_id not in topology._node_volumes[source_node]:
                topology._node_volumes[source_node].append(volume_id)
            topology.graph.nodes[volume_id]["node_id"] = source_node

        elif action == "qos":
            old_iops_limit = action_state["old_iops_limit"]
            if old_iops_limit is None:
                if "iops_limit" in topology.graph.nodes[volume_id]:
                    del topology.graph.nodes[volume_id]["iops_limit"]
            else:
                topology.graph.nodes[volume_id]["iops_limit"] = old_iops_limit

        elif action == "tier_change":
            old_tier = action_state["old_tier"]
            topology.graph.nodes[volume_id]["tier"] = old_tier
            
        logger.info("Rollback complete for volume %s", volume_id)

    def add_virtual_node(self, node_id: str, capacity_gb: Optional[float], tier: Optional[str], topology: TopologyGraph) -> Dict[str, Any]:
        """Add a new virtual/storage node to the topology (autoscaler helper).

        This is intentionally lightweight: it registers a new storage node that the
        autoscaler or operator can target for future migrations.
        """
        if topology.graph.has_node(node_id):
            logger.info("Virtual node %s already exists.", node_id)
            return {"action": "add_node", "node_id": node_id, "status": "exists"}

        topology.add_storage_node(node_id, capacity_gb=capacity_gb, tier=tier)
        logger.info("Added virtual node %s (cap=%s, tier=%s).", node_id, capacity_gb, tier)
        return {"action": "add_node", "node_id": node_id, "capacity_gb": capacity_gb, "tier": tier, "status": "success"}

    def expand_logical_pool(self, pool_id: str, new_node_id: str, topology: TopologyGraph) -> Dict[str, Any]:
        """Expand a logical pool by adding `new_node_id` into its topology footprint.

        Current simplified behaviour: ensure the node exists and mark the node as a candidate
        for the pool by annotating volumes moved later via migrations. Full data path
        attachment is out-of-scope for this prototype.
        """
        if not topology.graph.has_node(new_node_id):
            raise ValueError(f"Node {new_node_id} does not exist; create it first with add_virtual_node.")

        # Annotate the node as supporting the pool via node attribute 'pools'
        pools = topology.graph.nodes[new_node_id].get("pools", [])
        if pool_id not in pools:
            pools.append(pool_id)
            topology.graph.nodes[new_node_id]["pools"] = pools

        logger.info("Expanded pool %s with node %s.", pool_id, new_node_id)
        return {"action": "expand_pool", "pool_id": pool_id, "node_id": new_node_id, "status": "success"}
