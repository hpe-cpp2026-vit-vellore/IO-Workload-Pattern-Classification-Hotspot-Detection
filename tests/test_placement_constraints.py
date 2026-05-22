import pytest
from src.pipeline.topology_graph import TopologyGraph
from src.control_plane.rebalancer import Rebalancer


def test_replica_anti_affinity_validate():
    topo = TopologyGraph()
    topo.add_storage_node("nodeA")
    topo.add_storage_node("nodeB")
    topo.add_storage_node("nodeC")

    topo.add_volume("vol1", "nodeA")
    topo.add_volume("vol1_rep", "nodeB")

    # Register replica relationship
    topo.set_replica("vol1_rep", "vol1")

    # vol1 should NOT be allowed to migrate to nodeB (replica present)
    assert topo.validate_migration("vol1", "nodeB") is False

    # vol1 should be allowed to migrate to nodeC
    assert topo.validate_migration("vol1", "nodeC") is True

    # Replica should NOT be allowed to migrate to nodeA (primary present)
    assert topo.validate_migration("vol1_rep", "nodeA") is False


def test_rebalancer_block_on_placement():
    topo = TopologyGraph()
    topo.add_storage_node("n1")
    topo.add_storage_node("n2")
    topo.add_volume("p", "n1")
    topo.add_volume("r", "n2")
    topo.set_replica("r", "p")

    rebal = Rebalancer()
    # Attempt to migrate primary p to n2 should be blocked
    res = rebal.execute_migration("p", "n2", topo)
    assert res["status"] == "blocked_by_placement"

    # Migrate p to a new node should succeed
    topo.add_storage_node("n3")
    res2 = rebal.execute_migration("p", "n3", topo)
    assert res2["status"] == "success"
    assert topo.get_node_of_volume("p") == "n3"  # updated
