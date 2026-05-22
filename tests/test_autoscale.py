import pandas as pd
from src.pipeline.topology_graph import TopologyGraph
from src.control_plane.rebalancer import Rebalancer
from src.control_plane.monitor import ActionMonitor
from src.control_plane.decision_engine import DecisionEngine


class FakeHub:
    def __init__(self, topology, policy):
        self.topology = topology
        self.policy = policy

    def analyze_volume(self, volume_id, timestamp):
        # Return a forecast indicating capacity will be reached soon
        return {"days_to_fill": {"warning_85pct_days": 2.0, "critical_95pct_days": None}}


def test_autoscale_triggers_and_adds_node():
    topo = TopologyGraph()
    topo.add_storage_node("node1")
    topo.add_volume("volA", "node1", pool_id="poolA")

    policy = {"rebalance_policy": {"autoscale": {"enabled": True, "warning_days": 7, "min_interval_hours": 0, "max_new_nodes_per_run": 1, "new_node_capacity_gb": 100}}}

    hub = FakeHub(topo, policy)
    rebal = Rebalancer()
    monitor = ActionMonitor()
    engine = DecisionEngine(hub, rebal, monitor)

    # Run queued action processing (which calls autoscale check)
    ts = pd.Timestamp.now()
    res = engine.process_queued_actions(ts)

    # Ensure a new autoscale node was added to topology
    nodes = topo.all_nodes()
    autos_nodes = [n for n in nodes if n.startswith("autoscale-")]
    assert len(autos_nodes) == 1
    assert res["autoscale"]
    # And the node should be annotated with pool membership
    node_attr = topo.graph.nodes[autos_nodes[0]]
    assert "pools" in node_attr and "poolA" in node_attr["pools"]
    assert engine.last_autoscale_time is not None
    assert monitor.total_actions == 1
