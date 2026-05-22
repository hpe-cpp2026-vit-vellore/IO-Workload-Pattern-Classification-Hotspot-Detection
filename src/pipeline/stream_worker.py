"""
src/pipeline/stream_worker.py

Ingestion Stream Worker (HPE Blueprint Phase 5.3)
==================================================
Consumes real-time telemetry events from Redis Streams, performs model inference
and graph updates via InferenceHub, and caches results back to Redis.
"""

import sys
import time
import json
import logging
from pathlib import Path
import pandas as pd

# Setup pathing
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add anomaly model directory to sys.path for internal imports
ANOMALY_DIR = PROJECT_ROOT / "src" / "models" / "anomaly"
if str(ANOMALY_DIR) not in sys.path:
    sys.path.insert(0, str(ANOMALY_DIR))

from src.control_plane import InferenceHub, Rebalancer, ActionMonitor, DecisionEngine
from src.pipeline.telemetry_parser import parse_and_clip, load_or_create_bounds

def sync_topology_structure_from_redis(r_client, topology):
    try:
        assignments = r_client.hgetall("topology:volume_to_node")
        for vol_id, target_node in assignments.items():
            if topology.graph.has_node(vol_id) and target_node:
                source_node = topology.get_node_of_volume(vol_id)
                if source_node != target_node:
                    logger.info("Syncing topology structure: moving %s from %s to %s (from Redis)", vol_id, source_node, target_node)
                    if source_node and topology.graph.has_edge(vol_id, source_node):
                        topology.graph.remove_edge(vol_id, source_node)
                    topology.graph.add_edge(vol_id, target_node, relation="resides_on")
                    topology._volume_to_node[vol_id] = target_node
                    if source_node and vol_id in topology._node_volumes.get(source_node, []):
                        topology._node_volumes[source_node].remove(vol_id)
                    if target_node not in topology._node_volumes:
                        topology._node_volumes[target_node] = []
                    if vol_id not in topology._node_volumes[target_node]:
                        topology._node_volumes[target_node].append(vol_id)
                    topology.graph.nodes[vol_id]["node_id"] = target_node

        tiers = r_client.hgetall("topology:volume_tier")
        for vol_id, tier in tiers.items():
            if topology.graph.has_node(vol_id) and tier:
                topology.graph.nodes[vol_id]["tier"] = tier

        limits = r_client.hgetall("topology:volume_iops_limit")
        for vol_id, limit_str in limits.items():
            if topology.graph.has_node(vol_id):
                if limit_str == "None" or limit_str == "":
                    topology.graph.nodes[vol_id].pop("iops_limit", None)
                else:
                    try:
                        topology.graph.nodes[vol_id]["iops_limit"] = float(limit_str)
                    except ValueError:
                        pass
    except Exception as e:
        logger.error(f"Error syncing topology structure from Redis: {e}")

def _sync_policy_from_redis(r_client, engine_obj, monitor_obj):
    try:
        raw_policy = r_client.get("control_plane:policy")
        if raw_policy:
            policy = json.loads(raw_policy)
            engine_obj.policy = policy
            rebalance = policy.get("rebalance_policy", {})
            engine_obj.rebalance_policy = rebalance
            engine_obj.enabled = rebalance.get("enabled", engine_obj.enabled)
            engine_obj.dry_run_mode = rebalance.get("dry_run_mode", engine_obj.dry_run_mode)
            engine_obj.min_hotspot_score = rebalance.get("min_hotspot_score_to_trigger", engine_obj.min_hotspot_score)
            engine_obj.min_hotspot_duration = rebalance.get("min_hotspot_duration_minutes", engine_obj.min_hotspot_duration)
            engine_obj.max_moves_per_hour = rebalance.get("max_volumes_moved_per_hour", engine_obj.max_moves_per_hour)
            engine_obj.max_concurrent_migrations = rebalance.get("max_concurrent_migrations", engine_obj.max_concurrent_migrations)
            
            safety = policy.get("safety_guardrails", {})
            monitor_obj.rollback_threshold_pct = safety.get("rollback_if_target_latency_increases_pct", monitor_obj.rollback_threshold_pct)
            monitor_obj.rollback_timeout_minutes = safety.get("rollback_timeout_minutes", monitor_obj.rollback_timeout_minutes)
    except Exception as e:
        logger.warning("Error syncing policy from Redis: %s", e)

def _sync_control_plane_state_from_redis(r_client, engine_obj, monitor_obj):
    try:
        raw_monitors = r_client.hgetall("control_plane:active_monitors")
        monitors = {}
        for aid, raw_mon in raw_monitors.items():
            mon = json.loads(raw_mon)
            mon["timestamp"] = pd.to_datetime(mon["timestamp"])
            monitors[aid] = mon
        
        monitor_obj.actions = monitors
        monitor_obj.total_actions = len(monitors)
        monitor_obj.rolled_back_count = sum(1 for m in monitors.values() if m["status"] == "rolled_back")

        raw_history = r_client.get("control_plane:action_history")
        if raw_history:
            history = json.loads(raw_history)
            for act in history:
                act["timestamp"] = pd.to_datetime(act["timestamp"])
            engine_obj.action_history = history
            
        raw_queue = r_client.get("control_plane:action_queue")
        if raw_queue:
            queue = json.loads(raw_queue)
            for q in queue:
                q["timestamp"] = pd.to_datetime(q["timestamp"])
            engine_obj.action_queue = queue

        raw_autoscale = r_client.get("control_plane:autoscale_state")
        if raw_autoscale:
            autoscale_state = json.loads(raw_autoscale)
            last_autoscale = autoscale_state.get("last_autoscale_time")
            engine_obj.last_autoscale_time = pd.to_datetime(last_autoscale) if last_autoscale else None
    except Exception as e:
        logger.warning("Error syncing control plane state from Redis: %s", e)

def _persist_control_plane_state(r_client, engine_obj, monitor_obj):
    if engine_obj is None or monitor_obj is None or r_client is None:
        return

    def _serialize_dict(d):
        serialized = {}
        for k, v in d.items():
            if isinstance(v, pd.Timestamp):
                serialized[k] = v.isoformat()
            elif isinstance(v, dict):
                serialized[k] = _serialize_dict(v)
            elif isinstance(v, list):
                serialized[k] = [
                    _serialize_dict(item) if isinstance(item, dict) else (item.isoformat() if isinstance(item, pd.Timestamp) else item)
                    for item in v
                ]
            else:
                serialized[k] = v
        return serialized

    try:
        serialized_history = [_serialize_dict(act) for act in engine_obj.action_history]
        serialized_monitors = {aid: _serialize_dict(mon) for aid, mon in monitor_obj.actions.items()}
        serialized_queue = [_serialize_dict(q) for q in engine_obj.action_queue]
        autoscale_state = {
            "last_autoscale_time": engine_obj.last_autoscale_time.isoformat() if engine_obj.last_autoscale_time else None
        }

        r_client.set("control_plane:action_history", json.dumps(serialized_history))
        r_client.set("control_plane:action_queue", json.dumps(serialized_queue))
        r_client.set("control_plane:autoscale_state", json.dumps(autoscale_state))
        r_client.delete("control_plane:active_monitors")
        if serialized_monitors:
            r_client.hset("control_plane:active_monitors", mapping={aid: json.dumps(mon) for aid, mon in serialized_monitors.items()})
    except Exception as e:
        logger.error("Failed to persist control plane state to Redis: %s", e)

def _persist_topology_to_redis_worker(r_client, hub_obj, volume_id: str):
    try:
        node_id = hub_obj.topology.get_node_of_volume(volume_id)
        if node_id:
            r_client.hset("topology:volume_to_node", volume_id, node_id)
        
        tier = hub_obj.topology.graph.nodes[volume_id].get("tier")
        if tier:
            r_client.hset("topology:volume_tier", volume_id, tier)

        limit = hub_obj.topology.graph.nodes[volume_id].get("iops_limit")
        if limit is not None:
            r_client.hset("topology:volume_iops_limit", volume_id, str(limit))
        else:
            r_client.hset("topology:volume_iops_limit", volume_id, "None")
    except Exception as e:
        logger.error("Failed to persist topology to Redis for volume %s: %s", volume_id, e)

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] StreamWorker: %(message)s")
logger = logging.getLogger("stream_worker")

REDIS_PORT = 6379


import atexit
_wsl_keepalive_proc = None

def _start_wsl_keepalive(host: str):
    """Start a background WSL session to prevent idle VM shutdown when host is WSL IP."""
    global _wsl_keepalive_proc
    if host != "127.0.0.1" and host != "localhost" and _wsl_keepalive_proc is None:
        import subprocess
        try:
            creationflags = 0
            try:
                creationflags = subprocess.CREATE_NO_WINDOW
            except AttributeError:
                pass
            _wsl_keepalive_proc = subprocess.Popen(
                ["wsl", "sleep", "infinity"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags
            )
            logger.info("Started WSL keepalive process to prevent idle VM shutdown (PID: %s)", _wsl_keepalive_proc.pid)
            atexit.register(_stop_wsl_keepalive)
        except Exception as e:
            logger.warning("Failed to start WSL keepalive process: %s", e)

def _stop_wsl_keepalive():
    """Stop the background WSL keepalive process cleanly on exit."""
    global _wsl_keepalive_proc
    if _wsl_keepalive_proc is not None:
        try:
            _wsl_keepalive_proc.terminate()
            _wsl_keepalive_proc.wait(timeout=1)
            logger.info("Stopped WSL keepalive process.")
        except Exception:
            try:
                _wsl_keepalive_proc.kill()
            except Exception:
                pass
        _wsl_keepalive_proc = None

def _detect_redis_host() -> str:
    """Auto-detect the best Redis host address (handles WSL2 networking)."""
    import subprocess, socket
    candidates = []
    try:
        result = subprocess.run(
            ["wsl", "hostname", "-I"],
            capture_output=True, text=True, timeout=3
        )
        wsl_ip = result.stdout.strip().split()[0] if result.returncode == 0 else None
        if wsl_ip:
            candidates.append(wsl_ip)
    except Exception:
        pass
    candidates.append("127.0.0.1")

    for host in candidates:
        try:
            with socket.create_connection((host, REDIS_PORT), timeout=2):
                logger.info("Detected Redis reachable at %s:%s", host, REDIS_PORT)
                return host
        except OSError:
            continue
    logger.warning("Could not detect Redis host. Defaulting to 127.0.0.1.")
    return "127.0.0.1"


REDIS_HOST = _detect_redis_host()


def setup_consumer_group(r) -> None:
    """Create Redis Streams consumer group if it doesn't exist."""
    try:
        r.xgroup_create("telemetry:stream", "cg_control_plane", id="0", mkstream=True)
        logger.info("Created consumer group 'cg_control_plane' on stream 'telemetry:stream'.")
    except Exception as e:
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group 'cg_control_plane' already exists.")
        else:
            logger.error(f"Error creating consumer group: {e}")

def run_worker():
    # 1. Connect to Redis with retry loop
    r = None
    import redis
    while True:
        try:
            logger.info("Attempting to connect to Redis on %s:%s...", REDIS_HOST, REDIS_PORT)
            
            # Start WSL keepalive if connecting to WSL IP
            _start_wsl_keepalive(REDIS_HOST)
            
            import socket
            socket_keepalive_options = {}
            if hasattr(socket, "TCP_KEEPIDLE"):
                socket_keepalive_options[socket.TCP_KEEPIDLE] = 10
            if hasattr(socket, "TCP_KEEPINTVL"):
                socket_keepalive_options[socket.TCP_KEEPINTVL] = 5
            if hasattr(socket, "TCP_KEEPCNT"):
                socket_keepalive_options[socket.TCP_KEEPCNT] = 3

            r = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=5,
                socket_keepalive=True,
                socket_keepalive_options=socket_keepalive_options,
                retry_on_timeout=True,
                health_check_interval=15,
            )
            r.ping()
            logger.info("Connected to Redis successfully!")
            break
        except Exception as e:
            logger.warning(
                "Redis server not reachable at %s:%s: %s. Retrying in 5 seconds. "
                "If you are using TCP fallback, this worker is not required; FastAPI handles ingestion directly.",
                REDIS_HOST,
                REDIS_PORT,
                e
            )
            time.sleep(5)

    # 2. Initialize InferenceHub
    logger.info("Initializing InferenceHub and loading ML models...")
    hub = InferenceHub(project_root=PROJECT_ROOT)
    logger.info("InferenceHub initialized successfully.")

    # Initialize telemetry outlier bounds
    bounds = load_or_create_bounds(PROJECT_ROOT)

    # Initialize control plane components
    rebalancer = Rebalancer()
    monitor = ActionMonitor(
        rollback_threshold_pct=hub.policy.get("safety_guardrails", {}).get("rollback_if_target_latency_increases_pct", 20.0),
        rollback_timeout_minutes=hub.policy.get("safety_guardrails", {}).get("rollback_timeout_minutes", 5.0)
    )
    engine = DecisionEngine(hub, rebalancer, monitor)

    # 3. Setup stream consumer group
    setup_consumer_group(r)

    # 4. Ingestion loop
    logger.info("Ingestion consumer group loop started. Listening for events...")
    
    last_trim_time = time.time()
    last_analyzed_time = {}
    
    while True:
        try:
            # Read new messages from consumer group
            # > indicates we want messages that have never been delivered to other consumers
            response = r.xreadgroup(
                groupname="cg_control_plane",
                consumername="worker_1",
                streams={"telemetry:stream": ">"},
                count=50,
                block=1000
            )

            if not response:
                continue

            for stream_name, messages in response:
                pipe = r.pipeline(transaction=False)

                # Sync dynamic policy, control plane state, and topology assignments from Redis first
                sync_topology_structure_from_redis(r, hub.topology)
                _sync_policy_from_redis(r, engine, monitor)
                _sync_control_plane_state_from_redis(r, engine, monitor)

                last_ts = None
                
                for msg_id, fields in messages:
                    # Clean/parse values from Redis string fields using the C++ parser or Python fallback
                    try:
                        json_str = json.dumps(fields)
                        event = parse_and_clip(json_str, bounds)
                    except Exception as e:
                        logger.error(f"Failed to parse and clip message {msg_id}: {e}")
                        pipe.xack("telemetry:stream", "cg_control_plane", msg_id)
                        continue

                    # Ensure type/fallback safety for all columns to not break downstream components
                    STRING_COLS = {"volume_id", "node_id", "timestamp", "workload_label"}
                    for k, v in list(event.items()):
                        if k in STRING_COLS:
                            event[k] = str(v) if v is not None else ""
                        elif v is None or v == "" or v == "None" or v == "NaN" or v == "nan":
                            event[k] = 0.0
                        else:
                            try:
                                event[k] = float(v)
                            except (ValueError, TypeError):
                                event[k] = 0.0

                    volume_id = event.get("volume_id")
                    timestamp_str = event.get("timestamp")

                    if not volume_id or not timestamp_str:
                        # Invalid event format
                        pipe.xack("telemetry:stream", "cg_control_plane", msg_id)
                        continue

                    # Parse timestamp to pandas Timestamp
                    ts = pd.to_datetime(timestamp_str)
                    event["timestamp"] = ts
                    if last_ts is None or ts > last_ts:
                        last_ts = ts

                    # A. Update local live features dataframe for time-series context
                    new_row = pd.DataFrame([event])
                    common_cols = [c for c in hub.features_df.columns if c in new_row.columns]
                    new_row = new_row[common_cols]
                    hub.live_features_df = pd.concat([hub.live_features_df, new_row], ignore_index=True)

                    # B. Update topology metrics
                    hub.topology.update_volume_metrics(volume_id, event)

                    # C. Run real-time inference (Workload classification & anomaly detection)
                    now = time.time()
                    should_analyze = (volume_id not in last_analyzed_time or (now - last_analyzed_time[volume_id] >= 30.0))
                    
                    analysis = None
                    if should_analyze:
                        try:
                            analysis = hub.analyze_volume(volume_id, ts)
                            last_analyzed_time[volume_id] = now
                        except Exception as ex:
                            logger.error(f"Inference failed for {volume_id} at {timestamp_str}: {ex}")
                            analysis = {
                                "volume_id": volume_id,
                                "timestamp": timestamp_str,
                                "workload_type": "Unknown",
                                "hotspot_score": 0.0,
                                "noisy_neighbor_victims": {},
                                "days_to_fill": {"warning_85pct_days": None, "critical_95pct_days": None},
                                "bandwidth_forecast_24h": {"p50_latency_us": [], "p90_latency_us": [], "p95_latency_us": []},
                                "latency_risk_score": 0.0
                            }

                    # D. Write latest state to Redis Hashes
                    metric_fields = {k: str(v) for k, v in event.items() if k != "timestamp"}
                    metric_fields["timestamp"] = timestamp_str
                    
                    pipe.hset(f"volume:{volume_id}:metrics", mapping=metric_fields)
                    if analysis is not None:
                        pipe.hset(f"volume:{volume_id}:analysis", "data", json.dumps(analysis))

                        # Evaluate volume in DecisionEngine
                        try:
                            action_result = engine.evaluate_volume(volume_id, ts, analysis)
                            if action_result:
                                status = action_result.get("status")
                                if status in ("executed", "queued"):
                                    _persist_control_plane_state(r, engine, monitor)
                                    _persist_topology_to_redis_worker(r, hub, volume_id)
                        except Exception as ex:
                            logger.error(f"Decision engine evaluation failed for {volume_id} at {timestamp_str}: {ex}")

                    # E. Write to history list (rolling window of last 100 entries per volume)
                    # Serializing timestamp as string
                    history_event = dict(event)
                    history_event["timestamp"] = timestamp_str
                    history_json = json.dumps(history_event)
                    
                    pipe.lpush(f"volume:{volume_id}:history", history_json)
                    pipe.ltrim(f"volume:{volume_id}:history", 0, 99)

                    # F. Acknowledge message processing completion
                    pipe.xack("telemetry:stream", "cg_control_plane", msg_id)

                if last_ts is not None:
                    try:
                        queued_results = engine.process_queued_actions(last_ts)
                        if queued_results:
                            if queued_results.get("migrations") or queued_results.get("autoscale"):
                                _persist_control_plane_state(r, engine, monitor)
                                for q_res in queued_results.get("migrations", []):
                                    _persist_topology_to_redis_worker(r, hub, q_res["volume_id"])
                    except Exception as ex:
                        logger.error(f"Processing queued actions failed at {last_ts}: {ex}")

                pipe.execute()

            # Prevent live_features_df memory leak (trim df if it exceeds 15,000 rows, keeping last 200 per volume)
            current_time = time.time()
            if len(hub.live_features_df) > 15000 and (current_time - last_trim_time > 60):
                hub.live_features_df = hub.live_features_df.groupby("volume_id").tail(200).reset_index(drop=True)
                last_trim_time = current_time
                logger.info(f"Trimmed local live features dataframe. Size: {len(hub.live_features_df)} rows.")

        except Exception as e:
            logger.error(f"Error in main consumption loop: {e}")
            time.sleep(2)

if __name__ == "__main__":
    try:
        run_worker()
    except KeyboardInterrupt:
        logger.info("Stream worker stopped by user.")
