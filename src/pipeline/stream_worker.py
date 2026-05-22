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

from src.control_plane.inference_hub import InferenceHub

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
                
                for msg_id, fields in messages:
                    # Clean/parse values from Redis string fields
                    STRING_COLS = {"volume_id", "node_id", "timestamp", "workload_label"}
                    event = {}
                    for k, v in fields.items():
                        if k in STRING_COLS:
                            event[k] = v
                        elif v is None or v == "" or v == "None" or v == "NaN" or v == "nan":
                            event[k] = 0.0
                        else:
                            try:
                                if "." in v:
                                    event[k] = float(v)
                                else:
                                    event[k] = int(v)
                            except ValueError:
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

                    # E. Write to history list (rolling window of last 100 entries per volume)
                    # Serializing timestamp as string
                    history_event = dict(event)
                    history_event["timestamp"] = timestamp_str
                    history_json = json.dumps(history_event)
                    
                    pipe.lpush(f"volume:{volume_id}:history", history_json)
                    pipe.ltrim(f"volume:{volume_id}:history", 0, 99)

                    # F. Acknowledge message processing completion
                    pipe.xack("telemetry:stream", "cg_control_plane", msg_id)

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
