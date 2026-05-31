"""
api/main.py

FastAPI Application Layer (HPE Blueprint Phase 6)
===================================================
Provides REST API access to the storage control plane, telemetry analysis,
what-if simulators, decision engines, and system performance indicators.
"""

import sys
import time
import json
import shap
import uvicorn
import asyncio
import uuid
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Sibling model imports
ANOMALY_DIR = PROJECT_ROOT / "src" / "models" / "anomaly"
if str(ANOMALY_DIR) not in sys.path:
    sys.path.insert(0, str(ANOMALY_DIR))

from src.control_plane import InferenceHub, Rebalancer, ActionMonitor, DecisionEngine, WhatIfSimulator
from src.pipeline.telemetry_parser import parse_and_clip, load_or_create_bounds
from api.schemas.models import (
    SimulateCapacityRequest,
    SimulateMigrateRequest,
    SimulateQosRequest,
    SimulateTierRequest,
    PolicyUpdateRequest,
    RebalanceRequest,
    RollbackRequest
)

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api.main")

app = FastAPI(
    title="HPE Storage Control Plane API",
    description="ML-Powered Control Plane for Workload Classification, Hotspot Detection, and Rebalancing.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    import redis
except ImportError:  # Redis server/client is optional because TCP fallback is supported.
    redis = None

# Global engines
hub: Optional[InferenceHub] = None
rebalancer: Optional[Rebalancer] = None
monitor: Optional[ActionMonitor] = None
engine: Optional[DecisionEngine] = None
simulator: Optional[WhatIfSimulator] = None
explainer: Optional[shap.TreeExplainer] = None
bounds: Dict[str, Any] = {}

# Active SSE connections
active_streams: Dict[str, List[asyncio.Queue]] = {}

# Telemetry bus configuration
REDIS_PORT = 6379
TCP_FALLBACK_HOST = "127.0.0.1"
TCP_FALLBACK_PORT = 9000
# REDIS_HOST is detected lazily in startup_event() so WSL2 networking is ready
REDIS_HOST = "127.0.0.1"


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
    """Auto-detect the best Redis host address (handles WSL2 networking).

    Priority: WSL2 VM IP > 127.0.0.1.
    'localhost' is deliberately excluded because Windows resolves it
    through a flaky WSL2 NAT layer that drops long-lived TCP connections.
    """
    import os
    import subprocess, socket as _sock
    env_host = os.environ.get("REDIS_HOST", "").strip()
    if env_host:
        logger.info("Using REDIS_HOST from environment: %s", env_host)
        return env_host
    candidates = []
    try:
        result = subprocess.run(
            ["wsl", "hostname", "-I"],
            capture_output=True, text=True, timeout=5
        )
        wsl_ip = result.stdout.strip().split()[0] if result.returncode == 0 else None
        if wsl_ip:
            candidates.append(wsl_ip)
    except Exception:
        pass
    candidates.append("127.0.0.1")

    for host in candidates:
        try:
            with _sock.create_connection((host, REDIS_PORT), timeout=2):
                logger.info("Detected Redis reachable at %s:%s", host, REDIS_PORT)
                return host
        except OSError:
            logger.debug("Redis not reachable at %s:%s, trying next candidate.", host, REDIS_PORT)
            continue
    logger.warning("Could not detect Redis host. Defaulting to 127.0.0.1.")
    return "127.0.0.1"

# Redis Client / TCP fallback state
r: Optional[Any] = None
use_redis: bool = False
redis_error: Optional[str] = None
tcp_server: Optional[Any] = None

# Rate limiting for volume analyses (in seconds)
last_analyzed_time: Dict[str, float] = {}
analysis_tasks: Dict[str, asyncio.Task] = {}

class LiveTelemetryState:
    """In-memory live telemetry store separate from the historical/model dataframe."""

    def __init__(self, history_limit: int = 500) -> None:
        self.history_limit = history_limit
        self.latest_by_volume: Dict[str, Dict[str, Any]] = {}
        self.history_by_volume: Dict[str, deque] = {}
        self.events_received = 0
        self.current_tick: Optional[pd.Timestamp] = None
        self.latest_complete_tick: Optional[pd.Timestamp] = None
        self._expected_volume_count: Optional[int] = None
        self.first_received_at: Optional[pd.Timestamp] = None
        self.last_received_at: Optional[pd.Timestamp] = None

    @staticmethod
    def _copy_event(event: Dict[str, Any]) -> Dict[str, Any]:
        copied = dict(event)
        copied["timestamp"] = pd.to_datetime(copied["timestamp"])
        return copied

    def record(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        volume_id = event.get("volume_id")
        if not volume_id or "timestamp" not in event:
            return None

        copied = self._copy_event(event)
        volume_id = str(volume_id)
        copied["volume_id"] = volume_id

        self.latest_by_volume[volume_id] = copied
        self.history_by_volume.setdefault(volume_id, deque(maxlen=self.history_limit)).append(copied)
        self.events_received += 1
        self.current_tick = copied["timestamp"]

        # Track latest complete tick (all expected volumes have reported for this timestamp)
        if self._expected_volume_count is not None:
            tick_count = sum(
                1 for row in self.latest_by_volume.values()
                if row.get("timestamp") == self.current_tick
            )
            if tick_count >= self._expected_volume_count:
                self.latest_complete_tick = self.current_tick

        now = pd.Timestamp.now()
        if self.first_received_at is None:
            self.first_received_at = now
        self.last_received_at = now
        return copied

    def set_expected_volume_count(self, count: int) -> None:
        """Set the number of volumes expected per tick (e.g. 50)."""
        self._expected_volume_count = count

    def latest_rows(self) -> List[Dict[str, Any]]:
        return list(self.latest_by_volume.values())

    def current_tick_rows(self) -> List[Dict[str, Any]]:
        if self.current_tick is None:
            return []
        return [
            row for row in self.latest_by_volume.values()
            if row.get("timestamp") == self.current_tick
        ]

    def history(self, volume_id: str, limit: int) -> List[Dict[str, Any]]:
        rows = list(self.history_by_volume.get(volume_id, []))
        if limit:
            rows = rows[-limit:]
        return rows

    def status(self, expected_volume_count: Optional[int] = None) -> Dict[str, Any]:
        current_rows = self.current_tick_rows()
        current_tick_volume_count = len(current_rows)
        return {
            "events_received": self.events_received,
            "live_volume_count": len(self.latest_by_volume),
            "expected_volume_count": expected_volume_count,
            "current_tick": self.current_tick.isoformat() if self.current_tick is not None else None,
            "latest_complete_tick": self.latest_complete_tick.isoformat() if self.latest_complete_tick is not None else None,
            "current_tick_volume_count": current_tick_volume_count,
            "current_tick_complete": (
                current_tick_volume_count >= expected_volume_count
                if expected_volume_count is not None else False
            ),
            "first_received_at": self.first_received_at.isoformat() if self.first_received_at is not None else None,
            "last_received_at": self.last_received_at.isoformat() if self.last_received_at is not None else None,
        }

class RedisBackedCache(dict):
    """Custom dictionary that dynamically reads live analysis data from Redis, falling back to local memory."""
    _redis_error_logged_at: float = 0.0  # rate-limit Redis error logging

    def __getitem__(self, key):
        global use_redis, r, hub
        if use_redis and r is not None:
            try:
                data = r.hget(f"volume:{key}:analysis", "data")
                if data:
                    return json.loads(data)
            except Exception as e:
                # Rate-limit error logging to once every 30 seconds
                now = time.time()
                if now - RedisBackedCache._redis_error_logged_at > 30.0:
                    logger.warning("Redis cache read failing (will retry silently): %s", e)
                    RedisBackedCache._redis_error_logged_at = now
        try:
            return super().__getitem__(key)
        except KeyError:
            if hub is not None:
                val = hub.analyze_volume(key)
                super().__setitem__(key, val)
                return val
            raise

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        if hub is not None:
            all_vols = hub.features_df["volume_id"].unique()
        else:
            all_vols = super().keys()
        return [(v, self[v]) for v in all_vols]

    def values(self):
        if hub is not None:
            all_vols = hub.features_df["volume_id"].unique()
        else:
            all_vols = super().keys()
        return [self[v] for v in all_vols]

    def __contains__(self, key):
        if hub is not None:
            return key in hub.features_df["volume_id"].unique()
        return super().__contains__(key)

    def __iter__(self):
        if hub is not None:
            return iter(hub.features_df["volume_id"].unique())
        return super().__iter__()

    def __len__(self):
        if hub is not None:
            return len(hub.features_df["volume_id"].unique())
        return super().__len__()

# In-memory cache for live telemetry state (warmup)
cached_analysis = RedisBackedCache()
live_state = LiveTelemetryState()

def sync_topology_from_redis():
    """Synchronizes in-memory topology metrics and structure with latest metrics in Redis."""
    global use_redis, r, hub
    STRING_COLS = {"volume_id", "node_id", "timestamp", "workload_label"}
    if use_redis and r is not None and hub is not None:
        try:
            # 1. Sync metrics
            for vol_id in hub.features_df["volume_id"].unique():
                metrics_data = r.hgetall(f"volume:{vol_id}:metrics")
                if metrics_data:
                    typed_metrics = {}
                    for k, v in metrics_data.items():
                        if k in STRING_COLS:
                            typed_metrics[k] = v
                        elif v is None or v == "" or v == "None":
                            typed_metrics[k] = 0.0
                        else:
                            try:
                                if "." in v:
                                    typed_metrics[k] = float(v)
                                else:
                                    typed_metrics[k] = int(v)
                            except ValueError:
                                typed_metrics[k] = v
                    hub.topology.update_volume_metrics(vol_id, typed_metrics)

            # 2. Sync topology structure assignments
            assignments = r.hgetall("topology:volume_to_node")
            for vol_id, target_node in assignments.items():
                if hub.topology.graph.has_node(vol_id) and target_node:
                    source_node = hub.topology.get_node_of_volume(vol_id)
                    if source_node != target_node:
                        logger.info("Syncing topology structure: moving %s from %s to %s (from Redis)", vol_id, source_node, target_node)
                        if source_node and hub.topology.graph.has_edge(vol_id, source_node):
                            hub.topology.graph.remove_edge(vol_id, source_node)
                        hub.topology.graph.add_edge(vol_id, target_node, relation="resides_on")
                        hub.topology._volume_to_node[vol_id] = target_node
                        if source_node and vol_id in hub.topology._node_volumes.get(source_node, []):
                            hub.topology._node_volumes[source_node].remove(vol_id)
                        if target_node not in hub.topology._node_volumes:
                            hub.topology._node_volumes[target_node] = []
                        if vol_id not in hub.topology._node_volumes[target_node]:
                            hub.topology._node_volumes[target_node].append(vol_id)
                        hub.topology.graph.nodes[vol_id]["node_id"] = target_node

            # 3. Sync storage tiers
            tiers = r.hgetall("topology:volume_tier")
            for vol_id, tier in tiers.items():
                if hub.topology.graph.has_node(vol_id) and tier:
                    hub.topology.graph.nodes[vol_id]["tier"] = tier

            # 4. Sync QoS IOPS limits
            limits = r.hgetall("topology:volume_iops_limit")
            for vol_id, limit_str in limits.items():
                if hub.topology.graph.has_node(vol_id):
                    if limit_str == "None" or limit_str == "":
                        hub.topology.graph.nodes[vol_id].pop("iops_limit", None)
                    else:
                        try:
                            hub.topology.graph.nodes[vol_id]["iops_limit"] = float(limit_str)
                        except ValueError:
                            pass
        except Exception as e:
            logger.error(f"Error syncing topology from Redis: {e}")

def get_expected_volume_count() -> Optional[int]:
    if hub is None:
        return None
    try:
        return len(hub.topology.all_volumes())
    except Exception:
        return int(hub.features_df["volume_id"].nunique())

def _create_redis_client() -> Any:
    """Create a new Redis client with resilient WSL2-friendly settings."""
    detected_host = _detect_redis_host()
    logger.info("Creating Redis client targeting %s:%s", detected_host, REDIS_PORT)
    
    # Start WSL keepalive if connecting to WSL IP
    _start_wsl_keepalive(detected_host)
    
    import socket
    socket_keepalive_options = {}
    if hasattr(socket, "TCP_KEEPIDLE"):
        socket_keepalive_options[socket.TCP_KEEPIDLE] = 10
    if hasattr(socket, "TCP_KEEPINTVL"):
        socket_keepalive_options[socket.TCP_KEEPINTVL] = 5
    if hasattr(socket, "TCP_KEEPCNT"):
        socket_keepalive_options[socket.TCP_KEEPCNT] = 3

    return redis.Redis(
        host=detected_host,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=5,
        socket_keepalive=True,
        socket_keepalive_options=socket_keepalive_options,
        retry_on_timeout=True,
        health_check_interval=15,
    )


async def redis_stream_listener():
    """Background task reading from Redis Stream to push live SSE metrics.

    Automatically reconnects when the WSL2 NAT layer drops the connection.
    """
    global r, active_streams
    last_id = "$"
    STRING_COLS = {"volume_id", "node_id", "timestamp", "workload_label"}
    consecutive_errors = 0
    logger.info("Started FastAPI Redis Stream listener for live SSE updates.")
    while True:
        try:
            if r is None:
                await asyncio.sleep(2)
                continue
            res = await asyncio.to_thread(
                r.xread,
                streams={"telemetry:stream": last_id},
                count=10,
                block=1000
            )
            consecutive_errors = 0  # reset on success
            if not res:
                continue
            for stream_name, messages in res:
                for msg_id, fields in messages:
                    last_id = msg_id
                    event = {}
                    for k, v in fields.items():
                        if k in STRING_COLS:
                            event[k] = v
                        elif v is None or v == "" or v == "None":
                            event[k] = 0.0
                        else:
                            try:
                                if "." in v:
                                    event[k] = float(v)
                                else:
                                    event[k] = int(v)
                            except ValueError:
                                event[k] = v
                    
                    volume_id = event.get("volume_id")
                    if volume_id:
                        timestamp_str = event.get("timestamp")
                        event["timestamp"] = pd.to_datetime(timestamp_str)
                        live_state.record(event)
                        if hub is not None:
                            hub.topology.update_volume_metrics(volume_id, event)
                        if volume_id in active_streams:
                            payload = {
                                "timestamp": str(timestamp_str),
                                "total_iops": float(event.get("total_iops", 0.0) or 0.0),
                                "avg_latency_us": float(event.get("avg_latency_us", 0.0) or 0.0),
                                "queue_depth": float(event.get("queue_depth", 0.0) or 0.0),
                                "capacity_used_pct": float(event.get("capacity_used_pct", 0.0) or 0.0)
                            }
                            for q in active_streams[volume_id]:
                                await q.put(payload)
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors <= 3 or consecutive_errors % 30 == 0:
                logger.error("redis_stream_listener error #%d: %s", consecutive_errors, e)
            # After 3 consecutive failures, try to rebuild the connection
            if consecutive_errors >= 3:
                logger.warning("Rebuilding Redis connection after %d consecutive errors...", consecutive_errors)
                try:
                    r = _create_redis_client()
                    r.ping()
                    logger.info("Redis reconnection successful.")
                    consecutive_errors = 0
                except Exception as re_err:
                    logger.warning("Redis reconnection failed: %s. Will retry in 5s.", re_err)
            await asyncio.sleep(min(5, 2 * consecutive_errors))
MAX_ROWS_PER_VOLUME = 500  # model inference context window per volume

def _append_feature_rows(rows: List[Dict[str, Any]]) -> None:
    """Append live telemetry rows to hub.live_features_df for ML model inference.

    The historical hub.features_df remains immutable (Parquet snapshot).
    InferenceHub data-access methods search live_features_df first,
    then fall back to features_df.
    """
    if hub is None or not rows:
        return
    try:
        new_df = pd.DataFrame(rows)
        if "timestamp" in new_df.columns:
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
        # Only keep columns that exist in the original features dataframe
        common_cols = [c for c in hub.features_df.columns if c in new_df.columns]
        new_df = new_df[common_cols]
        hub.live_features_df = pd.concat([hub.live_features_df, new_df], ignore_index=True)

        # Periodic trim: keep only the last MAX_ROWS_PER_VOLUME rows per volume
        if len(hub.live_features_df) > MAX_ROWS_PER_VOLUME * 60:
            hub.live_features_df = (
                hub.live_features_df
                .sort_values("timestamp")
                .groupby("volume_id", group_keys=False)
                .tail(MAX_ROWS_PER_VOLUME)
                .reset_index(drop=True)
            )
    except Exception as e:
        logger.error("_append_feature_rows failed: %s", e)

async def _analyze_and_cache_volume(volume_id: str, ts: pd.Timestamp) -> None:
    """Run heavier per-volume inference outside the ingestion hot path."""
    global analysis_tasks, hub, engine, use_redis
    try:
        if hub is not None:
            analysis = await asyncio.to_thread(hub.analyze_volume, volume_id, ts)
            cached_analysis[volume_id] = analysis
            
            if engine is not None and engine.enabled:
                action_result = await asyncio.to_thread(engine.evaluate_volume, volume_id, ts, analysis)
                if action_result:
                    _persist_control_plane_state()
                    if use_redis:
                        _persist_topology_to_redis(volume_id)
                
                queued_results = await asyncio.to_thread(engine.process_queued_actions, ts)
                if queued_results:
                    if queued_results.get("migrations") or queued_results.get("autoscale"):
                        _persist_control_plane_state()
                        if use_redis:
                            for q_res in queued_results.get("migrations", []):
                                _persist_topology_to_redis(q_res["volume_id"])
    except Exception as e:
        logger.error("Background analysis failed for %s at %s: %s", volume_id, ts, e)
    finally:
        analysis_tasks.pop(volume_id, None)

def _schedule_volume_analysis(volume_id: str, ts: pd.Timestamp) -> None:
    """Throttle background analysis so telemetry ingestion keeps the API responsive."""
    now = time.time()
    if now - last_analyzed_time.get(volume_id, 0.0) < 30.0:
        return
    task = analysis_tasks.get(volume_id)
    if task is not None and not task.done():
        return

    last_analyzed_time[volume_id] = now
    analysis_tasks[volume_id] = asyncio.create_task(_analyze_and_cache_volume(volume_id, ts))

async def handle_tcp_client(reader, writer):
    """TCP Handler for playback agent streaming metrics directly to FastAPI when Redis is offline."""
    global hub, cached_analysis, active_streams, last_analyzed_time, bounds
    
    # Columns that should remain as strings
    STRING_COLS = {"volume_id", "node_id", "timestamp", "workload_label"}
    
    logger.info("TCP fallback client connected.")
    pending_rows: List[Dict[str, Any]] = []
    pending_analysis: Dict[str, pd.Timestamp] = {}
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                raw_line = line.decode("utf-8").strip()
                data = parse_and_clip(raw_line, bounds)
                volume_id = data.get("volume_id")
                timestamp_str = data.get("timestamp")
                if volume_id and timestamp_str:
                    ts = pd.to_datetime(timestamp_str)
                    
                    # Ensure type/fallback safety for all columns to not break downstream components
                    event = {}
                    for k, v in data.items():
                        if k in STRING_COLS:
                            event[k] = str(v) if v is not None else ""
                        elif v is None or v == "" or v == "None" or v == "NaN" or v == "nan":
                            event[k] = 0.0
                        else:
                            try:
                                event[k] = float(v)
                            except (ValueError, TypeError):
                                event[k] = 0.0
                    
                    event["timestamp"] = ts
                    live_state.record(event)
                    
                    pending_rows.append(event)
                    pending_analysis[volume_id] = ts
                    if len(pending_rows) >= 250:
                        _append_feature_rows(pending_rows)
                        pending_rows.clear()
                        for pending_volume_id, pending_ts in pending_analysis.items():
                            _schedule_volume_analysis(pending_volume_id, pending_ts)
                        pending_analysis.clear()
                    
                    # Update topology metrics
                    hub.topology.update_volume_metrics(volume_id, event)
                    
                    # Push to SSE streams
                    if volume_id in active_streams:
                        payload = {
                            "timestamp": timestamp_str,
                            "total_iops": float(event.get("total_iops", 0.0) or 0.0),
                            "avg_latency_us": float(event.get("avg_latency_us", 0.0) or 0.0),
                            "queue_depth": float(event.get("queue_depth", 0.0) or 0.0),
                            "capacity_used_pct": float(event.get("capacity_used_pct", 0.0) or 0.0)
                        }
                        for q in active_streams[volume_id]:
                            await q.put(payload)
            except Exception as ex:
                logger.error(f"Error processing TCP record: {ex}")
    except Exception as e:
        logger.error(f"TCP fallback client error: {e}")
    finally:
        _append_feature_rows(pending_rows)
        for pending_volume_id, pending_ts in pending_analysis.items():
            _schedule_volume_analysis(pending_volume_id, pending_ts)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        logger.info("TCP fallback client disconnected.")

async def _warmup_cache():
    """Pre-warm the analysis cache in the background so startup is non-blocking."""
    global hub, cached_analysis, last_analyzed_time
    all_vols = hub.features_df["volume_id"].unique()
    latest_ts = hub.features_df["timestamp"].max()
    logger.info(f"Background cache warmup started for {len(all_vols)} volumes...")
    
    # Pre-populate last_analyzed_time to prevent immediate CPU overload on stream start
    now = time.time()
    for vol in all_vols:
        last_analyzed_time[vol] = now

    for i, vol in enumerate(all_vols):
        try:
            # Offload heavy ML inference to a thread pool
            cached_analysis[vol] = await asyncio.to_thread(hub.analyze_volume, vol, latest_ts)
        except Exception as e:
            logger.error(f"Error pre-warming cache for volume {vol}: {e}")
        # Yield control to the event loop every 5 volumes so the server stays responsive
        if (i + 1) % 5 == 0:
            await asyncio.sleep(0)
    logger.info("Background cache warmup complete.")

async def _start_tcp_fallback_server() -> None:
    """Bind the TCP fallback listener and keep the server object alive."""
    global tcp_server
    tcp_server = await asyncio.start_server(
        handle_tcp_client,
        TCP_FALLBACK_HOST,
        TCP_FALLBACK_PORT
    )
    sockets = tcp_server.sockets or []
    bound = ", ".join(str(sock.getsockname()) for sock in sockets) or f"{TCP_FALLBACK_HOST}:{TCP_FALLBACK_PORT}"
    logger.info("TCP fallback server listening on %s.", bound)


def _load_control_plane_state():
    """Load rebalance history and monitors from JSON file and Redis (if available)."""
    global engine, monitor, use_redis, r
    # 1. Load from local JSON first
    history_file = PROJECT_ROOT / "data" / "processed" / "rebalance_history.json"
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
                history = data.get("action_history", [])
                monitors = data.get("active_monitors", {})
                autoscale_state = data.get("autoscale_state", {})
                
                # Deserialize timestamps
                for act in history:
                    act["timestamp"] = pd.to_datetime(act["timestamp"])
                for aid, mon in monitors.items():
                    mon["timestamp"] = pd.to_datetime(mon["timestamp"])
                    
                if engine is not None:
                    engine.action_history = history
                    last_autoscale = autoscale_state.get("last_autoscale_time")
                    engine.last_autoscale_time = pd.to_datetime(last_autoscale) if last_autoscale else None
                if monitor is not None:
                    monitor.actions = monitors
                    monitor.total_actions = len(monitors)
                    monitor.rolled_back_count = sum(1 for m in monitors.values() if m["status"] == "rolled_back")
            logger.info("Successfully loaded rebalance history from JSON file.")
        except Exception as e:
            logger.error("Failed to load rebalance history from JSON file: %s", e)

    # 2. Synchronize/Overwrite from Redis if active
    if use_redis and r is not None:
        try:
            raw_history = r.get("control_plane:action_history")
            if raw_history:
                history = json.loads(raw_history)
                for act in history:
                    act["timestamp"] = pd.to_datetime(act["timestamp"])
                if engine is not None:
                    engine.action_history = history
            
            raw_monitors = r.hgetall("control_plane:active_monitors")
            if raw_monitors:
                monitors = {}
                for aid, raw_mon in raw_monitors.items():
                    mon = json.loads(raw_mon)
                    mon["timestamp"] = pd.to_datetime(mon["timestamp"])
                    monitors[aid] = mon
                if monitor is not None:
                    monitor.actions = monitors
                    monitor.total_actions = len(monitors)
                    monitor.rolled_back_count = sum(1 for m in monitors.values() if m["status"] == "rolled_back")
            
            raw_queue = r.get("control_plane:action_queue")
            if raw_queue and engine is not None:
                queue = json.loads(raw_queue)
                for q in queue:
                    q["timestamp"] = pd.to_datetime(q["timestamp"])
                engine.action_queue = queue

            raw_autoscale = r.get("control_plane:autoscale_state")
            if raw_autoscale and engine is not None:
                autoscale_state = json.loads(raw_autoscale)
                last_autoscale = autoscale_state.get("last_autoscale_time")
                engine.last_autoscale_time = pd.to_datetime(last_autoscale) if last_autoscale else None
                
            logger.info("Successfully synchronized control plane state from Redis.")
        except Exception as e:
            logger.error("Failed to load control plane state from Redis: %s", e)


def _persist_control_plane_state():
    """Save rebalance history and monitors to JSON file and Redis (if available)."""
    global engine, monitor, use_redis, r
    if engine is None or monitor is None:
        return

    # Deep serialize timestamps to ISO format
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

    serialized_history = [_serialize_dict(act) for act in engine.action_history]
    serialized_monitors = {aid: _serialize_dict(mon) for aid, mon in monitor.actions.items()}
    serialized_queue = [_serialize_dict(q) for q in engine.action_queue]
    autoscale_state = {
        "last_autoscale_time": engine.last_autoscale_time.isoformat() if engine.last_autoscale_time else None
    }

    # Save to local JSON
    history_file = PROJECT_ROOT / "data" / "processed" / "rebalance_history.json"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp_file = history_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump({
                "action_history": serialized_history,
                "active_monitors": serialized_monitors,
                "action_queue": serialized_queue,
                "autoscale_state": autoscale_state
            }, f, indent=2)
        import os
        if history_file.exists():
            os.replace(str(temp_file), str(history_file))
        else:
            temp_file.rename(history_file)
    except Exception as e:
        logger.error("Failed to write rebalance history to JSON: %s", e)

    # Save to Redis if available
    if use_redis and r is not None:
        try:
            r.set("control_plane:action_history", json.dumps(serialized_history))
            r.set("control_plane:action_queue", json.dumps(serialized_queue))
            r.set("control_plane:autoscale_state", json.dumps(autoscale_state))
            r.delete("control_plane:active_monitors")
            if serialized_monitors:
                r.hset("control_plane:active_monitors", mapping={aid: json.dumps(mon) for aid, mon in serialized_monitors.items()})
        except Exception as e:
            logger.error("Failed to persist control plane state to Redis: %s", e)


def _sync_monitors_from_redis():
    """Sync monitors and history from Redis in background loop."""
    global monitor, engine, r
    if r is None or monitor is None or engine is None:
        return
    try:
        raw_monitors = r.hgetall("control_plane:active_monitors")
        monitors = {}
        for aid, raw_mon in raw_monitors.items():
            mon = json.loads(raw_mon)
            mon["timestamp"] = pd.to_datetime(mon["timestamp"])
            monitors[aid] = mon
        
        monitor.actions = monitors
        monitor.total_actions = len(monitors)
        monitor.rolled_back_count = sum(1 for m in monitors.values() if m["status"] == "rolled_back")

        raw_history = r.get("control_plane:action_history")
        if raw_history:
            history = json.loads(raw_history)
            for act in history:
                act["timestamp"] = pd.to_datetime(act["timestamp"])
            engine.action_history = history
            
        raw_queue = r.get("control_plane:action_queue")
        if raw_queue:
            queue = json.loads(raw_queue)
            for q in queue:
                q["timestamp"] = pd.to_datetime(q["timestamp"])
            engine.action_queue = queue
        raw_autoscale = r.get("control_plane:autoscale_state")
        if raw_autoscale:
            autoscale_state = json.loads(raw_autoscale)
            last_autoscale = autoscale_state.get("last_autoscale_time")
            engine.last_autoscale_time = pd.to_datetime(last_autoscale) if last_autoscale else None
    except Exception as e:
        logger.warning("Error syncing monitors from Redis in background loop: %s", e)


def _persist_topology_to_redis(volume_id: str):
    """Save dynamic topology adjustments of a volume to Redis."""
    global r, use_redis, hub
    if not use_redis or r is None or hub is None:
        return
    try:
        node_id = hub.topology.get_node_of_volume(volume_id)
        if node_id:
            r.hset("topology:volume_to_node", volume_id, node_id)
        
        tier = hub.topology.graph.nodes[volume_id].get("tier")
        if tier:
            r.hset("topology:volume_tier", volume_id, tier)

        limit = hub.topology.graph.nodes[volume_id].get("iops_limit")
        r.hset("topology:volume_iops_limit", volume_id, str(limit))
    except Exception as e:
        logger.error("Failed to persist topology to Redis for volume %s: %s", volume_id, e)


async def action_monitor_loop():
    """Background monitoring loop to check target latencies and trigger rollbacks."""
    global monitor, rebalancer, hub, use_redis, r, engine
    logger.info("Started FastAPI rebalance action monitoring loop.")
    while True:
        try:
            await asyncio.sleep(5)
            if monitor is None or rebalancer is None or hub is None:
                continue

            if use_redis and r is not None:
                _sync_monitors_from_redis()

            active_ids = [aid for aid, act in monitor.actions.items() if act["status"] == "monitoring"]
            if not active_ids:
                continue

            for aid in active_ids:
                action = monitor.actions[aid]
                vol_id = action["action_state"]["volume_id"]

                current_latency = 0.0
                live_event = live_state.latest_by_volume.get(vol_id)
                if live_event is not None:
                    current_latency = float(live_event.get("avg_latency_us", 0.0) or 0.0)
                else:
                    metrics_dict = hub.topology._volume_metrics.get(vol_id, {})
                    current_latency = float(metrics_dict.get("avg_latency_us", 0.0) or 0.0)

                elapsed_min = (pd.Timestamp.now() - action["timestamp"]).total_seconds() / 60.0

                new_status = monitor.update_metrics(
                    action_id=aid,
                    current_latency=current_latency,
                    elapsed_minutes=elapsed_min,
                    rebalancer=rebalancer,
                    topology=hub.topology
                )

                if new_status != "monitoring":
                    for h_action in engine.action_history:
                        if h_action.get("action_id") == aid:
                            h_action["status"] = new_status
                            break
                    _persist_control_plane_state()
                    if use_redis and r is not None:
                        _persist_topology_to_redis(vol_id)
        except Exception as e:
            logger.error("Error in action_monitor_loop: %s", e)


@app.on_event("startup")
async def startup_event():
    global hub, rebalancer, monitor, engine, simulator, explainer, cached_analysis, r, use_redis, redis_error, bounds, REDIS_HOST
    logger.info("Initializing Control Plane engines and loading ML models...")
    
    bounds = load_or_create_bounds(PROJECT_ROOT)
    hub = InferenceHub(project_root=PROJECT_ROOT)
    rebalancer = Rebalancer()
    monitor = ActionMonitor(
        rollback_threshold_pct=hub.policy.get("safety_guardrails", {}).get("rollback_if_target_latency_increases_pct", 20.0),
        rollback_timeout_minutes=hub.policy.get("safety_guardrails", {}).get("rollback_timeout_minutes", 5.0)
    )
    engine = DecisionEngine(hub, rebalancer, monitor)
    simulator = WhatIfSimulator(hub)
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(hub.classifier)

    # Wire expected volume count for complete-tick tracking
    expected_count = get_expected_volume_count()
    if expected_count:
        live_state.set_expected_volume_count(expected_count)
    
    # Detect Redis host NOW (not at import time) so WSL2 networking is fully ready.
    REDIS_HOST = _detect_redis_host()

    import os
    try:
        redis_retry_attempts = int(os.environ.get("REDIS_RETRY_ATTEMPTS", "5"))
    except ValueError:
        redis_retry_attempts = 5
    redis_retry_attempts = max(1, redis_retry_attempts)
    try:
        redis_retry_delay = float(os.environ.get("REDIS_RETRY_DELAY", "3"))
    except ValueError:
        redis_retry_delay = 3.0

    # Try to connect to Redis. If unavailable, the API owns a direct TCP fallback.
    redis_error = None
    if redis is None:
        use_redis = False
        redis_error = "Python package 'redis' is not installed."
        logger.warning("%s Activating TCP fallback mode on port %s.", redis_error, TCP_FALLBACK_PORT)
    else:
        for attempt in range(1, redis_retry_attempts + 1):
            try:
                r = _create_redis_client()
                r.ping()
                use_redis = True
                logger.info("Successfully connected to Redis at %s:%s. Redis mode active.", REDIS_HOST, REDIS_PORT)
                
                # Sync topology structure from Redis on start
                sync_topology_from_redis()
                break
            except Exception as e:
                redis_error = str(e)
                use_redis = False
                r = None
                if attempt < redis_retry_attempts:
                    logger.warning(
                        "Redis connect attempt %d/%d failed: %s. Retrying in %ss...",
                        attempt,
                        redis_retry_attempts,
                        redis_error,
                        redis_retry_delay
                    )
                    await asyncio.sleep(redis_retry_delay)
                else:
                    logger.warning(
                        "Could not connect to Redis at %s:%s: %s. Activating TCP fallback mode on port %s.",
                        REDIS_HOST,
                        REDIS_PORT,
                        redis_error,
                        TCP_FALLBACK_PORT
                    )

    # Load control plane state
    _load_control_plane_state()

    # Start live stream background tasks
    if use_redis:
        asyncio.create_task(redis_stream_listener())
    else:
        try:
            await _start_tcp_fallback_server()
        except OSError as e:
            logger.error(
                "Could not bind TCP fallback server on %s:%s: %s",
                TCP_FALLBACK_HOST,
                TCP_FALLBACK_PORT,
                e
            )
    
    # Start action monitor background loop
    asyncio.create_task(action_monitor_loop())
    
    # Pre-warm cache in the BACKGROUND so /health responds immediately
    asyncio.create_task(_warmup_cache())
            
    logger.info("Control Plane engines and models loaded. Server is ready (cache warming in background).")

@app.on_event("shutdown")
async def shutdown_event():
    """Close long-lived local servers cleanly on API shutdown."""
    global tcp_server
    if tcp_server is not None:
        tcp_server.close()
        await tcp_server.wait_closed()
        tcp_server = None


# Helper: Validate volume_id
def validate_volume(volume_id: str):
    if hub is None or volume_id not in hub.features_df["volume_id"].unique():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Volume '{volume_id}' not found in the storage cluster."
        )


# --- API Routes ---

@app.get("/health", status_code=status.HTTP_200_OK)
def get_health():
    """Simple health check endpoint."""
    return {
        "status": "healthy" if hub is not None else "starting",
        "service": "HPE Storage API",
        "loaded_models": ["lightgbm", "isolation_forest", "lstm_ae", "nbeats", "tft"],
        "telemetry_bus": {
            "mode": "redis_streams" if use_redis else "tcp_fallback",
            "redis": {
                "available": bool(use_redis),
                "host": REDIS_HOST,
                "port": REDIS_PORT,
                "error": redis_error,
            },
            "tcp_fallback": {
                "enabled": not use_redis,
                "host": TCP_FALLBACK_HOST,
                "port": TCP_FALLBACK_PORT,
                "listening": tcp_server is not None,
            },
        },
        "live_telemetry": live_state.status(get_expected_volume_count()),
    }


@app.get("/telemetry/status", status_code=status.HTTP_200_OK)
def get_telemetry_status():
    """Inspect API-owned live telemetry state without relying on dashboard pages."""
    return live_state.status(get_expected_volume_count())


@app.get("/kpi", status_code=status.HTTP_200_OK)
def get_kpi():
    """Aggregate KPIs across the entire storage pool for the dashboard overview."""
    if hub is None:
        return {
            "avg_latency_us": 0.0,
            "total_iops": 0.0,
            "total_actions": 0,
            "rolled_back_count": 0,
            "rollback_rate_pct": 0.0,
            "classifier_accuracy_target_pct": 95.0,
            "latency_variance_reduction_target_pct": 30.0,
            "is_live": False,
            "source": "not_ready",
            "current_tick": None,
            "live_volume_count": 0,
        }

    # Determine whether live telemetry is available
    is_live = live_state.events_received > 0

    if is_live:
        # Use the latest received event per volume for pool-wide KPIs
        latest_rows = live_state.latest_rows()
        latencies = [float(row.get("avg_latency_us", 0.0) or 0.0) for row in latest_rows]
        iops_vals = [float(row.get("total_iops", 0.0) or 0.0) for row in latest_rows]
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        total_iops = float(np.sum(iops_vals)) if iops_vals else 0.0
        current_tick = live_state.current_tick.isoformat() if live_state.current_tick is not None else None
        source = "live_telemetry"
        live_volume_count = len(latest_rows)
    else:
        # Fallback: use the static Parquet dataset's final timestamp
        latest_ts = hub.features_df["timestamp"].max()
        latest_df = hub.features_df[hub.features_df["timestamp"] == latest_ts]
        avg_latency = float(latest_df["avg_latency_us"].mean()) if not latest_df.empty else 0.0
        total_iops = float(latest_df["total_iops"].sum()) if not latest_df.empty else 0.0
        current_tick = latest_ts.isoformat() if latest_ts is not None else None
        source = "historical_parquet"
        live_volume_count = 0

    mon_summary = {
        "total_actions": 0,
        "rolled_back_count": 0,
        "rollback_rate_pct": 0.0,
    }
    if monitor is not None:
        mon_summary = monitor.get_summary()

    return {
        "avg_latency_us": round(avg_latency, 2),
        "total_iops": round(total_iops, 2),
        "total_actions": mon_summary["total_actions"],
        "rolled_back_count": mon_summary["rolled_back_count"],
        "rollback_rate_pct": mon_summary["rollback_rate_pct"],
        "classifier_accuracy_target_pct": 95.0,
        "latency_variance_reduction_target_pct": 30.0,
        "is_live": is_live,
        "source": source,
        "current_tick": current_tick,
        "live_volume_count": live_volume_count,
        "events_received": live_state.events_received,
        "stream_started_at": live_state.first_received_at.isoformat() if live_state.first_received_at is not None else None,
        "latest_complete_tick": live_state.latest_complete_tick.isoformat() if live_state.latest_complete_tick is not None else None,
    }


@app.get("/volumes", status_code=status.HTTP_200_OK)
def get_volumes():
    """Get all 50 volumes in the storage pool with their current status and hotspot scores."""
    result = []
    for vol_id, analysis in cached_analysis.items():
        # Get active tier
        tier = "HDD"
        if hub.topology.graph.has_node(vol_id):
            tier = hub.topology.graph.nodes[vol_id].get("tier", "HDD")

        # Enrich with live telemetry when available
        live_event = live_state.latest_by_volume.get(vol_id)
        if live_event is not None:
            current_iops = float(live_event.get("total_iops", 0.0) or 0.0)
            current_latency = float(live_event.get("avg_latency_us", 0.0) or 0.0)
            last_seen_ts = live_event.get("timestamp")
            last_seen = last_seen_ts.isoformat() if isinstance(last_seen_ts, pd.Timestamp) else str(last_seen_ts)
            vol_is_live = True
        else:
            current_iops = 0.0
            current_latency = 0.0
            last_seen = analysis.get("timestamp")
            vol_is_live = False

        # Include analysis freshness so dashboard can judge hotspot score age
        analysis_ts = analysis.get("timestamp")
        last_analyzed = last_analyzed_time.get(vol_id, 0.0)
        analysis_age_s = round(time.time() - last_analyzed, 1) if last_analyzed > 0 else None

        result.append({
            "volume_id": vol_id,
            "hotspot_score": analysis["hotspot_score"],
            "workload_type": analysis["workload_type"],
            "tier": tier,
            "capacity_used_pct": analysis["days_to_fill"].get("capacity_used_pct", 50.0),
            "latency_risk_score": analysis["latency_risk_score"],
            "current_iops": round(current_iops, 2),
            "current_latency_us": round(current_latency, 2),
            "last_seen_timestamp": last_seen,
            "is_live": vol_is_live,
            "last_analyzed_timestamp": analysis_ts,
            "analysis_freshness_s": analysis_age_s,
        })
    return result


@app.get("/volumes/{id}/metrics", status_code=status.HTTP_200_OK)
def get_volume_metrics(id: str, limit: int = Query(100, ge=1, le=500)):
    """Retrieve historical time-series telemetry data for a volume."""
    validate_volume(id)

    def _format_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a single telemetry row into the API response schema."""
        ts = row.get("timestamp")
        if isinstance(ts, pd.Timestamp):
            ts = ts.isoformat()
        return {
            "timestamp": str(ts),
            "read_iops": float(row.get("read_iops", 0.0) or 0.0),
            "write_iops": float(row.get("write_iops", 0.0) or 0.0),
            "total_iops": float(row.get("total_iops", 0.0) or 0.0),
            "read_throughput_mbps": float(row.get("read_throughput_mbps", 0.0) or 0.0),
            "write_throughput_mbps": float(row.get("write_throughput_mbps", 0.0) or 0.0),
            "avg_latency_us": float(row.get("avg_latency_us", 0.0) or 0.0),
            "read_latency_p95_us": float(row.get("read_latency_p95_us", 0.0) or 0.0),
            "write_latency_p95_us": float(row.get("write_latency_p95_us", 0.0) or 0.0),
            "queue_depth": float(row.get("queue_depth", 0.0) or 0.0),
            "capacity_used_pct": float(row.get("capacity_used_pct", 0.0) or 0.0),
        }

    # Priority 1: Redis history (when Redis is the active bus)
    if use_redis and r is not None:
        try:
            raw_items = r.lrange(f"volume:{id}:history", 0, limit - 1)
            records = [_format_row(json.loads(item)) for item in raw_items]
            records.reverse()
            return records
        except Exception as e:
            logger.error(f"Error reading metrics for {id} from Redis: {e}")

    # Priority 2: Live telemetry history (TCP fallback mode)
    live_history = live_state.history(id, limit)
    if live_history:
        return [_format_row(row) for row in live_history]

    # Priority 3: Static Parquet history (before playback starts)
    df_vol = hub.features_df[hub.features_df["volume_id"] == id].sort_values("timestamp").tail(limit)
    records = []
    for _, row in df_vol.iterrows():
        records.append(_format_row(row.to_dict()))
    return records


@app.get("/volumes/{id}/metrics/stream")
def stream_volume_metrics(id: str):
    """Server-Sent Events (SSE) stream of real-time volume metrics."""
    validate_volume(id)
    
    async def event_generator():
        q = asyncio.Queue()
        if id not in active_streams:
            active_streams[id] = []
        active_streams[id].append(q)
        
        try:
            while True:
                payload = await q.get()
                yield f"data: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if id in active_streams and q in active_streams[id]:
                active_streams[id].remove(q)
                if not active_streams[id]:
                    del active_streams[id]
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/volumes/{id}/workload", status_code=status.HTTP_200_OK)
def get_volume_workload(id: str):
    """Current workload pattern classification and confidence scores."""
    validate_volume(id)
    analysis = cached_analysis.get(id)
    if not analysis:
        analysis = hub.analyze_volume(id)
    return {
        "volume_id": id,
        "workload_type": analysis["workload_type"],
        "confidence": analysis["workload_confidence"],
        "arf_workload_type": analysis.get("arf_workload_type"),
        "arf_agrees": analysis.get("arf_agrees_with_lgbm"),
    }


@app.get("/model/drift-status", status_code=status.HTTP_200_OK)
def get_model_drift_status():
    """Report ARF availability and its agreement with the LightGBM classifier."""
    arf_loaded = bool(getattr(hub, "arf_classifier", None) is not None) if hub is not None else False
    if not arf_loaded:
        return {
            "arf_loaded": False,
            "lgbm_arf_agreement_rate": None,
            "disagreeing_volumes": [],
        }

    comparable = []
    disagreeing_volumes = []
    for volume_id, analysis in cached_analysis.items():
        agree = analysis.get("arf_agrees_with_lgbm")
        if agree is None:
            continue
        comparable.append(agree)
        if agree is False:
            disagreeing_volumes.append(volume_id)

    agreement_rate = round((sum(1 for item in comparable if item) / len(comparable)) * 100.0, 2) if comparable else 0.0
    return {
        "arf_loaded": True,
        "lgbm_arf_agreement_rate": agreement_rate,
        "disagreeing_volumes": disagreeing_volumes,
    }


@app.get("/volumes/{id}/explain", status_code=status.HTTP_200_OK)
def get_volume_explanation(id: str):
    """SHAP-based explainability details for the workload classification."""
    validate_volume(id)
    
    # Run SHAP on classifier
    # Use live tick when playback is active, otherwise fall back to latest in features_df
    if live_state.current_tick is not None:
        explain_ts = live_state.current_tick
    else:
        explain_ts = hub.features_df["timestamp"].max()
    row = hub.get_raw_feature_row(id, pd.to_datetime(explain_ts))
    
    classifier_feature_cols = hub.classifier_scaler.feature_names_in_.tolist()
    features_arr = row[classifier_feature_cols].to_numpy(dtype=np.float64).reshape(1, -1)
    features_log = np.sign(features_arr) * np.log1p(np.abs(features_arr))
    features_log_df = pd.DataFrame(features_log, columns=classifier_feature_cols)
    features_scaled = hub.classifier_scaler.transform(features_log_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=classifier_feature_cols)
    
    pred_class = int(hub.classifier.predict(features_scaled_df)[0])
    shap_vals = explainer.shap_values(features_scaled_df)
    
    if isinstance(shap_vals, list):
        cls_shap = shap_vals[pred_class][0]
    else:
        cls_shap = shap_vals[0, :, pred_class]
        
    contribs = []
    for name, val in zip(classifier_feature_cols, cls_shap):
        contribs.append({"feature": name, "shap_value": float(val)})
        
    # Sort contributions
    contribs = sorted(contribs, key=lambda x: abs(x["shap_value"]), reverse=True)
    
    top_reasons = contribs[:3]
    reasons = []
    for r in top_reasons:
        direction = "high" if r["shap_value"] > 0 else "low"
        reasons.append(f"{r['feature']} is {direction} (contribution: {round(r['shap_value'], 3)})")
        
    explanation_text = (
        f"Volume '{id}' classified as '{InferenceHub.analyze_volume.__globals__['LABEL_NAMES'][pred_class]}' "
        f"primarily because: " + ", ".join(reasons)
    )
    
    return {
        "volume_id": id,
        "predicted_class": InferenceHub.analyze_volume.__globals__["LABEL_NAMES"][pred_class],
        "explanation": explanation_text,
        "feature_contributions": contribs
    }


@app.get("/alerts", status_code=status.HTTP_200_OK)
def get_alerts():
    """All active alerts sorted by severity."""
    alerts = []
    for vol_id, analysis in cached_analysis.items():
        score = analysis["hotspot_score"]
        if score >= 40:
            severity = "Normal"
            if 40 <= score < 60:
                severity = "Warning"
            elif 60 <= score < 80:
                severity = "High"
            elif score >= 80:
                severity = "Critical"

            # Prefer live timestamp when available
            live_event = live_state.latest_by_volume.get(vol_id)
            if live_event is not None:
                ts = live_event.get("timestamp")
                alert_ts = ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts)
            else:
                alert_ts = analysis["timestamp"]

            alerts.append({
                "volume_id": vol_id,
                "hotspot_score": score,
                "severity": severity,
                "workload_type": analysis["workload_type"],
                "timestamp": alert_ts,
            })

    # Sort alerts by score descending
    alerts = sorted(alerts, key=lambda x: x["hotspot_score"], reverse=True)
    return alerts


@app.get("/noisy-neighbors", status_code=status.HTTP_200_OK)
def get_noisy_neighbors():
    """Returns detected noisy neighbor relationships across all nodes."""
    noisy_pairs = []
    for vol_id, analysis in cached_analysis.items():
        victims = analysis.get("noisy_neighbor_victims", {})
        if victims:
            noisy_pairs.append({
                "aggressor_id": vol_id,
                "workload_type": analysis["workload_type"],
                "hotspot_score": analysis["hotspot_score"],
                "victims": [
                    {"volume_id": v_id, "impact_score": round(v_score, 2)}
                    for v_id, v_score in victims.items()
                ]
            })
    return noisy_pairs


@app.get("/forecast/capacity", status_code=status.HTTP_200_OK)
def get_capacity_forecasts():
    """Retrieve Days-to-Fill (DTF) forecasts and predictions for the storage pool."""
    forecasts = []
    for vol_id, analysis in cached_analysis.items():
        dtf = analysis.get("days_to_fill", {})
        forecasts.append({
            "volume_id": vol_id,
            "warning_85pct_days": dtf.get("warning_85pct_days"),
            "critical_95pct_days": dtf.get("critical_95pct_days")
        })
    return forecasts


@app.get("/forecast/bandwidth", status_code=status.HTTP_200_OK)
def get_bandwidth_forecast(volume_id: str = Query(..., description="ID of the volume to query")):
    """24h hourly tail latency/bandwidth forecast for a specific volume."""
    validate_volume(volume_id)
    analysis = cached_analysis.get(volume_id)
    if not analysis:
        analysis = hub.analyze_volume(volume_id)
    return {
        "volume_id": volume_id,
        "forecast_24h": analysis.get("bandwidth_forecast_24h"),
        "demand_forecast_24h": analysis.get("demand_forecast_24h"),
    }


@app.get("/forecast/demand", status_code=200)
def get_demand_forecast(volume_id: str = Query(...)):
    validate_volume(volume_id)
    analysis = cached_analysis.get(volume_id)
    if not analysis:
        analysis = hub.analyze_volume(volume_id)
    demand = analysis.get("demand_forecast_24h")
    if demand is None:
        raise HTTPException(
            status_code=503,
            detail="Demand forecaster not loaded. Run scripts/train_all.py first."
        )
    return {"volume_id": volume_id, **demand}


@app.get("/forecast/dtf", status_code=status.HTTP_200_OK)
def get_dtf_urgency():
    """All volumes DTF sorted by urgency (most urgent capacity limits first)."""
    dtf_list = []
    for vol_id, analysis in cached_analysis.items():
        dtf = analysis.get("days_to_fill", {})
        crit_days = dtf.get("critical_95pct_days")
        warn_days = dtf.get("warning_85pct_days")
        
        # Sort key helper
        sort_key = crit_days if crit_days is not None else 9999.0
        
        dtf_list.append({
            "volume_id": vol_id,
            "warning_85pct_days": warn_days,
            "critical_95pct_days": crit_days,
            "sort_key": sort_key
        })
        
    dtf_list = sorted(dtf_list, key=lambda x: x["sort_key"])
    # Clean up sort_key
    for d in dtf_list:
        del d["sort_key"]
    return dtf_list


@app.get("/forecast/ttv", status_code=200)
def get_latency_ttv(volume_id: Optional[str] = Query(None)):
    """Returns time-to-violation estimates for latency SLOs."""
    if hub is None:
        raise HTTPException(status_code=503, detail="Hub not initialized.")

    if volume_id is not None:
        validate_volume(volume_id)
        analysis = cached_analysis.get(volume_id) or hub.analyze_volume(volume_id)
        return {"volume_id": volume_id, **analysis.get("latency_ttv", {})}

    results = []
    for vol_id, analysis in cached_analysis.items():
        ttv = analysis.get("latency_ttv", {})
        results.append({
            "volume_id": vol_id,
            "workload_type": analysis.get("workload_type"),
            **ttv
        })

    # Sort: critical first, then by hours_to_breach ascending, non-breaching last
    risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
    results.sort(key=lambda x: (
        risk_order.get(x.get("risk_level", "none"), 5),
        x.get("hours_to_breach") if x.get("hours_to_breach") is not None else 9999.0,
    ))
    return results


@app.get("/cluster/headroom", status_code=200)
def get_cluster_headroom():
    """Returns tier-level and pool-level capacity headroom for the full cluster."""
    if hub is None:
        raise HTTPException(status_code=503, detail="Hub not initialized.")
    # Sync latest live metrics into topology before computing headroom
    sync_topology_from_redis()
    return hub.get_cluster_headroom()


@app.get("/cluster/headroom/tier/{tier_name}", status_code=200)
def get_tier_headroom(tier_name: str):
    if hub is None:
        raise HTTPException(status_code=503, detail="Hub not initialized.")
    sync_topology_from_redis()
    headroom = hub.topology.get_tier_headroom()
    if tier_name not in headroom:
        raise HTTPException(
            status_code=404,
            detail=f"Tier '{tier_name}' not found. Available: {list(headroom.keys())}"
        )
    return {"tier": tier_name, **headroom[tier_name]}


# --- What-If Simulator Endpoints ---

@app.post("/simulate/capacity", status_code=status.HTTP_200_OK)
def post_simulate_capacity(req: SimulateCapacityRequest):
    """Simulate adding storage capacity (GB) to a volume."""
    validate_volume(req.volume_id)
    sync_topology_from_redis()
    try:
        return simulator.simulate_add_capacity_scenario(req.volume_id, req.added_gb)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/simulate/migrate", status_code=status.HTTP_200_OK)
def post_simulate_migrate(req: SimulateMigrateRequest):
    """Simulate moving a volume to another storage node."""
    validate_volume(req.volume_id)
    sync_topology_from_redis()
    try:
        return simulator.simulate_migration_scenario(req.volume_id, req.target_node)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/simulate/qos", status_code=status.HTTP_200_OK)
def post_simulate_qos(req: SimulateQosRequest):
    """Simulate capping a volume's IOPS limit."""
    validate_volume(req.volume_id)
    sync_topology_from_redis()
    try:
        return simulator.simulate_qos_shaping_scenario(req.volume_id, req.iops_limit)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/simulate/tier", status_code=status.HTTP_200_OK)
def post_simulate_tier(req: SimulateTierRequest):
    """Simulate upgrading or downgrading a volume's storage tier."""
    validate_volume(req.volume_id)
    sync_topology_from_redis()
    try:
        return simulator.simulate_tier_change_scenario(req.volume_id, req.new_tier)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --- Recommendations Endpoint ---

@app.get("/recommendations", status_code=status.HTTP_200_OK)
def get_recommendations():
    """Generates priority-sorted recommendations for active problems and forecasting limits."""
    sync_topology_from_redis()
    recs = []
    
    for vol_id, analysis in cached_analysis.items():
        # 1. Capacity warnings
        dtf = analysis.get("days_to_fill", {})
        crit_days = dtf.get("critical_95pct_days")
        warn_days = dtf.get("warning_85pct_days")
        
        if crit_days is not None and crit_days < 7.0:
            recs.append({
                "volume_id": vol_id,
                "priority": "CRITICAL",
                "message": f"Volume will breach capacity limit in {crit_days} days. Expand capacity or migrate cold data immediately."
            })
        elif warn_days is not None and warn_days < 30.0:
            recs.append({
                "volume_id": vol_id,
                "priority": "WARNING",
                "message": f"Volume will exceed 85% capacity threshold in {warn_days} days. Plan expansion."
            })
            
        # 2. Performance Hotspots
        score = analysis["hotspot_score"]
        if score >= 80:
            recs.append({
                "volume_id": vol_id,
                "priority": "HIGH",
                "message": f"Active hotspot detected (score: {score}) on volume. Severe performance anomaly."
            })
            
        # 3. Latency Risk SLO breaches (TTV-aware)
        ttv = analysis.get("latency_ttv", {})
        risk_level = ttv.get("risk_level", "none")
        hours = ttv.get("hours_to_breach")
        if risk_level == "critical":
            recs.append({
                "volume_id": vol_id,
                "priority": "CRITICAL",
                "message": (
                    f"Latency SLO breach imminent for {vol_id} "
                    f"({'<1h' if hours is not None and hours < 1 else 'already breached'}). "
                    f"Max p95 forecast: {ttv.get('max_p95_forecast_us', 0):.0f}us "
                    f"(SLO: {ttv.get('slo_threshold_us', 8000):.0f}us). "
                    f"Immediate QoS shaping or migration required."
                )
            })
        elif risk_level in ("high", "medium") and hours is not None:
            recs.append({
                "volume_id": vol_id,
                "priority": "HIGH" if risk_level == "high" else "MEDIUM",
                "message": (
                    f"Latency SLO breach predicted for {vol_id} in "
                    f"{hours:.1f}h. Max p95 forecast: "
                    f"{ttv.get('max_p95_forecast_us', 0):.0f}us. "
                    f"Consider preemptive migration or QoS shaping."
                )
            })

    # 4. Tier headroom warnings
    try:
        tier_headroom = hub.topology.get_tier_headroom()
        for tier_name, info in tier_headroom.items():
            used_pct = info.get("used_pct", 0.0)
            headroom_gb = info.get("headroom_gb", 0.0)
            if used_pct >= 90.0:
                recs.append({
                    "volume_id": "CLUSTER",
                    "priority": "CRITICAL",
                    "message": (
                        f"{tier_name} storage tier is {round(used_pct, 1)}% full "
                        f"({round(headroom_gb, 0)} GB remaining). "
                        f"Immediately add capacity or migrate volumes to other tiers."
                    )
                })
            elif used_pct >= 80.0:
                recs.append({
                    "volume_id": "CLUSTER",
                    "priority": "WARNING",
                    "message": (
                        f"{tier_name} storage tier is {round(used_pct, 1)}% full "
                        f"({round(headroom_gb, 0)} GB remaining). Plan capacity expansion."
                    )
                })
    except Exception as e:
        logger.error("Tier headroom recommendation failed: %s", e)
            
    # Sort recommendations: CRITICAL -> HIGH -> WARNING -> MEDIUM
    prio_order = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "MEDIUM": 3}
    recs = sorted(recs, key=lambda x: prio_order.get(x["priority"], 99))
    return recs


# --- Policy Management Endpoints ---

@app.get("/policy", status_code=status.HTTP_200_OK)
def get_policy():
    """View current active policies and thresholds."""
    return engine.policy


@app.put("/policy", status_code=status.HTTP_200_OK)
def update_policy(req: PolicyUpdateRequest):
    """Dynamically updates active policy configuration parameters."""
    if req.rebalance_policy:
        rebalance = req.rebalance_policy.dict(exclude_none=True)
        engine.rebalance_policy.update(rebalance)
        
        # Sync attributes
        if "enabled" in rebalance:
            engine.enabled = rebalance["enabled"]
            if rebalance["enabled"]:
                engine.circuit_breaker_tripped = False
                engine.circuit_breaker_tripped_at = None
                engine.circuit_breaker_reason = ""
        if "dry_run_mode" in rebalance:
            engine.dry_run_mode = rebalance["dry_run_mode"]
        if "min_hotspot_score_to_trigger" in rebalance:
            engine.min_hotspot_score = rebalance["min_hotspot_score_to_trigger"]
        if "min_hotspot_duration_minutes" in rebalance:
            engine.min_hotspot_duration = rebalance["min_hotspot_duration_minutes"]
        if "max_volumes_moved_per_hour" in rebalance:
            engine.max_moves_per_hour = rebalance["max_volumes_moved_per_hour"]
        if "max_concurrent_migrations" in rebalance:
            engine.max_concurrent_migrations = rebalance["max_concurrent_migrations"]
            
    if req.safety_guardrails:
        safety = req.safety_guardrails.dict(exclude_none=True)
        engine.policy["safety_guardrails"].update(safety)
        if "rollback_if_target_latency_increases_pct" in safety:
            monitor.rollback_threshold_pct = safety["rollback_if_target_latency_increases_pct"]
        if "rollback_timeout_minutes" in safety:
            monitor.rollback_timeout_minutes = safety["rollback_timeout_minutes"]
            
    # Persist policy changes to Redis if active
    if use_redis and r is not None:
        try:
            r.set("control_plane:policy", json.dumps(engine.policy))
        except Exception as e:
            logger.error("Failed to persist policy to Redis: %s", e)
            
    return {"status": "success", "message": "Policy parameters updated successfully.", "policy": engine.policy}


@app.get("/rebalance/circuit-breaker", status_code=200)
def get_circuit_breaker_status():
    if engine is None:
        return {"circuit_breaker_tripped": False, "engine_ready": False}
    return {
        "circuit_breaker_tripped": engine.circuit_breaker_tripped,
        "tripped_at": engine.circuit_breaker_tripped_at.isoformat()
                     if engine.circuit_breaker_tripped_at else None,
        "reason": engine.circuit_breaker_reason,
        "current_rollback_rate_pct": monitor.get_rollback_rate() if monitor else 0.0,
        "max_rollback_rate_pct": engine.max_rollback_rate_pct,
        "total_actions": monitor.total_actions if monitor else 0,
        "engine_enabled": engine.enabled,
    }


@app.post("/rebalance/circuit-breaker/reset", status_code=200)
def reset_circuit_breaker():
    """Manually reset the circuit breaker after investigating rollback causes."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")
    engine.circuit_breaker_tripped = False
    engine.circuit_breaker_tripped_at = None
    engine.circuit_breaker_reason = ""
    engine.enabled = engine.rebalance_policy.get("enabled", True)
    logger.info("Circuit breaker manually reset via API.")
    return {"status": "reset", "engine_enabled": engine.enabled}


# --- Execution Control Endpoints ---

@app.get("/rebalance/history", status_code=status.HTTP_200_OK)
def get_rebalance_history():
    """Retrieve all rebalancing actions executed or logged."""
    global engine, use_redis, r
    if engine is None:
        return []
    
    if use_redis and r is not None:
        _sync_monitors_from_redis()
        
    def _serialize_act(act):
        serialized = {}
        for k, v in act.items():
            if isinstance(v, pd.Timestamp):
                serialized[k] = v.isoformat()
            elif isinstance(v, dict):
                serialized[k] = _serialize_act(v)
            elif isinstance(v, list):
                serialized[k] = [
                    _serialize_act(item) if isinstance(item, dict) else (item.isoformat() if isinstance(item, pd.Timestamp) else item)
                    for item in v
                ]
            else:
                serialized[k] = v
        return serialized
        
    return [_serialize_act(act) for act in engine.action_history]


@app.get("/rebalance/monitors", status_code=status.HTTP_200_OK)
def get_rebalance_monitors():
    """Retrieve active and historical action monitors."""
    global monitor, use_redis, r
    if monitor is None:
        return {}
        
    if use_redis and r is not None:
        _sync_monitors_from_redis()
        
    def _serialize_mon(mon):
        serialized = {}
        for k, v in mon.items():
            if isinstance(v, pd.Timestamp):
                serialized[k] = v.isoformat()
            elif isinstance(v, dict):
                serialized[k] = _serialize_mon(v)
            elif isinstance(v, list):
                serialized[k] = [
                    _serialize_mon(item) if isinstance(item, dict) else (item.isoformat() if isinstance(item, pd.Timestamp) else item)
                    for item in v
                ]
            else:
                serialized[k] = v
        return serialized
        
    return {aid: _serialize_mon(mon) for aid, mon in monitor.actions.items()}


@app.post("/rebalance", status_code=status.HTTP_200_OK)
def trigger_rebalance(req: RebalanceRequest):
    """Manually invoke a rebalance action (migrate, QoS shaping, or tier-change)."""
    validate_volume(req.volume_id)
    sync_topology_from_redis()
    
    # Get current latency
    metrics_dict = hub.topology._volume_metrics.get(req.volume_id, {})
    pre_latency = metrics_dict.get("avg_latency_us", 1000.0)
    
    action_state = {}
    if req.action_type == "migrate":
        if req.target not in hub.topology.all_nodes():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Target node '{req.target}' does not exist.")
        action_state = rebalancer.execute_migration(req.volume_id, req.target, hub.topology)
    elif req.action_type == "qos":
        try:
            limit = float(req.target)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Target for QoS action must be a numeric IOPS limit.")
        action_state = rebalancer.execute_qos_shaping(req.volume_id, limit, hub.topology)
    elif req.action_type == "tier_change":
        action_state = rebalancer.execute_tier_change(req.volume_id, req.target, hub.topology)
        
    action_id = str(uuid.uuid4())
    
    monitor.register_action(
        action_id=action_id,
        action_state=action_state,
        pre_latency=pre_latency,
        timestamp=pd.Timestamp.now()
    )
    
    # Log to action history
    exec_record = {
        "action_id": action_id,
        "volume_id": req.volume_id,
        "action": req.action_type,
        "choice": {
            "action": req.action_type,
            "target_node": req.target if req.action_type == "migrate" else None,
            "iops_limit": float(req.target) if req.action_type == "qos" else None,
            "new_tier": req.target if req.action_type == "tier_change" else None,
            "expected_improvement": 0.0,
            "safe": True
        },
        "timestamp": pd.Timestamp.now(),
        "action_state": action_state,
        "status": "executed"
    }
    if engine is not None:
        engine.action_history.append(exec_record)
        
    # Persist changes
    _persist_control_plane_state()
    if use_redis:
        _persist_topology_to_redis(req.volume_id)
    
    return {
        "status": "success",
        "action_id": action_id,
        "action_state": action_state
    }


@app.post("/rollback", status_code=status.HTTP_200_OK)
def trigger_rollback(req: RollbackRequest):
    """Explicitly rollback an executed rebalance action."""
    action = monitor.actions.get(req.action_id)
    if not action:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Action ID '{req.action_id}' not found in active monitor registry.")
        
    try:
        rebalancer.rollback_action(action["action_state"], hub.topology)
        action["status"] = "rolled_back"
        monitor.rolled_back_count += 1
        
        # Update status in history log
        if engine is not None:
            for h_action in engine.action_history:
                if h_action.get("action_id") == req.action_id:
                    h_action["status"] = "rolled_back"
                    break
                    
        # Persist changes
        _persist_control_plane_state()
        if use_redis:
            _persist_topology_to_redis(action["action_state"]["volume_id"])
            
        return {"status": "success", "message": f"Action {req.action_id} successfully rolled back."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/topology", status_code=status.HTTP_200_OK)
def get_topology():
    """Retrieve network topology graph data formatted for visualization."""
    nodes = []
    edges = []
    
    # Export storage nodes
    for node_id in hub.topology.all_nodes():
        if node_id.startswith("node_"):
            nodes.append({
                "id": node_id,
                "label": node_id,
                "type": "storage_node",
                "tier": hub.topology.graph.nodes[node_id].get("tier", "SSD"),
                "capacity_gb": hub.topology.graph.nodes[node_id].get("capacity_gb", 50000.0)
            })
            
    # Export volumes
    for vol_id in hub.topology.graph.nodes:
        if vol_id.startswith("vol_"):
            node_owner = hub.topology.get_node_of_volume(vol_id)
            nodes.append({
                "id": vol_id,
                "label": vol_id,
                "type": "volume",
                "tier": hub.topology.graph.nodes[vol_id].get("tier", "HDD"),
                "node_owner": node_owner
            })
            if node_owner:
                edges.append({
                    "source": vol_id,
                    "target": node_owner,
                    "type": "resides_on"
                })
                
    return {"nodes": nodes, "edges": edges}


@app.get("/model/performance", status_code=status.HTTP_200_OK)
def get_model_performance():
    """Evaluate LightGBM model performance on validation data."""
    global hub
    label_names = {
        0: "DB_OLTP",
        1: "VM",
        2: "Backup",
        3: "AI_Training",
        4: "AI_Inference",
    }

    def _load_from_artifacts() -> Optional[Dict[str, Any]]:
        metrics_candidates = [
            PROJECT_ROOT / "models" / "classifier" / "lightgbm_tuned_metrics.json",
            PROJECT_ROOT / "models" / "classifier" / "lightgbm_metrics.json",
        ]

        for metrics_path in metrics_candidates:
            if not metrics_path.exists():
                continue
            try:
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception as exc:
                logger.warning("Failed to read metrics from %s: %s", metrics_path, exc)
                continue

            val_metrics = metrics.get("val") or metrics.get("validation")
            if not val_metrics:
                continue

            report = val_metrics.get("classification_report", {})
            accuracy = float(val_metrics.get("accuracy", report.get("accuracy", 0.0)))

            cm = None
            cm_path = val_metrics.get("confusion_matrix_path")
            if cm_path:
                cm_file = PROJECT_ROOT / cm_path
            else:
                cm_file = None

            if cm_file and cm_file.exists():
                cm_df = pd.read_csv(cm_file, index_col=0)
                cm = cm_df.values.tolist()
            else:
                fallback_names = [
                    "lightgbm_tuned_confusion_matrix_val.csv",
                    "lightgbm_confusion_matrix_val.csv",
                ]
                for name in fallback_names:
                    fallback = PROJECT_ROOT / "models" / "classifier" / name
                    if fallback.exists():
                        cm_df = pd.read_csv(fallback, index_col=0)
                        cm = cm_df.values.tolist()
                        break

            cm_norm = []
            if cm:
                for row in cm:
                    row_sum = sum(row)
                    if row_sum > 0:
                        cm_norm.append([round((val / row_sum) * 100.0, 2) for val in row])
                    else:
                        cm_norm.append([0.0 for _ in row])

            metrics_per_class: Dict[str, Any] = {}
            for cls_id, cls_name in label_names.items():
                cls_key = str(cls_id)
                if cls_key not in report:
                    continue
                cls_report = report[cls_key]
                metrics_per_class[cls_name] = {
                    "precision": round(float(cls_report.get("precision", 0.0)), 4),
                    "recall": round(float(cls_report.get("recall", 0.0)), 4),
                    "f1_score": round(float(cls_report.get("f1-score", 0.0)), 4),
                    "support": int(cls_report.get("support", 0)),
                }

            sample_count = int(sum(m.get("support", 0) for m in metrics_per_class.values()))

            return {
                "accuracy": round(accuracy, 4),
                "confusion_matrix": cm or [],
                "confusion_matrix_percentage": cm_norm,
                "metrics_per_class": metrics_per_class,
                "sample_count": sample_count,
            }

        return None

    artifact_payload = _load_from_artifacts()
    if artifact_payload:
        return artifact_payload

    if hub is None:
        raise HTTPException(status_code=503, detail="Model hub not initialized.")

    try:
        val_X_path = PROJECT_ROOT / "data" / "features" / "X_val.parquet"
        val_y_path = PROJECT_ROOT / "data" / "features" / "y_val.parquet"

        if not val_X_path.exists() or not val_y_path.exists():
            raise HTTPException(status_code=404, detail="Validation features parquet files not found.")

        X_val = pd.read_parquet(val_X_path)
        y_val = pd.read_parquet(val_y_path)

        if "label" not in y_val.columns:
            raise HTTPException(status_code=500, detail="Validation labels missing 'label' column.")

        y_true = y_val["label"].values.astype(int)

        if hasattr(hub.classifier, "feature_names_in_"):
            X_val = X_val[hub.classifier.feature_names_in_.tolist()]

        y_pred = hub.classifier.predict(X_val).astype(int)

        accuracy = float((y_pred == y_true).mean())

        num_classes = 5
        cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t][p] += 1

        cm_norm = []
        for row in cm:
            row_sum = sum(row)
            if row_sum > 0:
                cm_norm.append([round((val / row_sum) * 100.0, 2) for val in row])
            else:
                cm_norm.append([0.0 for _ in row])

        metrics_per_class = {}
        for c in range(num_classes):
            tp = cm[c][c]
            fp = sum(cm[i][c] for i in range(num_classes)) - tp
            fn = sum(cm[c][j] for j in range(num_classes)) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics_per_class[label_names[c]] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": int(tp + fn),
            }

        return {
            "accuracy": round(accuracy, 4),
            "confusion_matrix": cm,
            "confusion_matrix_percentage": cm_norm,
            "metrics_per_class": metrics_per_class,
            "sample_count": len(y_true),
        }
    except Exception as e:
        logger.error("Failed to compute model performance: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
