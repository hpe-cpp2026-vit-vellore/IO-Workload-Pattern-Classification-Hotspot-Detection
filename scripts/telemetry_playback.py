"""
scripts/telemetry_playback.py

Telemetry Playback Agent (Data Fetcher / Generator).
Chronologically streams telemetry records from the static Parquet features dataset.
Supports:
1. Redis Streams (primary, port 6379)
2. TCP Socket Loopback (fallback, port 9000)
"""

import sys
import time
import json
import logging
import socket
from pathlib import Path
import pandas as pd
import numpy as np

# Setup pathing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] PlaybackAgent: %(message)s")
logger = logging.getLogger("playback_agent")

PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "io_features.parquet"
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "io_features.csv"
REDIS_PORT = 6379
TCP_FALLBACK_HOST = "127.0.0.1"
TCP_FALLBACK_PORT = 9000


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
    """Auto-detect the best Redis host address.

    On Windows with WSL2, Redis runs inside a Linux VM whose loopback
    (127.0.0.1) is NOT the same as the Windows loopback.  We try:
      1. The WSL2 VM's real IP (obtained via ``wsl hostname -I``).
      2. ``localhost`` — Windows sometimes proxies this into WSL2.
      3. ``127.0.0.1`` — works only when Redis is native on Windows.
    """
    import subprocess
    # Try to get WSL2 IP
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


def load_dataset() -> pd.DataFrame:
    """Loads the processed features dataset (Parquet preferred, CSV fallback)."""
    if PARQUET_PATH.exists():
        logger.info(f"Loading dataset from Parquet: {PARQUET_PATH}")
        df = pd.read_parquet(PARQUET_PATH)
    elif CSV_PATH.exists():
        logger.info(f"Loading dataset from CSV: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    else:
        raise FileNotFoundError(f"No dataset found at {PARQUET_PATH} or {CSV_PATH}")
        
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Dataset loaded. Total rows: {len(df):,}, Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def clean_row_data(row_dict: dict) -> dict:
    """Convert numpy/pandas types to clean Python types suitable for JSON serialization."""
    # Columns that are string identifiers (not numeric)
    STRING_COLS = {"volume_id", "node_id", "timestamp", "workload_label"}
    cleaned = {}
    for k, v in row_dict.items():
        if isinstance(v, pd.Timestamp):
            cleaned[k] = v.isoformat()
        elif k in STRING_COLS:
            cleaned[k] = str(v) if not pd.isna(v) else ""
        elif pd.isna(v):
            # Emit None → becomes JSON null → Python None after json.loads
            cleaned[k] = None
        elif isinstance(v, (np.integer, np.floating)):
            cleaned[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, (float, int)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned

def connect_redis():
    """Return a Redis client if a local Redis server is reachable."""
    try:
        import redis
        
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

        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=5,
            socket_keepalive=True,
            socket_keepalive_options=socket_keepalive_options,
            health_check_interval=15,
        )
        client.ping()
        logger.info("Connected to Redis server on %s:%s. Stream playback enabled.", REDIS_HOST, REDIS_PORT)
        return client
    except Exception as e:
        logger.warning(
            "Could not connect to Redis at %s:%s: %s. Falling back to TCP socket mode on %s:%s.",
            REDIS_HOST,
            REDIS_PORT,
            e,
            TCP_FALLBACK_HOST,
            TCP_FALLBACK_PORT
        )
        return None

def wait_for_tcp_fallback(timeout_seconds: float = 30.0) -> bool:
    """Wait briefly for FastAPI's TCP fallback listener to become reachable."""
    deadline = time.time() + timeout_seconds
    last_error = None
    logger.info(
        "Waiting for FastAPI TCP fallback listener on %s:%s...",
        TCP_FALLBACK_HOST,
        TCP_FALLBACK_PORT
    )
    while time.time() < deadline:
        try:
            with socket.create_connection((TCP_FALLBACK_HOST, TCP_FALLBACK_PORT), timeout=1.0):
                logger.info("TCP fallback listener is reachable.")
                return True
        except OSError as e:
            last_error = e
            time.sleep(1.0)
    logger.warning("TCP fallback listener was not reachable after %.0fs: %s", timeout_seconds, last_error)
    return False

def run_playback():
    # 1. Connect to Redis (try)
    r = connect_redis()
    use_redis = r is not None
    if not use_redis:
        wait_for_tcp_fallback()

    # 2. Load data
    try:
        df = load_dataset()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Group by timestamp to stream all volumes co-located in time
    grouped = df.groupby("timestamp")
    timestamps = sorted(grouped.groups.keys())

    # 3. Stream Loop
    tcp_failure_count = 0
    while True:
        logger.info("Starting telemetry playback loop...")
        for ts in timestamps:
            tick_start = time.time()
            group_df = grouped.get_group(ts)
            
            # Prepare events for this tick
            events = []
            for _, row in group_df.iterrows():
                row_dict = row.to_dict()
                events.append(clean_row_data(row_dict))
                
            # Publish events
            if use_redis and r is not None:
                try:
                    # Use Redis pipeline for high-frequency bulk writes (O(1) round trips)
                    pipe = r.pipeline(transaction=False)
                    for event in events:
                        # Convert float/int values to string for Redis Stream fields
                        redis_fields = {k: str(v) for k, v in event.items()}
                        # Trimming stream to last 10,000 events to prevent memory bloat
                        pipe.xadd("telemetry:stream", redis_fields, maxlen=10000, approximate=True)
                    pipe.execute()
                    logger.info(f"Tick {ts}: Published {len(events)} volume telemetry events to Redis Stream.")
                except Exception as e:
                    logger.error(f"Redis write error: {e}. Attempting reconnection...")
                    try:
                        r.ping()
                    except Exception:
                        logger.warning("Redis server offline. Toggling to TCP Socket Fallback.")
                        use_redis = False
                        wait_for_tcp_fallback()
            
            if not use_redis:
                # Socket Fallback
                try:
                    with socket.create_connection((TCP_FALLBACK_HOST, TCP_FALLBACK_PORT), timeout=2.0) as s:
                        # Stream records as line-delimited JSON
                        for event in events:
                            s.sendall((json.dumps(event) + "\n").encode("utf-8"))
                    tcp_failure_count = 0
                    logger.info(
                        "Tick %s: Sent %s events over TCP socket fallback (%s:%s).",
                        ts,
                        len(events),
                        TCP_FALLBACK_HOST,
                        TCP_FALLBACK_PORT
                    )
                except Exception as e:
                    tcp_failure_count += 1
                    if tcp_failure_count == 1 or tcp_failure_count % 10 == 0:
                        logger.error(
                            "TCP socket fallback connect failed (%s failures): %s. Start FastAPI with "
                            "`venv\\Scripts\\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000`.",
                            tcp_failure_count,
                            e
                        )
            
            # Maintain 2s ingestion intervals (subtract compute time)
            elapsed = time.time() - tick_start
            sleep_time = max(0.1, 2.0 - elapsed)
            time.sleep(sleep_time)

if __name__ == "__main__":
    try:
        run_playback()
    except KeyboardInterrupt:
        logger.info("Playback agent stopped by user.")
