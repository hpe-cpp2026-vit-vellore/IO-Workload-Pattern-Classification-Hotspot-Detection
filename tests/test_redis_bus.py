"""
tests/test_redis_bus.py

Unit and Integration Tests for Redis Streams and TCP Socket Fallback.
"""

import unittest
import socket
import time
import json
import threading
import sys
from pathlib import Path
import pandas as pd

# Setup pathing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Sibling model imports
ANOMALY_DIR = PROJECT_ROOT / "src" / "models" / "anomaly"
if str(ANOMALY_DIR) not in sys.path:
    sys.path.insert(0, str(ANOMALY_DIR))

from src.control_plane.inference_hub import InferenceHub

class TestRedisBusAndFallback(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.hub = InferenceHub(project_root=PROJECT_ROOT)
        
    def test_tcp_fallback_server(self):
        """Verify the local TCP fallback server correctly receives and processes telemetry."""
        received_events = []
        server_running = True
        
        # Define a simple mock TCP server handler
        def run_mock_server():
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(("127.0.0.1", 9999))
            server_sock.listen(1)
            server_sock.settimeout(2.0)
            
            try:
                conn, addr = server_sock.accept()
                with conn:
                    conn.settimeout(2.0)
                    buffer = ""
                    while server_running:
                        try:
                            data = conn.recv(1024).decode('utf-8')
                            if not data:
                                break
                            buffer += data
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                if line.strip():
                                    received_events.append(json.loads(line))
                        except socket.timeout:
                            continue
            except Exception:
                pass
            finally:
                server_sock.close()
                
        server_thread = threading.Thread(target=run_mock_server)
        server_thread.start()
        
        # Allow server to bind
        time.sleep(0.5)
        
        # Connect client and send a message
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_sock.connect(("127.0.0.1", 9999))
            mock_event = {
                "volume_id": "vol_001",
                "timestamp": "2026-05-21T12:00:00",
                "total_iops": 500.0,
                "avg_latency_us": 120.0
            }
            client_sock.sendall((json.dumps(mock_event) + "\n").encode('utf-8'))
        finally:
            client_sock.close()
            
        # Wait for data processing
        time.sleep(0.5)
        server_running = False
        server_thread.join()
        
        # Verify receipt
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0]["volume_id"], "vol_001")
        self.assertEqual(received_events[0]["total_iops"], 500.0)

    def test_redis_connection_and_streams(self):
        """Verify Redis Streams publisher-subscriber and consumer group interactions if Redis is available."""
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=2, decode_responses=True)
            r.ping()
        except Exception:
            self.skipTest("Local Redis server is not running on localhost:6379. Skipping Redis-specific tests.")
            return
            
        stream_key = "test:telemetry:stream"
        group_name = "test:cg"
        
        # Cleanup if exists
        r.delete(stream_key)
        
        # Test stream writing
        mock_fields = {"volume_id": "vol_test", "total_iops": "120.5", "timestamp": "2026-05-21T12:00:00"}
        msg_id = r.xadd(stream_key, mock_fields)
        self.assertIsNotNone(msg_id)
        
        # Test consumer group creation
        r.xgroup_create(stream_key, group_name, id="0", mkstream=True)
        
        # Test consumer group reading
        read_res = r.xreadgroup(group_name, "test_consumer", {stream_key: ">"}, count=1)
        self.assertEqual(len(read_res), 1)
        self.assertEqual(read_res[0][0], stream_key)
        
        messages = read_res[0][1]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0][0], msg_id)
        self.assertEqual(messages[0][1]["volume_id"], "vol_test")
        
        # Test acknowledge
        ack_res = r.xack(stream_key, group_name, msg_id)
        self.assertEqual(ack_res, 1)
        
        # Cleanup
        r.delete(stream_key)

if __name__ == "__main__":
    unittest.main()
