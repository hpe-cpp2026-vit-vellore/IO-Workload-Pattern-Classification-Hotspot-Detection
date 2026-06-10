import redis
import json
import socket
from typing import Dict, Any, List
from src.infrastructure.interfaces import EventBus
from configs.settings import settings

class RedisBus(EventBus):
    def __init__(self):
        self.client = None

    def connect(self) -> None:
        socket_keepalive_options = {}
        if hasattr(socket, "TCP_KEEPIDLE"):
            socket_keepalive_options[socket.TCP_KEEPIDLE] = 10
        if hasattr(socket, "TCP_KEEPINTVL"):
            socket_keepalive_options[socket.TCP_KEEPINTVL] = 5
        if hasattr(socket, "TCP_KEEPCNT"):
            socket_keepalive_options[socket.TCP_KEEPCNT] = 3

        self.client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=5,
            socket_keepalive=True,
            socket_keepalive_options=socket_keepalive_options,
            retry_on_timeout=True,
            health_check_interval=15,
        )
        # Test connection
        self.client.ping()

    def publish(self, stream_name: str, payload: Dict[str, Any]) -> str:
        # Convert nested dicts to JSON strings for Redis hash compatibility
        flat_payload = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in payload.items()}
        return self.client.xadd(stream_name, flat_payload)

    def read_stream(self, stream_name: str, last_id: str = "$", count: int = 100) -> List[Any]:
        return self.client.xread({stream_name: last_id}, count=count, block=100)

    def get_latest_state(self, key: str) -> Dict[str, Any]:
        data = self.client.get(key)
        return json.loads(data) if data else {}

    def set_state(self, key: str, state: Dict[str, Any]) -> None:
        self.client.set(key, json.dumps(state))

    def __getattr__(self, name: str) -> Any:
        """Delegate missing attributes/methods to the underlying Redis client."""
        if self.client is not None:
            return getattr(self.client, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
