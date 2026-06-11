from abc import ABC, abstractmethod
from typing import Dict, Any, List

class EventBus(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the message bus."""
        pass

    @abstractmethod
    def publish(self, stream_name: str, payload: Dict[str, Any]) -> str:
        """Publish an event. Returns the message ID."""
        pass

    @abstractmethod
    def read_stream(self, stream_name: str, last_id: str, count: int = 100) -> List[Any]:
        """Read messages from a stream starting after last_id."""
        pass
        
    @abstractmethod
    def get_latest_state(self, key: str) -> Dict[str, Any]:
        """Retrieve a cached state object."""
        pass
        
    @abstractmethod
    def set_state(self, key: str, state: Dict[str, Any]) -> None:
        """Cache a state object."""
        pass

    @abstractmethod
    def publish_dlq(self, payload: str, error: str) -> None:
        """Publish a failed/poison message to the Dead-Letter Queue for offline inspection.
        
        Args:
            payload: The raw message payload that caused the failure (JSON string).
            error: The error message or traceback describing the failure.
        """
        pass
