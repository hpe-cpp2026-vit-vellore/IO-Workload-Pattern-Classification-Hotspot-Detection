from configs.settings import settings
from src.infrastructure.interfaces import EventBus
from src.infrastructure.redis_bus import RedisBus
from src.infrastructure.kafka_bus import KafkaBus

def get_event_bus() -> EventBus:
    """Returns the configured enterprise event bus based on settings."""
    if settings.bus_type == "redis":
        bus = RedisBus()
        bus.connect()
        return bus
    elif settings.bus_type == "kafka":
        bus = KafkaBus()
        bus.connect()
        return bus
    else:
        raise ValueError(f"Unknown bus type: {settings.bus_type}")
