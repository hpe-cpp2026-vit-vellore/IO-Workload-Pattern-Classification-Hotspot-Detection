import json
import uuid
from typing import Dict, Any, List
from confluent_kafka import Producer, Consumer
from src.infrastructure.interfaces import EventBus
from configs.settings import settings

class KafkaBus(EventBus):
    def __init__(self):
        self.producer = None
        self.consumer = None
        self._state_cache = {}  # In-memory KV fallback for state management

    def connect(self) -> None:
        """Establish connections to the Kafka cluster."""
        # Producer Configuration
        producer_conf = {'bootstrap.servers': settings.kafka_brokers}
        self.producer = Producer(producer_conf)

        # Consumer Configuration (Using Consumer Groups for horizontal scaling)
        consumer_conf = {
            'bootstrap.servers': settings.kafka_brokers,
            'group.id': 'hpe-control-plane-group',
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True
        }
        self.consumer = Consumer(consumer_conf)

    def publish(self, stream_name: str, payload: Dict[str, Any]) -> str:
        """Publish an event to a Kafka Topic."""
        msg_id = str(uuid.uuid4())
        # Flatten and serialize payload
        flat_payload = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in payload.items()}
        
        self.producer.produce(topic=stream_name, key=msg_id, value=json.dumps(flat_payload))
        self.producer.flush()
        return msg_id

    def read_stream(self, stream_name: str, last_id: str = "$", count: int = 100) -> List[Any]:
        """Read events from a Kafka Topic, returning data in the expected legacy stream format."""
        self.consumer.subscribe([stream_name])
        msgs = self.consumer.consume(num_messages=count, timeout=1.0)
        
        results = []
        for msg in msgs:
            if msg is None:
                continue
            if msg.error():
                continue
            
            try:
                val = json.loads(msg.value().decode('utf-8'))
                key = msg.key().decode('utf-8') if msg.key() else str(uuid.uuid4())
                results.append((key, val))
            except Exception:
                pass
                
        if not results:
            return []
            
        # Emulate the nested return structure expected by the stream worker: [[stream_name, [(msg_id, payload)]]]
        return [[stream_name, results]]

    def get_latest_state(self, key: str) -> Dict[str, Any]:
        """Retrieve state from the local replica cache."""
        return self._state_cache.get(key, {})

    def set_state(self, key: str, state: Dict[str, Any]) -> None:
        """Cache state locally and broadcast to a compacted Kafka topic."""
        self._state_cache[key] = state
        self.producer.produce(topic="hpe-control-plane-state", key=key, value=json.dumps(state))
        self.producer.flush()

    def publish_dlq(self, payload: str, error: str) -> None:
        """Publish failed payload to a dedicated Kafka DLQ topic.
        
        Messages are published to 'hpe_telemetry_dlq' topic for offline
        inspection and forensic debugging.
        """
        if not self.producer:
            return
        try:
            import logging
            import pandas as pd
            dlq_message = json.dumps({
                "payload": payload,
                "error": str(error),
                "timestamp": str(pd.Timestamp.now()),
            })
            self.producer.produce(
                "hpe_telemetry_dlq",
                value=dlq_message.encode("utf-8"),
            )
            self.producer.poll(0)
            logging.getLogger("kafka_bus").info(
                "Published poison message to DLQ topic 'hpe_telemetry_dlq'"
            )
        except Exception as dlq_err:
            logging.getLogger("kafka_bus").error(
                "Failed to publish to Kafka DLQ: %s", dlq_err
            )
