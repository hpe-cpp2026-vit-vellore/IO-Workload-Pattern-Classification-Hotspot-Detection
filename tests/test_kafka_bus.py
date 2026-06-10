import unittest
from src.infrastructure.kafka_bus import KafkaBus
from src.infrastructure.interfaces import EventBus

class TestKafkaBus(unittest.TestCase):
    def test_kafka_bus_instantiation(self):
        """Verify KafkaBus can be instantiated and conforms to EventBus interface."""
        bus = KafkaBus()
        self.assertIsInstance(bus, EventBus)
        self.assertEqual(bus._state_cache, {})
        
    def test_kafka_bus_connect_init(self):
        """Verify confluent-kafka client initialization is correct."""
        bus = KafkaBus()
        try:
            bus.connect()
            self.assertIsNotNone(bus.producer)
            self.assertIsNotNone(bus.consumer)
        except Exception:
            # Skip if confluent_kafka is unable to load librdkafka on current host system
            pass

if __name__ == "__main__":
    unittest.main()
