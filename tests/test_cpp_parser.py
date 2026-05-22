import unittest
import json
import sys
from pathlib import Path

# Setup pathing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.telemetry_parser import (
    parse_and_clip,
    CPP_AVAILABLE,
    load_or_create_bounds,
)


class TestCppTelemetryParser(unittest.TestCase):

    def setUp(self):
        self.bounds = {
            "low": {
                "total_iops": 0.0,
                "avg_latency_us": 10.0,
                "read_iops": 0.0,
                "write_iops": 0.0,
            },
            "high": {
                "total_iops": 10000.0,
                "avg_latency_us": 2000.0,
                "read_iops": 5000.0,
                "write_iops": 5000.0,
            }
        }

    def test_compilation_status(self):
        """Verify whether C++ or Python fallback is being used."""
        print(f"\n[INFO] C++ Telemetry Parser compiler detection status: CPP_AVAILABLE = {CPP_AVAILABLE}")

    def test_normal_parsing(self):
        """Verify that valid values inside bounds are preserved exactly."""
        event_str = '{"volume_id": "vol_123", "timestamp": "2026-05-22T12:00:00", "total_iops": 500.0, "avg_latency_us": 120.5, "node_id": "node_A"}'
        result = parse_and_clip(event_str, self.bounds)
        self.assertEqual(result["volume_id"], "vol_123")
        self.assertEqual(result["total_iops"], 500.0)
        self.assertEqual(result["avg_latency_us"], 120.5)
        self.assertEqual(result["node_id"], "node_A")

    def test_outlier_clipping_high(self):
        """Verify that values exceeding the high bound are clipped to the high bound."""
        # Force a massive outlier
        event_str = '{"volume_id": "vol_123", "total_iops": 999999.0, "avg_latency_us": 888888.0}'
        result = parse_and_clip(event_str, self.bounds)
        
        # Check high clipping
        self.assertEqual(result["total_iops"], self.bounds["high"]["total_iops"])
        self.assertEqual(result["avg_latency_us"], self.bounds["high"]["avg_latency_us"])

    def test_outlier_clipping_low(self):
        """Verify that values falling below the low bound are clipped to the low bound."""
        # Force a negative outlier
        event_str = '{"volume_id": "vol_123", "total_iops": -50.0, "avg_latency_us": -5.0}'
        result = parse_and_clip(event_str, self.bounds)
        
        # Check low clipping
        self.assertEqual(result["total_iops"], self.bounds["low"]["total_iops"])
        self.assertEqual(result["avg_latency_us"], self.bounds["low"]["avg_latency_us"])

    def test_handling_null_and_strings(self):
        """Verify that non-numeric values (like null/NaN) or non-clipped fields are preserved without error."""
        event_str = '{"volume_id": "vol_123", "total_iops": null, "avg_latency_us": "N/A", "capacity_used_pct": NaN}'
        result = parse_and_clip(event_str, self.bounds)
        self.assertIsNone(result["total_iops"])
        self.assertEqual(result["avg_latency_us"], "N/A")
        # In JSON representation NaN is handled as is (nan/NaN string or parsed properly)
        
    def test_empty_or_malformed_json_fallback(self):
        """Verify that malformed JSON is handled gracefully by falling back to standard loads."""
        malformed = '{"volume_id": "vol_123", "total_iops": 500' # Missing closing brace
        with self.assertRaises(Exception):
            parse_and_clip(malformed, self.bounds)


if __name__ == "__main__":
    unittest.main()
