# API Reference

This document covers all FastAPI endpoints implemented in `api/main.py`.

## Conventions
- Base URL: `http://localhost:8000`
- Content type for POST/PUT: `application/json`
- Path params are shown with braces, for example `{id}`.

---

## 1) Health and Telemetry

### GET /health
- Description: Service health, model load status, telemetry bus mode.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "status": "healthy",
  "service": "HPE Storage API",
  "loaded_models": ["lightgbm", "isolation_forest", "lstm_ae", "nbeats", "tft"],
  "telemetry_bus": {
    "mode": "redis_streams",
    "redis": {"available": true, "host": "redis", "port": 6379, "error": null},
    "tcp_fallback": {"enabled": false, "host": "127.0.0.1", "port": 9000, "listening": false}
  },
  "live_telemetry": {
    "events_received": 1200,
    "live_volume_count": 50,
    "current_tick": "2026-05-31T12:00:00"
  }
}
```

### GET /telemetry/status
- Description: Live telemetry ingestion status and completeness.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "events_received": 1200,
  "live_volume_count": 50,
  "expected_volume_count": 50,
  "current_tick": "2026-05-31T12:00:00",
  "latest_complete_tick": "2026-05-31T11:55:00",
  "current_tick_volume_count": 50,
  "current_tick_complete": true
}
```

### GET /kpi
- Description: Pool-wide KPI snapshot for dashboard overview.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "avg_latency_us": 1320.45,
  "total_iops": 452300.2,
  "total_actions": 12,
  "rolled_back_count": 0,
  "rollback_rate_pct": 0.0,
  "classifier_accuracy_target_pct": 95.0,
  "latency_variance_reduction_target_pct": 30.0,
  "is_live": true,
  "source": "live_telemetry",
  "current_tick": "2026-05-31T12:00:00",
  "live_volume_count": 50
}
```

---

## 2) Volume Data and Explainability

### GET /volumes
- Description: List all volumes with hotspot/workload/tier and live status.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
[
  {
    "volume_id": "vol_000",
    "hotspot_score": 42.18,
    "workload_type": "DB_OLTP",
    "tier": "NVMe",
    "capacity_used_pct": 50.0,
    "latency_risk_score": 0.14,
    "current_iops": 8120.3,
    "current_latency_us": 1180.5,
    "last_seen_timestamp": "2026-05-31T12:00:00",
    "is_live": true,
    "last_analyzed_timestamp": "2026-05-31T12:00:00",
    "analysis_freshness_s": 2.4
  }
]
```

### GET /volumes/{id}/metrics
- Description: Time-series telemetry for one volume.
- Query parameters:
  - `limit` (int, optional, default 100, min 1, max 500)
- Body parameters: none
- Sample response:
```json
[
  {
    "timestamp": "2026-05-31T11:55:00",
    "read_iops": 4300.0,
    "write_iops": 2200.0,
    "total_iops": 6500.0,
    "read_throughput_mbps": 540.2,
    "write_throughput_mbps": 301.9,
    "avg_latency_us": 1240.7,
    "read_latency_p95_us": 1820.4,
    "write_latency_p95_us": 1950.1,
    "queue_depth": 9.0,
    "capacity_used_pct": 0.67
  }
]
```

### GET /volumes/{id}/metrics/stream
- Description: SSE stream of live metrics for a volume.
- Query parameters: none
- Body parameters: none
- Sample response (SSE event payload):
```json
{
  "volume_id": "vol_000",
  "timestamp": "2026-05-31T12:00:00",
  "total_iops": 7000.0,
  "avg_latency_us": 1300.0
}
```

### GET /volumes/{id}/workload
- Description: Current workload prediction and confidence vector.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "volume_id": "vol_000",
  "workload_type": "DB_OLTP",
  "confidence": [0.92, 0.03, 0.01, 0.02, 0.02],
  "arf_workload_type": "DB_OLTP",
  "arf_agrees": true
}
```

### GET /volumes/{id}/explain
- Description: SHAP-based explanation for current classification.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "volume_id": "vol_000",
  "predicted_class": "DB_OLTP",
  "explanation": "Volume 'vol_000' classified as 'DB_OLTP' primarily because ...",
  "feature_contributions": [
    {"feature": "total_iops", "shap_value": 0.231},
    {"feature": "queue_depth", "shap_value": 0.118}
  ]
}
```

### GET /model/drift-status
- Description: ARF availability and ARF-vs-LightGBM agreement.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "arf_loaded": true,
  "lgbm_arf_agreement_rate": 93.2,
  "disagreeing_volumes": ["vol_011", "vol_024"]
}
```

### GET /model/performance
- Description: Validation performance summary (artifact-backed when available).
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "accuracy": 0.986,
  "confusion_matrix": [[100, 2], [3, 95]],
  "confusion_matrix_percentage": [[98.04, 1.96], [3.06, 96.94]],
  "metrics_per_class": {
    "DB_OLTP": {"precision": 0.96, "recall": 0.97, "f1_score": 0.97, "support": 10793}
  },
  "sample_count": 57600
}
```

---

## 3) Alerts and Forecasting

### GET /alerts
- Description: Active hotspot alerts sorted by score.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
[
  {
    "volume_id": "vol_036",
    "hotspot_score": 88.1,
    "severity": "Critical",
    "workload_type": "AI_Inference",
    "timestamp": "2026-05-31T12:00:00"
  }
]
```

### GET /noisy-neighbors
- Description: Aggressor-victim relationships from noisy-neighbor detector.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
[
  {
    "aggressor_id": "vol_036",
    "workload_type": "AI_Inference",
    "hotspot_score": 84.3,
    "victims": [
      {"volume_id": "vol_031", "impact_score": 22.4},
      {"volume_id": "vol_032", "impact_score": 15.7}
    ]
  }
]
```

### GET /forecast/capacity
- Description: DTF forecast summary for all volumes.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
[
  {"volume_id": "vol_000", "warning_85pct_days": 82.3, "critical_95pct_days": 103.0}
]
```

### GET /forecast/bandwidth
- Description: Latency quantile forecast and demand forecast for one volume.
- Query parameters:
  - `volume_id` (string, required)
- Body parameters: none
- Sample response:
```json
{
  "volume_id": "vol_000",
  "forecast_24h": {
    "p50_latency_us": [1200.1, 1195.4],
    "p90_latency_us": [1500.2, 1490.7],
    "p95_latency_us": [1700.9, 1684.3]
  },
  "demand_forecast_24h": {
    "iops_p95": [9000.0, 9200.0],
    "throughput_p95_mbps": [1200.0, 1220.0],
    "peak_iops": 9800.0,
    "peak_throughput_mbps": 1300.0,
    "forecast_hours": ["2026-05-31T12:00:00", "2026-05-31T13:00:00"]
  }
}
```

### GET /forecast/demand
- Description: IOPS and throughput quantile demand forecast for one volume.
- Query parameters:
  - `volume_id` (string, required)
- Body parameters: none
- Sample response:
```json
{
  "volume_id": "vol_000",
  "iops_p50": [7000.0, 7100.0],
  "iops_p95": [9000.0, 9200.0],
  "iops_p99": [9800.0, 9950.0],
  "throughput_p50_mbps": [850.0, 860.0],
  "throughput_p95_mbps": [1200.0, 1220.0],
  "throughput_p99_mbps": [1300.0, 1325.0],
  "peak_iops": 9950.0,
  "peak_throughput_mbps": 1325.0,
  "forecast_hours": ["2026-05-31T12:00:00", "2026-05-31T13:00:00"]
}
```

### GET /forecast/dtf
- Description: DTF urgency ranking across volumes.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
[
  {"volume_id": "vol_012", "warning_85pct_days": 12.4, "critical_95pct_days": 19.8}
]
```

### GET /forecast/ttv
- Description: Latency SLO time-to-violation (single volume or all volumes).
- Query parameters:
  - `volume_id` (string, optional)
- Body parameters: none
- Sample response (single volume):
```json
{
  "volume_id": "vol_000",
  "will_breach": false,
  "hours_to_breach": null,
  "breach_at_step": null,
  "max_p95_forecast_us": 3200.5,
  "slo_threshold_us": 8000.0,
  "current_headroom_us": 4799.5,
  "risk_level": "none"
}
```
- Sample response (all volumes):
```json
[
  {
    "volume_id": "vol_036",
    "workload_type": "AI_Inference",
    "will_breach": true,
    "hours_to_breach": 1.8,
    "breach_at_step": 2,
    "max_p95_forecast_us": 9300.0,
    "slo_threshold_us": 8000.0,
    "current_headroom_us": -1300.0,
    "risk_level": "high"
  }
]
```

---

## 4) Cluster Headroom

### GET /cluster/headroom
- Description: Tier-level and pool-level capacity headroom summary.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "tier_headroom": {
    "NVMe": {
      "total_capacity_gb": 30000.0,
      "used_capacity_gb": 22500.0,
      "used_pct": 75.0,
      "headroom_gb": 7500.0,
      "volume_count": 10,
      "critical_volumes": []
    }
  },
  "pool_headroom": {
    "pool_00_00": {
      "total_capacity_gb": 12000.0,
      "used_capacity_gb": 9000.0,
      "used_pct": 75.0,
      "headroom_gb": 3000.0,
      "volume_count": 6,
      "critical_volumes": []
    }
  }
}
```

### GET /cluster/headroom/tier/{tier_name}
- Description: Headroom for one tier (for example NVMe, SSD, HDD).
- Query parameters: none
- Body parameters: none
- Path parameters:
  - `tier_name` (string, required)
- Sample response:
```json
{
  "tier": "SSD",
  "total_capacity_gb": 85000.0,
  "used_capacity_gb": 64000.0,
  "used_pct": 75.29,
  "headroom_gb": 21000.0,
  "volume_count": 25,
  "critical_volumes": ["vol_044"]
}
```

---

## 5) What-If Simulations

### POST /simulate/capacity
- Description: Simulate adding capacity to a volume.
- Query parameters: none
- Body parameters:
  - `volume_id` (string, required)
  - `added_gb` (float > 0, required)
- Sample request:
```json
{"volume_id": "vol_000", "added_gb": 500.0}
```
- Sample response:
```json
{
  "volume_id": "vol_000",
  "added_gb": 500.0,
  "current_total_gb": 2000.0,
  "new_total_gb": 2500.0,
  "original_dtf_days": 22.4,
  "simulated_dtf_days": 30.8,
  "improvement_days": 8.4,
  "recommendation": "Adding 500.0GB extends time-to-full by 8.4 days."
}
```

### POST /simulate/migrate
- Description: Simulate migrating a volume to another node.
- Query parameters: none
- Body parameters:
  - `volume_id` (string, required)
  - `target_node` (string, required)
- Sample request:
```json
{"volume_id": "vol_000", "target_node": "node_03"}
```
- Sample response:
```json
{
  "volume_id": "vol_000",
  "source_node": "node_00",
  "target_node": "node_03",
  "predicted_source_latency_improvement_pct": 15.0,
  "target_utilization_increase_pct": 7.6,
  "migration_safe": true,
  "estimated_time_seconds": 2400,
  "recommendation": "Migration of vol_000 to node_03 is SAFE. Estimated transfer time: 2400s."
}
```

### POST /simulate/qos
- Description: Simulate QoS IOPS cap and estimated relief.
- Query parameters: none
- Body parameters:
  - `volume_id` (string, required)
  - `iops_limit` (float > 0, required)
- Sample request:
```json
{"volume_id": "vol_000", "iops_limit": 3000.0}
```
- Sample response:
```json
{
  "volume_id": "vol_000",
  "iops_limit": 3000.0,
  "current_iops": 8500.0,
  "throttling_pct": 64.71,
  "impact_on_self_throughput_reduction_pct": 64.71,
  "expected_noisy_neighbor_relief_pct": 19.41,
  "recommendation": "Throttling vol_000 to 3000.0 IOPS will reduce neighboring latency noise by 19.41%."
}
```

### POST /simulate/tier
- Description: Simulate tier change impact.
- Query parameters: none
- Body parameters:
  - `volume_id` (string, required)
  - `new_tier` (string, required)
- Sample request:
```json
{"volume_id": "vol_000", "new_tier": "NVMe"}
```
- Sample response:
```json
{
  "volume_id": "vol_000",
  "current_tier": "SSD",
  "new_tier": "NVMe",
  "expected_latency_improvement_pct": 50.0,
  "expected_throughput_gain_factor": 3.0,
  "recommendation": "Upgrading vol_000 to NVMe yields a predicted 50.0% latency reduction and 3.0x throughput bandwidth."
}
```

---

## 6) Recommendations and Policy

### GET /recommendations
- Description: Priority-sorted recommendation list (capacity, hotspots, TTV, tier headroom).
- Query parameters: none
- Body parameters: none
- Sample response:
```json
[
  {
    "volume_id": "vol_036",
    "priority": "CRITICAL",
    "message": "Latency SLO breach imminent for vol_036 (<1h). Max p95 forecast: 9300us (SLO: 8000us). Immediate QoS shaping or migration required."
  },
  {
    "volume_id": "CLUSTER",
    "priority": "WARNING",
    "message": "SSD storage tier is 82.5% full (9200 GB remaining). Plan capacity expansion."
  }
]
```

### GET /policy
- Description: Get active policy document used by control plane.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "rebalance_policy": {"enabled": true, "min_hotspot_score_to_trigger": 75},
  "safety_guardrails": {"rollback_if_target_latency_increases_pct": 20, "max_rollback_rate_pct": 1.0},
  "capacity_policy": {"latency_slo_threshold_us": 8000.0}
}
```

### PUT /policy
- Description: Update rebalance and/or safety policy fields.
- Query parameters: none
- Body parameters:
  - `rebalance_policy` (object, optional)
  - `safety_guardrails` (object, optional)
- Sample request:
```json
{
  "rebalance_policy": {"enabled": true, "max_volumes_moved_per_hour": 3},
  "safety_guardrails": {"max_rollback_rate_pct": 1.0}
}
```
- Sample response:
```json
{
  "status": "success",
  "message": "Policy parameters updated successfully.",
  "policy": {"rebalance_policy": {"enabled": true}, "safety_guardrails": {"max_rollback_rate_pct": 1.0}}
}
```

---

## 7) Rebalance and Rollback Controls

### GET /rebalance/circuit-breaker
- Description: Circuit breaker state and rollback-rate context.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "circuit_breaker_tripped": false,
  "tripped_at": null,
  "reason": "",
  "current_rollback_rate_pct": 0.0,
  "max_rollback_rate_pct": 1.0,
  "total_actions": 12,
  "engine_enabled": true
}
```

### POST /rebalance/circuit-breaker/reset
- Description: Manually reset circuit breaker.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{"status": "reset", "engine_enabled": true}
```

### GET /rebalance/history
- Description: Executed actions history.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
[
  {
    "action_id": "9f2f0f2a-2b98-4ad7-8f7d-2f4d15ea5ac0",
    "volume_id": "vol_000",
    "action": "migrate",
    "timestamp": "2026-05-31T12:00:00",
    "status": "executed"
  }
]
```

### GET /rebalance/monitors
- Description: Active and historical monitor records.
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "9f2f0f2a-2b98-4ad7-8f7d-2f4d15ea5ac0": {
    "status": "monitoring",
    "pre_latency": 1200.0,
    "current_latency": 1180.0,
    "elapsed_minutes": 1.5
  }
}
```

### POST /rebalance
- Description: Manually execute rebalance action.
- Query parameters: none
- Body parameters:
  - `volume_id` (string, required)
  - `action_type` (enum: `migrate`, `qos`, `tier_change`)
  - `target` (string, required; interpretation depends on action type)
- Sample request:
```json
{"volume_id": "vol_000", "action_type": "migrate", "target": "node_03"}
```
- Sample response:
```json
{
  "status": "success",
  "action_id": "8aebf3d2-6fbc-4d0b-b1e2-06d0d9fd8b77",
  "action_state": {"action": "migrate", "volume_id": "vol_000", "source_node": "node_00", "target_node": "node_03"}
}
```

### POST /rollback
- Description: Roll back an action by `action_id`.
- Query parameters: none
- Body parameters:
  - `action_id` (string UUID, required)
- Sample request:
```json
{"action_id": "8aebf3d2-6fbc-4d0b-b1e2-06d0d9fd8b77"}
```
- Sample response:
```json
{"status": "success", "message": "Action 8aebf3d2-6fbc-4d0b-b1e2-06d0d9fd8b77 successfully rolled back."}
```

---

## 8) Topology

### GET /topology
- Description: Graph-ready topology export (nodes and edges).
- Query parameters: none
- Body parameters: none
- Sample response:
```json
{
  "nodes": [
    {"id": "node_00", "label": "node_00", "type": "storage_node", "tier": "SSD", "capacity_gb": 50000.0},
    {"id": "vol_000", "label": "vol_000", "type": "volume", "tier": "NVMe", "node_owner": "node_00"}
  ],
  "edges": [
    {"source": "vol_000", "target": "node_00", "type": "resides_on"}
  ]
}
```
