"""
api/main.py

FastAPI Application Layer (HPE Blueprint Phase 6)
===================================================
Provides REST API access to the storage control plane, telemetry analysis,
what-if simulators, decision engines, and system performance indicators.
"""

import sys
import shap
import uvicorn
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
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

# Global engines
hub: Optional[InferenceHub] = None
rebalancer: Optional[Rebalancer] = None
monitor: Optional[ActionMonitor] = None
engine: Optional[DecisionEngine] = None
simulator: Optional[WhatIfSimulator] = None
explainer: Optional[shap.TreeExplainer] = None

# In-memory cache for live telemetry state (warmup)
cached_analysis: Dict[str, Dict[str, Any]] = {}

@app.on_event("startup")
async def startup_event():
    global hub, rebalancer, monitor, engine, simulator, explainer, cached_analysis
    logger.info("Initializing Control Plane engines and loading ML models...")
    
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
    
    # Warmup analysis cache with the latest timestamp in the dataset
    logger.info("Pre-warming volume analysis cache...")
    all_vols = hub.features_df["volume_id"].unique()
    latest_ts = hub.features_df["timestamp"].max()
    
    for vol in all_vols:
        try:
            cached_analysis[vol] = hub.analyze_volume(vol, latest_ts)
        except Exception as e:
            logger.error(f"Error pre-warming cache for volume {vol}: {e}")
            
    logger.info("Control Plane engines and models successfully loaded.")


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
    return {"status": "healthy", "service": "HPE Storage API", "loaded_models": ["lightgbm", "isolation_forest", "lstm_ae", "nbeats", "tft"]}


@app.get("/volumes", status_code=status.HTTP_200_OK)
def get_volumes():
    """Get all 50 volumes in the storage pool with their current status and hotspot scores."""
    result = []
    for vol_id, analysis in cached_analysis.items():
        # Get active tier
        tier = "HDD"
        if hub.topology.graph.has_node(vol_id):
            tier = hub.topology.graph.nodes[vol_id].get("tier", "HDD")
            
        result.append({
            "volume_id": vol_id,
            "hotspot_score": analysis["hotspot_score"],
            "workload_type": analysis["workload_type"],
            "tier": tier,
            "capacity_used_pct": analysis["days_to_fill"].get("capacity_used_pct", 50.0), # estimated
            "latency_risk_score": analysis["latency_risk_score"]
        })
    return result


@app.get("/volumes/{id}/metrics", status_code=status.HTTP_200_OK)
def get_volume_metrics(id: str, limit: int = Query(100, ge=1, le=500)):
    """Retrieve historical time-series telemetry data for a volume."""
    validate_volume(id)
    df_vol = hub.features_df[hub.features_df["volume_id"] == id].sort_values("timestamp").tail(limit)
    
    records = []
    for _, row in df_vol.iterrows():
        records.append({
            "timestamp": row["timestamp"].isoformat(),
            "read_iops": float(row["read_iops"]),
            "write_iops": float(row["write_iops"]),
            "total_iops": float(row["total_iops"]),
            "read_throughput_mbps": float(row["read_throughput_mbps"]),
            "write_throughput_mbps": float(row["write_throughput_mbps"]),
            "avg_latency_us": float(row["avg_latency_us"]),
            "read_latency_p95_us": float(row["read_latency_p95_us"]),
            "write_latency_p95_us": float(row["write_latency_p95_us"]),
            "queue_depth": float(row["queue_depth"]),
            "capacity_used_pct": float(row.get("capacity_used_pct", 0.0))
        })
    return records


@app.get("/volumes/{id}/metrics/stream")
def stream_volume_metrics(id: str):
    """Server-Sent Events (SSE) stream of real-time volume metrics."""
    validate_volume(id)
    
    async def event_generator():
        # Stream recent rows simulating continuous ingestion
        df_vol = hub.features_df[hub.features_df["volume_id"] == id].sort_values("timestamp").tail(30)
        for _, row in df_vol.iterrows():
            payload = {
                "timestamp": row["timestamp"].isoformat(),
                "total_iops": float(row["total_iops"]),
                "avg_latency_us": float(row["avg_latency_us"]),
                "queue_depth": float(row["queue_depth"]),
                "capacity_used_pct": float(row.get("capacity_used_pct", 0.0))
            }
            yield f"data: {payload}\n\n"
            await asyncio.sleep(1.0) # Yield every 1 second
            
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
        "confidence": analysis["workload_confidence"]
    }


@app.get("/volumes/{id}/explain", status_code=status.HTTP_200_OK)
def get_volume_explanation(id: str):
    """SHAP-based explainability details for the workload classification."""
    validate_volume(id)
    
    # Run SHAP on classifier
    row = hub.get_raw_feature_row(id, pd.to_datetime(hub.features_df["timestamp"].max()))
    
    classifier_feature_cols = hub.classifier_scaler.feature_names_in_.tolist()
    features_vec = row[classifier_feature_cols].values.reshape(1, -1)
    features_scaled = hub.classifier_scaler.transform(features_vec)
    
    pred_class = int(hub.classifier.predict(features_scaled)[0])
    shap_vals = explainer.shap_values(features_scaled)
    
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
                
            alerts.append({
                "volume_id": vol_id,
                "hotspot_score": score,
                "severity": severity,
                "workload_type": analysis["workload_type"],
                "timestamp": analysis["timestamp"]
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
        "forecast_24h": analysis["bandwidth_forecast_24h"]
    }


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


# --- What-If Simulator Endpoints ---

@app.post("/simulate/capacity", status_code=status.HTTP_200_OK)
def post_simulate_capacity(req: SimulateCapacityRequest):
    """Simulate adding storage capacity (GB) to a volume."""
    validate_volume(req.volume_id)
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
            
        # 3. Latency Risk SLO breaches
        risk_score = analysis.get("latency_risk_score", 0.0)
        if risk_score >= 0.8:
            recs.append({
                "volume_id": vol_id,
                "priority": "MEDIUM",
                "message": f"High risk of SLO latency breach (peak probability: {round(risk_score * 100, 1)}%) in next 24h."
            })
            
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
            monitor.rollback_threshold = safety["rollback_if_target_latency_increases_pct"]
        if "rollback_timeout_minutes" in safety:
            monitor.timeout = safety["rollback_timeout_minutes"]
            
    return {"status": "success", "message": "Policy parameters updated successfully.", "policy": engine.policy}


# --- Execution Control Endpoints ---

@app.post("/rebalance", status_code=status.HTTP_200_OK)
def trigger_rebalance(req: RebalanceRequest):
    """Manually invoke a rebalance action (migrate, QoS shaping, or tier-change)."""
    validate_volume(req.volume_id)
    
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
        
    action_id = str(uuid.uuid4()) if hasattr(uuid, "uuid4") else "manual-rebalance"
    import uuid
    action_id = str(uuid.uuid4())
    
    monitor.register_action(
        action_id=action_id,
        action_state=action_state,
        pre_latency=pre_latency,
        timestamp=pd.Timestamp.now()
    )
    
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


@app.get("/kpi", status_code=status.HTTP_200_OK)
def get_kpis():
    """Global system-wide KPIs."""
    mon_summary = monitor.get_summary()
    
    # Calculate average latency across all volumes
    all_latencies = [
        float(hub.topology._volume_metrics[v].get("avg_latency_us", 0))
        for v in hub.topology._volume_metrics
    ]
    avg_latency = float(np.mean(all_latencies)) if all_latencies else 1000.0
    
    # Total IOPS
    all_iops = [
        float(hub.topology._volume_metrics[v].get("total_iops", 0))
        for v in hub.topology._volume_metrics
    ]
    total_iops = float(np.sum(all_iops))
    
    return {
        "rollback_rate_pct": mon_summary["rollback_rate_pct"],
        "total_actions": mon_summary["total_actions"],
        "rolled_back_count": mon_summary["rolled_back_count"],
        "avg_latency_us": round(avg_latency, 2),
        "total_iops": round(total_iops, 2),
        "classifier_accuracy_target_pct": 95.0,
        "latency_variance_reduction_target_pct": 30.0
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
