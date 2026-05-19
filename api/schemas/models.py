"""
api/schemas/models.py

Pydantic Schemas for HPE Storage Control Plane API (HPE Blueprint Phase 6)
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

# --- Simulation Schemas ---

class SimulateCapacityRequest(BaseModel):
    volume_id: str = Field(..., description="ID of the volume to simulate capacity addition on")
    added_gb: float = Field(..., gt=0.0, description="Amount of capacity in GB to add")

class SimulateMigrateRequest(BaseModel):
    volume_id: str = Field(..., description="ID of the volume to migrate")
    target_node: str = Field(..., description="Target node ID to migrate the volume to")

class SimulateQosRequest(BaseModel):
    volume_id: str = Field(..., description="ID of the volume to apply QoS limit on")
    iops_limit: float = Field(..., gt=0.0, description="IOPS cap limit")

class SimulateTierRequest(BaseModel):
    volume_id: str = Field(..., description="ID of the volume to change tier on")
    new_tier: str = Field(..., description="Target tier (e.g. SSD, NVMe, HDD)")


# --- Policy Management Schemas ---

class RebalancePolicyUpdate(BaseModel):
    enabled: Optional[bool] = None
    dry_run_mode: Optional[bool] = None
    min_hotspot_score_to_trigger: Optional[float] = None
    min_hotspot_duration_minutes: Optional[float] = None
    max_volumes_moved_per_hour: Optional[int] = None
    max_concurrent_migrations: Optional[int] = None

class SafetyGuardrailsUpdate(BaseModel):
    rollback_if_target_latency_increases_pct: Optional[float] = None
    rollback_timeout_minutes: Optional[float] = None
    max_rollback_rate_pct: Optional[float] = None

class PolicyUpdateRequest(BaseModel):
    rebalance_policy: Optional[RebalancePolicyUpdate] = None
    safety_guardrails: Optional[SafetyGuardrailsUpdate] = None


# --- Action Control Schemas ---

class RebalanceRequest(BaseModel):
    volume_id: str = Field(..., description="ID of the volume to rebalance")
    action_type: str = Field(..., pattern="^(migrate|qos|tier_change)$", description="Type of rebalance action: migrate, qos, or tier_change")
    target: str = Field(..., description="Target node ID, QoS limit, or tier name depending on action_type")

class RollbackRequest(BaseModel):
    action_id: str = Field(..., description="UUID of the action to roll back")
