"""
dashboard/pages/4_Control_Plane.py

Control Plane Configuration & Automation Rebalancing Page.
Covers:
- What-If Capacity Extension Simulators.
- Automated Policy parameters configurations.
- Manual overrides (dispatch rebalancing actions, trigger manual rollbacks).
- Active rebalance execution monitors (with inline rollback actuators).
- Historical rebalancing operation logs.
"""

import time
import pandas as pd
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    apply_custom_css,
    get_api_data,
    post_api_data,
    put_api_data,
    render_sidebar_telemetry
)

st.set_page_config(
    page_title="Control Plane Management - HPE Storage Console",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Render standard sidebar
health = render_sidebar_telemetry()
if not health:
    st.error(f"API Connection Offline. Debug: {st.session_state.get('last_api_error', 'No exception recorded')}")
    st.stop()

# --- Page Headers ---
st.markdown("<h1>🛠️ Control Loop Rebalancing & Policy</h1>", unsafe_allow_html=True)
st.markdown("Manage automated rebalancing configurations, dispatch manual overrides, track active execution actuators, and run capacity simulations.", unsafe_allow_html=True)
st.markdown("---")

# Tab navigation for page 4
tab_sim, tab_policy, tab_override, tab_history = st.tabs([
    "📅 What-If Capacity Simulator",
    "⚙️ Automation Policy Settings",
    "⚡ Manual Execution Overrides",
    "📋 Monitor & Rebalance Logs"
])


# --- SUB-TAB 1: WHAT-IF CAPACITY EXTENSION SIMULATOR ---
with tab_sim:
    st.subheader("What-If Capacity Extension Scenario Simulator")
    st.markdown("Inject simulated storage capacity into an active volume to calculate the days-to-fill relief and recommendation changes.")
    
    dtf_list = get_api_data("/forecast/dtf") or []
    if not dtf_list:
        st.warning("Days-To-Fill projections unavailable. Make sure api server has run volume analysis.")
    else:
        df_dtf = pd.DataFrame(dtf_list)
        
        sim_col_l, sim_col_r = st.columns([1, 1])
        
        with sim_col_l:
            target_vol = st.selectbox("Select Target Volume for Simulation:", df_dtf["volume_id"].tolist())
            added_gb = st.slider("Additional Capacity to Inject (GB):", min_value=50, max_value=5000, step=50, value=500)
            
            if st.button("Calculate Relief Scenario"):
                payload = {"volume_id": target_vol, "added_gb": float(added_gb)}
                res = post_api_data("/simulate/capacity", payload)
                
                if "detail" in res:
                    st.error(f"Simulation execution failed: {res['detail']}")
                else:
                    with sim_col_r:
                        orig_dtf = res.get('original_dtf_days')
                        sim_dtf = res.get('simulated_dtf_days')
                        orig_str = f"{orig_dtf:.1f} days" if orig_dtf is not None else "N/A"
                        sim_str = f"{sim_dtf:.1f} days" if sim_dtf is not None else "N/A"
                        
                        st.markdown(f"""
                        <div class="metric-card" style="border-color:#00f0ff;">
                            <h3 style="color:#00f0ff; margin-top:0;">Simulation Results: {target_vol}</h3>
                            <p><strong>Current Size:</strong> {res.get('current_total_gb', 0.0):.1f} GB</p>
                            <p><strong>New Size:</strong> {res.get('new_total_gb', 0.0):.1f} GB</p>
                            <p><strong>Original DTF:</strong> {orig_str}</p>
                            <p><strong>Simulated DTF:</strong> {sim_str}</p>
                            <p><strong>Capacity Relief Extension:</strong> <span style="color:#00e676; font-weight:bold;">+{res.get('improvement_days', 0.0):.1f} days</span></p>
                            <hr style="border-color:rgba(255,255,255,0.05)"/>
                            <p><em>Recommendation: {res.get('recommendation', 'No Action')}</em></p>
                        </div>
                        """, unsafe_allow_html=True)


# --- SUB-TAB 2: AUTOMATION POLICY SETTINGS ---
with tab_policy:
    st.subheader("Rebalancing Automation & Safety Policies")
    st.markdown("Set thresholds, scheduling limits, and safety metrics for the rebalance loop actuator daemon.")
    
    policy = get_api_data("/policy") or {}
    
    with st.form("policy_config_form"):
        pcol_1, pcol_2 = st.columns(2)
        
        with pcol_1:
            st.markdown("##### Actuator Boundaries")
            enabled = st.checkbox("Enable Automated Control Loop", value=policy.get("rebalance_policy", {}).get("enabled", True))
            dry_run = st.checkbox("Dry Run Mode (Simulate without moving)", value=policy.get("rebalance_policy", {}).get("dry_run_mode", False))
            min_score = st.slider("Min Hotspot Score to Action:", min_value=10, max_value=100, value=int(policy.get("rebalance_policy", {}).get("min_hotspot_score_to_trigger", 70)))
            
        with pcol_2:
            st.markdown("##### Schedule & Safety bounds")
            min_duration = st.slider("Min Hotspot Duration before moving (min):", min_value=1, max_value=30, value=int(policy.get("rebalance_policy", {}).get("min_hotspot_duration_minutes", 2)))
            max_moves = st.slider("Max Volume Migrations Per Hour:", min_value=1, max_value=20, value=int(policy.get("rebalance_policy", {}).get("max_volumes_moved_per_hour", 3)))
            rollback_pct = st.slider("Rollback Target Latency Limit (% increase):", min_value=5, max_value=100, value=int(policy.get("safety_guardrails", {}).get("rollback_if_target_latency_increases_pct", 20)))
            
        if st.form_submit_button("Update Active Parameters"):
            payload = {
                "rebalance_policy": {
                    "enabled": enabled,
                    "dry_run_mode": dry_run,
                    "min_hotspot_score_to_trigger": float(min_score),
                    "min_hotspot_duration_minutes": float(min_duration),
                    "max_volumes_moved_per_hour": int(max_moves)
                },
                "safety_guardrails": {
                    "rollback_if_target_latency_increases_pct": float(rollback_pct)
                }
            }
            res = put_api_data("/policy", payload)
            if "status" in res and res["status"] == "success":
                st.success("🟢 Rebalance daemon safety parameters updated.")
                time.sleep(1.0)
                st.rerun()
            else:
                st.error("Failed to commit settings update.")


# --- SUB-TAB 3: MANUAL EXECUTION OVERRIDES ---
with tab_override:
    st.subheader("Manual Control Plane Overrides")
    st.markdown("Directly issue command instructions to the actuators, or initiate rollbacks on running actions.")
    
    ocol1, ocol2 = st.columns(2)
    
    with ocol1:
        st.markdown("##### Dispatch Rebalance Actuator Action")
        with st.form("manual_rebalance_form"):
            vol_id = st.text_input("Volume ID:", "vol_003")
            action_type = st.selectbox("Action Type:", ["migrate", "qos", "tier_change", "reschedule_job"])
            target = st.text_input("Target Target (Node name, IOPS Limit, Tier, or Time slot):", "node_02")
            
            if st.form_submit_button("Dispatch Operation"):
                payload = {
                    "volume_id": vol_id,
                    "action_type": action_type,
                    "target": target
                }
                res = post_api_data("/rebalance", payload)
                if "status" in res and res["status"] == "success":
                    st.success(f"🟢 Actuator command dispatched. Action ID: {res['action_id']}")
                else:
                    st.error(f"Dispatch failed: {res.get('detail', 'Unknown error')}")
                    
    with ocol2:
        st.markdown("##### Dispatch Manual Rollback")
        with st.form("manual_rollback_form"):
            action_id = st.text_input("Action ID to Rollback:")
            
            if st.form_submit_button("Trigger Actuator Rollback"):
                payload = {"action_id": action_id}
                res = post_api_data("/rollback", payload)
                if "status" in res and res["status"] == "success":
                    st.success(res["message"])
                else:
                    st.error(f"Rollback execution failed: {res.get('detail', 'Unknown error')}")


# --- SUB-TAB 4: MONITOR & REBALANCE LOGS ---
with tab_history:
    # 1. Operational monitor counts
    monitors = get_api_data("/rebalance/monitors") or {}
    history = get_api_data("/rebalance/history") or []
    
    active_monitors_count = sum(1 for m in monitors.values() if m.get("status") == "monitoring")
    success_count = sum(1 for m in monitors.values() if m.get("status") == "success")
    rolled_back_count = sum(1 for m in monitors.values() if m.get("status") == "rolled_back")
    total_monitors = len(monitors)
    rollback_rate = (rolled_back_count / total_monitors * 100.0) if total_monitors > 0 else 0.0
    
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Monitors</div>
            <div class="metric-value">{active_monitors_count}</div>
            <div class="metric-subtitle">Under execution watchdog tracking</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Success Acts</div>
            <div class="metric-value">{success_count}</div>
            <div class="metric-subtitle">Migrations completed cleanly</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rollbacks Tripped</div>
            <div class="metric-value">{rolled_back_count}</div>
            <div class="metric-subtitle">Breach guardrails tripped rollbacks</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rollback Rate</div>
            <div class="metric-value">{rollback_rate:.2f}%</div>
            <div class="metric-subtitle">Watchdog target < 1.00%</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Active monitors table
    st.subheader("Active Watchdog Execution Monitors")
    active_monitors = [m for m in monitors.values() if m.get("status") == "monitoring"]
    
    if not active_monitors:
        st.info("No volume actuator movements currently under active watchdog observation.")
    else:
        h_cols = st.columns([2, 1, 1, 1.2, 1.8, 1.8, 1])
        h_cols[0].markdown("**Action ID**")
        h_cols[1].markdown("**Volume**")
        h_cols[2].markdown("**Type**")
        h_cols[3].markdown("**Elapsed**")
        h_cols[4].markdown("**Pre/Live Latency**")
        h_cols[5].markdown("**Target Details**")
        h_cols[6].markdown("**Action**")
        
        st.markdown("---")
        for mon in active_monitors:
            aid = mon["action_id"]
            vol_id = mon["action_state"].get("volume_id", "—")
            act_type = mon["action_state"].get("action", "—")
            elapsed = f"{mon.get('elapsed_minutes', 0.0):.1f} min"
            pre_lat = f"{mon.get('pre_latency', 0.0):.1f} µs"
            curr_lat = f"{mon.get('current_latency', 0.0):.1f} µs"
            
            if act_type == "migrate":
                target_details = f"Node: {mon['action_state'].get('target_node', '—')}"
            elif act_type == "qos":
                target_details = f"Limit: {mon['action_state'].get('new_iops_limit', '—')} IOPS"
            elif act_type == "tier_change":
                target_details = f"Tier: {mon['action_state'].get('new_tier', '—')}"
            elif act_type == "reschedule_job":
                target_details = f"Workload: {mon['action_state'].get('workload_type', '—')}"
            else:
                target_details = "—"
                
            r_cols = st.columns([2, 1, 1, 1.2, 1.8, 1.8, 1])
            r_cols[0].caption(aid)
            r_cols[1].write(vol_id)
            r_cols[2].write(act_type)
            r_cols[3].write(elapsed)
            r_cols[4].write(f"{pre_lat} → {curr_lat}")
            r_cols[5].write(target_details)
            
            if r_cols[6].button("Rollback", key=f"rb_page_{aid}"):
                payload = {"action_id": aid}
                res = post_api_data("/rollback", payload)
                if "status" in res and res["status"] == "success":
                    st.success(f"Dispatched rollback for {vol_id}!")
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error(f"Failed: {res.get('detail', 'Unknown error')}")

    # 3. Operational Log Table
    st.subheader("Rebalancing History Logger")
    if not history:
        st.info("No logged control plane events.")
    else:
        hist_df = pd.DataFrame(history)
        if "timestamp" in hist_df.columns:
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], format="mixed", errors="coerce")
            hist_df = hist_df.sort_values("timestamp", ascending=False)
            
        display_rows = []
        for _, row in hist_df.iterrows():
            ts_str = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row["timestamp"], pd.Timestamp) else str(row["timestamp"])
            
            # Safely handle the DataFrame parsing of dictionary columns where NaNs can be cast as floats
            action_state = row.get("action_state")
            if not isinstance(action_state, dict):
                action_state = {}
                
            details = ""
            action_name = action_state.get("action", row.get("action", "—"))
            
            if action_name == "migrate":
                details = f"Target Node: {action_state.get('target_node')}"
            elif action_name == "qos":
                details = f"IOPS Limit: {action_state.get('new_iops_limit', action_state.get('iops_limit'))}"
            elif action_name == "tier_change":
                details = f"New Tier: {action_state.get('new_tier')}"
            elif action_name == "reschedule_job":
                details = f"Workload: {action_state.get('workload_type')}"
            elif action_name == "autoscale_add_node":
                details = f"New Node: {row.get('node_id')} (Reason: {row.get('reason')})"
                
            display_rows.append({
                "Time": ts_str,
                "Action ID": row.get("action_id", "—"),
                "Volume ID": row.get("volume_id", "—"),
                "Action Type": action_name,
                "Status": row.get("status", "—"),
                "Details": details
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, height=300)
