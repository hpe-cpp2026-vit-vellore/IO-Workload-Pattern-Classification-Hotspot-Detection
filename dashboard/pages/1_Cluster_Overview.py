"""
dashboard/pages/1_Cluster_Overview.py

Cluster Overview & Telemetry Dashboard.
Presents high-level KPIs, high-frequency live telemetry (isolated via st.fragment),
a pixel-dense Storage Admin volumes list with inline SVG sparklines, and prescriptive control plane recommendations.
"""

import time
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    apply_custom_css,
    apply_dark_theme,
    get_api_data,
    render_sidebar_telemetry,
    generate_sparkline_svg
)

st.set_page_config(
    page_title="Cluster Overview - HPE Storage Console",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styling
apply_custom_css()

# Render standard sidebar telemetry
health = render_sidebar_telemetry()
if not health:
    st.error("API Connection Offline.")
    st.stop()

# Title
st.markdown("<h1>⚡ Storage Pool Live Status</h1>", unsafe_allow_html=True)
st.markdown("Real-time telemetry aggregation, workload profiling, and system-wide performance indices.", unsafe_allow_html=True)
st.markdown("---")

# Initialize sparkline history cache in session state if not present
if "sparkline_history" not in st.session_state:
    st.session_state["sparkline_history"] = {}

# --- High-Frequency Telemetry Fragment ---
@st.fragment
def live_telemetry_view():
    kpi_data = get_api_data("/kpi") or {}
    volumes_list = get_api_data("/volumes") or []
    alerts = get_api_data("/alerts") or []
    capacity_plan = get_api_data("/capacity/plan") or {}
    
    is_live = kpi_data.get("is_live", False)
    source = kpi_data.get("source", "unknown")
    current_tick = kpi_data.get("current_tick")
    live_vol_count = kpi_data.get("live_volume_count", 0)
    events_count = kpi_data.get("events_received", 0)
    
    # 1. Live status bar
    if is_live:
        tick_display = current_tick if current_tick else "—"
        st.markdown(f"""
        <div style="
            background: rgba(0, 230, 118, 0.08);
            border: 1px solid rgba(0, 230, 118, 0.25);
            border-radius: 10px;
            padding: 10px 18px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 24px;
            font-size: 13px;
        ">
            <span style="color: #00e676; font-weight: 700; animation: pulse 1.5s infinite;">● LIVE TELEMETRY</span>
            <span style="color: #c9d1d9;">Simulated Tick: <strong style="color:#00f0ff;">{tick_display}</strong></span>
            <span style="color: #8b949e;">Source: {source}</span>
            <span style="color: #8b949e;">Volumes Active: <strong style="color:#ffffff;">{live_vol_count}</strong></span>
            <span style="color: #8b949e;">Events Count: <strong style="color:#ffffff;">{events_count:,}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background: rgba(255, 145, 0, 0.06);
            border: 1px solid rgba(255, 145, 0, 0.2);
            border-radius: 10px;
            padding: 10px 18px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 24px;
            font-size: 13px;
        ">
            <span style="color: #ff9100; font-weight: 700;">◉ HISTORICAL PLAYBACK</span>
            <span style="color: #c9d1d9;">Showing static Parquet snapshot — start telemetry generator for live stream</span>
            <span style="color: #8b949e;">Source: {source}</span>
        </div>
        """, unsafe_allow_html=True)

    # 2. KPI Cards row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_lat = kpi_data.get("avg_latency_us", 0.0)
        lat_color = "#ff1744" if avg_lat > 1500 else "#00e676" if avg_lat < 800 else "#ff9100"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Latency</div>
            <div class="metric-value" style="color:{lat_color};">{avg_lat:.1f} <span style="font-size:16px;">µs</span></div>
            <div class="metric-subtitle">SLO target < 1500 µs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        total_iops = kpi_data.get("total_iops", 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Aggregate IOPS</div>
            <div class="metric-value">{int(total_iops):,}</div>
            <div class="metric-subtitle">Active Storage Pool IOPS</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        total_actions = kpi_data.get("total_actions", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rebalance Actions</div>
            <div class="metric-value">{total_actions}</div>
            <div class="metric-subtitle">Dispatched optimizations</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        rollback_rate = kpi_data.get("rollback_rate_pct", 0.0)
        rb_color = "#ff1744" if rollback_rate > 1.0 else "#00e676"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rollback Rate</div>
            <div class="metric-value" style="color:{rb_color};">{rollback_rate:.2f}%</div>
            <div class="metric-subtitle">SLO target < 1.0%</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Storage Admin High-Density Table
    st.subheader("📋 Storage Admin - High-Density Volume Status")
    if volumes_list:
        # Update rolling latency history for sparklines
        for vol in volumes_list:
            vol_id = vol["volume_id"]
            lat = vol.get("current_latency_us") or 0.0
            if vol_id not in st.session_state["sparkline_history"]:
                st.session_state["sparkline_history"][vol_id] = []
            history = st.session_state["sparkline_history"][vol_id]
            history.append(lat)
            if len(history) > 12:
                history.pop(0)
                
        # Build dense table
        table_rows = []
        for vol in volumes_list:
            vol_id = vol["volume_id"]
            workload = vol.get("workload_type", "Unknown")
            tier = vol.get("tier", "HDD")
            iops = vol.get("current_iops") or 0.0
            latency = vol.get("current_latency_us") or 0.0
            hs_score = vol.get("hotspot_score") or 0.0
            
            # Status styling
            if hs_score >= 70 or latency > 1500:
                badge = '<span class="badge badge-critical">Critical</span>'
                color = "#ff1744"
            elif hs_score >= 40 or latency > 1000:
                badge = '<span class="badge badge-warning">Warning</span>'
                color = "#ff9100"
            else:
                badge = '<span class="badge badge-healthy">Healthy</span>'
                color = "#00e676"
                
            # Sparkline
            history = st.session_state["sparkline_history"].get(vol_id, [])
            sparkline_svg = generate_sparkline_svg(history, color)
            
            table_rows.append(f"""<tr>
<td style="font-family: monospace; font-weight: bold; color: #00f0ff;">{vol_id}</td>
<td>{workload}</td>
<td>{tier}</td>
<td>{int(iops):,}</td>
<td style="color: {color}; font-weight: 600;">{latency:.1f} µs</td>
<td>{hs_score:.1f}</td>
<td>{badge}</td>
<td>{sparkline_svg}</td>
</tr>""")
            
        rows_content = "\n".join(table_rows)
        html_table = f"""<table class="dense-table">
<thead>
<tr>
<th>Volume ID</th>
<th>Workload Class</th>
<th>Tier</th>
<th>IOPS</th>
<th>Latency</th>
<th>Hotspot Score</th>
<th>Status</th>
<th>Latency Trend (12 Ticks)</th>
</tr>
</thead>
<tbody>
{rows_content}
</tbody>
</table>"""
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.info("No active volume metadata returned.")

    st.markdown("---")

    # 4. Prescriptive Action Items + Hotspot Score Grid
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.subheader("💡 Prescriptive Capacity & Performance Plan")
        recommendations = capacity_plan.get("recommendations", [])
        if recommendations:
            for rec in recommendations:
                urgency = rec.get("urgency", "INFO")
                rec_type = rec.get("rec_type", "NO_ACTION")
                title = rec.get("title", "Plan Recommendation")
                desc = rec.get("description", "")
                headroom = rec.get("estimated_headroom_gained_days", 0.0)
                actionable = rec.get("auto_actionable", False)
                
                if urgency == "CRITICAL":
                    border_color = "#ff1744"
                    bg_color = "rgba(255, 23, 68, 0.05)"
                elif urgency in ("HIGH", "MEDIUM"):
                    border_color = "#ff9100"
                    bg_color = "rgba(255, 145, 0, 0.05)"
                else:
                    border_color = "#00e676"
                    bg_color = "rgba(0, 230, 118, 0.05)"
                    
                act_badge = '<span class="badge badge-healthy">Auto-Actionable</span>' if actionable else '<span class="badge badge-warning">Approval Required</span>'
                
                st.markdown(f"""
                <div style="
                    background: {bg_color};
                    border: 1px solid rgba(255, 255, 255, 0.04);
                    border-left: 5px solid {border_color};
                    border-radius: 8px;
                    padding: 16px;
                    margin-bottom: 12px;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <strong style="color: #ffffff; font-size: 14px;">{title}</strong>
                        {act_badge}
                    </div>
                    <p style="margin: 0 0 8px 0; font-size: 13px; color: #c9d1d9;">{desc}</p>
                    <div style="font-size: 11px; color: #8b949e;">
                        <span>Type: <strong style="color: #00f0ff;">{rec_type}</strong></span> &nbsp;|&nbsp;
                        <span>Urgency: <strong style="color: {border_color};">{urgency}</strong></span> &nbsp;|&nbsp;
                        <span>Reclaimed Headroom: <strong style="color: #00e676;">+{headroom:.1f} days</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🟢 All storage nodes and latency SLO limits are currently inside safe levels.")

    with col_r:
        st.subheader("🚨 High-Priority Telemetry Alerts")
        if alerts:
            for alert in alerts[:5]:
                severity = alert["severity"]
                vol_id = alert["volume_id"]
                score = alert["hotspot_score"]
                wl = alert["workload_type"]
                ts = alert["timestamp"]
                
                if severity == "Critical":
                    st.markdown(f"""
                    <div class="alert-panel">
                        <strong style="color:#ff1744;">CRITICAL ALERT: Volume {vol_id}</strong><br/>
                        Hotspot Score: {score:.1f} | Workload: {wl}<br/>
                        <span style="font-size:10px; color:#8b949e;">Timestamp: {ts}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-warning-panel">
                        <strong style="color:#ff9100;">{severity.upper()} ALERT: Volume {vol_id}</strong><br/>
                        Hotspot Score: {score:.1f} | Workload: {wl}<br/>
                        <span style="font-size:10px; color:#8b949e;">Timestamp: {ts}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("🟢 No active anomalies or SLA breaches reported.")

    # Auto-rerun telemetry fragment block if live data is running
    if is_live:
        time.sleep(3)
        st.rerun()

# Run the live telemetry view fragment
live_telemetry_view()

# --- Workload Profile Plot ---
# We keep static plots outside of st.fragment to avoid heavy graph rebuild calculations on high-frequency runs
volumes_list = get_api_data("/volumes") or []
if volumes_list:
    st.subheader("Cluster Workload Pattern Distribution")
    vols_df = pd.DataFrame(volumes_list)
    dist = vols_df["workload_type"].value_counts().reset_index()
    dist.columns = ["Workload Pattern", "Volume Count"]
    
    fig_pie = px.pie(
        dist,
        names="Workload Pattern",
        values="Volume Count",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    apply_dark_theme(fig_pie)
    fig_pie.update_layout(height=350)
    st.plotly_chart(fig_pie, use_container_width=True)
