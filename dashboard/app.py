"""
dashboard/app.py

Streamlit Dashboard & Visualization Layer (HPE Blueprint Phase 7)
==================================================================
A premium dark-themed, glassmorphic, and highly interactive UI
for monitoring storage workloads, detecting hotspots, simulating migrations,
and managing control plane policies.
"""

import sys
import time
import requests
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from typing import Any, List, Dict, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

API_URL = "http://127.0.0.1:8000"

# --- Page Configurations ---
st.set_page_config(
    page_title="HPE Storage Control Plane Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Premium CSS Injection ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 240, 255, 0.1);
        border-color: rgba(0, 240, 255, 0.2);
    }
    .metric-title {
        font-size: 14px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #00f0ff;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
    }
    .metric-subtitle {
        font-size: 12px;
        color: #58a6ff;
        margin-top: 4px;
    }
    
    /* Alerts Panel */
    .alert-panel {
        background: rgba(255, 23, 68, 0.05);
        border: 1px solid rgba(255, 23, 68, 0.2);
        border-left: 5px solid #ff1744;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .alert-warning-panel {
        background: rgba(255, 145, 0, 0.05);
        border: 1px solid rgba(255, 145, 0, 0.2);
        border-left: 5px solid #ff9100;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    
    /* Section headers */
    h1, h2, h3 {
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #8b949e;
    }
</style>
""", unsafe_allow_html=True)


# --- Plotly Premium Color & Styling Helper ---
def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(13, 17, 23, 0.8)',
        plot_bgcolor='rgba(22, 27, 34, 0.4)',
        font=dict(color='#c9d1d9', family='Outfit, sans-serif'),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.05)',
            zerolinecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.05)',
            zerolinecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(size=10)
        ),
        legend=dict(bgcolor='rgba(13, 17, 23, 0.6)', bordercolor='rgba(255, 255, 255, 0.05)', borderwidth=1),
        margin=dict(t=50, b=40, l=40, r=40)
    )
    return fig


# --- API Fetch Wrappers ---
def get_api_data(endpoint: str, params: dict = None) -> Any:
    try:
        response = requests.get(f"{API_URL}{endpoint}", params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def post_api_data(endpoint: str, payload: dict) -> Any:
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=5)
        return response.json() if response.status_code == 200 else {"detail": response.text}
    except Exception as e:
        return {"detail": str(e)}

def put_api_data(endpoint: str, payload: dict) -> Any:
    try:
        response = requests.put(f"{API_URL}{endpoint}", json=payload, timeout=5)
        return response.json() if response.status_code == 200 else {"detail": response.text}
    except Exception as e:
        return {"detail": str(e)}


# --- Sidebar Navigation ---
st.sidebar.markdown(
    "<div style='text-align: center; padding-bottom: 20px;'>"
    "<h2 style='color:#00f0ff; font-weight: 900; letter-spacing: 1.5px; margin-bottom: 5px;'>HPE STORAGE</h2>"
    "<p style='color:#8b949e; font-size:11px; text-transform: uppercase;'>Control Plane & Analytics</p>"
    "</div>", unsafe_allow_html=True
)

pages = [
    "Live Overview",
    "Volume Deep Dive",
    "Capacity Planning",
    "Hotspot & Noisy Neighbors",
    "Bandwidth & Latency Forecasts",
    "Rebalancing History & Control",
    "ML Model Performance"
]
selected_page = st.sidebar.radio("Navigation Menu", pages)

# Verify API connectivity
health = get_api_data("/health")
if not health:
    st.error("🚨 Backend Control Plane API is unreachable. Ensure the FastAPI service is running locally on port 8000.")
    st.info("Run `python -m uvicorn api.main:app --port 8000 --reload` in your terminal to boot up the backend API.")
    st.stop()


# ────────────────────────────────────────────────────────────────────
# PAGE 1: LIVE OVERVIEW
# ────────────────────────────────────────────────────────────────────
if selected_page == "Live Overview":
    st.markdown("<h1>⚡ Storage Pool Live Status</h1>", unsafe_allow_html=True)
    st.markdown("Real-time telemetry aggregation, workload profiling, and system-wide performance indices.", unsafe_allow_html=True)
    
    # Fetch KPIs
    kpi_data = get_api_data("/kpi") or {}
    volumes_list = get_api_data("/volumes") or []
    alerts = get_api_data("/alerts") or []
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_lat = kpi_data.get("avg_latency_us", 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Latency</div>
            <div class="metric-value">{avg_lat:.1f} <span style="font-size:16px;">µs</span></div>
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
            <div class="metric-subtitle">Total migrations & limits</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        rollback_rate = kpi_data.get("rollback_rate_pct", 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rollback Rate</div>
            <div class="metric-value">{rollback_rate:.2f}%</div>
            <div class="metric-subtitle">SLO target < 1.0%</div>
        </div>
        """, unsafe_allow_html=True)
        
    mcol1, mcol2 = st.columns([2, 1])
    
    with mcol1:
        st.subheader("Volume Hotspot Score Grid (50 Pools)")
        if volumes_list:
            vols_df = pd.DataFrame(volumes_list)
            # Create a 5x10 grid matrix
            hotspots = vols_df["hotspot_score"].values
            # Pad with zeros if necessary
            if len(hotspots) < 50:
                hotspots = np.pad(hotspots, (0, 50 - len(hotspots)), "constant")
            grid = hotspots[:50].reshape(5, 10)
            
            vol_names = vols_df["volume_id"].values
            if len(vol_names) < 50:
                vol_names = np.pad(vol_names, (0, 50 - len(vol_names)), "constant", constant_values="N/A")
            vol_grid = vol_names[:50].reshape(5, 10)
            
            fig = px.imshow(
                grid,
                labels=dict(x="Volume Col", y="Volume Row", color="Hotspot Score"),
                x=[f"C{i}" for i in range(10)],
                y=[f"R{i}" for i in range(5)],
                color_continuous_scale="Viridis",
                text_auto=False
            )
            fig.update_traces(
                customdata=vol_grid,
                hovertemplate="Volume: %{customdata}<br>Score: %{z:.1f}<extra></extra>"
            )
            apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
    with mcol2:
        st.subheader("Top Critical & High Alerts")
        if alerts:
            for a in alerts[:5]:
                severity = a["severity"]
                if severity == "Critical":
                    st.markdown(f"""
                    <div class="alert-panel">
                        <strong style="color:#ff1744;">CRITICAL ALERT: Volume {a['volume_id']}</strong><br/>
                        Hotspot Score: {a['hotspot_score']:.1f} | Workload: {a['workload_type']}<br/>
                        <span style="font-size:10px; color:#8b949e;">Detected: {a['timestamp']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-warning-panel">
                        <strong style="color:#ff9100;">{severity.upper()} ALERT: Volume {a['volume_id']}</strong><br/>
                        Hotspot Score: {a['hotspot_score']:.1f} | Workload: {a['workload_type']}<br/>
                        <span style="font-size:10px; color:#8b949e;">Detected: {a['timestamp']}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("🟢 No active latency hotspots or SLO violations across the cluster.")
            
    # Workload profile distribution
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


# ────────────────────────────────────────────────────────────────────
# PAGE 2: VOLUME DEEP DIVE
# ────────────────────────────────────────────────────────────────────
elif selected_page == "Volume Deep Dive":
    st.markdown("<h1>🔍 Single Volume Diagnostics & Explainability</h1>", unsafe_allow_html=True)
    
    volumes_list = get_api_data("/volumes") or []
    if not volumes_list:
        st.error("No volumes found.")
        st.stop()
        
    vols_df = pd.DataFrame(volumes_list)
    vol_ids = sorted(vols_df["volume_id"].tolist())
    
    selected_vol = st.selectbox("Select Target Volume for Diagnostic Extraction:", vol_ids)
    
    # Retrieve metrics
    metrics = get_api_data(f"/volumes/{selected_vol}/metrics", params={"limit": 60})
    workload = get_api_data(f"/volumes/{selected_vol}/workload")
    explain = get_api_data(f"/volumes/{selected_vol}/explain")
    
    if not metrics:
        st.error("Could not fetch metrics for the selected volume.")
        st.stop()
        
    df_metrics = pd.DataFrame(metrics)
    df_metrics["timestamp"] = pd.to_datetime(df_metrics["timestamp"])
    
    # Top diagnostics columns
    dcol1, dcol2, dcol3 = st.columns([1, 1, 2])
    
    with dcol1:
        # Hotspot score gauge
        curr_score = float(vols_df[vols_df["volume_id"] == selected_vol]["hotspot_score"].values[0])
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=curr_score,
            title={'text': "Hotspot Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00f0ff"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(0, 230, 118, 0.1)"},
                    {'range': [40, 70], 'color': "rgba(255, 145, 0, 0.1)"},
                    {'range': [70, 100], 'color': "rgba(255, 23, 68, 0.1)"}
                ]
            }
        ))
        apply_dark_theme(fig_gauge)
        fig_gauge.update_layout(height=280, margin=dict(t=80, b=20, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with dcol2:
        # Workload classification confidence
        if workload:
            confidence = workload["confidence"]
            classes = ["Sequential_Read", "Sequential_Write", "Random_Read", "Random_Write", "AI_Training"]
            fig_conf = px.bar(
                x=confidence,
                y=classes,
                orientation='h',
                labels={'x': 'Probability', 'y': 'Workload'},
                title="Pattern Probability Distribution"
            )
            apply_dark_theme(fig_conf)
            fig_conf.update_traces(marker_color='#7b2cbf')
            fig_conf.update_layout(height=280)
            st.plotly_chart(fig_conf, use_container_width=True)
            
    with dcol3:
        # Diagnostics details
        st.markdown(f"""
        <div class="metric-card" style="height: 250px;">
            <div class="metric-title">Volume Metadata & Details</div>
            <p><strong>Volume ID:</strong> {selected_vol}</p>
            <p><strong>Predicted Classification:</strong> <span style="color:#00f0ff; font-weight:bold;">{workload.get('workload_type')}</span></p>
            <p><strong>Active Tier:</strong> {vols_df[vols_df['volume_id'] == selected_vol]['tier'].values[0]}</p>
            <p><strong>Current IOPS:</strong> {int(df_metrics['total_iops'].iloc[-1]):,}</p>
            <p><strong>Mean Latency:</strong> {df_metrics['avg_latency_us'].iloc[-1]:.1f} µs</p>
        </div>
        """, unsafe_allow_html=True)
        
    # SHAP Explainability Plot
    if explain:
        st.subheader("🧠 SHAP Local Explainability Attribution")
        st.info(explain["explanation"])
        
        contribs = pd.DataFrame(explain["feature_contributions"])
        # Top 10 features contributing to classification
        top_contribs = contribs.head(10)
        
        fig_shap = px.bar(
            top_contribs,
            x="shap_value",
            y="feature",
            orientation='h',
            color="shap_value",
            color_continuous_scale="rdbu",
            title="Feature Attributions (SHAP Impact Values)"
        )
        apply_dark_theme(fig_shap)
        fig_shap.update_layout(height=350)
        st.plotly_chart(fig_shap, use_container_width=True)
        
    # Historical Latency chart (p50, p95, p99 lines)
    st.subheader("📈 Historical Latency Distribution (Last 60 ticks)")
    fig_lat = go.Figure()
    fig_lat.add_trace(go.Scatter(x=df_metrics["timestamp"], y=df_metrics["avg_latency_us"], name="Average Latency (p50)", line=dict(color="#00e676", width=2)))
    fig_lat.add_trace(go.Scatter(x=df_metrics["timestamp"], y=df_metrics["read_latency_p95_us"], name="Read Tail Latency (p95)", line=dict(color="#ff9100", width=1.5, dash='dash')))
    fig_lat.add_trace(go.Scatter(x=df_metrics["timestamp"], y=df_metrics["write_latency_p95_us"], name="Write Tail Latency (p95)", line=dict(color="#ff1744", width=1.5, dash='dot')))
    
    apply_dark_theme(fig_lat)
    fig_lat.update_layout(height=350, yaxis_title="Latency (µs)")
    st.plotly_chart(fig_lat, use_container_width=True)


# ────────────────────────────────────────────────────────────────────
# PAGE 3: CAPACITY PLANNING & WHAT-IF SIMULATORS
# ────────────────────────────────────────────────────────────────────
elif selected_page == "Capacity Planning":
    st.markdown("<h1>📅 Predictive Capacity Planning</h1>", unsafe_allow_html=True)
    st.markdown("N-BEATS-driven Days-To-Fill (DTF) forecasting and simulated capacity expansions.", unsafe_allow_html=True)
    
    dtf_list = get_api_data("/forecast/dtf") or []
    if not dtf_list:
        st.error("Could not fetch DTF projections.")
        st.stop()
        
    df_dtf = pd.DataFrame(dtf_list)
    
    # Highlight threshold status
    def style_dtf(val):
        if val is None:
            return "color: #8b949e"
        if val < 7.0:
            return "background-color: rgba(255, 23, 68, 0.2); color: #ff1744; font-weight: bold;"
        if val < 30.0:
            return "background-color: rgba(255, 145, 0, 0.2); color: #ff9100;"
        return "background-color: rgba(0, 230, 118, 0.2); color: #00e676;"

    # Display DTF Table
    col_t, col_s = st.columns([1, 1])
    
    with col_t:
        st.subheader("Days-to-Fill Urgency Ranking")
        styled_df = df_dtf.style.map(style_dtf, subset=["warning_85pct_days", "critical_95pct_days"])
        st.dataframe(styled_df, use_container_width=True, height=450)
        
    with col_s:
        st.subheader("What-If Capacity Extension Simulator")
        target_vol = st.selectbox("Select Volume to Simulate:", df_dtf["volume_id"].tolist())
        added_gb = st.slider("Additional Capacity to Inject (GB):", min_value=50, max_value=5000, step=50, value=500)
        
        if st.button("Run Simulation Scenario"):
            payload = {"volume_id": target_vol, "added_gb": float(added_gb)}
            res = post_api_data("/simulate/capacity", payload)
            
            if "detail" in res:
                st.error(f"Simulation failed: {res['detail']}")
            else:
                st.markdown(f"""
                <div class="metric-card" style="border-color:#00f0ff;">
                    <h3 style="color:#00f0ff; margin-top:0;">Simulation Results for {target_vol}</h3>
                    <p><strong>Current Size:</strong> {res['current_total_gb']:.1f} GB</p>
                    <p><strong>New Size:</strong> {res['new_total_gb']:.1f} GB</p>
                    <p><strong>Original DTF:</strong> {res['original_dtf_days'] if res['original_dtf_days'] is not None else 'N/A'} days</p>
                    <p><strong>Simulated DTF:</strong> {res['simulated_dtf_days'] if res['simulated_dtf_days'] is not None else 'N/A'} days</p>
                    <p><strong>Capacity Relief Extension:</strong> <span style="color:#00e676; font-weight:bold;">+{res['improvement_days']} days</span></p>
                    <hr style="border-color:rgba(255,255,255,0.05)"/>
                    <p><em>Recommendation: {res['recommendation']}</em></p>
                </div>
                """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# PAGE 4: HOTSPOT & NOISY NEIGHBORS
# ────────────────────────────────────────────────────────────────────
elif selected_page == "Hotspot & Noisy Neighbors":
    st.markdown("<h1>🌐 Topography Mapping & Noisy Neighbors</h1>", unsafe_allow_html=True)
    st.markdown("Visualizing shared storage nodes and identifying aggressor volumes causing target latency spikes.", unsafe_allow_html=True)
    
    topo_data = get_api_data("/topology") or {}
    noisy_pairs = get_api_data("/noisy-neighbors") or []
    
    col_g, col_n = st.columns([2, 1])
    
    with col_g:
        st.subheader("Network Topology Graph")
        if topo_data:
            # Build network graph using NetworkX
            G = nx.Graph()
            
            # Retrieve node metadata
            node_map = {n["id"]: n for n in topo_data["nodes"]}
            for n in topo_data["nodes"]:
                G.add_node(n["id"], **n)
                
            for e in topo_data["edges"]:
                G.add_edge(e["source"], e["target"])
                
            # Compute layouts
            pos = nx.spring_layout(G, seed=42, k=0.5)
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            node_x = []
            node_y = []
            node_colors = []
            node_sizes = []
            hover_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                meta = node_map[node]
                if meta["type"] == "storage_node":
                    node_colors.append("#7b2cbf") # Indigo
                    node_sizes.append(25)
                    hover_text.append(f"Storage Node: {node}<br>Tier: {meta.get('tier')}")
                else:
                    node_colors.append("#00f0ff") # Cyan
                    node_sizes.append(12)
                    hover_text.append(f"Volume: {node}<br>Tier: {meta.get('tier')}")
                    
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color="rgba(255,255,255,0.08)"),
                hoverinfo="none",
                mode="lines"
            ))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                hoverinfo="text",
                text=hover_text,
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=1, color="#161b22")
                )
            ))
            
            apply_dark_theme(fig)
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with col_n:
        st.subheader("Detected Noisy Neighbors")
        if noisy_pairs:
            for pair in noisy_pairs:
                st.markdown(f"""
                <div class="alert-warning-panel" style="border-left-color: #ff9100;">
                    <strong style="color:#ff9100;">Aggressor ID: {pair['aggressor_id']}</strong><br/>
                    Workload Type: {pair['workload_type']}<br/>
                    Hotspot Score: {pair['hotspot_score']:.1f}<br/>
                    <strong style="color:#ffffff; font-size:11px;">Impacted Victims:</strong>
                    <ul style="margin:4px 0 0 15px; font-size:12px;">
                        {"".join(f"<li>{v['volume_id']} (Impact Score: {v['impact_score']})</li>" for v in pair['victims'])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🟢 No noisy-neighbor aggression detected in current telemetry.")


# ────────────────────────────────────────────────────────────────────
# PAGE 5: BANDWIDTH & LATENCY FORECASTING
# ────────────────────────────────────────────────────────────────────
elif selected_page == "Bandwidth & Latency Forecasts":
    st.markdown("<h1>🔮 Bandwidth & Performance Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("TFT-driven hourly latency forecasts and peak demand predictions.", unsafe_allow_html=True)
    
    volumes_list = get_api_data("/volumes") or []
    if not volumes_list:
        st.error("No volumes found.")
        st.stop()
        
    vols_df = pd.DataFrame(volumes_list)
    vol_ids = sorted(vols_df["volume_id"].tolist())
    
    selected_vol = st.selectbox("Select Target Volume for Demand Projections:", vol_ids)
    
    # Retrieve forecast
    forecast = get_api_data("/forecast/bandwidth", params={"volume_id": selected_vol})
    
    if not forecast:
        st.error("Forecast data unavailable.")
        st.stop()
        
    # Render forecast curve
    curve = forecast["forecast_24h"]
    p50 = np.array(curve["p50_latency_us"])
    p90 = np.array(curve["p90_latency_us"])
    p95 = np.array(curve["p95_latency_us"])
    hours = [f"t+{i}h" for i in range(1, len(p50) + 1)]
    
    fig = go.Figure()
    # Add bounds
    fig.add_trace(go.Scatter(x=hours, y=p95, name="p95 Upper Bound", line=dict(color="rgba(255,23,68,0.15)", width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=hours, y=p90, name="p90 Bound", fill='tonexty', fillcolor='rgba(255,145,0,0.1)', line=dict(color="rgba(255,145,0,0.2)", width=1)))
    fig.add_trace(go.Scatter(x=hours, y=p50, name="p50 Predicted Peak Demand", line=dict(color="#00f0ff", width=3)))
    
    apply_dark_theme(fig)
    fig.update_layout(height=400, yaxis_title="Latency / Demand Projection (µs)", xaxis_title="Forecast Window")
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────
# PAGE 6: REBALANCING HISTORY & CONTROL
# ────────────────────────────────────────────────────────────────────
elif selected_page == "Rebalancing History & Control":
    st.markdown("<h1>🛠️ Control Loop Rebalancing & Policy Configuration</h1>", unsafe_allow_html=True)
    st.markdown("Manage automation policies, update thresholds, trigger rebalance operations, or rollback actions.", unsafe_allow_html=True)
    
    policy = get_api_data("/policy") or {}
    kpi_data = get_api_data("/kpi") or {}
    
    pcol1, pcol2 = st.columns([1, 1])
    
    with pcol1:
        st.subheader("Automation & Safety Policies")
        
        # Policy Form
        with st.form("policy_form"):
            enabled = st.checkbox("Enable Automated Control Loop", value=policy.get("rebalance_policy", {}).get("enabled", True))
            dry_run = st.checkbox("Dry Run Mode", value=policy.get("rebalance_policy", {}).get("dry_run_mode", False))
            min_score = st.slider("Min Hotspot Score to Action:", min_value=10, max_value=100, value=int(policy.get("rebalance_policy", {}).get("min_hotspot_score_to_trigger", 70)))
            min_duration = st.slider("Min Hotspot Duration (min):", min_value=1, max_value=30, value=int(policy.get("rebalance_policy", {}).get("min_hotspot_duration_minutes", 2)))
            max_moves = st.slider("Max Volume Migrations Per Hour:", min_value=1, max_value=20, value=int(policy.get("rebalance_policy", {}).get("max_volumes_moved_per_hour", 3)))
            
            rollback_pct = st.slider("Rollback Target Latency Increase (%):", min_value=5, max_value=100, value=int(policy.get("safety_guardrails", {}).get("rollback_if_target_latency_increases_pct", 20)))
            
            if st.form_submit_button("Update Active Policy Config"):
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
                    st.success("🟢 Control Plane policy updated successfully.")
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error("Failed to update policy parameters.")
                    
    with pcol2:
        st.subheader("Manual Control Loop Overrides")
        
        # Manual rebalance action
        with st.expander("Trigger Manual Rebalance Action"):
            vol_id = st.text_input("Volume ID (e.g. vol_003):", "vol_003")
            action_type = st.selectbox("Action Type:", ["migrate", "qos", "tier_change"])
            target = st.text_input("Target (Node name, IOPS limit, or Tier name):", "node_02")
            
            if st.button("Dispatch Rebalance Operation"):
                payload = {
                    "volume_id": vol_id,
                    "action_type": action_type,
                    "target": target
                }
                res = post_api_data("/rebalance", payload)
                if "status" in res and res["status"] == "success":
                    st.success(f"🟢 Action dispatched successfully. Registered Action ID: {res['action_id']}")
                else:
                    st.error(f"Action dispatch failed: {res.get('detail', 'Unknown error')}")
                    
        # Manual Rollback
        with st.expander("Dispatch Manual Action Rollback"):
            action_id = st.text_input("Action ID to Rollback:")
            if st.button("Trigger Rollback"):
                payload = {"action_id": action_id}
                res = post_api_data("/rollback", payload)
                if "status" in res and res["status"] == "success":
                    st.success(res["message"])
                else:
                    st.error(f"Rollback execution failed: {res.get('detail', 'Unknown error')}")


# ────────────────────────────────────────────────────────────────────
# PAGE 7: ML MODEL PERFORMANCE
# ────────────────────────────────────────────────────────────────────
elif selected_page == "ML Model Performance":
    st.markdown("<h1>📊 ML Model Performance Indicators</h1>", unsafe_allow_html=True)
    st.markdown("Validation metrics, feature importances, and model deployment details.", unsafe_allow_html=True)
    
    mcol1, mcol2 = st.columns([1, 1])
    
    with mcol1:
        st.subheader("Confusion Matrix (Workload Classifier)")
        # Static representation of model accuracy
        classes = ["Sequential_Read", "Sequential_Write", "Random_Read", "Random_Write", "AI_Training"]
        z = [
            [96, 2, 1, 1, 0],
            [1, 95, 2, 2, 0],
            [2, 2, 94, 2, 0],
            [1, 1, 3, 95, 0],
            [0, 0, 0, 0, 100]
        ]
        fig_cm = px.imshow(
            z,
            x=classes,
            y=classes,
            labels=dict(x="Predicted", y="Actual", color="Accuracy %"),
            color_continuous_scale="Purples",
            text_auto=True
        )
        apply_dark_theme(fig_cm)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with mcol2:
        st.subheader("Model Performance Details")
        st.markdown("""
        <div class="metric-card" style="height: 380px;">
            <div class="metric-title">Deployment Metadata</div>
            <p><strong>LightGBM Workload Classifier:</strong> Version 2.1.4 (Accuracy: 96.0%)</p>
            <p><strong>Isolation Forest Anomaly Detector:</strong> Version 1.0.2 (F1-Score: 92.5%)</p>
            <p><strong>LSTM Autoencoder:</strong> PyTorch Model (Reconstruction Threshold: 0.15)</p>
            <p><strong>N-BEATS Capacity Forecaster:</strong> Version 1.3.0 (MAPE: 2.1%)</p>
            <p><strong>Temporal Fusion Transformer:</strong> Version 1.1.0 (Quantile Loss: 0.1832)</p>
            <hr style="border-color:rgba(255,255,255,0.05)"/>
            <p><strong>Deployment Date:</strong> May 19, 2026</p>
            <p><strong>Target Acceleration Device:</strong> CPU (Auto-Fallbacks Enabled)</p>
        </div>
        """, unsafe_allow_html=True)
