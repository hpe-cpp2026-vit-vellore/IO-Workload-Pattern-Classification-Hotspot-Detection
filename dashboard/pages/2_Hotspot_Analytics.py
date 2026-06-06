"""
dashboard/pages/2_Hotspot_Analytics.py

Hotspot & Diagnostics Page.
Covers:
- Volume diagnostics selection, probability distributions, SHAP explainability, and multi-quantile latency profiles.
- Topographical Network mapping of nodes and volumes (using NetworkX).
- Noisy Neighbor aggressor-victim detection tables.
- ML validation performance reports (cached).
"""

import time
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
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
    render_sidebar_telemetry
)

st.set_page_config(
    page_title="Hotspot Analytics - HPE Storage Console",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Render standard sidebar
health = render_sidebar_telemetry()
if not health:
    st.error("API Connection Offline.")
    st.stop()

# --- Page Headers ---
st.markdown("<h1>🔍 Hotspot Diagnostics & Noisy Neighbors</h1>", unsafe_allow_html=True)
st.markdown("Topography mapping, ensemble anomaly detection, SHAP local explainability, and noisy neighbor evaluations.", unsafe_allow_html=True)
st.markdown("---")

# Caching ML Model performance endpoint (static validation metrics)
@st.cache_data(ttl=300)
def get_cached_model_performance():
    return get_api_data("/model/performance")

# Tab navigation for page 2 sub-sections
tab_diag, tab_topo, tab_ml = st.tabs([
    "🔍 Volume Diagnostics & SHAP",
    "🌐 Topography & Noisy Neighbors",
    "📊 Classifier Performance"
])

# --- SUB-TAB 1: VOLUME DIAGNOSTICS & SHAP ---
with tab_diag:
    st.subheader("Targeted Volume Diagnostic Extraction")
    
    volumes_list = get_api_data("/volumes") or []
    if not volumes_list:
        st.warning("No volumes found to query.")
    else:
        vols_df = pd.DataFrame(volumes_list)
        vol_ids = sorted(vols_df["volume_id"].tolist())
        selected_vol = st.selectbox("Select Target Volume for Diagnostic Analysis:", vol_ids)
        
        # Details metadata card
        vol_row = vols_df[vols_df["volume_id"] == selected_vol].iloc[0]
        vol_is_live = bool(vol_row.get("is_live", False))
        vol_current_iops = float(vol_row.get("current_iops", 0.0) or 0.0)
        vol_current_latency = float(vol_row.get("current_latency_us", 0.0) or 0.0)
        vol_last_seen = vol_row.get("last_seen_timestamp", "—")
        
        if vol_is_live:
            st.markdown(f"""
            <div style="
                background: rgba(0, 230, 118, 0.08); border: 1px solid rgba(0, 230, 118, 0.25);
                border-radius: 8px; padding: 8px 14px; margin-bottom: 14px;
                display: flex; align-items: center; gap: 18px; font-size: 12px;
            ">
                <span style="color: #00e676; font-weight: 700;">● LIVE TELEMETRY</span>
                <span style="color: #c9d1d9;">Last seen: <strong style="color:#00f0ff;">{vol_last_seen}</strong></span>
                <span style="color: #8b949e;">IOPS: <strong>{int(vol_current_iops):,}</strong></span>
                <span style="color: #8b949e;">Latency: <strong>{vol_current_latency:.1f} µs</strong></span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: rgba(255, 145, 0, 0.06); border: 1px solid rgba(255, 145, 0, 0.2);
                border-radius: 8px; padding: 8px 14px; margin-bottom: 14px;
                font-size: 12px; color: #ff9100;
            ">
                ◉ HISTORICAL DATA — Displaying stored profile analytics.
            </div>
            """, unsafe_allow_html=True)

        metrics = get_api_data(f"/volumes/{selected_vol}/metrics", params={"limit": 60})
        workload = get_api_data(f"/volumes/{selected_vol}/workload")
        explain = get_api_data(f"/volumes/{selected_vol}/explain")
        
        if not metrics:
            st.error("No metrics history returned for this volume.")
        else:
            df_metrics = pd.DataFrame(metrics)
            df_metrics["timestamp"] = pd.to_datetime(df_metrics["timestamp"])
            
            dcol1, dcol2, dcol3 = st.columns([1, 1, 2])
            
            with dcol1:
                # Hotspot Score gauge chart
                curr_score = float(vol_row.get("hotspot_score", 0.0) or 0.0)
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
                # Workload probabilities
                if workload:
                    confidence = workload["confidence"]
                    classes = ["DB_OLTP", "VM", "Backup", "AI_Training", "AI_Inference"]
                    fig_conf = px.bar(
                        x=confidence,
                        y=classes,
                        orientation='h',
                        labels={'x': 'Probability', 'y': 'Workload'},
                        title="Ensemble Pattern Confidence"
                    )
                    apply_dark_theme(fig_conf)
                    fig_conf.update_traces(marker_color='#7b2cbf')
                    fig_conf.update_layout(height=280)
                    st.plotly_chart(fig_conf, use_container_width=True)
                    
            with dcol3:
                display_iops = int(vol_current_iops) if vol_is_live else int(df_metrics['total_iops'].iloc[-1])
                display_latency = vol_current_latency if vol_is_live else float(df_metrics['avg_latency_us'].iloc[-1])
                data_label = "Live" if vol_is_live else "Historical"
                
                arf_info_line = ""
                if workload and workload.get('arf_workload_type') is not None:
                    arf_agrees = workload.get('arf_agrees')
                    arf_info_line = (
                        f"<p><strong>Active Rollback Filter (ARF):</strong> {workload.get('arf_workload_type')}"
                        f" <span style='color:#8b949e;'>(agrees: {'yes' if arf_agrees is True else 'no' if arf_agrees is False else 'n/a'})</span></p>"
                    )
                    
                st.markdown(f"""
                <div class="metric-card" style="height: 250px; overflow-y: auto;">
                    <div class="metric-title">Volume Identification</div>
                    <p><strong>Volume:</strong> {selected_vol}</p>
                    <p><strong>Workload Pattern:</strong> <span style="color:#00f0ff; font-weight:bold;">{workload.get('workload_type') if workload else '—'}</span></p>
                    {arf_info_line}
                    <p><strong>Active Tier:</strong> {vol_row['tier']}</p>
                    <p><strong>IOPS ({data_label}):</strong> {display_iops:,}</p>
                    <p><strong>Mean Latency ({data_label}):</strong> {display_latency:.1f} µs</p>
                </div>
                """, unsafe_allow_html=True)

            # Local Explainability Plot (SHAP)
            if explain:
                st.subheader("🧠 SHAP Feature Attribution Impact")
                st.info(explain["explanation"])
                
                contribs = pd.DataFrame(explain["feature_contributions"])
                top_contribs = contribs.head(10)
                
                fig_shap = px.bar(
                    top_contribs,
                    x="shap_value",
                    y="feature",
                    orientation='h',
                    color="shap_value",
                    color_continuous_scale="rdbu",
                    title="SHAP Local Explanations (Feature Attribution Impact)"
                )
                apply_dark_theme(fig_shap)
                fig_shap.update_layout(height=350)
                st.plotly_chart(fig_shap, use_container_width=True)

            # Latency Curve over time
            chart_source = "Live" if vol_is_live else "Historical"
            st.subheader(f"📈 Latency Curves — Last {len(df_metrics)} ticks ({chart_source})")
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Scatter(x=df_metrics["timestamp"], y=df_metrics["avg_latency_us"], name="Average Latency (p50)", line=dict(color="#00e676", width=2)))
            fig_lat.add_trace(go.Scatter(x=df_metrics["timestamp"], y=df_metrics["read_latency_p95_us"], name="Read Tail Latency (p95)", line=dict(color="#ff9100", width=1.5, dash='dash')))
            fig_lat.add_trace(go.Scatter(x=df_metrics["timestamp"], y=df_metrics["write_latency_p95_us"], name="Write Tail Latency (p95)", line=dict(color="#ff1744", width=1.5, dash='dot')))
            
            apply_dark_theme(fig_lat)
            fig_lat.update_layout(height=350, yaxis_title="Latency (µs)")
            st.plotly_chart(fig_lat, use_container_width=True)


# --- SUB-TAB 2: TOPOGRAPHY & NOISY NEIGHBORS ---
with tab_topo:
    topo_data = get_api_data("/topology") or {}
    noisy_pairs = get_api_data("/noisy-neighbors") or []
    
    col_g, col_n = st.columns([2, 1])
    
    with col_g:
        st.subheader("Storage Pool Network Topography")
        if topo_data and volumes_list:
            vol_lookup = {v["volume_id"]: v for v in volumes_list}
            
            # Draw Network
            G = nx.Graph()
            node_map = {n["id"]: n for n in topo_data["nodes"]}
            for n in topo_data["nodes"]:
                G.add_node(n["id"], **n)
            for e in topo_data["edges"]:
                G.add_edge(e["source"], e["target"])
                
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
                    child_volumes = [v for v in G.neighbors(node) if v.startswith("vol_")]
                    child_iops = sum(vol_lookup.get(v, {}).get("current_iops", 0.0) or 0.0 for v in child_volumes)
                    child_latencies = [float(vol_lookup.get(v, {}).get("current_latency_us", 0.0) or 0.0) for v in child_volumes if vol_lookup.get(v, {}).get("current_latency_us")]
                    avg_lat = float(np.mean(child_latencies)) if child_latencies else 0.0
                    hotspot_count = sum(1 for v in child_volumes if float(vol_lookup.get(v, {}).get("hotspot_score", 0.0) or 0.0) >= 40)
                    
                    node_colors.append("#7b2cbf")
                    node_sizes.append(25)
                    hover_text.append(
                        f"Storage Node: {node}<br>Tier: {meta.get('tier')}"
                        f"<br>Aggregate IOPS: {int(child_iops):,}"
                        f"<br>Avg Latency: {avg_lat:.1f} µs"
                        f"<br>Hotspot Volumes: {hotspot_count}/{len(child_volumes)}"
                    )
                else:
                    vinfo = vol_lookup.get(node, {})
                    hs = float(vinfo.get("hotspot_score", 0.0) or 0.0)
                    v_iops = float(vinfo.get("current_iops", 0.0) or 0.0)
                    v_lat = float(vinfo.get("current_latency_us", 0.0) or 0.0)
                    v_live = vinfo.get("is_live", False)
                    
                    if hs >= 70:
                        color = "#ff1744"
                    elif hs >= 40:
                        color = "#ff9100"
                    else:
                        color = "#00e676"
                        
                    node_colors.append(color)
                    node_sizes.append(14)
                    live_tag = "● Live" if v_live else "◉ Historical"
                    hover_text.append(
                        f"Volume: {node}<br>Tier: {meta.get('tier')}"
                        f"<br>Hotspot: {hs:.1f}"
                        f"<br>IOPS: {int(v_iops):,}"
                        f"<br>Latency: {v_lat:.1f} µs"
                        f"<br>{live_tag}"
                    )
                    
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
        else:
            st.info("Topology map data unavailable.")
            
    with col_n:
        st.subheader("Noisy Neighbors Contention")
        if noisy_pairs:
            for pair in noisy_pairs:
                st.markdown(f"""
                <div class="alert-warning-panel" style="border-left-color: #ff9100;">
                    <strong style="color:#ff9100;">Aggressor ID: {pair['aggressor_id']}</strong><br/>
                    Workload Type: {pair['workload_type']}<br/>
                    Hotspot Score: {pair['hotspot_score']:.1f}<br/>
                    <strong style="color:#ffffff; font-size:11px;">Victims Impacted:</strong>
                    <ul style="margin:4px 0 0 15px; font-size:12px;">
                        {"".join(f"<li>{v['volume_id']} (Impact Score: {v['impact_score']})</li>" for v in pair['victims'])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🟢 No noisy neighbor aggression patterns detected.")


# --- SUB-TAB 3: CLASSIFIER PERFORMANCE (CACHED) ---
with tab_ml:
    st.subheader("ML Model Evaluation Reports (Validation Set)")
    
    perf_data = get_cached_model_performance()
    if not perf_data:
        st.error("Could not load validation metrics from control plane.")
    else:
        accuracy = perf_data.get("accuracy", 0.0) * 100.0
        sample_count = perf_data.get("sample_count", 0)
        cm_perc = perf_data.get("confusion_matrix_percentage", [])
        metrics_per_class = perf_data.get("metrics_per_class", {})
        
        classes = ["DB_OLTP", "VM", "Backup", "AI_Training", "AI_Inference"]
        
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Validation Accuracy</div>
                <div class="metric-value">{accuracy:.2f}%</div>
                <div class="metric-subtitle">Aggregate classification validation score</div>
            </div>
            """, unsafe_allow_html=True)
        with pcol2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Validation Sample Count</div>
                <div class="metric-value">{sample_count:,}</div>
                <div class="metric-subtitle">Time-series validation records evaluated</div>
            </div>
            """, unsafe_allow_html=True)
            
        mcol1, mcol2 = st.columns([1, 1])
        with mcol1:
            st.subheader("Confusion Matrix (%)")
            if cm_perc:
                fig_cm = px.imshow(
                    cm_perc,
                    x=classes,
                    y=classes,
                    labels=dict(x="Predicted Class", y="Actual Class", color="Percentage (%)"),
                    color_continuous_scale="Purples",
                    text_auto=True
                )
                apply_dark_theme(fig_cm)
                st.plotly_chart(fig_cm, use_container_width=True)
                
        with mcol2:
            st.subheader("Class-Wise Performance Details")
            if metrics_per_class:
                class_rows = []
                for cls_name, cls_metrics in metrics_per_class.items():
                    class_rows.append({
                        "Workload Class": cls_name,
                        "Precision": f"{cls_metrics.get('precision', 0.0)*100:.2f}%",
                        "Recall": f"{cls_metrics.get('recall', 0.0)*100:.2f}%",
                        "F1-Score": f"{cls_metrics.get('f1_score', 0.0)*100:.2f}%",
                        "Support (Samples)": f"{cls_metrics.get('support', 0):,}"
                    })
                st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)
                
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Model Architecture Metadata</div>
                <p><strong>Temporal Workload LightGBM:</strong> Evaluates running 10-tick rolling features.</p>
                <p><strong>Isolation Forest Outlier:</strong> Anomaly detection on queues and service time.</p>
                <p><strong>LSTM Autoencoder:</strong> PyTorch based reconstruction loss thresholding for SLA risk detection.</p>
                <hr style="border-color:rgba(255,255,255,0.05)"/>
                <p><strong>Device Target:</strong> CPU (Auto-Fallbacks enabled).</p>
            </div>
            """, unsafe_allow_html=True)
