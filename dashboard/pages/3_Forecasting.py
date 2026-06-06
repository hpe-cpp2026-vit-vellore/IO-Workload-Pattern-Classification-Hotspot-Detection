"""
dashboard/pages/3_Forecasting.py

Forecasting & Projections Page.
Covers:
- N-BEATS-driven Days-To-Fill (DTF) storage capacity projections.
- TFT-driven hourly tail latency performance projections with uncertainty bands.
"""

import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    page_title="Forecasting & Projections - HPE Storage Console",
    page_icon="🔮",
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
st.markdown("<h1>🔮 Deep Learning Storage Projections</h1>", unsafe_allow_html=True)
st.markdown("Evaluating cluster-wide Days-To-Fill metrics and forecasting hourly latency/bandwidth demands.", unsafe_allow_html=True)
st.markdown("---")

# Cache DTF rankings to speed up UI transitions
@st.cache_data(ttl=30)
def get_cached_dtf_list():
    return get_api_data("/forecast/dtf") or []

col_t, col_s = st.columns([1, 1])

# --- N-BEATS Days-To-Fill Table ---
with col_t:
    st.subheader("📅 N-BEATS - Days-to-Fill (DTF) Rankings")
    dtf_list = get_cached_dtf_list()
    
    if not dtf_list:
        st.info("No Days-To-Fill projections returned from backend.")
    else:
        df_dtf = pd.DataFrame(dtf_list)
        
        # Color coding for warning levels
        def style_dtf(val):
            if val is None:
                return "color: #8b949e"
            if val < 7.0:
                return "background-color: rgba(255, 23, 68, 0.2); color: #ff1744; font-weight: bold;"
            if val < 30.0:
                return "background-color: rgba(255, 145, 0, 0.2); color: #ff9100;"
            return "background-color: rgba(0, 230, 118, 0.2); color: #00e676;"

        styled_df = df_dtf.style.map(style_dtf, subset=["warning_85pct_days", "critical_95pct_days"])
        st.dataframe(styled_df, use_container_width=True, height=450)

# --- TFT Latency/Bandwidth Demand Projections ---
with col_s:
    st.subheader("📈 TFT - 24-Hour Tail Latency Projections")
    
    volumes_list = get_api_data("/volumes") or []
    if not volumes_list:
        st.warning("No active volumes to select.")
    else:
        vols_df = pd.DataFrame(volumes_list)
        vol_ids = sorted(vols_df["volume_id"].tolist())
        selected_vol = st.selectbox("Select Volume for Performance Projections:", vol_ids)
        
        # Retrieve TFT forecast curves
        forecast = get_api_data("/forecast/bandwidth", params={"volume_id": selected_vol})
        
        if not forecast or "forecast_24h" not in forecast:
            st.error("Quantile demand projection curves unavailable for this volume.")
        else:
            curve = forecast["forecast_24h"]
            p50 = np.array(curve.get("p50_latency_us", []))
            p90 = np.array(curve.get("p90_latency_us", []))
            p95 = np.array(curve.get("p95_latency_us", []))
            
            if len(p50) == 0:
                st.info("No forecast points generated yet.")
            else:
                hours = [f"t+{i}h" for i in range(1, len(p50) + 1)]
                
                fig = go.Figure()
                
                # Add uncertainty envelope bounds
                fig.add_trace(go.Scatter(
                    x=hours, y=p95,
                    name="p95 Bound (SLA Threshold)",
                    line=dict(color="rgba(255,23,68,0.15)", width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=hours, y=p90,
                    name="p90 Bound (Peak Range)",
                    fill='tonexty',
                    fillcolor='rgba(255,145,0,0.1)',
                    line=dict(color="rgba(255,145,0,0.2)", width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=hours, y=p50,
                    name="p50 Predicted Peak Demand",
                    line=dict(color="#00f0ff", width=3)
                ))
                
                apply_dark_theme(fig)
                fig.update_layout(
                    height=380,
                    yaxis_title="Predicted Latency (µs)",
                    xaxis_title="Forecast Window Hour",
                    legend=dict(x=0.02, y=0.98)
                )
                st.plotly_chart(fig, use_container_width=True)
