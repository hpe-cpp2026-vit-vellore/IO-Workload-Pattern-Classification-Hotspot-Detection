"""
dashboard/utils.py

Shared utilities, layout configurations, styling, API wrappers, and telemetry
monitoring for the modular HPE Storage Control Plane Dashboard.
"""

import os
import sys
import time
import base64
import requests
import subprocess
from pathlib import Path
from typing import Any, Optional, Dict, List
from urllib.parse import urlparse
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("HPE_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_LOG_PATH = PROJECT_ROOT / "api_server.log"

def apply_custom_css():
    """Injects premium glassmorphic dark-theme styles, Outfit font, and scrollbar styles."""
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
        
        /* Custom High-Density Table Styles */
        .dense-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: rgba(22, 27, 34, 0.6);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 1.5rem;
        }
        .dense-table th {
            background: #161b22;
            color: #8b949e;
            font-weight: 600;
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .dense-table td {
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: #c9d1d9;
            vertical-align: middle;
        }
        .dense-table tr:hover {
            background: rgba(0, 240, 255, 0.02);
        }
        
        /* Status Badges */
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-healthy {
            background: rgba(0, 230, 118, 0.15);
            color: #00e676;
            border: 1px solid rgba(0, 230, 118, 0.3);
        }
        .badge-warning {
            background: rgba(255, 145, 0, 0.15);
            color: #ff9100;
            border: 1px solid rgba(255, 145, 0, 0.3);
        }
        .badge-critical {
            background: rgba(255, 23, 68, 0.15);
            color: #ff1744;
            border: 1px solid rgba(255, 23, 68, 0.3);
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

def apply_dark_theme(fig):
    """Applies premium dark-mode styling to Plotly Figures."""
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

# --- API Fetch wrappers ---

# Use a global session to enable HTTP Keep-Alive and prevent TCP port exhaustion
_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=1)
_session.mount("http://", adapter)
_session.mount("https://", adapter)

@st.cache_data(ttl=1500) # Cache token for 25 mins (expires in 30 mins)
def _get_auth_token(api_url: str) -> str:
    try:
        response = _session.post(
            f"{api_url}/token", 
            data={"username": "admin", "password": "hpe_admin_2026"},
            timeout=5.0
        )
        if response.status_code == 200:
            return response.json().get("access_token")
    except Exception:
        pass
    return ""

def get_api_data(endpoint: str, params: dict = None, timeout: float = 5.0) -> Any:
    """Helper to query local FastAPI endpoints using Keep-Alive."""
    try:
        headers = {}
        if not endpoint.startswith("/health"):
            token = _get_auth_token(API_URL)
            if token:
                headers["Authorization"] = f"Bearer {token}"
        response = _session.get(f"{API_URL}{endpoint}", params=params, headers=headers, timeout=timeout)
        if response.status_code == 200:
            st.session_state["last_api_error"] = None
            return response.json()
        st.session_state["last_api_error"] = f"HTTP {response.status_code}: {response.text[:300]}"
        return None
    except Exception as e:
        st.session_state["last_api_error"] = str(e)
        return None

def post_api_data(endpoint: str, payload: dict) -> Any:
    """Helper to post parameters to FastAPI endpoints using Keep-Alive."""
    try:
        headers = {}
        if not endpoint.startswith("/health"):
            token = _get_auth_token(API_URL)
            if token:
                headers["Authorization"] = f"Bearer {token}"
        response = _session.post(f"{API_URL}{endpoint}", json=payload, headers=headers, timeout=15.0)
        return response.json() if response.status_code == 200 else {"detail": response.text}
    except Exception as e:
        return {"detail": str(e)}

def put_api_data(endpoint: str, payload: dict) -> Any:
    """Helper to update policy configurations at FastAPI endpoints using Keep-Alive."""
    try:
        headers = {}
        if not endpoint.startswith("/health"):
            token = _get_auth_token(API_URL)
            if token:
                headers["Authorization"] = f"Bearer {token}"
        response = _session.put(f"{API_URL}{endpoint}", json=payload, headers=headers, timeout=15.0)
        return response.json() if response.status_code == 200 else {"detail": response.text}
    except Exception as e:
        return {"detail": str(e)}

# --- Local API autostart logic ---
def _local_python_executable() -> str:
    venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

def start_backend_api() -> bool:
    """Start local FastAPI control plane when dashboard is opened."""
    cmd = [
        _local_python_executable(),
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        with open(API_LOG_PATH, "a", encoding="utf-8") as log_file:
            subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        st.session_state["backend_autostart_cmd"] = " ".join(cmd)
        return True
    except Exception as e:
        st.session_state["backend_autostart_error"] = str(e)
        return False

def wait_for_api(seconds: float = 60.0) -> Optional[dict]:
    """Poll the API health endpoint until it is ready."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        health = get_api_data("/health", timeout=20.0)
        if health:
            return health
        time.sleep(1.0)
    return None

def can_autostart_backend() -> bool:
    parsed = urlparse(API_URL)
    return parsed.hostname in {"127.0.0.1", "localhost"} and (parsed.port in {None, 8000})

# --- Sidebar rendering ---
def render_sidebar_telemetry():
    """Renders high-density sidebar header, connection state, and active circuit breakers."""
    st.sidebar.markdown(
        "<div style='text-align: center; padding-bottom: 20px;'>"
        "<h2 style='color:#00f0ff; font-weight: 900; letter-spacing: 1.5px; margin-bottom: 5px;'>HPE STORAGE</h2>"
        "<p style='color:#8b949e; font-size:11px; text-transform: uppercase;'>Control Plane & Analytics</p>"
        "</div>", unsafe_allow_html=True
    )
    
    # Connection Check (Increased timeout to 5.0 to survive heavy GIL-locked PyTorch simulations)
    health = get_api_data("/health", timeout=5.0)
    if not health:
        st.sidebar.error("API Connection: UNREACHABLE")
        return None
        
    telemetry_bus = health.get("telemetry_bus", {})
    if telemetry_bus:
        if telemetry_bus.get("mode") == "redis_streams":
            st.sidebar.success("Telemetry: Redis Streams")
        else:
            redis_error = telemetry_bus.get("redis", {}).get("error")
            fallback = telemetry_bus.get("tcp_fallback", {})
            if fallback.get("listening"):
                st.sidebar.warning("Telemetry: TCP Fallback")
            else:
                st.sidebar.error("Telemetry: Fallback offline")
            if redis_error:
                st.sidebar.caption(f"Redis offline: {redis_error}")
    else:
        st.sidebar.caption("Telemetry status unknown.")
        
    circuit_breaker = get_api_data("/rebalance/circuit-breaker") or {}
    if circuit_breaker.get("circuit_breaker_tripped"):
        st.sidebar.error("Circuit Breaker: TRIPPED")
        reason = circuit_breaker.get("reason", "Circuit breaker tripped.")
        st.markdown(
            f"""
            <div style="
                background: rgba(255, 23, 68, 0.10);
                border: 1px solid rgba(255, 23, 68, 0.35);
                border-left: 6px solid #ff1744;
                border-radius: 10px;
                padding: 12px 16px;
                margin: 8px 0 18px 0;
            ">
                <strong style="color:#ff6b81;">Circuit Breaker Tripped</strong><br/>
                <span style="color:#c9d1d9; font-size:12px;">{reason}</span><br/>
                <span style="color:#8b949e; font-size:11px;">Rebalancing features are suspended.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.success("Circuit Breaker: OK")
        
    return health

# --- Sparkline SVG Generator ---
def generate_sparkline_svg(values: List[float], color: str = "#00f0ff") -> str:
    """Generates a base64 encoded inline SVG sparkline line chart to render directly in HTML cells."""
    if not values or len(values) < 2:
        # Return empty spacer SVG
        return '<svg width="100" height="20"></svg>'
    
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1.0
    
    width = 100
    height = 20
    padding = 2
    points = []
    
    for idx, val in enumerate(values):
        x = (idx / (len(values) - 1)) * (width - 2 * padding) + padding
        # Invert y axis so 0 is bottom
        y = height - (((val - min_val) / val_range) * (height - 2 * padding) + padding)
        points.append(f"{x:.1f},{y:.1f}")
        
    points_str = " ".join(points)
    svg = f'<svg width="{width}" height="{height}" style="vertical-align: middle;"><polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points_str}"/></svg>'
    return svg
