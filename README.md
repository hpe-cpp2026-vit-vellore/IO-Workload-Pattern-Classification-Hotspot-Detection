# 🧠 HPE Storage Control Plane: IO Workload Pattern Classification & Hotspot Detection

### *Enterprise-grade Autonomous Storage Control Plane for SLO Compliance, Predictive Rebalancing, and Capacity Planning.*

---

## 🚀 Executive Summary & Architecture Highlights

This repository contains an intelligent, predictive **Storage Control Plane** designed for the HPE Blueprint. It leverages deep learning and streaming machine learning to forecast capacity limits and latency breaches, proactively actuating volume rebalancing before Service Level Objectives (SLOs) are breached.

### Key Engineering & ML Highlights:

* **Dual-Track Ingestion Infrastructure:**
  * **Local Track:** Lightweight Docker Compose deployment using **Redis Streams** as the messaging backbone.
  * **Production Track:** Enterprise-scale **Kubernetes (K8s)** deployment leveraging **Apache Kafka** partitioned telemetry ingestion.
* **Disaggregated Inference Architecture:**
  * Supports a local monolithic coordinator ([InferenceHub](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/inference_hub.py)) for lightweight deployments.
  * Native proxy routing ([RemoteInferenceClient](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/remote_inference_client.py)) redirects heavy inference requests to dedicated GPU clusters running **Triton** or **KServe**.
* **High-Throughput Hybrid Pipeline:** Telemetry JSON stream is parsed and outlier-clipped by a C++ extension ([telemetry_parser.cpp](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/cpp/telemetry_parser.cpp)) with python fallback.
* **State Management Persistence:** Seamlessly transitions from local memory structures to time-series optimized **TimescaleDB** hypertables for windowed metrics.
* **$O(1)$ Optimization for What-If Analysis:** High-performance simulations mapping rebalance candidates are kept under sub-millisecond latencies by replacing global dataframe scans with hash-mapped dictionary lookups.
* **Observability & Proactive Safeguards:**
  * Automatic **Prometheus ServiceMonitor** hooks reporting real-time system performance.
  * **Closed-loop control loop** that rolls back migration actions if the target volume's average latency increases by $\ge 20\%$ post-move.
  * **Circuit Breaker** safeguard that disables auto-rebalancing if the action rollback rate exceeds $1\%$.

---

## 📂 Repository Directory Layout

```
.
├── api/                             # FastAPI Gateway (37 REST Endpoints)
│   ├── main.py                      # REST Routing, TCP Fallback, SSE Streams
│   └── schemas/
│       └── models.py                # Pydantic Schemas for Requests/Responses
├── configs/                         # Configuration Center
│   ├── policy.yaml                  # System SLO limits & guardrail thresholds
│   └── settings.py                  # Pydantic BaseSettings (12-Factor App config)
├── dashboard/                       # Streamlit UI Layer
│   ├── Home.py                      # UI Home & Authentication Gate
│   ├── utils.py                     # API HTTP Requests & SVG Sparklines
│   └── pages/
│       ├── 1_Cluster_Overview.py    # Main KPIs & Volume Status Pools
│       ├── 2_Hotspot_Analytics.py   # SHAP Explainability & Topology Maps
│       ├── 3_Forecasting.py         # N-BEATS Capacity & TFT Latency Graphs
│       └── 4_Control_Plane.py       # Rebalance policy triggers & history audit
├── data/                            # Local Data Store (Gitignored except bounds)
│   └── processed/
│       └── bounds.json              # IQR clipping bounds computed during training
├── deploy/                          # Deployment Target Manifests
│   ├── helm/
│   │   └── hpe-control-plane/       # Production K8s charts (HPA, Prometheus)
│   ├── k8s/                         # Static manifests (Dashboard Service etc.)
│   └── monitoring/                  # Prometheus & Grafana Configuration
├── docs/                            # Design Documentation
│   ├── images/
│   │   └── architecture_diagram.png # Updated System Architecture Diagram
│   ├── production_architecture.md   # Enterprise Deployment Guidelines
│   └── api_reference.md             # REST API Reference
├── models/                          # Trained Weights & Scalers (Gitignored)
├── scripts/                         # Automation & Training Pipelines
│   ├── telemetry_playback.py        # Plays back dataset as a real-time stream
│   └── train_all.py                 # Fits ML models, scaling bounds, & exports stats
├── src/                             # Python Logic Core
│   ├── cpp/
│   │   └── telemetry_parser.cpp     # C++ high-speed JSON parser & IQR clipper
│   ├── infrastructure/              # Communication Bus & State Storage
│   │   ├── interfaces.py            # EventBus Abstract Base Class
│   │   ├── redis_bus.py             # Redis Streams driver + DLQ
│   │   ├── kafka_bus.py             # Apache Kafka consumer/producer
│   │   ├── bus_factory.py           # Ingestion factory chooser
│   │   ├── timescale_client.py      # TimescaleDB relational connector
│   │   ├── security.py              # JWT authentication tokens
│   │   └── tracing.py               # OpenTelemetry instrumentation
│   ├── pipeline/                    # Stream Processors
│   │   ├── telemetry_parser.py      # C++ library binder (ctypes fallback)
│   │   ├── preprocessor.py          # Scaling, windowing, and features
│   │   ├── data_loader.py           # Forecaster sliding window datasets
│   │   ├── topology_graph.py        # NetworkX nodes ↔ volumes topology mapping
│   │   └── stream_worker.py         # Stream worker ingestion consumer loop
│   ├── models/                      # ML Model Zoo
│   │   ├── classifier/              # LightGBM Classifier & ARF-ADWIN Drift
│   │   ├── anomaly/                 # 3-Tier Ensemble & Noisy Neighbor Detector
│   │   └── forecasting/             # N-BEATS (Capacity), TFT (Latency Quantiles)
│   └── control_plane/               # Decisions, Actuators, & Safety Loops
│       ├── inference_hub.py         # Monolithic coordinator class
│       ├── remote_inference_client.py# Triton GPU serving client proxy
│       ├── decision_engine.py       # Policy evaluation & circuit checks
│       ├── rebalancer.py            # Migration and QoS shaper actions
│       ├── monitor.py               # Watchdogs, action loops, and rollbacks
│       ├── simulator.py             # What-if simulations for the engine
│       ├── actuators.py             # Actuator layer (Stub & CSI K8s driver)
│       └── capacity_planner.py      # Capacity plans and autoscale evaluations
└── tests/                           # Pytest suites
```

---

## 🛠️ Local Development Setup

### 1. Install Dependencies
Ensure you have Python 3.11+ installed. Create a virtual environment and install the requirements:
```bash
# Create and activate virtual env
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Local Runbook (Three Terminals)
We execute the system locally using three parallel processes:

* **Terminal 1: FastAPI Control Plane Server**
  ```bash
  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
  ```
  *(Note: If a Redis server is not running on 127.0.0.1:6379, the API automatically triggers a **TCP Fallback Listener** on port 9000 to ingest metrics directly from the playback simulator.)*

* **Terminal 2: Streamlit Dashboard UI**
  ```bash
  streamlit run dashboard/Home.py --server.port 8501
  ```

* **Terminal 3: Telemetry Playback Streamer**
  ```bash
  python scripts/telemetry_playback.py
  ```

---

## 🧠 ML Model Training & Experiment Tracking

The system contains 7 distinct machine learning algorithms designed to train locally and track hyperparameter metrics via **MLflow**.

### Run Model Training Pipeline:
To generate training datasets, fit classifier scalers, compute outlier bounds, and export trained models (`.pth` and `.pkl` weights):
```bash
python scripts/train_all.py
```

### Launch MLflow UI:
To audit runs, parameter charts, and validation losses:
```bash
mlflow ui --port 5000
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. All metrics are stored locally in the gitignored `mlruns/` directory.

---

## ☸️ Enterprise Production Deployment (Kubernetes/Helm)

When transitioning to enterprise datacenters, the system acts as a strict **12-Factor App**. Injecting environment variables overrides the default settings configured in [settings.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/configs/settings.py).

### Core Settings Overrides:
* `ENVIRONMENT`: Set to `"production"` (swaps stub actuator for K8s CSI Driver).
* `BUS_TYPE`: Set to `"kafka"` (swaps Redis Streams for Kafka telemetry brokers).
* `DB_TYPE`: Set to `"timescaledb"` (swaps local file cache for TimescaleDB Hypertables).
* `INFERENCE_MODE`: Set to `"remote"` (swaps monolithic local InferenceHub for Triton GPU serving client proxy).

### Deploying via Helm:
A comprehensive Helm chart is located in [deploy/helm/hpe-control-plane/](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/deploy/helm/hpe-control-plane).

To deploy:
```bash
helm upgrade --install hpe-storage-control-plane ./deploy/helm/hpe-control-plane \
  --namespace hpe-storage --create-namespace \
  --values ./deploy/helm/hpe-control-plane/values.yaml
```

The values default to:
* **HPA:** Scalable deployment from 3 to 50 REST API replicas based on $75\%$ target CPU load.
* **Resources:** `2Gi` memory requests and `4Gi` limits per API pod to support robust PyTorch Tensor allocations.
* **ServiceMonitor:** Hooked automatically for **Prometheus Operator** discovery to monitor endpoints.

---

## 🔒 Closed-Loop Guardrails & Circuit Breaker

To prevent erratic actions in storage arrays, the Control Plane integrates state-of-the-art closed-loop safety limits:

1. **Watchdog Thread:** Located in [monitor.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/monitor.py), detects hung migrations and monitors execution timeouts.
2. **Auto-Rollback Engine:** If a target volume's average latency increases by **$\ge 20\%$** within the monitor window post-rebalance, the rebalancer automatically triggers a rollback operation.
3. **Tripping the Circuit Breaker:** The Decision Engine monitors the ratio of rolled back actions. If the rollback rate exceeds **$1\%$** (after a minimum of 10 actions), the circuit breaker trips, disabling autonomous migrations, setting the engine state to `circuit_breaker_tripped`, and alerting storage administrators.

---

## 📑 REST API Endpoint Reference (Major Routes)

| Method | Endpoint | Description | Layer |
|:---|:---|:---|:---|
| `POST` | `/token` | Generates OAuth2 JWT token for API access | Security |
| `GET` | `/health` | In-depth diagnostics of Redis, TCP Fallback, and DB | System |
| `GET` | `/health/live` | Kubernetes Liveness Probe | System |
| `GET` | `/health/ready` | K8s Readiness Probe (verifies models are loaded in RAM) | System |
| `GET` | `/kpi` | Retrieves pool-wide KPIs (average latency, total IOPS) | Metrics |
| `GET` | `/volumes` | Details of all 50 volumes, hotspot scores, workload labels | Control Plane |
| `GET` | `/volumes/{id}/metrics` | Time-series telemetry points for a specific volume | Metrics |
| `GET` | `/volumes/{id}/workload` | Detailed workload classifier outputs with confidence arrays | ML Models |
| `GET` | `/volumes/{id}/explain` | SHAP explainability feature attributions for diagnostics | ML Models |
| `GET` | `/noisy-neighbors` | Scans co-located volumes to map aggressor-victim pairs | ML Models |
| `GET` | `/forecast/capacity` | capacity projections calculated by N-BEATS | ML Models |
| `GET` | `/forecast/dtf` | Days-to-Fill prediction metrics (warning at 85%, critical at 95%) | ML Models |
| `GET` | `/forecast/ttv` | Time-to-Violation hourly forecasts for latency SLO limits | ML Models |
| `POST` | `/simulate/migrate` | Simulates rebalancing candidate moves with $O(1)$ calculations | Simulation |
| `POST` | `/rebalance` | Dispatches manual rebalance commands for a volume | Control Plane |
| `POST` | `/rollback` | Commands immediate manual rollback for an action ID | Control Plane |
| `GET` | `/topology` | Exports placement mappings (volumes ↔ nodes) as JSON | System |

*For the complete detailed API specifications, see the [API Reference Guide](docs/api_reference.md).*
