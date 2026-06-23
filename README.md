# 🧠 HPE Storage Control Plane: IO Workload Pattern Classification & Hotspot Detection

### *Enterprise-grade Autonomous Storage Control Plane for SLO Compliance, Predictive Rebalancing, and Capacity Planning.*

---

## ⚡ Quick Start (2 Minutes)

Get the system running with one command:

```bash
# 1. Clone and navigate to project directory
git clone <repo-url>
cd IO-Workload-Pattern-Classification-Hotspot-Detection

# 2. Start all services with Docker Compose
docker compose up -d

# 3. Open the dashboard in your browser
open http://localhost:8501
```

**That's it!** The system will:
- ✅ Start Redis, API, Dashboard, Stream Worker, and Telemetry Generator
- ✅ Generate and process synthetic telemetry data
- ✅ Display live KPIs, hotspot analytics, and forecasts

> **Note:** First startup takes 2-3 minutes to build Docker images. Trained model artifacts are required (see [ML Model Training](#-ml-model-training--experiment-tracking) section).

---

## 🚀 Executive Summary

This repository contains an intelligent, predictive **Storage Control Plane** designed for the HPE Blueprint. It leverages deep learning and streaming machine learning to forecast capacity limits and latency breaches, proactively actuating volume rebalancing before Service Level Objectives (SLOs) are breached.

### Key Engineering & ML Highlights:

* **Dual-Track Ingestion Infrastructure:**
  * **Local Track:** Lightweight Docker Compose deployment using **Redis Streams** as the messaging backbone.
  * **Production Track:** Enterprise-scale **Kubernetes (K8s)** deployment leveraging **Apache Kafka** partitioned telemetry ingestion.
* **9 Machine Learning Models** organized into 4 categories:
  * **Workload Classification:** LightGBM (baseline & Optuna-tuned), ARF+ADWIN streaming classifier
  * **Anomaly Detection:** 3-tier ensemble (Statistical + IsolationForest + LSTM Autoencoder), Noisy Neighbor detector
  * **Forecasting:** N-BEATS (capacity), Temporal Fusion Transformer (latency), Demand Forecaster (IOPS/throughput)
* **Disaggregated Inference Architecture:**
  * Supports a local monolithic coordinator ([InferenceHub](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/inference_hub.py)) for lightweight deployments.
  * Native proxy routing ([RemoteInferenceClient](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/remote_inference_client.py)) redirects heavy inference requests to dedicated GPU clusters running **Triton** or **KServe**.
* **High-Throughput Hybrid Pipeline:** Telemetry JSON stream is parsed and outlier-clipped by a C++ extension ([telemetry_parser.cpp](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/cpp/telemetry_parser.cpp)) with Python fallback.
* **State Management Persistence:** Seamlessly transitions from local memory structures to time-series optimized **TimescaleDB** hypertables for windowed metrics.
* **$O(1)$ Optimization for What-If Analysis:** High-performance simulations mapping rebalance candidates are kept under sub-millisecond latencies by replacing global dataframe scans with hash-mapped dictionary lookups.
* **Observability & Proactive Safeguards:**
  * Automatic **Prometheus ServiceMonitor** hooks reporting real-time system performance.
  * **Closed-loop control** that rolls back migration actions if the target volume's average latency increases by $\ge 20\%$ post-move.
  * **Circuit Breaker** safeguard that disables auto-rebalancing if the action rollback rate exceeds $1\%$.

## Architecture

For a comprehensive system architecture diagram and production deployment details, see the [Production Architecture Guide](docs/production_architecture.md).

---

## 📋 Prerequisites

Before running this project, ensure you have:

### Required:
- **Python**: 3.11 or higher
- **Docker**: 20.10+ and Docker Compose v2+
- **RAM**: Minimum 8GB (16GB recommended for ML training)
- **Disk Space**: ~5GB for models and datasets

### Optional (for production):
- **Kubernetes**: 1.28+
- **Helm**: 3.0+
- **Apache Kafka**: 3.5+ (for production message bus)
- **TimescaleDB**: 2.13+ (for production time-series storage)
- **Prometheus**: 2.45+ (for metrics collection)
- **Grafana**: 10.0+ (for visualization)

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
│   ├── raw/                         # Synthetic telemetry CSV
│   ├── processed/                   # Engineered features Parquet
│   └── features/                    # Train/test split datasets
├── deploy/                          # Deployment Target Manifests
│   ├── helm/
│   │   └── hpe-control-plane/       # Production K8s charts (HPA, Prometheus)
│   ├── k8s/                         # Static manifests (Dashboard Service etc.)
│   └── monitoring/                  # Prometheus & Grafana Configuration
├── docs/                            # Design Documentation
│   ├── images/
│   │   └── architecture_diagram.png # System Architecture Diagram
│   ├── production_architecture.md   # Enterprise Deployment Guidelines
│   ├── api_reference.md             # REST API Reference
│   └── project_report.md            # Full Technical Report
├── models/                          # Trained Weights & Scalers (Gitignored)
│   ├── classifier/                  # LightGBM, ARF+ADWIN models
│   ├── anomaly/                     # Ensemble detector models
│   ├── forecasting/                 # N-BEATS, TFT, Demand forecaster
│   ├── scaler.pkl                   # StandardScaler for preprocessing
│   └── bounds.json                  # IQR outlier bounds
├── notebooks/                       # Jupyter Notebooks for Analysis
│   ├── 01_eda_and_data_exploration.ipynb
│   ├── 02_workload_classifier_evaluation.ipynb
│   ├── 03_hotspot_and_anomaly_detection_demo.ipynb
│   └── 04_forecasting_and_capacity_planning.ipynb
├── scripts/                         # Automation & Training Pipelines
│   ├── telemetry_playback.py        # Plays back dataset as a real-time stream
│   └── train_all.py                 # Fits ML models, scaling bounds, & exports stats
├── src/                             # Python Logic Core
│   ├── cpp/
│   │   └── telemetry_parser.cpp     # C++ high-speed JSON parser & IQR clipper
│   ├── data/                        # Data generation and feature engineering
│   │   ├── data_generator.py        # Synthetic telemetry generator
│   │   └── feature_engineer.py      # Feature extraction pipeline
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
└── tests/                           # Pytest suites (12+ test modules)
```

---

## 🐳 Docker Compose Deployment (Recommended)

The easiest way to run the entire system is using Docker Compose with all services orchestrated:

```bash
# Build and start all services
docker compose up --build

# Or run in detached mode (background)
docker compose up -d

# View logs from all services
docker compose logs -f

# View logs from specific service
docker compose logs -f api

# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

### Services Started:
Docker Compose orchestrates **5 containers**:

1. **redis** - Message broker and state store (port 6379)
2. **api** - FastAPI REST server with 4 workers (port 8000)
3. **dashboard** - Streamlit UI (port 8501)
4. **stream-worker** - Telemetry ingestion consumer and ML inference engine
5. **telemetry-generator** - Simulated telemetry playback from Parquet files

### Access Points:
- **Dashboard UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics (Prometheus format)

### Important Notes:
- Model artifacts must exist in `models/` directory before starting
- Run `python scripts/train_all.py` first if models are missing
- First build takes 2-3 minutes; subsequent starts are instant
- All data persists in Docker volumes (survives restarts)

---
## 🛠️ Local Development Setup (Manual)

For development and debugging, you can run services individually.

### 1. Install Dependencies
Ensure you have Python 3.11+ installed. Create a virtual environment and install the requirements:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Generate Data and Train Models
Before running the system, generate synthetic data and train ML models:

```bash
# Generate synthetic telemetry data (30 days, 50 volumes)
python src/data/data_generator.py

# Engineer features with rolling statistics
python src/data/feature_engineer.py

# Train all ML models and export artifacts
python scripts/train_all.py
```

This creates:
- `data/raw/telemetry_raw.csv` - Raw synthetic telemetry
- `data/processed/io_features.parquet` - Engineered features
- `models/` - Trained model weights (.pkl, .pth files)
- `models/bounds.json` - IQR outlier bounds

### 3. Local Runbook (Four Processes)

Run these in separate terminals:

**Terminal 1: Start Redis**
```bash
# Using Docker
docker run -p 6379:6379 redis:7-alpine

# Or using local Redis
redis-server
```

**Terminal 2: FastAPI Control Plane Server**
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 3: Stream Worker (Required for Production Mode)**
```bash
python -m src.pipeline.stream_worker
```

The stream worker:
- Consumes telemetry from `telemetry:stream` via consumer group `cg_control_plane`
- Parses and clips telemetry using C++/Python hybrid parser
- Runs ML inference every 30 seconds via InferenceHub
- Executes control plane decisions via DecisionEngine
- Persists results to Redis hash keys and action history

**Terminal 4: Streamlit Dashboard UI**
```bash
streamlit run dashboard/Home.py --server.port 8501
```

**Terminal 5: Telemetry Playback (Simulator)**
```bash
python scripts/telemetry_playback.py
```

> **Note:** If Redis is not running on 127.0.0.1:6379, the API automatically triggers a **TCP Fallback Listener** on port 9000 to ingest metrics directly from the playback simulator (development mode only).

---
## 📊 Data Generation & Feature Engineering

The system uses synthetic telemetry data simulating 5 workload archetypes across 50 volumes.

### Generate Synthetic Dataset:
```bash
python src/data/data_generator.py
```

**Output:**
- `data/raw/telemetry_raw.csv` - 30 days of synthetic telemetry at 5-minute intervals
- Simulates 5 workload types: DB_OLTP, VM, Backup, AI_Training, AI_Inference
- 50 volumes with realistic IOPS, latency, and throughput patterns

### Feature Engineering:
```bash
python src/data/feature_engineer.py
```

**Generated Features:**
- Rolling statistics: 5-minute, 30-minute, 1-hour windows
- `read_write_ratio` - Read vs write operation balance
- `io_size_entropy` - Randomness in IO request sizes
- `iops_per_queue` - Queue depth efficiency metrics
- `sequential_ratio` - Sequential vs random access patterns
- `latency_cv` - Coefficient of variation for latency stability

**Output:**
- `data/processed/io_features.parquet` - Processed features ready for training
- Chronological split: 21 days training, 9 days testing

---

## 🧠 ML Model Training & Experiment Tracking

The system contains **9 distinct machine learning models** organized into 4 categories.

### Run Model Training Pipeline:
To generate training datasets, fit classifier scalers, compute outlier bounds, and export trained models:

```bash
python scripts/train_all.py
```

**This trains:**

1. **Workload Classification:**
   - LightGBM Baseline (`lightgbm_model.pkl`)
   - LightGBM Tuned with Optuna HPO (`lightgbm_tuned_model.pkl`)
   - ARF+ADWIN Streaming Classifier (`arf_model.pkl`)
   - StandardScaler (`scaler.pkl`)

2. **Anomaly Detection:**
   - Statistical Hotspot Detector (24h rolling z-score)
   - Isolation Forest (200 trees, 5% contamination)
   - LSTM Autoencoder (PyTorch, 64 hidden units, 8 latent)
   - Noisy Neighbor Detector (co-location mapping)

3. **Forecasting:**
   - N-BEATS (`nbeats_model.pth`) - Capacity forecasting, 20d → 7d
   - Temporal Fusion Transformer (`tft_model.pth`) - Latency quantiles, 24h → 6h
   - Demand Forecaster (`demand_forecaster.pkl`) - IOPS/throughput 24h ahead

### Launch MLflow UI:
To audit runs, parameter charts, and validation losses:

```bash
mlflow ui --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

**MLflow tracks:**
- Hyperparameters for each model
- Training/validation metrics
- Model artifacts and checkpoints
- Feature importance rankings
- Confusion matrices and ROC curves

All experiment data is stored locally in the gitignored `mlruns/` directory.

---
## 📓 Exploratory Notebooks

Interactive Jupyter notebooks are available in the `notebooks/` directory for in-depth analysis:

1. **[EDA & Data Exploration](notebooks/01_eda_and_data_exploration.ipynb)**  
   Analyzes synthetic telemetry distributions, workload patterns, and feature correlations.

2. **[Workload Classifier Evaluation](notebooks/02_workload_classifier_evaluation.ipynb)**  
   Evaluates LightGBM and ARF+ADWIN classifier performance with confusion matrices and accuracy metrics.

3. **[Hotspot & Anomaly Detection Demo](notebooks/03_hotspot_and_anomaly_detection_demo.ipynb)**  
   Demonstrates the 3-tier ensemble detector with SHAP explainability and feature attribution.

4. **[Forecasting & Capacity Planning](notebooks/04_forecasting_and_capacity_planning.ipynb)**  
   Visualizes N-BEATS capacity forecasts, TFT latency predictions, and Days-to-Fill calculations.

### Running Notebooks:
```bash
# Launch Jupyter
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab notebooks/
```

---

## 🧪 Testing

The project includes comprehensive test coverage for all core components.

### Run All Tests:
```bash
# Run pytest with coverage report
pytest tests/ -v --cov=src --cov=api --cov-report=html

# View HTML coverage report
open htmlcov/index.html
```

### Run Specific Test Suites:
```bash
# Test control plane decision logic
pytest tests/test_decision_engine.py -v

# Test API security and JWT authentication
pytest tests/test_api_security.py -v

# Test infrastructure components
pytest tests/test_redis_bus.py tests/test_kafka_bus.py -v

# Test C++ telemetry parser
pytest tests/test_cpp_parser.py -v

# Test actuators and capacity planner
pytest tests/test_actuators_and_planner.py -v

# Test Kubernetes CSI actuator
pytest tests/test_csi_actuator.py -v

# Test production pipeline end-to-end
pytest tests/test_production_pipeline.py -v
```

### Test Categories:
- **Unit Tests**: Individual component testing (parsers, detectors, models)
- **Integration Tests**: End-to-end pipeline testing (ingestion → inference → actuation)
- **Security Tests**: JWT authentication, authorization, and token validation
- **Infrastructure Tests**: Message bus (Redis/Kafka), database connectors (TimescaleDB)
- **Control Plane Tests**: Decision engine, rebalancer, rollback logic, circuit breaker

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

```bash
# Install with default values
helm upgrade --install hpe-storage-control-plane ./deploy/helm/hpe-control-plane \
  --namespace hpe-storage --create-namespace \
  --values ./deploy/helm/hpe-control-plane/values.yaml

# Install with custom production values
helm upgrade --install hpe-storage-control-plane ./deploy/helm/hpe-control-plane \
  --namespace hpe-storage --create-namespace \
  --set environment=production \
  --set busType=kafka \
  --set kafka.bootstrapServers="kafka-broker-1:9092,kafka-broker-2:9092" \
  --set timescaledb.host="timescale.prod.svc.cluster.local" \
  --set inferenceMode=remote \
  --set triton.endpoint="http://triton-inference:8000"
```

### Helm Chart Features:
- **HPA (Horizontal Pod Autoscaler)**: Scales API replicas from 3 to 50 based on 75% target CPU load
- **Resource Limits**: 2Gi memory requests, 4Gi limits per API pod for PyTorch tensors
- **ServiceMonitor**: Auto-discovery by Prometheus Operator for `/metrics` endpoint
- **Liveness/Readiness Probes**: K8s health checks for zero-downtime deployments
- **PersistentVolumeClaims**: For model artifacts and configuration persistence
- **ConfigMaps**: For policy.yaml and environment-specific settings
- **Secrets**: For JWT keys, database credentials, and Kafka certificates

### Verify Deployment:
```bash
# Check pod status
kubectl get pods -n hpe-storage

# View logs
kubectl logs -n hpe-storage deployment/hpe-storage-control-plane-api -f

# Check HPA status
kubectl get hpa -n hpe-storage

# Access dashboard via port-forward
kubectl port-forward -n hpe-storage svc/hpe-storage-control-plane-dashboard 8501:8501
```

---
## 🔒 Closed-Loop Guardrails & Circuit Breaker

To prevent erratic actions in storage arrays, the Control Plane integrates state-of-the-art closed-loop safety limits:

1. **Watchdog Thread:** Located in [monitor.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/monitor.py), detects hung migrations and monitors execution timeouts (300s stall detection).

2. **Auto-Rollback Engine:** If a target volume's average latency increases by **$\ge 20\%$** within the monitor window post-rebalance, the rebalancer automatically triggers a rollback operation to restore the previous topology.

3. **Tripping the Circuit Breaker:** The Decision Engine monitors the ratio of rolled back actions. If the rollback rate exceeds **$1\%$** (after a minimum of 10 actions), the circuit breaker trips, disabling autonomous migrations, setting the engine state to `circuit_breaker_tripped`, and alerting storage administrators.

### Safety Thresholds (configurable in `configs/policy.yaml`):
```yaml
safety_guardrails:
  rollback_if_target_latency_increases_pct: 20  # Auto-rollback threshold
  rollback_timeout_minutes: 15                   # Monitoring window
  max_rollback_rate_pct: 1.0                     # Circuit breaker trip point
```

---

## 🔧 Configuration

### CORS Settings

For local development, CORS defaults to allow all origins (`*`).  
For production or shared deployments, set the `CORS_ORIGINS` environment variable:

**On Linux/macOS:**
```bash
CORS_ORIGINS="http://localhost:8501,https://dashboard.company.com" \
  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**On Windows (PowerShell):**
```powershell
$env:CORS_ORIGINS = "http://localhost:8501,https://dashboard.company.com"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**In Docker Compose:**
Edit `docker-compose.yml`:
```yaml
api:
  environment:
    CORS_ORIGINS: "http://localhost:8501,https://dashboard.company.com"
```

### Policy Configuration

Edit `configs/policy.yaml` to adjust:
- **Rebalancing thresholds**: Minimum hotspot score to trigger actions
- **Rate limits**: Max volumes moved per hour, max concurrent migrations
- **Safety guardrails**: Rollback thresholds, circuit breaker sensitivity
- **Capacity thresholds**: Warning/critical levels, Days-to-Fill limits
- **QoS policies**: IOPS caps for backup workloads, noisy neighbor thresholds

Changes take effect immediately (hot-reload via Redis state sync).

---
## 📑 REST API Endpoint Reference

The FastAPI server exposes 37+ endpoints for control plane operations.

### Core Endpoints:

| Method | Endpoint | Description | Category |
|:---|:---|:---|:---|
| `POST` | `/token` | Generates OAuth2 JWT token for API access | Security |
| `GET` | `/health` | In-depth diagnostics of Redis, TCP Fallback, and DB | System |
| `GET` | `/health/live` | Kubernetes Liveness Probe | System |
| `GET` | `/health/ready` | K8s Readiness Probe (verifies models are loaded) | System |
| `GET` | `/metrics` | Prometheus metrics export | Observability |
| `GET` | `/kpi` | Pool-wide KPIs (avg latency, total IOPS, rollback rate) | Metrics |
| `GET` | `/volumes` | All volumes with hotspot scores and workload labels | Control Plane |
| `GET` | `/volumes/{id}/metrics` | Time-series telemetry points for a volume | Metrics |
| `GET` | `/volumes/{id}/workload` | Workload classifier outputs with confidence arrays | ML Models |
| `GET` | `/volumes/{id}/explain` | SHAP explainability feature attributions | ML Models |
| `GET` | `/volumes/{id}/forecast` | Combined N-BEATS + TFT forecast results | ML Models |
| `GET` | `/noisy-neighbors` | Aggressor-victim pairs from co-located analysis | ML Models |
| `GET` | `/forecast/capacity` | Capacity projections (N-BEATS) | ML Models |
| `GET` | `/forecast/dtf` | Days-to-Fill predictions (85% warning, 95% critical) | ML Models |
| `GET` | `/forecast/ttv` | Time-to-Violation hourly latency SLO forecasts | ML Models |
| `GET` | `/topology` | Volume-to-node placement mappings (NetworkX export) | System |
| `GET` | `/policy` | Current rebalance policy and safety guardrails | Control Plane |
| `PUT` | `/policy` | Update rebalance policy and guardrail thresholds | Control Plane |
| `POST` | `/simulate/migrate` | Simulate volume migration ($O(1)$ calculation) | Simulation |
| `POST` | `/simulate/qos` | Simulate QoS IOPS cap impact | Simulation |
| `POST` | `/simulate/tier` | Simulate tier change (SSD ↔ NVMe ↔ HDD) | Simulation |
| `POST` | `/rebalance/trigger` | Dispatch manual rebalance for a volume | Control Plane |
| `POST` | `/rollback` | Command immediate rollback for an action ID | Control Plane |
| `GET` | `/rebalance/monitors` | Active action monitors with latency tracking | Control Plane |
| `GET` | `/rebalance/history` | Historical action log with timestamps and outcomes | Control Plane |
| `GET` | `/capacity/plan` | Autoscaling recommendations and node additions | Capacity Planning |

*For complete API specifications with request/response schemas, see the [API Reference Guide](docs/api_reference.md) or visit `/docs` (Swagger UI) when the server is running.*

---
## 🔧 Troubleshooting

### Common Issues:

**Problem:** `ModuleNotFoundError: No module named 'src'`  
**Solution:** Ensure you're running commands from the project root directory, not from subdirectories.

**Problem:** API returns "Models not loaded in RAM" or 500 errors  
**Solution:** Run `python scripts/train_all.py` to generate model artifacts first. Models must exist in `models/` directory.

**Problem:** Docker build fails with "models/ directory not found"  
**Solution:** Train models locally first or mount pre-trained models into containers. Models are gitignored and must be generated.

**Problem:** `redis.exceptions.ConnectionError: Connection refused`  
**Solution:** Ensure Redis is running. Test with `redis-cli ping` (should return `PONG`). Start Redis with `docker run -p 6379:6379 redis:7-alpine`.

**Problem:** Dashboard shows "Connection Error" or "Cannot connect to API"  
**Solution:** Verify API is running on port 8000: `curl http://localhost:8000/health`. Check `HPE_API_URL` environment variable is set correctly.

**Problem:** Stream worker not processing messages  
**Solution:** 
- Check Redis Stream has messages: `redis-cli XLEN telemetry:stream`
- Verify consumer group exists: `redis-cli XINFO GROUPS telemetry:stream`
- Check stream worker logs for errors

**Problem:** Telemetry playback script fails with "FileNotFoundError"  
**Solution:** Generate data first with `python src/data/data_generator.py`. Check that `data/processed/io_features.parquet` exists.

**Problem:** CUDA/GPU errors during training  
**Solution:** PyTorch models (N-BEATS, TFT, LSTM) will fall back to CPU if CUDA is unavailable. For GPU training, ensure CUDA toolkit is installed and `torch.cuda.is_available()` returns `True`.

**Problem:** MLflow UI shows no experiments  
**Solution:** Ensure `mlruns/` directory exists and training scripts completed successfully. Check `mlflow ui --backend-store-uri file:///path/to/mlruns`.

**Problem:** Kubernetes pods stuck in `CrashLoopBackOff`  
**Solution:** 
- Check pod logs: `kubectl logs -n hpe-storage <pod-name>`
- Verify ConfigMaps and Secrets are created
- Ensure model artifacts are available via PVC or init containers
- Check resource limits (may need more memory for ML models)

**Problem:** Circuit breaker tripped unexpectedly  
**Solution:** Review rollback history in dashboard or via `GET /rebalance/history`. Adjust `max_rollback_rate_pct` in `configs/policy.yaml` if threshold is too aggressive.

**Problem:** High memory usage in API container  
**Solution:** ML models (especially LightGBM, LSTM, TFT) consume significant RAM. Increase Docker memory limits or reduce `--workers` count in uvicorn command.

### Getting Help:
- Check logs: `docker compose logs -f` or `kubectl logs -f <pod-name>`
- Review documentation: `docs/production_architecture.md`
- Inspect Redis state: `redis-cli` → `KEYS *` → `HGETALL <key>`
- API diagnostics: `curl http://localhost:8000/health` (shows detailed component status)

---
## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Production Architecture Guide](docs/production_architecture.md)** - Enterprise deployment patterns, dual-track architecture, scaling strategies
- **[API Reference](docs/api_reference.md)** - Complete REST API documentation with request/response schemas
- **[Project Report](docs/project_report.md)** - Full technical report covering ML methodologies, system design, and evaluation metrics
- **[Architecture Diagram](docs/images/architecture_diagram.png)** - Visual system architecture with all components and data flows

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Workflow:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with clear, atomic commits
4. Write or update tests for new functionality
5. Ensure all tests pass (`pytest tests/ -v`)
6. Update documentation as needed
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to your branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request with a detailed description

### Code Style:
- **Python**: Follow PEP 8 style guide
- **Type Hints**: Use type annotations for all function signatures
- **Docstrings**: Use Google-style docstrings for classes and public methods
- **Testing**: Maintain >80% code coverage for new code
- **Commits**: Use conventional commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, etc.)

### Testing Requirements:
```bash
# Run tests with coverage
pytest tests/ -v --cov=src --cov=api

# Run linting
pylint src/ api/

# Format code
black src/ api/ tests/
```

### Areas for Contribution:
- Additional ML models (transformer-based classifiers, attention mechanisms)
- Enhanced visualization dashboards
- Integration with real storage arrays (HPE Primera, Nimble, 3PAR)
- Performance optimizations
- Additional actuator implementations (REST APIs, CLI tools)
- Documentation improvements and tutorials

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

This project was developed as part of the HPE Blueprint initiative for intelligent storage management.

**Key Technologies:**
- **ML Frameworks:** scikit-learn, LightGBM, PyTorch, River
- **Deep Learning:** N-BEATS, Temporal Fusion Transformer, LSTM Autoencoders
- **Infrastructure:** FastAPI, Streamlit, Redis, Apache Kafka, TimescaleDB
- **Orchestration:** Docker, Kubernetes, Helm
- **Observability:** Prometheus, Grafana, OpenTelemetry, MLflow

---

## 📬 Contact & Support

For questions, issues, or enterprise support inquiries, please:
- Open an issue in this repository
- Review existing documentation in `docs/`
- Check the troubleshooting section above

---

**Built with ❤️ for intelligent storage automation**
