# IO Workload Pattern Classification & Hotspot Detection
## HPE CPP 2026 — Project Report

### 1. Executive Summary
This project implements an ML-powered storage control plane for classifying IO workloads, detecting hotspots/anomalies, forecasting capacity and latency risk, and driving policy-guarded corrective actions. The target operating context is a 50-volume, multi-node storage cluster where mixed workloads compete for shared resources and can trigger tail-latency spikes, noisy-neighbor interference, and capacity pressure.

The implemented system combines offline training and live inference: LightGBM and ARF+ADWIN for workload understanding, a three-model anomaly ensemble for hotspot intelligence, N-BEATS and TFT for predictive planning, and quantile demand forecasting for peak/95th/99th bandwidth demand estimation. It exposes recommendations, simulations, and control endpoints through FastAPI and a Streamlit dashboard, with Redis Streams or TCP fallback for telemetry flow.

Using the current saved artifacts, the tuned LightGBM model reports 95.12% test accuracy (models/classifier/lightgbm_tuned_metrics.json), ARF prequential accuracy is 86.46% on test stream order (models/classifier/arf_metrics.json), and the anomaly ensemble reports 11,883 detections with calibrated detector fusion stats (models/anomaly/ensemble/ensemble_stats.json). These results demonstrate a functional end-to-end control-plane prototype aligned with CPP3 scope.

### 2. Problem Statement
The CPP3 objective is to build an intelligent storage control plane that can: (1) classify workload patterns from telemetry, (2) detect and forecast hotspots/noisy-neighbor contention, and (3) trigger safe automated rebalancing while honoring latency, throughput, fairness, and rollback guardrails.

In practical terms, this means turning raw time-series telemetry into continuously updated decisions: where risk is emerging, how soon SLOs may be violated, what action has the best expected impact, and whether the system should rollback or stop automation when safety thresholds are crossed.

### 3. System Architecture
The implementation follows the CPP architecture layers and maps them to concrete modules.

- Data Platform:
  - Data generation: `src/data/data_generator.py`
  - Feature engineering and store creation: `src/data/feature_engineer.py`, `data/processed/io_features.parquet`
  - Topology & capacity graph: `src/pipeline/topology_graph.py`
  - Message bus integration and live telemetry path: Redis/TCP handling in `api/main.py`
- Control Plane:
  - Model inference hub: `src/control_plane/inference_hub.py`
  - Decision/policy engine: `src/control_plane/decision_engine.py`
  - Action monitor + rollback: `src/control_plane/monitor.py`
  - Action executor: `src/control_plane/rebalancer.py`
  - What-if simulator: `src/control_plane/simulator.py`
- API Layer:
  - FastAPI application and endpoint orchestration: `api/main.py`
  - Request schemas: `api/schemas/models.py`
- Dashboard:
  - Streamlit UI: `dashboard/app.py`
- Model Layer:
  - Classifier: `src/models/classifier/*`
  - Anomaly detection: `src/models/anomaly/*`
  - Forecasting (N-BEATS/TFT/Demand): `src/models/forecasting/*`

Interconnection flow:
1. Telemetry enters API (Redis Streams or TCP fallback), updates live state and topology metrics.
2. `InferenceHub` reads current features and runs classifiers/anomaly/forecast models.
3. `DecisionEngine` applies policies, persistence checks, simulation scoring, and executes actions through `Rebalancer`.
4. `ActionMonitor` enforces rollback and circuit-breaker safety.
5. API serves status/recommendations/simulations to dashboard and operators.

### 4. Dataset and Feature Engineering
Synthetic dataset profile (from `src/data/data_generator.py`):
- Volumes: 50
- Nodes: 5
- Time span: 30 days
- Sampling interval: 5 minutes
- Total rows: 432,000
- Workload classes: DB_OLTP, VM, Backup, AI_Training, AI_Inference

Engineered feature groups (from `src/data/feature_engineer.py`):

- Basic derived:
  - `iops_per_queue`, `total_throughput_mbps`, `write_pressure`
  - `read_latency_jitter`, `write_latency_jitter`, `avg_latency_us`
  - `iops_latency_score`, `capacity_burn_rate`, `capacity_headroom_gb`
- Time-based and cyclical:
  - `hour`, `day_of_week`, `is_weekend`, `day_of_month`
  - `hour_sin`, `hour_cos`, `day_sin`, `day_cos`
- Rolling window (15m, 30m, 1h):
  - `<metric>_roll_15m_mean/std`, `<metric>_roll_30m_mean/std`, `<metric>_roll_1h_mean/std`
  - Metrics: `total_iops`, `avg_latency_us`, `total_throughput_mbps`
- Lag features (5m, 15m, 30m, 60m):
  - `<metric>_lag_5m/15m/30m/60m`
  - Metrics: `total_iops`, `avg_latency_us`, `capacity_used_pct`
- Rate-of-change:
  - `<metric>_delta`, `<metric>_pct_change`
  - Metrics: `total_iops`, `avg_latency_us`, `capacity_used_pct`
- IO size entropy:
  - `io_size_entropy` computed per volume-hour bucket

Data generator design choices:
- Latent-factor generation (`generate_metrics_from_latent`) to avoid trivial label leakage.
- Time-varying workload schedule windows per volume (`build_workload_schedule`) for non-stationary behavior.
- Concept bleed enabled before anomaly injection: `bleed_ratio=0.03` (`apply_concept_bleed`).
- Noisy-neighbor injection: `n_events=500`, each event duration 3 intervals (15 real minutes) in `inject_noisy_neighbor_events`.

### 5. Workload Classification
LightGBM baseline (`src/models/classifier/lightgbm_baseline.py`):
- Objective: multiclass
- `n_estimators=300`, `learning_rate=0.1`, `num_leaves=31`
- `random_state=42`, `n_jobs=-1`

LightGBM tuned with Optuna (`src/models/classifier/lightgbm_tuned.py`):
- Configured search trials: `N_TRIALS=60`
- Sampler: TPE (`TPESampler(seed=42, multivariate=True)`)
- Pruner: Hyperband
- Early stopping: `EARLY_STOP=15`
- Search space includes:
  - `learning_rate`, `max_depth`, `num_leaves`
  - `min_child_samples`, `reg_alpha`, `reg_lambda`, `min_split_gain`
  - `colsample_bytree`, `subsample`, `subsample_freq`
- Artifact record (`models/classifier/lightgbm_tuned_metrics.json`) currently shows:
  - `n_trials_total=39`, `n_trials_completed=25`, `best_trial=31`

Actual tuned test metrics (from `models/classifier/lightgbm_tuned_metrics.json`):
- Test accuracy: **0.9511805555555556 (95.12%)**

Per-class test metrics:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| DB_OLTP (0) | 0.9083 | 0.8963 | 0.9023 | 14964 |
| VM (1) | 0.9579 | 0.9589 | 0.9584 | 13375 |
| Backup (2) | 0.9908 | 0.9898 | 0.9903 | 14615 |
| AI_Training (3) | 0.9837 | 0.9858 | 0.9848 | 14233 |
| AI_Inference (4) | 0.9178 | 0.9283 | 0.9230 | 14813 |

Target check (CPP criterion: ≥95% precision/recall):
- Overall test accuracy target is met (95.12%).
- Per-class precision/recall ≥95% is **not met for all classes** in current artifact.

ARF+ADWIN streaming classifier (`src/models/classifier/arf_adwin.py`):
- `n_models=25`, `grace_period=100`
- ADWIN drift delta: `0.001` (warning delta `0.01`)
- Evaluation style: prequential (test-then-train)

Actual ARF prequential accuracy (`models/classifier/arf_metrics.json`):
- Train final accuracy: **0.890519**
- Val final accuracy: **0.881962**
- Test final accuracy: **0.864639**
- Drift events: train `1`, val `0`, test `0`

### 6. Hotspot Detection and Anomaly Detection
Ensemble composition (`src/models/anomaly/ensemble_detector.py`):
- Statistical detector:
  - Rolling window: 24h
  - Sigma threshold: 3.0 (z-score)
- Isolation Forest:
  - `n_estimators=200`
  - `contamination=0.05`
- LSTM Autoencoder:
  - `sequence_length=12`

Fusion logic:
- Weighted score fusion:
  - `w_stat=0.35`, `w_if=0.35`, `w_lstm=0.30`
- Consensus gate:
  - `min_agreement=1`
  - alarm threshold: `40.0`

Actual ensemble artifact stats (`models/anomaly/ensemble/ensemble_stats.json`):
- Ensemble total detections: **11,883**
- Statistical detector detections: **854**
- Isolation Forest anomalies: **9,245**
- Isolation Forest anomaly rate: **0.0713**
- LSTM anomalies: **9,444**
- LSTM anomaly rate: **0.0592**
- LSTM threshold: **0.8584738969802856** (95th percentile)

Noisy-neighbor detection:
- Implemented via aggressor/victim analysis in `src/models/anomaly/noisy_neighbor.py` and invoked from `InferenceHub`.
- Configured latency z-threshold in hub initialization: **2.0**.

### 7. Predictive Capacity and Performance Planning
N-BEATS capacity forecasting (`src/models/forecasting/dtf_forecaster.py`):
- Input size: 20 days
- Forecast size: 7 days
- Extended to 60 days for DTF via autoregressive roll-forward
- Artifact (`models/forecasting/nbeats_training_stats.json`):
  - Best validation loss: **0.0006973701273091138**
  - Parameters: **501,363**

TFT latency forecasting (`src/models/forecasting/tft_forecaster.py`):
- Input: 24 hourly steps
- Forecast horizon: 6 hours
- Quantiles: p50/p90/p95
- Artifact (`models/forecasting/tft_training_stats.json`):
  - Best validation loss: **0.11326554754776741**
  - Parameters: **50,716**

Quantile demand forecasting (`src/models/forecasting/demand_forecaster.py`):
- Per-volume linear quantile regressors for IOPS and throughput
- Quantiles: `q=0.5`, `q=0.95`, `q=0.99`
- 24-hour forward output includes peak demand estimates:
  - `peak_iops`, `peak_throughput_mbps`

Time-to-violation (TTV):
- Implemented in `compute_latency_ttv` in `src/control_plane/inference_hub.py`.
- Uses TFT p95 forecast to compute first breach time vs SLO (`capacity_policy.latency_slo_threshold_us`).
- Risk levels:
  - `none`: no predicted breach
  - `low`: breach >5h
  - `medium`: breach in 3-5h
  - `high`: breach in 1-3h
  - `critical`: breach <1h or immediate

Tier headroom tracking:
- Implemented in `src/pipeline/topology_graph.py`:
  - `get_tier_headroom()`
  - `get_pool_headroom()`
- Aggregates total/used/headroom GB, used %, critical volumes.

### 8. Control Plane and Automated Rebalancing
Decision engine (`src/control_plane/decision_engine.py`):
- Hotspot persistence guard:
  - trigger score threshold from policy (`min_hotspot_score_to_trigger`, default 75)
  - minimum persistence: **2 minutes** (`min_hotspot_duration_minutes`)
- Action simulation and best-action selection:
  - migration, QoS, tier change scored by expected improvement
- Enforced limits:
  - max migrations/hour: **3**
  - max concurrent migrations: **1**

Action monitor (`src/control_plane/monitor.py`):
- Pre/post latency tracking per action
- Rollback if latency increase > **20%**
- Monitoring timeout: **5 minutes**

Circuit breaker:
- Trips when rollback rate exceeds **1.0%** (`max_rollback_rate_pct`)
- Auto-disables engine until reset endpoint or policy re-enable

Autoscale (`DecisionEngine._check_and_trigger_autoscale`):
- Trigger by DTF warning days threshold and cluster capacity threshold
- Creates virtual node(s) with policy limits:
  - min interval 24h
  - max new nodes/run 1

### 9. What-If Simulations
Implemented in `src/control_plane/simulator.py` and exposed in API:
- Capacity add simulation (`/simulate/capacity`):
  - returns before/after DTF and improvement days
- Migration simulation (`/simulate/migrate`):
  - returns safety, expected source improvement, target impact, estimated transfer time
- QoS shaping simulation (`/simulate/qos`):
  - returns throttling %, self impact, and expected neighbor relief
- Tier-change simulation (`/simulate/tier`):
  - returns expected latency improvement and throughput gain factor

### 10. Results and Success Criteria Evaluation
| Criterion | Target | Achieved | Evidence |
|---|---|---|---|
| Workload classification quality | ≥95% precision/recall | **Partially met**: test accuracy 95.12%, but not all class precision/recall values are ≥95% | `models/classifier/lightgbm_tuned_metrics.json` |
| Latency spike and utilization variance reduction | 20–40% reduction | Not directly quantified in persisted artifact files | Control logic and simulation paths in `src/control_plane/decision_engine.py`, `src/control_plane/simulator.py`; KPI endpoint `GET /kpi` |
| Hotspot detection within seconds + preemptive scheduling | Detect quickly and schedule moves | Functional implementation present; detector outputs and rebalancer queueing active | `src/models/anomaly/statistical_detector.py`, `src/control_plane/decision_engine.py`, `models/anomaly/ensemble/ensemble_stats.json` |
| Zero SLO breaches during automated actions; rollback rate <1% | rollback rate <1% and safe actions | Safety controls implemented (rollback + circuit breaker at 1.0%); no persisted campaign-level outcome artifact in repo | `src/control_plane/monitor.py`, `src/control_plane/decision_engine.py`, `configs/policy.yaml` |

### 11. API Reference Summary
| Method | Path | Description |
|---|---|---|
| GET | /health | Service/model/telemetry health |
| GET | /telemetry/status | Live telemetry state |
| GET | /kpi | Cluster KPI summary |
| GET | /volumes | Volume status list |
| GET | /volumes/{id}/metrics | Historical metrics for a volume |
| GET | /volumes/{id}/metrics/stream | SSE live stream for one volume |
| GET | /volumes/{id}/workload | Current workload classification |
| GET | /volumes/{id}/explain | SHAP feature contribution explanation |
| GET | /model/drift-status | ARF vs LightGBM agreement |
| GET | /model/performance | Validation performance summary |
| GET | /alerts | Active hotspot alerts |
| GET | /noisy-neighbors | Aggressor/victim relationships |
| GET | /forecast/capacity | DTF forecast list |
| GET | /forecast/bandwidth | 24h latency forecast (+ demand) for one volume |
| GET | /forecast/demand | 24h IOPS/throughput quantile demand for one volume |
| GET | /forecast/dtf | DTF urgency ranking |
| GET | /forecast/ttv | Latency SLO time-to-violation forecast |
| GET | /cluster/headroom | Tier + pool capacity headroom |
| GET | /cluster/headroom/tier/{tier_name} | Headroom for one storage tier |
| POST | /simulate/capacity | What-if capacity addition |
| POST | /simulate/migrate | What-if migration |
| POST | /simulate/qos | What-if QoS cap |
| POST | /simulate/tier | What-if tier change |
| GET | /recommendations | Priority recommendations |
| GET | /policy | Current active policy |
| PUT | /policy | Update rebalance/safety policy |
| GET | /rebalance/circuit-breaker | Circuit breaker status |
| POST | /rebalance/circuit-breaker/reset | Circuit breaker manual reset |
| GET | /rebalance/history | Executed action history |
| GET | /rebalance/monitors | Active/historical monitor state |
| POST | /rebalance | Manually trigger rebalance action |
| POST | /rollback | Roll back a prior action |
| GET | /topology | Topology graph for visualization |

Detailed endpoint docs are provided in `docs/api_reference.md`.

### 12. Running the Project
1. Install requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run full training pipeline

```bash
python3 scripts/train_all.py
```

3. Start API, dashboard, and telemetry playback (local)

```bash
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```bash
python3 -m streamlit run dashboard/app.py
```

```bash
python3 scripts/telemetry_playback.py
```

Optional Redis stream worker when using Redis mode:

```bash
python3 -m src.pipeline.stream_worker
```

4. Docker startup

```bash
docker compose up --build
```

### 13. Technology Stack
Dependencies from `requirements.txt`:

- pandas==2.2.3: tabular telemetry processing and feature engineering
- numpy>=2.2.4: numerical operations, vectorization, time-series transforms
- pyarrow==23.0.1: parquet read/write backend
- scipy>=1.10.0: scientific utilities used by ML stack
- scikit-learn>=1.4.0: preprocessing, metrics, Isolation Forest, QuantileRegressor
- lightgbm>=4.3.0: gradient-boosted multiclass workload classifier
- shap>=0.40.0: model explainability for workload predictions
- river==0.24.2: online ARF+ADWIN streaming classifier
- optuna>=4.0.0: hyperparameter optimization for tuned LightGBM
- joblib>=1.5.0: model serialization/deserialization
- torch>=2.0.0: deep learning models (LSTM-AE, N-BEATS, TFT)
- mlflow>=2.10.0: experiment tracking support
- matplotlib>=3.5.0: offline plots and diagnostics
- seaborn>=0.12.0: statistical plotting utilities
- plotly>=5.0.0: interactive graph and visualization output
- streamlit>=1.34.0: dashboard frontend
- networkx>=3.3: topology/capacity graph representation
- fastapi>=0.110.0: API service framework
- uvicorn[standard]>=0.29.0: ASGI server runtime
- requests>=2.31.0: HTTP client utilities
- pyyaml>=6.0: policy configuration parsing
- tqdm>=4.60.0: training progress bars
- redis>=5.0.1: Redis stream/cache integration

### 14. References
CPP3-provided references:
1. https://www.deeplearning.ai/
2. https://pypi.org/
3. https://www.kaggle.com/
4. https://wasabi.com/learn/ai-workloads
5. https://colab.research.google.com/

Additional implementation references:
6. LightGBM documentation: https://lightgbm.readthedocs.io/
7. Optuna documentation: https://optuna.readthedocs.io/
8. River documentation (ARF, ADWIN): https://riverml.xyz/
9. N-BEATS paper: Oreshkin et al., 2019 (arXiv:1905.10437)
10. Temporal Fusion Transformers paper: Lim et al., 2019 (arXiv:1912.09363)
11. Isolation Forest paper: Liu, Ting, Zhou, 2008 (ICDM)
12. FastAPI documentation: https://fastapi.tiangolo.com/
