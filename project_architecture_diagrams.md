# IO Workload Pattern Classification & Hotspot Detection — Architecture Diagrams

Complete system architecture derived from line-by-line analysis of the full codebase.

---

## 1. High-Level System Block Diagrams

The system implements a **Dual-Track Architecture** allowing the same codebase to run locally or scale seamlessly to datacenters.

### 1.1. Local Prototype System Block Diagram (Docker + Redis)

This diagram shows the local orchestration using Docker Compose and Redis Streams.

```mermaid
graph TB
    subgraph Docker["Docker Compose Cluster"]
        direction TB
        
        subgraph TelGen["telemetry-generator"]
            TP["telemetry_playback.py<br/>Reads Parquet → publishes<br/>to Redis Stream"]
        end
        
        subgraph RedisNode["Redis 7 Alpine"]
            RS["telemetry:stream<br/>(Redis Stream)"]
            RH["volume:*:metrics<br/>volume:*:analysis<br/>volume:*:history<br/>(Hash/List keys)"]
            RP["control_plane:policy<br/>control_plane:action_history<br/>control_plane:active_monitors<br/>control_plane:circuit_breaker<br/>topology:volume_to_node<br/>(State keys)"]
            DLQ["hpe_telemetry_dlq<br/>(Dead-Letter Queue)"]
        end
        
        subgraph SW["stream-worker"]
            SWP["stream_worker.py<br/>Consumer Group: cg_control_plane"]
        end
        
        subgraph API["api (FastAPI × 4 workers)"]
            MAIN["api/main.py<br/>97KB · 2381 lines<br/>Uvicorn --workers 4"]
        end
        
        subgraph Dash["dashboard (Streamlit)"]
            HOME["Home.py"]
            P1["1_Cluster_Overview.py"]
            P2["2_Hotspot_Analytics.py"]
            P3["3_Forecasting.py"]
            P4["4_Control_Plane.py"]
        end
    end

    TP -->|XADD| RS
    RS -->|XREADGROUP| SWP
    SWP -->|HSET / SET| RH
    SWP -->|SET / HSET| RP
    SWP -.->|publish_dlq| DLQ
    MAIN -->|GET / HGETALL| RH
    MAIN -->|GET / SET| RP
    P1 -->|HTTP GET/PUT| MAIN
    P2 -->|HTTP GET| MAIN
    P3 -->|HTTP GET| MAIN
    P4 -->|HTTP GET/PUT/POST| MAIN

    style Docker fill:#0d1117,stroke:#30363d,color:#c9d1d9
    style RedisNode fill:#1a1a2e,stroke:#e94560,color:#ffffff
    style SW fill:#16213e,stroke:#0f3460,color:#ffffff
    style API fill:#1a1a2e,stroke:#7b2cbf,color:#ffffff
    style Dash fill:#16213e,stroke:#00e676,color:#ffffff
```

### 1.2. Datacenter Production System Block Diagram (Kubernetes + Kafka + Triton)

This diagram shows the enterprise deployment target running on Kubernetes, leveraging Kafka for high-throughput ingestion, TimescaleDB for time-series persistence, and a disaggregated Triton/KServe inference serving pool.

```mermaid
graph TB
    subgraph K8s["Kubernetes Production Cluster"]
        direction TB
        
        subgraph TelGen["telemetry-playback (Replica: 1)"]
            TP["telemetry_playback.py<br/>Reads telemetry data → publishes<br/>to Kafka Telemetry Topic"]
        end
        
        subgraph KafkaNode["Apache Kafka Enterprise Broker"]
            RS["telemetry-topic<br/>(Partitioned Ingestion)"]
            DLQ["hpe_telemetry_dlq<br/>(Kafka Dead-Letter Topic)"]
        end
        
        subgraph SW["stream-worker (Replicas: 3+)"]
            SWP["stream_worker.py<br/>Consumer Group: cg_control_plane"]
        end
        
        subgraph API["api-control-plane (FastAPI × 3-50 Pods HPA)"]
            MAIN["api/main.py<br/>REST endpoints with JWT auth"]
        end
        
        subgraph Dash["dashboard-ui (Streamlit Service)"]
            HOME["Home.py"]
            P1["1_Cluster_Overview.py"]
            P2["2_Hotspot_Analytics.py"]
            P3["3_Forecasting.py"]
            P4["4_Control_Plane.py"]
        end
        
        subgraph DB["TimescaleDB StatefulSet"]
            TS["TimescaleClient<br/>Hypertable partition queries"]
        end
        
        subgraph Serving["ML Serving Cluster (Triton / KServe)"]
            T_LGB["LightGBM Classifier Model"]
            T_ENS["Anomaly Ensemble Models"]
            T_TFT["TFT Latency Forecaster"]
            T_NB["N-BEATS Capacity Forecaster"]
        end
    end

    TP -->|Kafka Producer| RS
    RS -->|Kafka Consumer| SWP
    SWP -->|gRPC / HTTP| Serving
    SWP -->|INSERT / UPDATE| TS
    SWP -.->|publish_dlq| DLQ
    MAIN -->|SELECT / windowed queries| TS
    MAIN -->|gRPC / HTTP| Serving
    P1 -->|HTTP GET/PUT| MAIN
    P2 -->|HTTP GET| MAIN
    P3 -->|HTTP GET| MAIN
    P4 -->|HTTP GET/PUT/POST| MAIN

    style K8s fill:#0d1117,stroke:#30363d,color:#c9d1d9
    style KafkaNode fill:#1a1a2e,stroke:#e94560,color:#ffffff
    style SW fill:#16213e,stroke:#0f3460,color:#ffffff
    style API fill:#1a1a2e,stroke:#7b2cbf,color:#ffffff
    style Dash fill:#16213e,stroke:#00e676,color:#ffffff
    style DB fill:#1a1a2e,stroke:#00e676,color:#ffffff
    style Serving fill:#1a1a2e,stroke:#ff9100,color:#ffffff
```

---

## 2. ML Model Hierarchy Block Diagram

Shows every trained model, its type, and how they feed into the InferenceHub.

```mermaid
graph LR
    subgraph InferenceHub["InferenceHub (inference_hub.py)"]
        direction TB
        AV["analyze_volume()"]
    end

    subgraph Classification["Workload Classification"]
        LGB["LightGBM Tuned<br/>(lightgbm_tuned_model.pkl)<br/>5-class: DB_OLTP, VM,<br/>Backup, AI_Training,<br/>AI_Inference"]
        ARF["ARF+ADWIN<br/>(arf_model.pkl)<br/>Streaming drift detector<br/>Online learning with<br/>pseudo-labels"]
        SC["StandardScaler<br/>(scaler.pkl)<br/>log1p transform"]
    end

    subgraph Anomaly["3-Tier Ensemble Anomaly (ensemble_detector.py)"]
        direction TB
        SD["Tier 1: StatisticalDetector<br/>Rolling 24h z-score<br/>IQR outlier bounds<br/>Per-volume baseline"]
        IF["Tier 2: IsolationForest<br/>200 trees, 5% contamination<br/>10 feature dimensions<br/>Global cross-volume"]
        LSTM["Tier 3: LSTM Autoencoder<br/>PyTorch · 64 hidden · 8 latent<br/>12-step sequences<br/>Reconstruction error"]
        FUSE["Weighted Fusion<br/>w_stat=0.35 · w_if=0.35<br/>w_lstm=0.30<br/>+ Consensus Gate"]
    end

    subgraph Forecasting["Forecasting Models"]
        NB["N-BEATS<br/>(nbeats_model.pth)<br/>3 stacks × 3 blocks<br/>Input: 20 days → 7 days<br/>Capacity: DTF 85%/95%"]
        TFT["Temporal Fusion Transformer<br/>(tft_model.pth)<br/>24h input → 6h forecast<br/>Quantiles: p50, p90, p95<br/>Multi-head attention"]
        DF["DemandForecaster<br/>(demand_forecaster.pkl)<br/>Quantile regressors<br/>IOPS/Throughput 24h"]
    end

    subgraph Detection["Noisy Neighbor"]
        NN["NoisyNeighborDetector<br/>Z-score baselines<br/>Aggressor-victim mapping<br/>Co-located node scan"]
    end

    SC --> LGB
    SC --> ARF
    LGB --> AV
    ARF --> AV
    SD --> FUSE
    IF --> FUSE
    LSTM --> FUSE
    FUSE --> AV
    NB --> AV
    TFT --> AV
    DF --> AV
    NN --> AV

    style InferenceHub fill:#7b2cbf,stroke:#ffffff,color:#ffffff
    style Classification fill:#1a1a2e,stroke:#00f0ff,color:#ffffff
    style Anomaly fill:#1a1a2e,stroke:#ff1744,color:#ffffff
    style Forecasting fill:#1a1a2e,stroke:#ff9100,color:#ffffff
    style Detection fill:#1a1a2e,stroke:#00e676,color:#ffffff
```

---

## 3. Control Plane Closed-Loop Block Diagram

Shows the full observe → decide → act → monitor feedback loop.

```mermaid
graph TB
    subgraph Observe["OBSERVE Layer"]
        TEL["Telemetry Stream<br/>(Redis Streams / Kafka)"]
        PARSE["TelemetryParser<br/>C++ binary / Python fallback<br/>IQR outlier clipping"]
        TOPO["TopologyGraph<br/>(NetworkX)<br/>Nodes ↔ Volumes<br/>Tier assignments"]
    end

    subgraph Decide["DECIDE Layer"]
        IH["InferenceHub / RemoteClient<br/>analyze_volume()"]
        DE["DecisionEngine<br/>evaluate_volume()"]
        SIM["What-If Simulator<br/>(simulator.py × simulate_actions())"]
        CPL["CapacityPlanner<br/>(capacity_planner.py × generate_recs())"]
        CB["Circuit Breaker<br/>Rollback rate > 1%<br/>→ DISABLE engine"]
    end

    subgraph Act["ACT Layer"]
        REB["Rebalancer<br/>execute_migration()<br/>execute_qos_shaping()<br/>execute_tier_change()"]
        ACT["Actuator Layer<br/>StubActuator (dev)<br/>CSIActuator (K8s prod)<br/>ArrayAPIActuator (REST)"]
    end

    subgraph Monitor["MONITOR Layer"]
        AM["ActionMonitor<br/>register_action()<br/>update_metrics()"]
        WD["Watchdog Thread<br/>Stall detection (300s)<br/>Phase tracking"]
        RB["Rollback Engine<br/>Latency increase > 20%<br/>→ revert topology"]
    end

    TEL --> PARSE --> IH
    TOPO --> IH
    IH -->|hotspot_score ≥ threshold| DE
    DE -->|evaluate actions| SIM
    SIM -->|simulation ROI| DE
    CPL -->|autoscaling recommendations| DE
    DE -->|rate limits OK| REB
    DE -.->|rate limit hit| Q["Action Queue"]
    Q -.->|process_queued_actions| REB
    REB --> ACT
    ACT --> AM
    AM --> WD
    AM -->|latency breach| RB
    RB -->|rollback_action| REB
    AM -->|rollback_rate| CB
    CB -.->|trip| DE

    style Observe fill:#16213e,stroke:#00f0ff,color:#ffffff
    style Decide fill:#1a1a2e,stroke:#7b2cbf,color:#ffffff
    style Act fill:#16213e,stroke:#ff9100,color:#ffffff
    style Monitor fill:#1a1a2e,stroke:#ff1744,color:#ffffff
```

---

## 4. Sequence Diagrams — Telemetry Event to Action

To maintain readability and prevent the diagrams from shrinking, this complex sequence has been broken down into three distinct phases.

### Phase 1: Ingestion & Preprocessing
```mermaid
sequenceDiagram
    participant TG as TelGen
    participant RS as Redis
    participant SW as Worker
    participant TP as Parser
    participant RD as State

    TG->>RS: XADD telemetry:stream
    
    rect rgb(22, 33, 62)
        Note over SW: Consumer Group Loop
        SW->>RS: XREADGROUP
        RS-->>SW: [{msg_id, fields}]
        
        SW->>RD: Sync policy, topology, state
        RD-->>SW: Latest policy
    end

    rect rgb(26, 26, 46)
        Note over SW,TP: Message Processing
        SW->>TP: parse_and_clip()
        TP-->>SW: Clipped dict
        SW->>SW: live_features_df.concat(event)
        SW->>SW: topology.update_metrics()
    end
```

### Phase 2: ML Inference & Anomaly Detection
```mermaid
sequenceDiagram
    participant SW as Worker
    participant IH as InferenceHub
    participant LGB as LightGBM
    participant ARF as ARF+ADWIN
    participant ED as Ensemble
    participant SD as StatDet
    participant IF as IsoForest
    participant LSTM as LSTM
    participant NN as NoisyNeigh
    participant NB as N-BEATS
    participant TFT as TFT Model

    rect rgb(123, 44, 191)
        Note over SW,TFT: Inference (1 per 30s)
        SW->>IH: analyze_volume()
        
        IH->>LGB: predict_proba()
        IH->>ARF: predict_one()
        
        IH->>ED: detect()
        ED->>SD: detect_hotspot()
        ED->>IF: detect()
        ED->>LSTM: detect()
        ED->>ED: fuse_scores()
        ED-->>IH: hotspot_score
        
        IH->>NN: detect_event()
        IH->>NB: forecast(capacity)
        IH->>TFT: forward(24h)
        IH-->>SW: Full analysis result dict
    end
```

### Phase 3: Control Plane Decision & Execution
```mermaid
sequenceDiagram
    participant SW as Worker
    participant DE as DecisionEngine
    participant REB as Rebalancer
    participant ACT as Actuator
    participant AM as ActionMonitor
    participant RD as State
    participant RS as Redis

    rect rgb(22, 33, 62)
        Note over SW,AM: Decision & Execution
        SW->>DE: evaluate_volume()
        DE->>DE: simulate_actions()
        
        alt Rate limits OK
            DE->>REB: execute_migration()
            REB->>ACT: execute_move()
            ACT->>AM: update_phase(EXECUTING)
            ACT-->>REB: success
            REB->>REB: Update topology
            DE->>AM: register_action()
        else Rate limit exceeded
            DE->>DE: action_queue.append()
        end
    end

    rect rgb(26, 26, 46)
        Note over SW,AM: Monitoring
        SW->>AM: update_metrics()
        alt Latency increase > 20%
            AM->>REB: rollback_action()
        else Elapsed ≥ timeout
            AM-->>SW: success
        end
    end

    SW->>RD: Persist history
    SW->>RS: XACK msg_id
```

---

## 5. Dashboard ↔ API Sequence Diagram

Shows how each dashboard page queries the FastAPI backend.

```mermaid
sequenceDiagram
    participant User as Storage Admin
    participant ST as Streamlit Dashboard
    participant API as FastAPI (api/main.py)
    participant RD as Redis
    participant IH as InferenceHub
    participant TOPO as TopologyGraph

    Note over ST: Page 1 — Cluster Overview (auto-refresh 3s)
    loop st.fragment(run_every=3)
        ST->>API: GET /kpi
        API->>RD: HGETALL volume:*:metrics
        API->>IH: monitor.get_summary()
        API-->>ST: {avg_latency, total_iops, rollback_rate, ...}
        
        ST->>API: GET /volumes
        API->>RD: HGETALL volume:*:metrics + analysis
        API->>TOPO: topology data
        API-->>ST: [{volume_id, tier, iops, latency, hotspot_score, ...}]
        
        ST->>API: GET /alerts
        API-->>ST: [{severity, volume_id, hotspot_score, ...}]
        
        ST->>API: GET /capacity/plan
        API->>IH: capacity_planner.generate_recommendations()
        API-->>ST: {recommendations: [...]}
        
        ST->>API: GET /policy
        API->>RD: GET control_plane:policy
        API-->>ST: {rebalance_policy, safety_guardrails, ...}

        ST->>ST: Render KPI cards + Volume table + SVG Sparklines
    end

    Note over ST: Page 2 — Hotspot Analytics
    User->>ST: Select volume (e.g. vol_000)
    ST->>API: GET /volumes/vol_000/metrics?limit=60
    API-->>ST: [60 metric rows]
    ST->>API: GET /volumes/vol_000/workload
    API->>IH: analyze_volume("vol_000")
    API-->>ST: {workload_type, confidence[], arf_agrees, ...}
    ST->>API: GET /volumes/vol_000/explain
    API->>IH: SHAP feature attributions
    API-->>ST: {feature_contributions: [...]}
    ST->>API: GET /topology
    API->>TOPO: Export nodes + edges
    API-->>ST: {nodes: [...], edges: [...]}
    ST->>API: GET /noisy-neighbors
    API-->>ST: [{aggressor_id, victims: [...]}]

    Note over ST: Page 3 — Forecasting
    ST->>API: GET /volumes/vol_000/forecast
    API->>IH: N-BEATS + TFT inference
    API-->>ST: {capacity_forecast, latency_quantiles, dtf, ttv}

    Note over ST: Page 4 — Control Plane
    User->>ST: Update Policy (PUT)
    ST->>API: PUT /policy {rebalance_policy, safety_guardrails}
    API->>RD: SET control_plane:policy (JSON)
    API->>IH: engine.min_hotspot_score = new_value
    API-->>ST: Updated policy

    User->>ST: Trigger Manual Rebalance
    ST->>API: POST /rebalance/trigger {volume_id}
    API->>IH: engine.evaluate_volume(vol_id, now)
    API-->>ST: {action_id, status, action_state}

    ST->>API: GET /rebalance/monitors
    API->>IH: monitor.actions
    API-->>ST: {action_id: {status, elapsed, latency}}

    ST->>API: GET /rebalance/history
    API->>IH: engine.action_history
    API-->>ST: [{action_id, volume_id, action, status, timestamp}]
```

---

## 6. Data Flow Block Diagram — Ingestion Pipeline

Shows the complete path from raw Parquet data through feature engineering to model training.

```mermaid
graph LR
    subgraph DataGen["Data Generation"]
        DG["data_generator.py<br/>5 workload archetypes<br/>50 volumes × 5min intervals<br/>30 days synthetic telemetry"]
    end

    subgraph FeatureEng["Feature Engineering"]
        FE["feature_engineer.py<br/>Rolling stats: 5m, 30m, 1h<br/>read_write_ratio<br/>io_size_entropy<br/>iops_per_queue<br/>sequential_ratio"]
    end

    subgraph Storage["Data Storage"]
        PQ["io_features.parquet<br/>Processed features<br/>Chronological split:<br/>21d train / 9d test"]
        TS["TimescaleDB<br/>(Optional)<br/>Hypertable partitions"]
    end

    subgraph Training["Model Training"]
        T1["lightgbm_tuned.py<br/>Optuna HPO<br/>→ lightgbm_tuned_model.pkl"]
        T2["arf_adwin.py<br/>River streaming<br/>→ arf_model.pkl"]
        T3["ensemble_detector.py<br/>fit_isolation_forest()<br/>fit_lstm()<br/>→ ensemble/models/"]
        T4["nbeats_model.py<br/>PyTorch training<br/>→ nbeats_model.pth"]
        T5["tft_forecaster.py<br/>PyTorch training<br/>→ tft_model.pth"]
        T6["demand_forecaster.py<br/>Quantile regression<br/>→ demand_forecaster.pkl"]
    end

    subgraph Bounds["Outlier Bounds"]
        BND["telemetry_parser.py<br/>Q1, Q3, IQR from train<br/>→ bounds.json"]
    end

    DG -->|Raw CSV| FE
    FE -->|Engineered features| PQ
    FE -.->|Optional| TS
    PQ --> T1
    PQ --> T2
    PQ --> T3
    PQ --> T4
    PQ --> T5
    PQ --> T6
    PQ --> BND

    style DataGen fill:#16213e,stroke:#00f0ff,color:#ffffff
    style FeatureEng fill:#1a1a2e,stroke:#ff9100,color:#ffffff
    style Storage fill:#16213e,stroke:#00e676,color:#ffffff
    style Training fill:#1a1a2e,stroke:#7b2cbf,color:#ffffff
    style Bounds fill:#16213e,stroke:#ff1744,color:#ffffff
```

---

## 7. Module Dependency Map

Every Python module and its direct imports within the project.

```mermaid
graph LR
    subgraph Infrastructure["src/infrastructure/"]
        INT["interfaces.py<br/>(EventBus ABC)"]
        RB["redis_bus.py"]
        KB["kafka_bus.py"]
        BF["bus_factory.py"]
        TC["timescale_client.py"]
        SEC["security.py<br/>JWT Auth"]
        TR["tracing.py<br/>OpenTelemetry"]
    end

    subgraph Pipeline["src/pipeline/"]
        TLP["telemetry_parser.py<br/>C++/Python hybrid"]
        DL["data_loader.py"]
        PP["preprocessor.py"]
        TG["topology_graph.py<br/>NetworkX graph"]
        SWK["stream_worker.py<br/>Main ingestion loop"]
    end

    subgraph Models["src/models/"]
        LGT["classifier/lightgbm_tuned.py"]
        ARFA["classifier/arf_adwin.py"]
        SDD["anomaly/statistical_detector.py"]
        IFD["anomaly/isolation_forest.py"]
        LAE["anomaly/lstm_autoencoder.py"]
        END["anomaly/ensemble_detector.py"]
        NND["anomaly/noisy_neighbor.py"]
        NBM["forecasting/nbeats_model.py"]
        TFM["forecasting/tft_model.py"]
        TFF["forecasting/tft_forecaster.py"]
        DMF["forecasting/demand_forecaster.py"]
    end

    subgraph ControlPlane["src/control_plane/"]
        IHB["inference_hub.py"]
        RIC["remote_inference_client.py"]
        DEN["decision_engine.py"]
        RBL["rebalancer.py"]
        MON["monitor.py"]
        ACT2["actuators.py"]
        SIM["simulator.py"]
        CPL["capacity_planner.py"]
    end

    subgraph APILayer["api/"]
        APM["main.py<br/>(FastAPI app)"]
    end

    subgraph DashLayer["dashboard/"]
        HM["Home.py"]
        UT["utils.py"]
        CO["pages/1_Cluster_Overview.py"]
        HA["pages/2_Hotspot_Analytics.py"]
        FC["pages/3_Forecasting.py"]
        CP["pages/4_Control_Plane.py"]
    end

    %% Infrastructure deps
    RB --> INT
    KB --> INT
    BF --> RB
    BF --> KB

    %% Pipeline deps
    SWK --> TLP
    SWK --> IHB
    SWK --> RBL
    SWK --> MON
    SWK --> DEN
    SWK --> TR

    %% Control plane deps
    IHB --> END
    IHB --> TG
    IHB --> NND
    IHB --> NBM
    IHB --> TFM
    IHB --> TFF
    IHB --> DMF
    RIC --> TG
    RIC --> TC
    DEN --> IHB
    DEN --> RBL
    DEN --> MON
    RBL --> TG
    RBL --> ACT2
    MON --> RBL
    MON --> TG

    %% Ensemble deps
    END --> SDD
    END --> IFD
    END --> LAE

    %% API deps
    APM --> IHB
    APM --> RIC
    APM --> DEN
    APM --> MON
    APM --> RBL
    APM --> ACT2
    APM --> SIM
    APM --> CPL
    APM --> SEC
    APM --> BF
    APM --> TR

    %% Dashboard deps
    CO --> UT
    HA --> UT
    FC --> UT
    CP --> UT
    UT -.->|HTTP| APM

    style Infrastructure fill:#1a1a2e,stroke:#e94560,color:#ffffff
    style Pipeline fill:#16213e,stroke:#00f0ff,color:#ffffff
    style Models fill:#1a1a2e,stroke:#ff9100,color:#ffffff
    style ControlPlane fill:#16213e,stroke:#7b2cbf,color:#ffffff
    style APILayer fill:#1a1a2e,stroke:#00e676,color:#ffffff
    style DashLayer fill:#16213e,stroke:#00e676,color:#ffffff
```

---

## 8. File-to-Module Summary Table

| Layer | File | Lines | Role |
|-------|------|------:|------|
| **Infrastructure** | [interfaces.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/infrastructure/interfaces.py) | 38 | `EventBus` abstract base class |
| | [redis_bus.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/infrastructure/redis_bus.py) | 87 | Redis Streams publish/subscribe + DLQ |
| | [kafka_bus.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/infrastructure/kafka_bus.py) | 108 | Apache Kafka consumer/producer event bus |
| | [bus_factory.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/infrastructure/bus_factory.py) | 17 | Factory to instantiate Redis or Kafka event bus |
| | [timescale_client.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/infrastructure/timescale_client.py) | 81 | TimescaleDB hypertable query and ingestion wrapper |
| | [security.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/infrastructure/security.py) | 43 | JWT token generation/validation |
| | [tracing.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/infrastructure/tracing.py) | 54 | OpenTelemetry span propagation |
| **Pipeline** | [telemetry_parser.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/pipeline/telemetry_parser.py) | 350 | C++/Python hybrid JSON parser + IQR outlier clipping |
| | [topology_graph.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/pipeline/topology_graph.py) | 476 | NetworkX graph: nodes ↔ volumes, tier mgmt, placement |
| | [preprocessor.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/pipeline/preprocessor.py) | 248 | Telemetry scaling, rolling feature computation, robust bounds |
| | [data_loader.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/pipeline/data_loader.py) | 94 | Dataset split and batch loading for forecasters |
| | [stream_worker.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/pipeline/stream_worker.py) | 617 | Main ingestion loop: consume → parse → infer → decide → persist |
| **ML Models** | [lightgbm_tuned.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/classifier/lightgbm_tuned.py) | 386 | Optuna-tuned LightGBM 5-class workload classifier |
| | [arf_adwin.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/classifier/arf_adwin.py) | 319 | River ARF + ADWIN drift detection streaming classifier |
| | [statistical_detector.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/anomaly/statistical_detector.py) | 416 | Per-volume 24h rolling z-score anomaly detector |
| | [isolation_forest.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/anomaly/isolation_forest.py) | 590 | Sklearn Isolation Forest with 10-feature vectors |
| | [lstm_autoencoder.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/anomaly/lstm_autoencoder.py) | 924 | PyTorch LSTM Autoencoder (12-step × 10-feature) |
| | [ensemble_detector.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/anomaly/ensemble_detector.py) | 1120 | Weighted fusion + consensus gate + meta-learner |
| | [noisy_neighbor.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/anomaly/noisy_neighbor.py) | 498 | Aggressor-victim detection via co-located node scan |
| | [nbeats_model.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/forecasting/nbeats_model.py) | 392 | Neural Basis Expansion for capacity forecasting |
| | [tft_model.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/forecasting/tft_model.py) | 203 | Temporal Fusion Transformer architecture |
| | [tft_forecaster.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/forecasting/tft_forecaster.py) | 526 | TFT data prep + hourly aggregation + scaler |
| | [demand_forecaster.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/forecasting/demand_forecaster.py) | 263 | IOPS and throughput quantile regression model |
| | [dtf_forecaster.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/models/forecasting/dtf_forecaster.py) | 603 | Days-to-fill capacity forecasting training/prediction |
| **Control Plane** | [inference_hub.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/inference_hub.py) | 577 | Central ML coordinator: loads all models, runs analyze_volume() |
| | [remote_inference_client.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/remote_inference_client.py) | 152 | Client proxy for disaggregated inference serving (GPU clusters) |
| | [decision_engine.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/decision_engine.py) | 529 | Policy evaluation, action simulation, rate limiting, circuit breaker |
| | [rebalancer.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/rebalancer.py) | 227 | Execute/rollback migrations, QoS, tier changes on topology |
| | [monitor.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/monitor.py) | 268 | Post-action latency tracking, rollback triggering, watchdog thread |
| | [actuators.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/actuators.py) | 277 | Stub/CSI/ArrayAPI actuator abstraction for physical moves |
| | [simulator.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/simulator.py) | 217 | What-If Simulator for policy migration/shaping evaluations |
| | [capacity_planner.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/src/control_plane/capacity_planner.py) | 233 | Generates capacity plan, pool headroom, and autoscale recs |
| **API** | [main.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/api/main.py) | 2380 | FastAPI app: 37 active endpoints, JWT auth, CORS, Redis sync |
| **Dashboard** | [Home.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/dashboard/Home.py) | 138 | Streamlit landing page and credentials configuration |
| | [utils.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/dashboard/utils.py) | 393 | Dashboard HTTP API client and SVG sparkline helpers |
| | [1_Cluster_Overview.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/dashboard/pages/1_Cluster_Overview.py) | 336 | Live KPIs, SVG sparklines, volume status table |
| | [2_Hotspot_Analytics.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/dashboard/pages/2_Hotspot_Analytics.py) | 418 | Topology map, SHAP, diagnostics, noisy neighbors, ML perf |
| | [3_Forecasting.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/dashboard/pages/3_Forecasting.py) | 136 | Capacity/latency forecaster charts and days-to-fill UI |
| | [4_Control_Plane.py](file:///home/akash-t-s-m/projects-may/akash.t.s.m-projects/IO-Workload-Pattern-Classification-Hotspot-Detection/dashboard/pages/4_Control_Plane.py) | 329 | Policy config, manual overrides, active monitors, history |