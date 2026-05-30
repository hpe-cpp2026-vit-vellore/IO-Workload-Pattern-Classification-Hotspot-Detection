# IO-Workload-Pattern-Classification-Hotspot-Detection
ML-based IO Workload Pattern Classification &amp; Hotspot Detection system

## Local Dashboard Runbook

Use three terminals from the project root:

```powershell
venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```powershell
venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

```powershell
venv\Scripts\python.exe scripts\telemetry_playback.py
```

Redis is optional for local demos. If no Redis server is running on `127.0.0.1:6379`, the API starts a TCP fallback listener on `127.0.0.1:9000`, and `scripts\telemetry_playback.py` sends telemetry there directly. To use Redis Streams instead, start a real Redis server first, then also run:

```powershell
venv\Scripts\python.exe -m src.pipeline.stream_worker
```

## Docker Notes

The Docker setup bind-mounts the `models/` directory at runtime and does not bake model artifacts into the image. On a fresh clone, `docker compose up` will fail until the trained artifacts are present (for example, `models/anomaly/ensemble/lstm_ae_model.pth` and related ensemble stats/config). Run the training pipeline first or place the prebuilt artifacts into `models/` before starting the containers.
