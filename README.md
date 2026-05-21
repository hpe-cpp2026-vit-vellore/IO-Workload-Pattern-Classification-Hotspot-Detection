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
