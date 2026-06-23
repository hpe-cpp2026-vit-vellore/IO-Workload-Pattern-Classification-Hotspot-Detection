import re

with open("api/main.py", "r") as f:
    code = f.read()

# 1. Remove background tasks (@repeat_every blocks)
code = re.sub(r'@app\.on_event\("startup"\)\s*@repeat_every\(seconds=1\)\s*def _simulate_system_tick\(\).*?(?=@app\.on_event)', '', code, flags=re.DOTALL)
code = re.sub(r'@app\.on_event\("startup"\)\s*@repeat_every\(seconds=10\)\s*def _persist_control_plane_state\(\).*?(?=@app\.on_event)', '', code, flags=re.DOTALL)
code = re.sub(r'@app\.on_event\("startup"\)\s*@repeat_every\(seconds=60\)\s*def _update_monitors\(\).*?(?=@app\.on_event)', '', code, flags=re.DOTALL)

# 2. Make _analyze_and_cache_volume empty or remove it
code = re.sub(r'async def _analyze_and_cache_volume.*?def _schedule_volume_analysis', 'def _schedule_volume_analysis', code, flags=re.DOTALL)

# 3. Remove _schedule_volume_analysis implementation
schedule_pattern = r'def _schedule_volume_analysis.*?try:.*?except RuntimeError:.*?pass'
code = re.sub(schedule_pattern, 'def _schedule_volume_analysis(volume_id: str, ts):\n    pass', code, flags=re.DOTALL)

# 4. Remove engine/monitor usage from GET /kpi
kpi_replacement = """
    mon_summary = {
        "total_actions": 0,
        "rolled_back_count": 0,
        "rollback_rate_pct": 0.0,
    }
    if use_redis and r is not None:
        try:
            history = r.get_latest_state("control_plane:action_history") or []
            total_actions = len(history)
            
            import json
            raw_monitors = r.hgetall("control_plane:active_monitors") or {}
            rolled_back = 0
            for raw_mon in raw_monitors.values():
                mon = json.loads(raw_mon)
                if mon.get("status") == "rolled_back":
                    rolled_back += 1
            
            rate = (rolled_back / total_actions * 100.0) if total_actions > 0 else 0.0
            mon_summary = {
                "total_actions": total_actions,
                "rolled_back_count": rolled_back,
                "rollback_rate_pct": rate
            }
        except Exception:
            pass

    return {
"""
code = re.sub(r'mon_summary = \{\s*"total_actions": 0,.*?return \{', kpi_replacement, code, flags=re.DOTALL)

with open("api/main_refactored.py", "w") as f:
    f.write(code)
print("done")
