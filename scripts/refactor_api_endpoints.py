import re

with open("api/main.py", "r") as f:
    code = f.read()

# 1. Refactor GET /rebalance/history
history_replacement = """
    global use_redis, r
    if use_redis and r is not None:
        try:
            history = r.get_latest_state("control_plane:action_history") or []
            return history
        except Exception:
            return []
    return []
"""
code = re.sub(r'    global engine, use_redis, r\n    if engine is None:\n        return \[\]\n.*?(?=@app\.get\("/rebalance/monitors)', history_replacement + "\n\n", code, flags=re.DOTALL)

# 2. Refactor GET /rebalance/monitors
monitors_replacement = """
    global use_redis, r
    if use_redis and r is not None:
        import json
        try:
            raw_monitors = r.hgetall("control_plane:active_monitors") or {}
            monitors = {}
            for aid, raw_mon in raw_monitors.items():
                mon = json.loads(raw_mon)
                monitors[aid] = mon
            return monitors
        except Exception:
            return {}
    return {}
"""
code = re.sub(r'    global monitor, use_redis, r\n    if monitor is None:\n        return \{\}\n.*?(?=@app\.post\("/rebalance",)', monitors_replacement + "\n\n", code, flags=re.DOTALL)

# 3. Refactor POST /rebalance/execute
# In the original code, `monitor.register_action` and `engine.action_history.append` are called.
# We will replace them with Redis appends.
execute_replacement = """
    import uuid
    import pandas as pd
    import json
    
    action_id = str(uuid.uuid4())
    now_str = pd.Timestamp.now().isoformat()
    
    act_mon = {
        "action_state": action_state,
        "pre_latency": pre_latency,
        "current_latency": pre_latency,
        "status": "monitoring",
        "timestamp": now_str,
        "elapsed_minutes": 0.0
    }
    
    exec_record = {
        "action_id": action_id,
        "volume_id": req.volume_id,
        "action": req.action_type,
        "choice": {
            "action": req.action_type,
            "target": req.target,
            "expected_improvement": 0.0,
            "safe": True
        },
        "timestamp": now_str,
        "action_state": action_state,
        "status": "executed"
    }

    if use_redis and r is not None:
        try:
            r.hset("control_plane:active_monitors", action_id, json.dumps(act_mon))
            history = r.get_latest_state("control_plane:action_history") or []
            history.append(exec_record)
            r.set_state("control_plane:action_history", history)
            
            queue = r.get_latest_state("control_plane:action_queue") or []
            queue.append(exec_record)
            r.set_state("control_plane:action_queue", queue)
        except Exception as e:
            logger.error(f"Failed to save manual action to Redis: {e}")

    logger.info("Manual Action %s executed and queued. Type: %s", action_id, req.action_type)
    return {"status": "executed", "action_id": action_id, "state": action_state}
"""

code = re.sub(r'    action_id = str\(uuid\.uuid4\(\)\)\n\s+monitor\.register_action\(.*?(?=@app\.post\("/rebalance/autoscale")', execute_replacement + "\n\n", code, flags=re.DOTALL)


with open("api/main_refactored_2.py", "w") as f:
    f.write(code)
print("done")
