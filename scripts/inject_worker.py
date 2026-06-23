import re

with open("src/pipeline/stream_worker.py", "r") as f:
    code = f.read()

# We need to add a periodic check in the main loop to update monitors.
# In `run_worker()`, there is a variable `last_trim_time = time.time()`.
# We can add `last_monitor_update_time = time.time()`
init_vars = "    last_trim_time = time.time()\n    last_monitor_update_time = time.time()\n"
code = re.sub(r'\s+last_trim_time = time\.time\(\)', "\n" + init_vars, code, count=1)

# At the end of the `while True:` loop inside `run_worker()`, we can insert the monitor check.
# Right around where `last_trim_time` is checked.
trim_logic = r'            if len\(hub\.live_features_df\) > 15000 and \(current_time - last_trim_time > 60\):'
monitor_logic = """
            if current_time - last_monitor_update_time > 60:
                last_monitor_update_time = current_time
                monitor_changed = False
                for action_id, act in list(monitor.actions.items()):
                    if act.get("status") == "monitoring":
                        volume_id = act.get("action_state", {}).get("volume_id")
                        if not volume_id:
                            continue
                        
                        metrics_dict = hub.topology._volume_metrics.get(volume_id, {})
                        current_latency = metrics_dict.get("avg_latency_us", 1000.0)
                        
                        # Calculate elapsed
                        start_ts = pd.to_datetime(act["timestamp"])
                        elapsed = (pd.Timestamp.now() - start_ts).total_seconds() / 60.0
                        
                        new_status = monitor.update_metrics(
                            action_id,
                            current_latency,
                            elapsed,
                            rebalancer,
                            hub.topology
                        )
                        if new_status != "monitoring":
                            monitor_changed = True
                            
                            # Also update the history array so that `engine.action_history` has the right status
                            for hist_act in engine.action_history:
                                if hist_act.get("action_id") == action_id:
                                    hist_act["status"] = new_status
                                    break
                                    
                if monitor_changed:
                    _persist_control_plane_state(r, engine, monitor)
                    
            if len(hub.live_features_df) > 15000 and (current_time - last_trim_time > 60):
"""

code = code.replace("            if len(hub.live_features_df) > 15000 and (current_time - last_trim_time > 60):", monitor_logic)

with open("src/pipeline/stream_worker_refactored.py", "w") as f:
    f.write(code)
print("done")
