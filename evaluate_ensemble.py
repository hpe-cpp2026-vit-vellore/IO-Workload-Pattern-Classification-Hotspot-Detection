"""Evaluate ensemble detector accuracy."""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load data
df = pd.read_parquet("data/processed/io_features.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["volume_id", "timestamp"]).reset_index(drop=True)

split_idx = int(len(df) * 0.7)
test_df = df.iloc[split_idx:].reset_index(drop=True)

# Ground truth
latency_threshold = test_df["avg_latency_us"].mean() + 3 * test_df["avg_latency_us"].std()
iops_high = test_df["total_iops"].mean() + 3 * test_df["total_iops"].std()
iops_low = test_df["total_iops"].mean() - 3 * test_df["total_iops"].std()
throughput_threshold = test_df["total_throughput_mbps"].mean() + 3 * test_df["total_throughput_mbps"].std()
p99_threshold = test_df["read_latency_p99_us"].mean() + 3 * test_df["read_latency_p99_us"].std()

ground_truth = (
    (test_df["avg_latency_us"] > latency_threshold) |
    (test_df["total_iops"] > iops_high) |
    (test_df["total_iops"] < iops_low) |
    (test_df["total_throughput_mbps"] > throughput_threshold) |
    (test_df["read_latency_p99_us"] > p99_threshold)
).values

# Load ensemble predictions
ensemble_df = pd.read_csv("models/anomaly/ensemble_scores.csv")
y_pred = ensemble_df["is_anomaly"].values

# Metrics
acc = accuracy_score(ground_truth, y_pred)
prec = precision_score(ground_truth, y_pred, zero_division=0)
rec = recall_score(ground_truth, y_pred, zero_division=0)
f1 = f1_score(ground_truth, y_pred, zero_division=0)
cm = confusion_matrix(ground_truth, y_pred)

tn, fp, fn, tp = cm.ravel()

print("=" * 70)
print(" ENSEMBLE DETECTOR ACCURACY")
print("=" * 70)
print(f"\nAccuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"  TN: {tn:,}  FP: {fp:,}")
print(f"  FN: {fn:,}  TP: {tp:,}")
print("=" * 70)
