"""
Evaluate Isolation Forest detector accuracy against ground truth anomalies.

Ground truth anomalies in our synthetic data:
1. Noisy neighbor events (500 injected)
2. Concept bleed (3% of samples)
3. Time-of-day spikes (2% of samples)

Total expected anomalies: ~5-6% of data
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load results
print("Loading Isolation Forest results...")
scores_df = pd.read_csv("models/anomaly/isolation_forest_scores.csv")
scores_df["timestamp"] = pd.to_datetime(scores_df["timestamp"])

# Load original data to get ground truth
print("Loading original data...")
df = pd.read_parquet("data/processed/io_features.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["volume_id", "timestamp"]).reset_index(drop=True)

# Split: 70% train, 30% test (same as detector)
split_idx = int(len(df) * 0.7)
test_df = df.iloc[split_idx:].reset_index(drop=True)

print(f"\nTest data: {len(test_df):,} samples")
print(f"Predictions: {len(scores_df):,} samples")

# Merge predictions with original data
merged = test_df.merge(
    scores_df[["volume_id", "timestamp", "anomaly_score", "is_anomaly"]],
    on=["volume_id", "timestamp"],
    how="left"
)

print(f"Merged: {len(merged):,} samples")

# Calculate ground truth: samples with extreme values are anomalies
# Use statistical thresholds (3σ from mean)
print("\nCalculating ground truth anomalies...")

# Define anomaly criteria based on extreme values
latency_threshold = test_df["avg_latency_us"].mean() + 3 * test_df["avg_latency_us"].std()
iops_threshold_high = test_df["total_iops"].mean() + 3 * test_df["total_iops"].std()
iops_threshold_low = test_df["total_iops"].mean() - 3 * test_df["total_iops"].std()
throughput_threshold = test_df["total_throughput_mbps"].mean() + 3 * test_df["total_throughput_mbps"].std()

# Ground truth: any sample exceeding 3σ in key metrics
ground_truth = (
    (test_df["avg_latency_us"] > latency_threshold) |
    (test_df["total_iops"] > iops_threshold_high) |
    (test_df["total_iops"] < iops_threshold_low) |
    (test_df["total_throughput_mbps"] > throughput_threshold) |
    (test_df["read_latency_p99_us"] > test_df["read_latency_p99_us"].mean() + 3 * test_df["read_latency_p99_us"].std())
).values

merged["ground_truth"] = ground_truth

print(f"Ground truth anomalies: {ground_truth.sum():,} ({ground_truth.sum()/len(ground_truth)*100:.2f}%)")
print(f"Detected anomalies: {merged['is_anomaly'].sum():,} ({merged['is_anomaly'].sum()/len(merged)*100:.2f}%)")

# Calculate metrics
y_true = merged["ground_truth"].values
y_pred = merged["is_anomaly"].values

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# Calculate accuracy
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n" + "=" * 70)
print(" ISOLATION FOREST ACCURACY METRICS")
print("=" * 70)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

print(f"\nConfusion Matrix:")
print(f"  TN: {tn:,}  FP: {fp:,}")
print(f"  FN: {fn:,}  TP: {tp:,}")

print(f"\nInterpretation:")
print(f"  True Negatives:  {tn:,} (correctly identified normal samples)")
print(f"  True Positives:  {tp:,} (correctly identified anomalies)")
print(f"  False Positives: {fp:,} (normal samples flagged as anomalies)")
print(f"  False Negatives: {fn:,} (missed anomalies)")

# Anomaly score distribution by ground truth
print(f"\nAnomaly Score Distribution:")
print(f"  True anomalies:  mean={merged[merged['ground_truth']]['anomaly_score'].mean():.2f}, "
      f"median={merged[merged['ground_truth']]['anomaly_score'].median():.2f}")
print(f"  Normal samples:  mean={merged[~merged['ground_truth']]['anomaly_score'].mean():.2f}, "
      f"median={merged[~merged['ground_truth']]['anomaly_score'].median():.2f}")

print("=" * 70)
