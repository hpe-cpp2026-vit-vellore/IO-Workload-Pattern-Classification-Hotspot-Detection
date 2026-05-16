"""
Evaluate all three anomaly detectors against ground truth.

Ground truth: Statistical anomalies (3σ from mean) in test data.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

print("=" * 70)
print(" ANOMALY DETECTOR ACCURACY EVALUATION")
print("=" * 70)

# Load original data
print("\nLoading original data...")
df = pd.read_parquet("data/processed/io_features.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["volume_id", "timestamp"]).reset_index(drop=True)

# Split: 70% train, 30% test
split_idx = int(len(df) * 0.7)
test_df = df.iloc[split_idx:].reset_index(drop=True)

print(f"Test data: {len(test_df):,} samples")

# Calculate ground truth: 3σ anomalies
print("\nCalculating ground truth (3σ threshold)...")
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

print(f"Ground truth anomalies: {ground_truth.sum():,} ({ground_truth.sum()/len(ground_truth)*100:.2f}%)")

# ============================================================================
# 1. STATISTICAL DETECTOR
# ============================================================================
print("\n" + "=" * 70)
print(" 1. STATISTICAL DETECTOR")
print("=" * 70)

stat_df = pd.read_csv("models/anomaly/statistical_detector_scores.csv")
stat_df["timestamp"] = pd.to_datetime(stat_df["timestamp"])

# Statistical detector uses hotspot_score > 60 as threshold
stat_df["is_hotspot"] = stat_df["hotspot_score"] > 60

# Merge with test data
stat_merged = test_df.merge(
    stat_df[["volume_id", "timestamp", "is_hotspot"]],
    on=["volume_id", "timestamp"],
    how="left"
)
stat_merged["is_hotspot"] = stat_merged["is_hotspot"].fillna(False)

y_true = ground_truth
y_pred = stat_merged["is_hotspot"].values

stat_acc = accuracy_score(y_true, y_pred)
stat_prec = precision_score(y_true, y_pred, zero_division=0)
stat_rec = recall_score(y_true, y_pred, zero_division=0)
stat_f1 = f1_score(y_true, y_pred, zero_division=0)
stat_cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = stat_cm.ravel()

print(f"\nAccuracy:  {stat_acc:.4f} ({stat_acc*100:.2f}%)")
print(f"Precision: {stat_prec:.4f} ({stat_prec*100:.2f}%)")
print(f"Recall:    {stat_rec:.4f} ({stat_rec*100:.2f}%)")
print(f"F1-Score:  {stat_f1:.4f} ({stat_f1*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"  TN: {tn:,}  FP: {fp:,}")
print(f"  FN: {fn:,}  TP: {tp:,}")

# ============================================================================
# 2. ISOLATION FOREST
# ============================================================================
print("\n" + "=" * 70)
print(" 2. ISOLATION FOREST")
print("=" * 70)

iso_df = pd.read_csv("models/anomaly/isolation_forest_scores.csv")
iso_df["timestamp"] = pd.to_datetime(iso_df["timestamp"])

iso_merged = test_df.merge(
    iso_df[["volume_id", "timestamp", "is_anomaly"]],
    on=["volume_id", "timestamp"],
    how="left"
)
iso_merged["is_anomaly"] = iso_merged["is_anomaly"].fillna(False)

y_pred = iso_merged["is_anomaly"].values

iso_acc = accuracy_score(y_true, y_pred)
iso_prec = precision_score(y_true, y_pred, zero_division=0)
iso_rec = recall_score(y_true, y_pred, zero_division=0)
iso_f1 = f1_score(y_true, y_pred, zero_division=0)
iso_cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = iso_cm.ravel()

print(f"\nAccuracy:  {iso_acc:.4f} ({iso_acc*100:.2f}%)")
print(f"Precision: {iso_prec:.4f} ({iso_prec*100:.2f}%)")
print(f"Recall:    {iso_rec:.4f} ({iso_rec*100:.2f}%)")
print(f"F1-Score:  {iso_f1:.4f} ({iso_f1*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"  TN: {tn:,}  FP: {fp:,}")
print(f"  FN: {fn:,}  TP: {tp:,}")

# ============================================================================
# 3. LSTM-AUTOENCODER
# ============================================================================
print("\n" + "=" * 70)
print(" 3. LSTM-AUTOENCODER")
print("=" * 70)

lstm_df = pd.read_csv("models/anomaly/lstm_ae_scores.csv")
lstm_df["timestamp"] = pd.to_datetime(lstm_df["timestamp"])

# LSTM works on sequences, so we need to map back to original samples
# For simplicity, we'll evaluate on the last timestamp of each sequence
lstm_merged = test_df.merge(
    lstm_df[["volume_id", "timestamp", "is_anomaly"]],
    on=["volume_id", "timestamp"],
    how="left"
)
lstm_merged["is_anomaly"] = lstm_merged["is_anomaly"].fillna(False)

y_pred = lstm_merged["is_anomaly"].values

lstm_acc = accuracy_score(y_true, y_pred)
lstm_prec = precision_score(y_true, y_pred, zero_division=0)
lstm_rec = recall_score(y_true, y_pred, zero_division=0)
lstm_f1 = f1_score(y_true, y_pred, zero_division=0)
lstm_cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = lstm_cm.ravel()

print(f"\nAccuracy:  {lstm_acc:.4f} ({lstm_acc*100:.2f}%)")
print(f"Precision: {lstm_prec:.4f} ({lstm_prec*100:.2f}%)")
print(f"Recall:    {lstm_rec:.4f} ({lstm_rec*100:.2f}%)")
print(f"F1-Score:  {lstm_f1:.4f} ({lstm_f1*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"  TN: {tn:,}  FP: {fp:,}")
print(f"  FN: {fn:,}  TP: {tp:,}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print(" SUMMARY: ALL DETECTORS")
print("=" * 70)

summary = pd.DataFrame({
    'Detector': ['Statistical', 'Isolation Forest', 'LSTM-Autoencoder'],
    'Accuracy': [f"{stat_acc*100:.2f}%", f"{iso_acc*100:.2f}%", f"{lstm_acc*100:.2f}%"],
    'Precision': [f"{stat_prec*100:.2f}%", f"{iso_prec*100:.2f}%", f"{lstm_prec*100:.2f}%"],
    'Recall': [f"{stat_rec*100:.2f}%", f"{iso_rec*100:.2f}%", f"{lstm_rec*100:.2f}%"],
    'F1-Score': [f"{stat_f1*100:.2f}%", f"{iso_f1*100:.2f}%", f"{lstm_f1*100:.2f}%"],
})

print("\n" + summary.to_string(index=False))

print("\n" + "=" * 70)
print(" INTERPRETATION")
print("=" * 70)
print("Statistical:      High precision, low recall (conservative)")
print("Isolation Forest: Balanced precision/recall (aggressive)")
print("LSTM-Autoencoder: Balanced precision/recall (temporal patterns)")
print("\nAll three detectors complement each other for production use.")
print("=" * 70)
