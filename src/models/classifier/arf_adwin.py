"""
src/models/classifier/arf_adwin.py

Adaptive Random Forest (ARF) with ADWIN drift detection for streaming IO
workload classification.

Blueprint (Phase 2.2):
  - river.forest.ARFClassifier  (river 0.24.2)
  - drift_detector  : ADWIN(delta=0.001)  — triggers tree replacement
  - warning_detector: ADWIN(delta=0.01)   — spawns background tree early
  - n_models        : 25
  - grace_period    : 100
  - Evaluation      : Prequential (interleaved test-then-train)

Performance note:
  With 5-minute polling intervals, the full dataset is ~432K rows — comfortably
  processable by ARF in under 30 minutes. No subsampling needed.

Produces:
  models/classifier/arf_metrics.json
  models/classifier/arf_prequential_accuracy.csv
"""

from __future__ import annotations

import json
import joblib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from river import drift, metrics
from river.forest import ARFClassifier
from tqdm import tqdm

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.data_loader import load_split

MODEL_DIR = ROOT / "models" / "classifier"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Label map ─────────────────────────────────────────────────────────────────
LABEL_NAMES: dict[int, str] = {
    0: "DB_OLTP",
    1: "VM",
    2: "Backup",
    3: "AI_Training",
    4: "AI_Inference",
}

# ── Hyperparameters (Blueprint §2.2) ─────────────────────────────────────────
ARF_CONFIG = {
    "n_models":      25,     # number of trees in the ensemble
    "grace_period":  100,    # observations between leaf split attempts
    "delta":         0.001,  # ADWIN drift sensitivity
    "warning_delta": 0.01,   # ADWIN warning sensitivity (looser)
    "seed":          42,
}

# Accuracy snapshot every N instances (for the prequential accuracy curve)
SNAPSHOT_INTERVAL = 5_000


def build_model() -> ARFClassifier:
    """
    Construct ARF with ADWIN drift & warning detectors per Blueprint §2.2.

    Note: In river 0.24.2, ARFClassifier lives in river.forest (not
    river.ensemble). The class-level `delta` controls Hoeffding-tree internal
    splits; setting it equal to the ADWIN drift delta ensures consistent
    sensitivity throughout the tree lifecycle.
    """
    return ARFClassifier(
        n_models=ARF_CONFIG["n_models"],
        grace_period=ARF_CONFIG["grace_period"],
        drift_detector=drift.ADWIN(delta=ARF_CONFIG["delta"]),
        warning_detector=drift.ADWIN(delta=ARF_CONFIG["warning_delta"]),
        delta=ARF_CONFIG["delta"],
        seed=ARF_CONFIG["seed"],
    )


def _per_class_metrics(cm: metrics.ConfusionMatrix) -> dict[str, dict]:
    """
    Derive per-class precision, recall, F1 from a river ConfusionMatrix.
    Uses tp/fp/fn counts directly — compatible with river 0.24.2 API.
    """
    per_class: dict[str, dict] = {}
    for label_int, label_name in LABEL_NAMES.items():
        tp = cm.true_positives(label_int)
        fp = cm.false_positives(label_int)
        fn = cm.false_negatives(label_int)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        per_class[label_name] = {
            "precision": round(precision, 6),
            "recall":    round(recall,    6),
            "f1":        round(f1,        6),
        }
    return per_class


def prequential_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model: ARFClassifier,
    split_name: str,
    learn: bool = True,
) -> tuple[dict, list[dict]]:
    """
    Prequential (interleaved test-then-train) evaluation.

    For every instance in chronological order:
      1. Predict  — test BEFORE the model has seen this instance.
      2. Train    — update the model with the true label (if learn=True).

    Parameters
    ----------
    X          : feature DataFrame (already scaled, chronologically ordered)
    y          : integer label Series
    model      : ARFClassifier (mutated in-place during learning)
    split_name : label for logging / output
    learn      : if False, only test (no weight updates) — used for test split

    Returns
    -------
    summary   : dict with final accuracy, per-class F1, drift events.
    snapshots : list of {n_seen, accuracy} for the prequential accuracy curve.
    """
    print(f"\n[Prequential] {split_name} ({len(X):,} instances)...")

    acc_metric   = metrics.Accuracy()
    report_cm    = metrics.ConfusionMatrix()
    snapshots: list[dict] = []
    drift_events = 0
    prev_n_models = len(model.models)

    X_arr = X.to_numpy(dtype=np.float32)
    y_arr = y.to_numpy(dtype=int)
    cols  = list(X.columns)

    for i, (x_row, y_true) in enumerate(
        tqdm(zip(X_arr, y_arr), total=len(X_arr), desc=f"  {split_name}")
    ):
        x_dict = dict(zip(cols, x_row.tolist()))

        # 1. Test (predict before learning)
        y_pred = model.predict_one(x_dict)
        if y_pred is not None:
            acc_metric.update(y_true, y_pred)
            report_cm.update(y_true, y_pred)

        # 2. Train (learn from true label)
        if learn:
            model.learn_one(x_dict, y_true)

        # Track ADWIN-triggered tree replacements
        curr_n_models = len(model.models)
        if curr_n_models != prev_n_models:
            drift_events += 1
            prev_n_models = curr_n_models

        if (i + 1) % SNAPSHOT_INTERVAL == 0:
            snapshots.append({
                "n_seen":   i + 1,
                "accuracy": round(float(acc_metric.get()), 6),
            })

    final_acc = float(acc_metric.get())
    per_class = _per_class_metrics(report_cm)

    summary = {
        "split":          split_name,
        "n_instances":    len(X),
        "final_accuracy": round(final_acc, 6),
        "drift_events":   drift_events,
        "per_class":      per_class,
        "arf_config":     ARF_CONFIG,
    }
    print(f"  ✅ prequential accuracy : {final_acc:.4f}")
    print(f"  ↺  ADWIN drift events  : {drift_events}")
    return summary, snapshots


def main() -> None:
    print("=" * 70)
    print(" ARF + ADWIN Streaming Classifier  (river 0.24.2)")
    print(" Blueprint Phase 2.2  —  Prequential Evaluation")
    print("=" * 70)

    # ── Load splits ───────────────────────────────────────────────────────────
    print("\nLoading preprocessed splits...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    print(f"  train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")

    y_train = y_train.astype(int)
    y_val   = y_val.astype(int)
    y_test  = y_test.astype(int)

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"\nBuilding ARF (n_models={ARF_CONFIG['n_models']}, "
          f"grace_period={ARF_CONFIG['grace_period']}, "
          f"ADWIN delta={ARF_CONFIG['delta']})...")
    model = build_model()

    # ── Prequential on TRAIN (full training set) ──────────────────────────────
    train_summary, train_snapshots = prequential_evaluate(
        X_train, y_train, model, "train", learn=True
    )

    # ── Prequential on VAL (model continues online learning) ──────────────────
    val_summary, val_snapshots = prequential_evaluate(
        X_val, y_val, model, "val", learn=True
    )

    # ── Prequential on TEST (final hold-out — predict-only, no leakage) ───────
    test_summary, test_snapshots = prequential_evaluate(
        X_test, y_test, model, "test", learn=False
    )

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    all_metrics = {
        "train": train_summary,
        "val":   val_summary,
        "test":  test_summary,
    }
    metrics_path = MODEL_DIR / "arf_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved  → {metrics_path.relative_to(ROOT)}")

    # ── Save prequential accuracy curve CSV ───────────────────────────────────
    all_snapshots = []
    for split_name, snaps in [
        ("train", train_snapshots),
        ("val",   val_snapshots),
        ("test",  test_snapshots),
    ]:
        for s in snaps:
            all_snapshots.append({"split": split_name, **s})

    curve_path = MODEL_DIR / "arf_prequential_accuracy.csv"
    pd.DataFrame(all_snapshots).to_csv(curve_path, index=False)
    print(f"Accuracy curve → {curve_path.relative_to(ROOT)}")

    # ── Save trained ARF model ────────────────────────────────────────────────
    model_path = MODEL_DIR / "arf_model.pkl"
    joblib.dump(model, model_path)
    print(f"ARF model serialized → {model_path.relative_to(ROOT)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" RESULTS SUMMARY")
    print("=" * 70)
    for label, summary in [
        ("Train (full)",  train_summary),
        ("Val   (full)",  val_summary),
        ("Test  (full)",  test_summary),
    ]:
        acc    = summary["final_accuracy"]
        target = "✅" if acc >= 0.95 else "❌ below ≥95% HPE target"
        print(f"  {label:<18} prequential accuracy: {acc:.4f}  {target}")
    print()
    print("  Per-class F1 (Test):") 
    for cls, m in test_summary["per_class"].items():
        f1   = m["f1"]
        flag = "✅" if f1 >= 0.95 else "⚠️ "
        print(f"    {cls:<15} F1={f1:.4f}  {flag}")
    print("=" * 70)


if __name__ == "__main__":
    main()
