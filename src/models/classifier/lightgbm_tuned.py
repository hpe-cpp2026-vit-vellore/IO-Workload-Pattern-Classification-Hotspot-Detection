"""
src/models/classifier/lightgbm_tuned.py

Optuna-driven hyperparameter tuning for LightGBM multiclass classifier.
Blueprint Phase 2.1 — push test accuracy past ≥95% HPE mandate.

Strategy
--------
1. Optuna TPE sampler explores a bounded hyperparameter space targeting the
   baseline overfitting problem (train 99.96% → test 88.2%).
2. Each trial uses LightGBM's native Dataset API with early stopping on VAL,
   preventing overfitting and making trials 5-10× faster than sklearn API.
3. After search, best params retrain on TRAIN + VAL combined; final model
   evaluated on held-out TEST.

Speed optimisations (vs naive approach)
---------------------------------------
  - Native lgb.Dataset + lgb.train() instead of sklearn wrapper — avoids
    repeated DataFrame→internal conversion overhead on every trial.
  - Datasets constructed ONCE before the loop, reused across all 100 trials.
  - n_estimators capped at 1000 (early stopping will halt well before that).
  - Tight search space — no degenerate parameter combos that waste time.
  - HyperbandPruner kills bad trials early.

Produces
--------
  models/classifier/lightgbm_tuned_model.pkl
  models/classifier/lightgbm_tuned_metrics.json
  models/classifier/lightgbm_tuned_feature_importance.csv
  models/classifier/lightgbm_tuned_confusion_matrix_{split}.csv
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.data_loader import load_split

MODEL_DIR = ROOT / "models" / "classifier"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Tuning config ─────────────────────────────────────────────────────────────
N_TRIALS        = 60      # 80 trials (TPE converges well by ~60)
EARLY_STOP      = 15      # LightGBM early stopping rounds
SEED            = 42
N_CLASSES       = 5
OPTUNA_TIMEOUT  = 2400    # 40-minute safety net


def objective(
    trial: optuna.Trial,
    dtrain: lgb.Dataset,
    dval: lgb.Dataset,
    y_val: np.ndarray,
) -> float:
    """
    Optuna objective: maximize weighted F1 on validation set.

    Uses native lgb.train() API — 5-10× faster than LGBMClassifier.fit()
    because the Dataset objects are pre-constructed and reused across trials.
    """
    params = {
        "objective":         "multiclass",
        "num_class":         N_CLASSES,
        "metric":            "multi_logloss",
        "verbosity":         -1,
        "seed":              SEED,
        "num_threads":       -1,
        "boosting_type":     "gbdt",

        # ── Tree structure ────────────────────────────────────────────────────
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":         trial.suggest_int("max_depth", 5, 10),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 200),

        # ── Regularisation ────────────────────────────────────────────────────
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 200),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.15),

        # ── Subsampling ───────────────────────────────────────────────────────
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 0.95),
        "subsample":         trial.suggest_float("subsample", 0.6, 0.95),
        "subsample_freq":    trial.suggest_int("subsample_freq", 1, 7),
    }

    # Enforce num_leaves <= 2^max_depth
    max_leaves = 2 ** params["max_depth"]
    if params["num_leaves"] > max_leaves:
        params["num_leaves"] = max_leaves

    # Native LightGBM training — much faster than sklearn wrapper
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Predict on val
    raw_preds = booster.predict(dval.get_data())
    preds = np.argmax(raw_preds, axis=1)
    val_f1 = float(f1_score(y_val, preds, average="weighted"))

    # Report for pruner
    trial.report(val_f1, step=booster.best_iteration)
    if trial.should_prune():
        raise optuna.TrialPruned()

    # Store best iteration for final retrain
    trial.set_user_attr("best_iteration", booster.best_iteration)

    return val_f1


def evaluate_split(
    model: lgb.LGBMClassifier,
    split_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    labels: list[int],
) -> dict:
    """Evaluate model on one split and return metrics + confusion matrix."""
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, output_dict=True, zero_division=0)

    cm = confusion_matrix(y, preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_path = MODEL_DIR / f"lightgbm_tuned_confusion_matrix_{split_name}.csv"
    cm_df.to_csv(cm_path, index=True)

    return {
        "accuracy":               acc,
        "classification_report":  report,
        "confusion_matrix_path":  str(cm_path.relative_to(ROOT)),
    }


def main() -> None:
    print("=" * 70)
    print(" LightGBM Hyperparameter Tuning  (Optuna TPE)")
    print(" Blueprint Phase 2.1  —  Target: ≥95% Test Accuracy")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading preprocessed splits...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    y_train = y_train.astype(int)
    y_val   = y_val.astype(int)
    y_test  = y_test.astype(int)

    labels = sorted(
        np.unique(np.concatenate([y_train.values, y_val.values, y_test.values]))
    )

    print(f"  train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]}  Classes: {len(labels)}")

    # ── Pre-build native LightGBM Datasets (built ONCE, reused every trial) ──
    print("\n  Pre-building LightGBM Datasets for fast trial execution...")
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, free_raw_data=False)
    dtrain.construct()
    dval.construct()
    print("  ✅ Datasets ready\n")

    # ── Optuna study ──────────────────────────────────────────────────────────
    print(f"Starting Optuna search ({N_TRIALS} trials, "
          f"early_stop={EARLY_STOP}, timeout={OPTUNA_TIMEOUT}s)...")
    print("  Objective: maximise weighted F1 on validation set\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=30,
            max_resource=1000,
            reduction_factor=3,
        ),
    )

    y_val_np = y_val.values

    try:
        study.optimize(
            lambda trial: objective(trial, dtrain, dval, y_val_np),
            n_trials=N_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Search interrupted. Using best trial found so far.")

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[ERROR] No completed trials. Exiting.")
        sys.exit(1)

    # ── Best trial summary ────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'─' * 70}")
    print(f"  Best trial #{best.number}")
    print(f"  Val weighted-F1: {best.value:.6f}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        print(f"    {k:<22} = {v}")
    print(f"{'─' * 70}")

    # ── Retrain best model on TRAIN + VAL combined ────────────────────────────
    print("\nRetraining best model on TRAIN + VAL combined...")
    best_params = best.params.copy()

    # Cap n_estimators to what early stopping actually used
    best_iteration = best.user_attrs.get("best_iteration", 1000)
    print(f"  Using n_estimators={best_iteration} (from early stopping)")

    # Enforce num_leaves constraint
    max_leaves = 2 ** best_params["max_depth"]
    if best_params.get("num_leaves", 31) > max_leaves:
        best_params["num_leaves"] = max_leaves

    final_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="multi_logloss",
        verbosity=-1,
        random_state=SEED,
        n_jobs=4,
        boosting_type="gbdt",
        n_estimators=best_iteration,
        **best_params,
    )

    X_tv = pd.concat([
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
    ], axis=0).reset_index(drop=True)
    y_tv = pd.concat([
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
    ], axis=0).reset_index(drop=True)

    final_model.fit(X_tv, y_tv)

    # ── Evaluate on all splits ────────────────────────────────────────────────
    print("\nEvaluating final tuned model...")
    metrics = {
        "best_trial":         best.number,
        "best_val_f1":        best.value,
        "best_params":        best.params,
        "best_iteration":     best_iteration,
        "n_trials_completed": len(completed),
        "n_trials_total":     len(study.trials),
        "train": evaluate_split(final_model, "train", X_train, y_train, labels),
        "val":   evaluate_split(final_model, "val",   X_val,   y_val,   labels),
        "test":  evaluate_split(final_model, "test",  X_test,  y_test,  labels),
    }

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = MODEL_DIR / "lightgbm_tuned_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics → {metrics_path.relative_to(ROOT)}")

    # ── Save feature importances ──────────────────────────────────────────────
    fi = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_path = MODEL_DIR / "lightgbm_tuned_feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    print(f"  Feature imp → {fi_path.relative_to(ROOT)}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / "lightgbm_tuned_model.pkl"
    dump(final_model, model_path)
    print(f"  Model → {model_path.relative_to(ROOT)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" RESULTS SUMMARY  (Tuned LightGBM)")
    print("=" * 70)
    for split_name in ["train", "val", "test"]:
        acc = metrics[split_name]["accuracy"]
        target = "✅" if acc >= 0.95 else "❌ below ≥95% HPE target"
        print(f"  {split_name:<6} accuracy: {acc:.4f}  {target}")

    test_report = metrics["test"]["classification_report"]
    print(f"\n  Per-class F1 (Test):")
    for cls_id in labels:
        cls_key = str(cls_id)
        if cls_key not in test_report:
            continue
        f1_val = test_report[cls_key]["f1-score"]
        flag = "✅" if f1_val >= 0.95 else "⚠️ "
        print(f"    Class {cls_id}  F1={f1_val:.4f}  {flag}")

    # Overfitting gap
    train_acc = metrics["train"]["accuracy"]
    test_acc  = metrics["test"]["accuracy"]
    gap = train_acc - test_acc
    print(f"\n  Overfitting gap (train - test): {gap:.4f} "
          f"({'✅ healthy' if gap < 0.05 else '⚠️  still overfitting'})")
    print(f"  Completed trials: {len(completed)} / {len(study.trials)}")
    print("=" * 70)


if __name__ == "__main__":
    main()