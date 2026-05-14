"""
src/models/classifier/lightgbm_tuned.py

Optuna-driven hyperparameter tuning for LightGBM multiclass classifier.
Blueprint Phase 2.1 — push test accuracy past ≥95% HPE mandate.

Strategy
--------
1. Optuna TPE sampler explores a carefully bounded hyperparameter space that
   targets the baseline's overfitting problem (train 99.96% → test 88.2%).
2. Each trial trains a LightGBM model with early stopping on the VAL set,
   preventing overfitting and dramatically speeding up trials.
3. After Optuna finishes, the best parameters are retrained on TRAIN + VAL
   combined, and the final model is evaluated on the held-out TEST set.

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
N_TRIALS        = 100     # number of Optuna trials
EARLY_STOP      = 50      # LightGBM early stopping rounds
SEED            = 42
N_CLASSES       = 5
OPTUNA_TIMEOUT  = 3600    # max seconds (1 hour safety net)


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    Optuna objective: maximize weighted F1 on the validation set.

    We optimize weighted-F1 instead of raw accuracy because the HPE mandate
    requires strong performance across ALL 5 workload classes, not just the
    majority class. A model with high accuracy but poor minority-class recall
    would fail in production.

    The hyperparameter space is engineered to combat the overfitting observed
    in the baseline (99.96% train / 88.2% test):
      - Aggressive regularization (reg_alpha, reg_lambda, min_child_samples)
      - Feature & row subsampling (colsample_bytree, subsample)
      - Constrained tree complexity (max_depth, num_leaves)
    """
    params = {
        "objective":         "multiclass",
        "num_class":         N_CLASSES,
        "metric":            "multi_logloss",
        "verbosity":         -1,
        "random_state":      SEED,
        "n_jobs":            -1,
        "boosting_type":     trial.suggest_categorical(
                                 "boosting_type", ["gbdt", "dart"]
                             ),

        # ── Tree structure (controls overfitting) ─────────────────────────────
        "n_estimators":      trial.suggest_int("n_estimators", 200, 2000, step=100),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth":         trial.suggest_int("max_depth", 4, 12),
        "num_leaves":        trial.suggest_int("num_leaves", 16, 256),

        # ── Regularization (prevent memorization) ─────────────────────────────
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),

        # ── Subsampling (force generalization) ────────────────────────────────
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq":    trial.suggest_int("subsample_freq", 1, 10),
    }

    # Enforce num_leaves <= 2^max_depth to avoid degenerate trees
    max_possible_leaves = 2 ** params["max_depth"]
    if params["num_leaves"] > max_possible_leaves:
        params["num_leaves"] = max_possible_leaves

    model = lgb.LGBMClassifier(**params)

    # Early stopping on val set — stops training when val loss plateaus,
    # which is the single most effective anti-overfitting technique
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    preds = model.predict(X_val)
    return float(f1_score(y_val, preds, average="weighted"))


def evaluate_split(
    model: lgb.LGBMClassifier,
    split_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    labels: list[int],
) -> dict:
    """Evaluate model on a split and return metrics + confusion matrix."""
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, preds, labels=labels)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_path = MODEL_DIR / f"lightgbm_tuned_confusion_matrix_{split_name}.csv"
    cm_df.to_csv(cm_path, index=True)

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix_path": str(cm_path.relative_to(ROOT)),
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

    labels = sorted(np.unique(y_train))

    print(f"  train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]}  Classes: {len(labels)}")

    # ── Optuna study ──────────────────────────────────────────────────────────
    print(f"\nStarting Optuna search ({N_TRIALS} trials, "
          f"early_stop={EARLY_STOP}, timeout={OPTUNA_TIMEOUT}s)...")
    print("  Objective: maximize weighted F1 on validation set\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=True,
    )

    # ── Best trial summary ────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'─' * 70}")
    print(f"  Best trial #{best.number}")
    print(f"  Val weighted-F1: {best.value:.6f}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        print(f"    {k:<22} = {v}")
    print(f"{'─' * 70}")

    # ── Retrain best model on TRAIN + VAL (maximum data for final model) ──────
    print("\nRetraining best model on TRAIN + VAL combined...")
    best_params = best.params.copy()

    # Enforce num_leaves constraint
    max_possible_leaves = 2 ** best_params["max_depth"]
    if best_params.get("num_leaves", 31) > max_possible_leaves:
        best_params["num_leaves"] = max_possible_leaves

    final_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="multi_logloss",
        verbosity=-1,
        random_state=SEED,
        n_jobs=-1,
        **best_params,
    )

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    final_model.fit(X_train_val, y_train_val)

    # ── Evaluate on all splits ────────────────────────────────────────────────
    print("\nEvaluating final tuned model...")
    metrics = {
        "best_trial":  best.number,
        "best_val_f1": best.value,
        "best_params": best.params,
        "n_trials":    len(study.trials),
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
        cls_data = test_report[str(cls_id)]
        f1_val = cls_data["f1-score"]
        flag = "✅" if f1_val >= 0.95 else "⚠️ "
        print(f"    Class {cls_id}  F1={f1_val:.4f}  {flag}")

    # Overfitting gap
    train_acc = metrics["train"]["accuracy"]
    test_acc  = metrics["test"]["accuracy"]
    gap = train_acc - test_acc
    print(f"\n  Overfitting gap (train - test): {gap:.4f} "
          f"({'✅ healthy' if gap < 0.05 else '⚠️  still overfitting'})")
    print("=" * 70)


if __name__ == "__main__":
    main()
