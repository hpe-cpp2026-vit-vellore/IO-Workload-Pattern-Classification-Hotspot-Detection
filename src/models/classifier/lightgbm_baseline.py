"""
src/models/classifier/lightgbm_baseline.py
Baseline LightGBM classifier for IO workload types.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.data_loader import load_split

MODEL_DIR = ROOT / "models" / "classifier"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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
    cm_path = MODEL_DIR / f"lightgbm_confusion_matrix_{split_name}.csv"
    cm_df.to_csv(cm_path, index=True)

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix_path": str(cm_path.relative_to(ROOT)),
    }


def main() -> None:
    print("Loading preprocessed splits...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    labels = sorted(np.unique(y_train))

    print("Training LightGBM baseline...")
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(labels),
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Evaluating...")
    metrics = {
        "train": evaluate_split(model, "train", X_train, y_train, labels),
        "val": evaluate_split(model, "val", X_val, y_val, labels),
        "test": evaluate_split(model, "test", X_test, y_test, labels),
    }

    metrics_path = MODEL_DIR / "lightgbm_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fi = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / "lightgbm_feature_importance.csv", index=False)

    model_path = MODEL_DIR / "lightgbm_model.pkl"
    dump(model, model_path)

    print("\nLightGBM baseline complete.")
    print(f"  Model: {model_path.relative_to(ROOT)}")
    print(f"  Metrics: {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
