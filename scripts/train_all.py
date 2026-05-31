#!/usr/bin/env python3
"""Run the full training pipeline in dependency order."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IO_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "io_features.parquet"

STEPS = [
    (1, "Data Generation", PROJECT_ROOT / "src" / "data" / "data_generator.py"),
    (2, "Feature Engineering", PROJECT_ROOT / "src" / "data" / "feature_engineer.py"),
    (3, "Preprocessing", PROJECT_ROOT / "src" / "pipeline" / "preprocessor.py"),
    (4, "LightGBM Baseline", PROJECT_ROOT / "src" / "models" / "classifier" / "lightgbm_baseline.py"),
    (5, "LightGBM Tuned", PROJECT_ROOT / "src" / "models" / "classifier" / "lightgbm_tuned.py"),
    (6, "ARF + ADWIN", PROJECT_ROOT / "src" / "models" / "classifier" / "arf_adwin.py"),
    (7, "Anomaly Detector Ensemble", PROJECT_ROOT / "src" / "models" / "anomaly" / "evaluate_all_detectors.py"),
    (8, "DTF Forecaster (N-BEATS)", PROJECT_ROOT / "src" / "models" / "forecasting" / "dtf_forecaster.py"),
    (9, "TFT Forecaster", PROJECT_ROOT / "src" / "models" / "forecasting" / "tft_forecaster.py"),
]

ARTIFACTS = [
    ("models/scaler.pkl", PROJECT_ROOT / "models" / "scaler.pkl"),
    ("models/classifier/lightgbm_model.pkl", PROJECT_ROOT / "models" / "classifier" / "lightgbm_model.pkl"),
    ("models/classifier/lightgbm_metrics.json", PROJECT_ROOT / "models" / "classifier" / "lightgbm_metrics.json"),
    ("models/classifier/lightgbm_tuned_model.pkl", PROJECT_ROOT / "models" / "classifier" / "lightgbm_tuned_model.pkl"),
    ("models/classifier/lightgbm_tuned_metrics.json", PROJECT_ROOT / "models" / "classifier" / "lightgbm_tuned_metrics.json"),
    ("models/classifier/arf_model.pkl", PROJECT_ROOT / "models" / "classifier" / "arf_model.pkl"),
    ("models/classifier/arf_metrics.json", PROJECT_ROOT / "models" / "classifier" / "arf_metrics.json"),
    ("models/classifier/arf_prequential_accuracy.csv", PROJECT_ROOT / "models" / "classifier" / "arf_prequential_accuracy.csv"),
    ("models/anomaly/ensemble/models", PROJECT_ROOT / "models" / "anomaly" / "ensemble" / "models"),
    ("models/forecasting/nbeats_model.pth", PROJECT_ROOT / "models" / "forecasting" / "nbeats_model.pth"),
    ("models/forecasting/dtf_forecast.json", PROJECT_ROOT / "models" / "forecasting" / "dtf_forecast.json"),
    ("models/forecasting/tft_model.pth", PROJECT_ROOT / "models" / "forecasting" / "tft_model.pth"),
]


def parse_steps(value: str) -> set[int]:
    steps: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise argparse.ArgumentTypeError("Steps must be comma-separated integers.")
        num = int(part)
        if num < 1 or num > len(STEPS):
            raise argparse.ArgumentTypeError("Steps must be between 1 and 9.")
        steps.add(num)
    if not steps:
        raise argparse.ArgumentTypeError("No valid steps provided.")
    return steps


def prompt_skip_data() -> bool:
    prompt = (
        "data/processed/io_features.parquet already exists. "
        "Skip data generation and feature engineering (steps 1-2)? [y/N]: "
    )
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def run_step(step_num: int, title: str, script_path: Path) -> None:
    print(f"=== STEP {step_num}/9: {title} ===")
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(
            f"ERROR: Step {step_num}/9 ({title}) failed with return code {result.returncode}.",
            file=sys.stderr,
        )
        sys.exit(result.returncode if result.returncode else 1)


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{int(size)} B"


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def print_summary() -> None:
    print("\nModel Artifacts Summary")
    print(f"{'Artifact':<60} {'Status':<8} Size")
    print("-" * 85)
    for label, path in ARTIFACTS:
        if path.exists():
            size = _human_size(_path_size(path))
            status = "OK"
        else:
            size = "-"
            status = "MISSING"
        print(f"{label:<60} {status:<8} {size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full training pipeline.")
    parser.add_argument("--skip-data", action="store_true", help="Skip steps 1-2 without prompting.")
    parser.add_argument(
        "--steps",
        type=parse_steps,
        help="Comma-separated step numbers to run (example: --steps 4,5,6).",
    )
    args = parser.parse_args()

    selected_steps = args.steps or set(range(1, len(STEPS) + 1))

    skip_data = args.skip_data
    if not skip_data and IO_FEATURES_PATH.exists() and ({1, 2} & selected_steps):
        if prompt_skip_data():
            skip_data = True

    if skip_data:
        selected_steps -= {1, 2}

    for step_num, title, script_path in STEPS:
        if step_num not in selected_steps:
            continue
        run_step(step_num, title, script_path)

    print_summary()


if __name__ == "__main__":
    main()
