#!/usr/bin/env python3
"""
scripts/train_all.py

Orchestrates the entire ML training pipeline from scratch in correct dependency order.

Pipeline Steps:
1. Data Generation       → data/synthetic/io_workload_data.parquet
2. Feature Engineering   → data/processed/io_features.parquet
3. Preprocessing         → data/features/*.parquet + models/scaler.pkl
4. LightGBM Baseline     → models/classifier/lightgbm_model.pkl
5. LightGBM Tuned        → models/classifier/lightgbm_tuned_model.pkl
6. ARF+ADWIN             → models/classifier/arf_model.pkl
7. Anomaly Detection     → models/anomaly/ensemble/models/
8. N-BEATS Forecasting   → models/forecasting/nbeats_model.pth
9. TFT Forecasting       → models/forecasting/tft_model.pth
10. Demand Forecasting    → models/forecasting/demand_forecaster.pkl

Usage:
    python scripts/train_all.py                    # Full pipeline
    python scripts/train_all.py --skip-data        # Skip steps 1-2
    python scripts/train_all.py --steps 4,5,6      # Run only specific steps
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── Project root (resolve relative to this script) ──────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ── Pipeline step definitions ────────────────────────────────────────────────
PIPELINE_STEPS = [
    {
        "id": 1,
        "name": "Data Generation",
        "script": "src/data/data_generator.py",
        "output": "data/synthetic/io_workload_data.parquet",
        "description": "Generate 432K rows of synthetic IO workload data",
    },
    {
        "id": 2,
        "name": "Feature Engineering",
        "script": "src/data/feature_engineer.py",
        "output": "data/processed/io_features.parquet",
        "description": "Engineer ML-ready features with rolling windows and lag features",
    },
    {
        "id": 3,
        "name": "Preprocessing & Splitting",
        "script": "src/pipeline/preprocessor.py",
        "output": "data/features/X_train.parquet",
        "description": "Chronological train/val/test split with robust scaling",
    },
    {
        "id": 4,
        "name": "LightGBM Baseline",
        "script": "src/models/classifier/lightgbm_baseline.py",
        "output": "models/classifier/lightgbm_model.pkl",
        "description": "Train baseline LightGBM classifier",
    },
    {
        "id": 5,
        "name": "LightGBM Tuned (Optuna)",
        "script": "src/models/classifier/lightgbm_tuned.py",
        "output": "models/classifier/lightgbm_tuned_model.pkl",
        "description": "Hyperparameter tuning with Optuna (60 trials)",
    },
    {
        "id": 6,
        "name": "ARF+ADWIN Streaming",
        "script": "src/models/classifier/arf_adwin.py",
        "output": "models/classifier/arf_model.pkl",
        "description": "Adaptive Random Forest with ADWIN drift detection",
    },
    {
        "id": 7,
        "name": "Anomaly Detection Ensemble",
        "script": "src/models/anomaly/evaluate_all_detectors.py",
        "output": "models/anomaly/statistical_detector_scores.csv",
        "description": "Train and evaluate all anomaly detectors",
    },
    {
        "id": 8,
        "name": "N-BEATS Capacity Forecasting",
        "script": "src/models/forecasting/dtf_forecaster.py",
        "output": "models/forecasting/nbeats_model.pth",
        "description": "Days-to-Fill (DTF) capacity forecasting with N-BEATS",
    },
    {
        "id": 9,
        "name": "TFT Latency Forecasting",
        "script": "src/models/forecasting/tft_forecaster.py",
        "output": "models/forecasting/tft_model.pth",
        "description": "Temporal Fusion Transformer for tail latency risk prediction",
    },
    {
        "id": 10,
        "name": "Demand Forecasting",
        "script": "src/models/forecasting/demand_forecaster.py",
        "output": "models/forecasting/demand_forecaster.pkl",
        "description": "Quantile regression forecasts for IOPS and throughput demand",
    },
]


# ── Helper functions ─────────────────────────────────────────────────────────
def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    width = 80
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width)


def print_step_header(step_num: int, total: int, name: str) -> None:
    """Print a step header."""
    print_header(f"STEP {step_num}/{total}: {name}", char="=")


def check_file_exists(path: Path) -> bool:
    """Check if a file or directory exists."""
    return path.exists()


def get_file_size(path: Path) -> str:
    """Get human-readable file size."""
    if not path.exists():
        return "N/A"
    
    if path.is_dir():
        # For directories, count files
        try:
            file_count = sum(1 for _ in path.rglob("*") if _.is_file())
            return f"{file_count} files"
        except Exception:
            return "directory"
    
    size_bytes = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def run_step(step: dict, python_exe: str) -> bool:
    """
    Run a single pipeline step as a subprocess.
    
    Returns:
        True if successful, False otherwise
    """
    script_path = PROJECT_ROOT / step["script"]
    
    if not script_path.exists():
        print(f"❌ ERROR: Script not found: {script_path}")
        return False
    
    print(f"\n📝 Description: {step['description']}")
    print(f"🔧 Running: {script_path.relative_to(PROJECT_ROOT)}")
    print(f"⏱️  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        # Run the script as a subprocess using the same Python interpreter
        result = subprocess.run(
            [python_exe, str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=False,
            capture_output=False,  # Stream output to console in real-time
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"\n❌ FAILED with exit code {result.returncode}")
            print(f"⏱️  Duration: {elapsed:.1f}s")
            return False
        
        print(f"\n✅ SUCCESS")
        print(f"⏱️  Duration: {elapsed:.1f}s")
        
        # Verify output file was created
        output_path = PROJECT_ROOT / step["output"]
        if check_file_exists(output_path):
            size = get_file_size(output_path)
            print(f"📦 Output: {step['output']} ({size})")
        else:
            print(f"⚠️  Warning: Expected output not found: {step['output']}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def prompt_skip_data() -> bool:
    """Prompt user whether to skip data generation and feature engineering."""
    print("\n" + "─" * 80)
    print("  Data files already exist:")
    
    features_path = PROJECT_ROOT / "data/processed/io_features.parquet"
    if features_path.exists():
        size = get_file_size(features_path)
        print(f"    ✓ {features_path.relative_to(PROJECT_ROOT)} ({size})")
    
    print("\n  Skip data generation and feature engineering? (saves ~5-10 minutes)")
    print("─" * 80)
    
    while True:
        response = input("  Skip steps 1-2? [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        if response in ("n", "no", ""):
            return False
        print("  Please enter 'y' or 'n'")


def print_summary(completed_steps: List[dict], failed_step: Optional[dict]) -> None:
    """Print final summary of pipeline execution."""
    print_header("PIPELINE SUMMARY", char="=")
    
    if failed_step:
        print(f"\n❌ Pipeline FAILED at step {failed_step['id']}: {failed_step['name']}")
        print(f"\n✅ Completed steps ({len(completed_steps)}):")
    else:
        print(f"\n✅ Pipeline completed successfully! All {len(completed_steps)} steps finished.")
        print(f"\n📦 Generated model files:")
    
    # Print table of completed steps with file sizes
    print("\n" + "─" * 80)
    print(f"  {'Step':<4} {'Name':<35} {'Output File':<25} {'Size':<15}")
    print("─" * 80)
    
    for step in completed_steps:
        output_path = PROJECT_ROOT / step["output"]
        size = get_file_size(output_path)
        exists = "✓" if check_file_exists(output_path) else "✗"
        print(f"  {step['id']:<4} {step['name']:<35} {exists} {step['output'].split('/')[-1]:<23} {size:<15}")
    
    print("─" * 80)
    
    if failed_step:
        print(f"\n💡 To resume from step {failed_step['id']}, run:")
        print(f"   python scripts/train_all.py --steps {failed_step['id']}-9")
    else:
        print("\n🎉 All models trained successfully!")
        print("   Next steps:")
        print("   - Run the API server: make up")
        print("   - View the dashboard: http://localhost:8501")
        print("   - Evaluate models: python evaluate_ensemble.py")


def parse_step_range(step_str: str) -> List[int]:
    """
    Parse step specification string into list of step IDs.
    
    Examples:
        "4,5,6" → [4, 5, 6]
        "4-6"   → [4, 5, 6]
        "4"     → [4]
    """
    steps = []
    for part in step_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            steps.extend(range(int(start), int(end) + 1))
        else:
            steps.append(int(part))
    return sorted(set(steps))


def main() -> int:
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Orchestrate the entire ML training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_all.py                    # Run full pipeline
  python scripts/train_all.py --skip-data        # Skip data generation
  python scripts/train_all.py --steps 4,5,6      # Run only steps 4, 5, 6
  python scripts/train_all.py --steps 4-9        # Run steps 4 through 9
    python scripts/train_all.py --steps 10         # Run demand forecasting only
        """,
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation and feature engineering (steps 1-2)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Run only specific steps (e.g., '4,5,6' or '4-6')",
    )
    
    args = parser.parse_args()
    
    # Determine which steps to run
    if args.steps:
        try:
            step_ids = parse_step_range(args.steps)
            steps_to_run = [s for s in PIPELINE_STEPS if s["id"] in step_ids]
            if not steps_to_run:
                print(f"❌ ERROR: No valid steps found in: {args.steps}")
                return 1
        except ValueError as e:
            print(f"❌ ERROR: Invalid step specification: {args.steps}")
            print(f"   {e}")
            return 1
    else:
        steps_to_run = PIPELINE_STEPS.copy()
        
        # Check if we should skip data steps
        skip_data = args.skip_data
        if not skip_data:
            # Check if data already exists and prompt user
            features_path = PROJECT_ROOT / "data/processed/io_features.parquet"
            if features_path.exists():
                skip_data = prompt_skip_data()
        
        if skip_data:
            print("\n⏭️  Skipping steps 1-2 (data generation and feature engineering)")
            steps_to_run = [s for s in steps_to_run if s["id"] > 2]
    
    # Print pipeline overview
    print_header("IO WORKLOAD ML TRAINING PIPELINE", char="=")
    print(f"\n📋 Pipeline Overview:")
    print(f"   Total steps: {len(PIPELINE_STEPS)}")
    print(f"   Steps to run: {len(steps_to_run)}")
    print(f"   Python: {sys.executable}")
    print(f"   Working directory: {PROJECT_ROOT}")
    
    if steps_to_run:
        print(f"\n   Steps: {', '.join(str(s['id']) for s in steps_to_run)}")
    
    # Confirm before starting
    if not args.steps and not args.skip_data:
        print("\n" + "─" * 80)
        input("  Press Enter to start, or Ctrl+C to cancel...")
    
    # Execute pipeline
    completed_steps = []
    failed_step = None
    total_start = time.time()
    
    for i, step in enumerate(steps_to_run, 1):
        print_step_header(i, len(steps_to_run), step["name"])
        
        success = run_step(step, sys.executable)
        
        if success:
            completed_steps.append(step)
        else:
            failed_step = step
            break
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print_summary(completed_steps, failed_step)
    
    print(f"\n⏱️  Total pipeline time: {total_elapsed / 60:.1f} minutes")
    
    return 0 if failed_step is None else 1


if __name__ == "__main__":
    sys.exit(main())
