"""
src/pipeline/data_loader.py

Unified entrypoint to load any preprocessed split with a single call.

Usage
-----
    from src.pipeline.data_loader import load_split

    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "features"

VALID_SPLITS = {"train", "val", "test"}


def load_split(
    split: str,
    as_numpy: bool = False,
) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
    """
    Load a preprocessed feature/label split from disk.

    Parameters
    ----------
    split    : one of "train", "val", "test"
    as_numpy : if True, returns (np.ndarray, np.ndarray) instead of DataFrames

    Returns
    -------
    X : pd.DataFrame or np.ndarray  — scaled feature matrix
    y : pd.Series   or np.ndarray   — integer class labels (0–4)

    Raises
    ------
    ValueError  : if split is not "train", "val", or "test"
    FileNotFoundError : if preprocessor.py has not been run yet
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")

    x_path = FEAT_DIR / f"X_{split}.parquet"
    y_path = FEAT_DIR / f"y_{split}.parquet"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Split files not found at {FEAT_DIR}. "
            "Run `python src/pipeline/preprocessor.py` first."
        )

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path).squeeze()   # DataFrame → Series

    if as_numpy:
        return X.to_numpy(), y.to_numpy()

    return X, y


def load_all_splits(
    as_numpy: bool = False,
) -> dict[str, tuple]:
    """
    Convenience function: load all three splits at once.

    Returns
    -------
    {
        "train": (X_train, y_train),
        "val"  : (X_val,   y_val),
        "test" : (X_test,  y_test),
    }
    """
    return {split: load_split(split, as_numpy=as_numpy) for split in VALID_SPLITS}


# ── Quick sanity check when run directly ─────────────────────────────────────
if __name__ == "__main__":
    print("Loading all splits...")
    for split in ("train", "val", "test"):
        X, y = load_split(split)
        print(f"  {split:>5} → X: {X.shape}  y: {y.shape}  "
              f"classes: {sorted(y.unique())}")
    print("✅ data_loader OK")
