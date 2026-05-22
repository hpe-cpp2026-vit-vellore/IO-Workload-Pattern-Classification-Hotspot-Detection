import os
import sys
import json
import ctypes
import shutil
import platform
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("telemetry_parser")

# Setup project pathing
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Shared library setup
SYSTEM = platform.system()
if SYSTEM == "Windows":
    LIB_EXT = ".dll"
elif SYSTEM == "Darwin":
    LIB_EXT = ".dylib"
else:
    LIB_EXT = ".so"

CPP_SOURCE_PATH = PROJECT_ROOT / "src" / "cpp" / "telemetry_parser.cpp"
LIB_DEST_PATH = PROJECT_ROOT / "models" / f"telemetry_parser{LIB_EXT}"

_cpp_lib = None
_cached_bounds = None
CPP_AVAILABLE = False


def load_or_create_bounds(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """
    Loads outlier bounds from models/bounds.json. If missing, computes them from the
    historical dataset (using first 21 days training split) to avoid data leakage
    and to prevent breaking execution.
    """
    global _cached_bounds
    if _cached_bounds is not None:
        return _cached_bounds

    bounds_path = project_root / "models" / "bounds.json"
    if bounds_path.exists():
        try:
            with open(bounds_path, "r") as f:
                _cached_bounds = json.load(f)
                logger.info("Loaded outlier bounds from models/bounds.json")
                return _cached_bounds
        except Exception as e:
            logger.warning(f"Error reading models/bounds.json: {e}. Will attempt re-creation.")

    # Bounds missing or unreadable -> compute from raw features
    logger.info("Outlier bounds file missing. Re-computing training bounds from historical telemetry...")
    try:
        import pandas as pd
        import numpy as np

        features_pq = project_root / "data" / "processed" / "io_features.parquet"
        if not features_pq.exists():
            features_pq = project_root / "data" / "processed" / "io_features.csv"

        if features_pq.exists():
            if features_pq.suffix == ".parquet":
                df = pd.read_parquet(features_pq)
            else:
                df = pd.read_csv(features_pq)

            # Match CPP3 chronological split objective (train on first 21 days only)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            start_date = df["timestamp"].min().normalize()
            train_df = df[df["timestamp"] < (start_date + pd.Timedelta(days=21))]

            # Identify numeric features
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            if "label" in numeric_cols:
                numeric_cols.remove("label")

            q1 = train_df[numeric_cols].quantile(0.25)
            q3 = train_df[numeric_cols].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr

            _cached_bounds = {
                "low": low.to_dict(),
                "high": high.to_dict()
            }

            bounds_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bounds_path, "w") as f:
                json.dump(_cached_bounds, f, indent=4)
            logger.info("Computed and saved outlier bounds to models/bounds.json")
            return _cached_bounds
    except Exception as e:
        logger.error(f"Failed to auto-generate outlier bounds: {e}")

    _cached_bounds = {"low": {}, "high": {}}
    return _cached_bounds


def find_msvc_env_script() -> Optional[str]:
    """Attempts to find the path to vcvarsall.bat in Visual Studio/Build Tools installations."""
    if SYSTEM != "Windows":
        return None

    # 1. Try vswhere.exe in standard location
    vswhere_path = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere_path.exists():
        try:
            res = subprocess.run(
                [str(vswhere_path), "-latest", "-property", "installationPath"],
                capture_output=True,
                text=True,
                check=True
            )
            install_path = res.stdout.strip()
            if install_path:
                vcvars = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                if vcvars.exists():
                    return str(vcvars)
        except Exception as e:
            logger.debug(f"vswhere check failed: {e}")

    # 2. Try standard fallback locations
    roots = [
        Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")),
        Path(os.environ.get("ProgramFiles", "C:\\Program Files")),
    ]
    for r in roots:
        vs_dir = r / "Microsoft Visual Studio"
        if vs_dir.exists():
            for vcvars in vs_dir.rglob("vcvarsall.bat"):
                if vcvars.exists():
                    return str(vcvars)

    return None


def compile_cpp_library() -> bool:
    """Attempts to compile the telemetry_parser.cpp source into a shared library DLL/SO."""
    if not CPP_SOURCE_PATH.exists():
        logger.warning(f"C++ source file not found at {CPP_SOURCE_PATH}. Compilation skipped.")
        return False

    LIB_DEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Look for g++
    gpp = shutil.which("g++")
    if gpp:
        try:
            cmd = [gpp, "-O3", "-shared", "-fPIC", str(CPP_SOURCE_PATH), "-o", str(LIB_DEST_PATH)]
            logger.info(f"Compiling telemetry parser with: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("C++ telemetry parser compiled successfully using g++.")
            return True
        except Exception as e:
            logger.warning(f"Compilation with g++ failed: {e}")

    # 2. Look for clang++
    clang = shutil.which("clang++")
    if clang:
        try:
            cmd = [clang, "-O3", "-shared", "-fPIC", str(CPP_SOURCE_PATH), "-o", str(LIB_DEST_PATH)]
            logger.info(f"Compiling telemetry parser with: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("C++ telemetry parser compiled successfully using clang++.")
            return True
        except Exception as e:
            logger.warning(f"Compilation with clang++ failed: {e}")

    # 3. Look for cl.exe (MSVC)
    cl = shutil.which("cl")
    vcvars_path = find_msvc_env_script()

    if cl or vcvars_path:
        try:
            if cl and not vcvars_path:
                cmd = [cl, "/LD", "/O2", "/EHsc", str(CPP_SOURCE_PATH), f"/Fe:{LIB_DEST_PATH}"]
                logger.info(f"Compiling telemetry parser with: {' '.join(cmd)}")
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            else:
                arch = "x64" if sys.maxsize > 2**32 else "x86"
                cmd_str = f'call "{vcvars_path}" {arch} && cl.exe /LD /O2 /EHsc "{str(CPP_SOURCE_PATH)}" /Fe:"{str(LIB_DEST_PATH)}"'
                logger.info(f"Compiling telemetry parser via MSVC env setup: {cmd_str}")
                subprocess.run(cmd_str, shell=True, capture_output=True, text=True, check=True)

            logger.info("C++ telemetry parser compiled successfully using MSVC.")
            
            # Clean up MSVC temporary obj/lib/exp files
            obj_file = PROJECT_ROOT / "telemetry_parser.obj"
            lib_file = LIB_DEST_PATH.with_suffix(".lib")
            exp_file = LIB_DEST_PATH.with_suffix(".exp")
            for f in (obj_file, lib_file, exp_file):
                if f.exists():
                    try:
                        os.remove(f)
                    except Exception as e:
                        logger.warning(f"Could not remove temporary compiler artifact {f}: {e}")
            return True
        except Exception as e:
            logger.warning(f"Compilation with MSVC failed: {e}")

    logger.warning("No suitable C++ compiler (g++, clang++, cl) found. Falling back to pure Python ingestion.")
    return False


def init_parser() -> bool:
    """Initializes the parser by compiling (if needed/possible) and loading the C++ library."""
    global _cpp_lib, CPP_AVAILABLE

    # Compile library if it does not exist
    if not LIB_DEST_PATH.exists():
        compile_cpp_library()

    if LIB_DEST_PATH.exists():
        try:
            _cpp_lib = ctypes.CDLL(str(LIB_DEST_PATH))
            # Define function signature
            _cpp_lib.parse_and_clip_json.argtypes = [
                ctypes.c_char_p,                 # json_str
                ctypes.POINTER(ctypes.c_char_p), # keys
                ctypes.POINTER(ctypes.c_double), # low_bounds
                ctypes.POINTER(ctypes.c_double), # high_bounds
                ctypes.c_int,                    # bounds_count
                ctypes.c_char_p,                 # out_json
                ctypes.c_int                     # out_len
            ]
            _cpp_lib.parse_and_clip_json.restype = ctypes.c_bool
            CPP_AVAILABLE = True
            logger.info("C++ telemetry parser loaded successfully via ctypes.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load compiled library: {e}. Falling back to Python.")
            _cpp_lib = None
            CPP_AVAILABLE = False
    else:
        CPP_AVAILABLE = False
    return False


def python_fallback_parse_and_clip(json_str: str, bounds: Dict[str, Any]) -> Dict[str, Any]:
    """Pure Python fallback for JSON parsing and outlier clipping."""
    data = json.loads(json_str)
    low_b = bounds.get("low", {})
    high_b = bounds.get("high", {})

    for k, v in data.items():
        if k in low_b and k in high_b and v is not None:
            try:
                orig_type = type(v)
                val = float(v)
                low = float(low_b[k])
                high = float(high_b[k])
                if val < low:
                    val = low
                elif val > high:
                    val = high
                data[k] = orig_type(val) if orig_type in (int, float) else val
            except (ValueError, TypeError):
                pass
    return data


def parse_and_clip(json_str: str, bounds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a telemetry event JSON string and clips numerical outliers.
    Uses C++ library if available, else falls back gracefully to Python.
    """
    if not bounds or "low" not in bounds or "high" not in bounds:
        return json.loads(json_str)

    if CPP_AVAILABLE and _cpp_lib is not None:
        try:
            keys_list = [k.encode("utf-8") for k in bounds["low"].keys()]
            keys_arr = (ctypes.c_char_p * len(keys_list))(*keys_list)
            
            low_vals = [float(bounds["low"][k]) for k in bounds["low"].keys()]
            low_arr = (ctypes.c_double * len(low_vals))(*low_vals)
            
            high_vals = [float(bounds["high"][k]) for k in bounds["low"].keys()]
            high_arr = (ctypes.c_double * len(high_vals))(*high_vals)

            # C++ outputs the clipped string into this buffer
            out_buf_len = max(len(json_str) * 3, 8192)
            out_buf = ctypes.create_string_buffer(out_buf_len)

            success = _cpp_lib.parse_and_clip_json(
                json_str.encode("utf-8"),
                keys_arr,
                low_arr,
                high_arr,
                len(keys_list),
                out_buf,
                out_buf_len
            )

            if success:
                return json.loads(out_buf.value.decode("utf-8"))
            else:
                logger.debug("C++ parsing failed or buffer too small; falling back to Python.")
        except Exception as e:
            logger.debug(f"Exception during C++ parse_and_clip call: {e}")

    # Graceful fallback
    return python_fallback_parse_and_clip(json_str, bounds)


# Initialize on import
init_parser()
