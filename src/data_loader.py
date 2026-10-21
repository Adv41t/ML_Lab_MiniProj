"""Data loading, whitespace parsing, schema validation, and CSV export for NASA C-MAPSS."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import (
    ALL_COLUMNS,
    INDEX_COLS,
    SETTING_COLS,
    SENSOR_COLS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_TXT_NAME,
    TEST_TXT_NAME,
    RUL_TXT_NAME,
)


class DataValidationError(Exception):
    """Exception raised when data fails schema or integrity validation."""
    pass


def find_file_path(filename: str, preferred_dir: Optional[Path] = None) -> Optional[Path]:
    """Look for a dataset file in preferred_dir, RAW_DATA_DIR, or project root."""
    search_dirs = [preferred_dir, RAW_DATA_DIR, Path("."), Path(__file__).resolve().parent.parent]
    for d in search_dirs:
        if d is not None:
            cand = d / filename
            if cand.exists() and cand.is_file():
                return cand.resolve()
    return None


def load_raw_txt(
    file_path: Path | str,
    export_csv: bool = True,
    csv_out_path: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load, parse, validate, and type-cast NASA C-MAPSS whitespace-delimited TXT file.
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Validated DataFrame and validation report.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    # Step 1: Whitespace separation parse
    try:
        raw_df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    except Exception as e:
        raise DataValidationError(f"Failed to parse whitespace separated file {path.name}: {e}")

    # Check for empty file
    if raw_df.empty:
        raise DataValidationError(f"Data file {path.name} is empty.")

    # Some versions have trailing empty columns if there's trailing whitespace
    # We validate meaningful column counts
    if raw_df.shape[1] > 26:
        # Check if excess columns are completely NaN or whitespace
        excess = raw_df.iloc[:, 26:]
        if excess.isna().all().all():
            raw_df = raw_df.iloc[:, :26]
        else:
            raise DataValidationError(
                f"File {path.name} contains {raw_df.shape[1]} columns, expected exactly 26 columns."
            )
    elif raw_df.shape[1] < 26:
        raise DataValidationError(
            f"File {path.name} contains only {raw_df.shape[1]} columns, expected exactly 26 columns."
        )

    # Assign exact column schema
    raw_df.columns = ALL_COLUMNS

    # Step 2: Validate Data Integrity & Quality
    report: Dict[str, Any] = {
        "source_file": str(path.name),
        "total_records": len(raw_df),
        "total_columns": len(raw_df.columns),
        "columns": ALL_COLUMNS,
        "missing_values": int(raw_df.isna().sum().sum()),
        "non_finite_values": int(np.isinf(raw_df.to_numpy()).sum()),
        "duplicate_engine_cycle_pairs": 0,
        "invalid_engine_ids": 0,
        "invalid_cycles": 0,
        "num_engines": 0,
        "min_cycle": 0,
        "max_cycle": 0,
        "avg_cycles_per_engine": 0.0,
        "is_chronologically_sorted": False,
        "status": "VALID",
        "warnings": [],
    }

    if report["missing_values"] > 0:
        report["warnings"].append(f"Found {report['missing_values']} missing (NaN) values.")
        raise DataValidationError(f"Data contains {report['missing_values']} missing values.")

    if report["non_finite_values"] > 0:
        report["warnings"].append(f"Found {report['non_finite_values']} non-finite values.")
        raise DataValidationError(f"Data contains {report['non_finite_values']} infinite values.")

    # Convert types explicitly
    try:
        raw_df["engine_id"] = raw_df["engine_id"].astype(int)
        raw_df["cycle"] = raw_df["cycle"].astype(int)
        for col in SETTING_COLS + SENSOR_COLS:
            raw_df[col] = raw_df[col].astype(float)
    except Exception as e:
        raise DataValidationError(f"Failed to cast columns to proper numeric types: {e}")

    # Check invalid engine identifiers and cycle numbers
    invalid_engines = (raw_df["engine_id"] <= 0).sum()
    invalid_cycles = (raw_df["cycle"] <= 0).sum()
    report["invalid_engine_ids"] = int(invalid_engines)
    report["invalid_cycles"] = int(invalid_cycles)

    if invalid_engines > 0:
        raise DataValidationError(f"Found {invalid_engines} non-positive engine IDs.")
    if invalid_cycles > 0:
        raise DataValidationError(f"Found {invalid_cycles} non-positive cycle values.")

    # Check duplicate (engine_id, cycle) pairs
    dup_count = int(raw_df.duplicated(subset=["engine_id", "cycle"]).sum())
    report["duplicate_engine_cycle_pairs"] = dup_count
    if dup_count > 0:
        report["warnings"].append(f"Detected {dup_count} duplicate engine-cycle records.")
        raise DataValidationError(f"Detected {dup_count} duplicate engine-cycle records.")

    # Check sorting
    is_sorted = raw_df["engine_id"].is_monotonic_increasing and (
        raw_df.groupby("engine_id")["cycle"].apply(lambda s: s.is_monotonic_increasing).all()
    )
    report["is_chronologically_sorted"] = bool(is_sorted)
    if not is_sorted:
        report["warnings"].append("Records were not strictly sorted; sorting by engine_id and cycle.")
        raw_df = raw_df.sort_values(by=["engine_id", "cycle"]).reset_index(drop=True)

    # Engine statistics
    report["num_engines"] = int(raw_df["engine_id"].nunique())
    report["min_cycle"] = int(raw_df["cycle"].min())
    report["max_cycle"] = int(raw_df["cycle"].max())
    report["avg_cycles_per_engine"] = float(round(len(raw_df) / report["num_engines"], 2))

    # Export to processed CSV if requested
    if export_csv:
        if csv_out_path is None:
            out_name = path.stem + ".csv"
            csv_out_path = PROCESSED_DATA_DIR / out_name
        else:
            csv_out_path = Path(csv_out_path)

        csv_out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(csv_out_path, index=False)
        report["exported_csv"] = str(csv_out_path)

    return raw_df, report


def load_official_rul(rul_file_path: Optional[Path | str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load and validate the official test ground-truth RUL file (e.g. RUL_FD001.txt)."""
    if rul_file_path is None:
        path = find_file_path(RUL_TXT_NAME)
        if path is None:
            raise FileNotFoundError(f"Official RUL file '{RUL_TXT_NAME}' could not be found.")
    else:
        path = Path(rul_file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Official RUL file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) == 0:
        raise DataValidationError("RUL file is empty.")

    try:
        rul_vals = np.array([float(x) for x in lines], dtype=float)
    except Exception as e:
        raise DataValidationError(f"Failed to parse numeric RUL values: {e}")

    if np.isnan(rul_vals).any() or np.isinf(rul_vals).any():
        raise DataValidationError("Official RUL values contain NaN or Inf.")

    if (rul_vals < 0).any():
        raise DataValidationError("Official RUL values contain negative numbers.")

    metadata = {
        "file": str(path.name),
        "count": len(rul_vals),
        "min": float(rul_vals.min()),
        "max": float(rul_vals.max()),
        "mean": float(round(rul_vals.mean(), 2)),
        "median": float(np.median(rul_vals)),
    }
    return rul_vals, metadata
