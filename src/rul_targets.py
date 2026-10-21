"""RUL target construction for NASA C-MAPSS training and test datasets."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.config import RUL_CAP, MAINTENANCE_WARNING_THRESHOLD


def compute_training_rul(
    df: pd.DataFrame,
    rul_cap: int = RUL_CAP,
    binary_threshold: int = MAINTENANCE_WARNING_THRESHOLD,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute run-to-failure Remaining Useful Life (RUL) for training engines.
    
    Formula:
        RUL = max_observed_cycle_for_engine - current_cycle
        RUL_clipped = min(RUL, rul_cap)
        label_binary = 1 if RUL <= binary_threshold else 0 (Maintenance Needed vs Safe)
    """
    df_out = df.copy()

    # Calculate max cycle per engine
    max_cycles = df_out.groupby("engine_id")["cycle"].max().to_dict()
    df_out["max_cycle"] = df_out["engine_id"].map(max_cycles)

    # Compute raw RUL
    df_out["RUL"] = df_out["max_cycle"] - df_out["cycle"]

    # Compute clipped piecewise linear RUL
    df_out["RUL_clipped"] = np.minimum(df_out["RUL"], rul_cap)

    # Historical binary maintenance label
    df_out["label_binary"] = (df_out["RUL"] <= binary_threshold).astype(int)

    metadata: Dict[str, Any] = {
        "rul_cap": rul_cap,
        "binary_threshold": binary_threshold,
        "total_engines": int(df_out["engine_id"].nunique()),
        "total_records": len(df_out),
        "min_raw_rul": int(df_out["RUL"].min()),
        "max_raw_rul": int(df_out["RUL"].max()),
        "mean_raw_rul": float(round(df_out["RUL"].mean(), 2)),
        "mean_clipped_rul": float(round(df_out["RUL_clipped"].mean(), 2)),
        "maintenance_needed_count": int(df_out["label_binary"].sum()),
        "safe_count": int((df_out["label_binary"] == 0).sum()),
        "maintenance_ratio": float(round(df_out["label_binary"].mean(), 4)),
    }

    return df_out, metadata


def align_official_test_rul(
    test_df: pd.DataFrame,
    official_rul_vector: np.ndarray,
    rul_cap: int = RUL_CAP,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Align official ground truth RUL with test engines.
    
    In the NASA C-MAPSS FD001 test set, each engine's degradation is truncated
    before failure. The official RUL vector specifies the remaining cycles at the
    FINAL observed cycle for each engine.
    
    Returns:
        Tuple:
            - test_df_full: Full test DataFrame with reconstructed cycle-by-cycle RUL.
            - test_final_cycles: Exactly 1 row per test engine at its maximum observed cycle.
            - metadata: Alignment summary.
    """
    test_out = test_df.copy()
    num_test_engines = test_out["engine_id"].nunique()

    if len(official_rul_vector) != num_test_engines:
        raise ValueError(
            f"Official RUL vector length ({len(official_rul_vector)}) does not match "
            f"number of unique test engines ({num_test_engines})."
        )

    # Extract max observed test cycle per engine
    test_max_cycles = test_out.groupby("engine_id")["cycle"].max().reset_index()
    test_max_cycles.rename(columns={"cycle": "max_test_cycle"}, inplace=True)
    test_max_cycles.sort_values(by="engine_id", inplace=True)

    # Map engine_id (1-indexed) to official final RUL
    rul_map = {
        int(engine_id): float(official_rul_vector[idx])
        for idx, engine_id in enumerate(test_max_cycles["engine_id"])
    }

    test_out["max_test_cycle"] = test_out["engine_id"].map(
        dict(zip(test_max_cycles["engine_id"], test_max_cycles["max_test_cycle"]))
    )
    test_out["official_final_rul"] = test_out["engine_id"].map(rul_map)

    # Reconstruct true RUL at every prior cycle
    # At cycle c: RUL(c) = official_final_rul + (max_test_cycle - c)
    test_out["RUL"] = test_out["official_final_rul"] + (test_out["max_test_cycle"] - test_out["cycle"])
    test_out["RUL_clipped"] = np.minimum(test_out["RUL"], rul_cap)
    test_out["label_binary"] = (test_out["RUL"] <= MAINTENANCE_WARNING_THRESHOLD).astype(int)

    # Final cycles: exactly 1 row per test engine at cycle == max_test_cycle
    test_final_cycles = (
        test_out[test_out["cycle"] == test_out["max_test_cycle"]]
        .sort_values(by="engine_id")
        .reset_index(drop=True)
    )

    metadata: Dict[str, Any] = {
        "num_test_engines": num_test_engines,
        "total_test_records": len(test_out),
        "rul_cap": rul_cap,
        "official_rul_min": float(official_rul_vector.min()),
        "official_rul_max": float(official_rul_vector.max()),
        "official_rul_mean": float(round(official_rul_vector.mean(), 2)),
        "aligned_final_cycles_count": len(test_final_cycles),
    }

    return test_out, test_final_cycles, metadata
