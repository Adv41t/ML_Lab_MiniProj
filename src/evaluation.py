"""Shared model evaluation module - Single Source of Truth for ENGINE HEALTH AI."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import RESULTS_DIR, MAINTENANCE_WARNING_THRESHOLD


def compute_nasa_scoring(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute official NASA PHM 2008 asymmetric prognostics score.
    
    Penalty function:
        d_i = y_pred_i - y_true_i
        penalty = exp(-d_i / 13) - 1   if d_i < 0 (early prediction)
        penalty = exp(d_i / 10) - 1    if d_i >= 0 (late prediction: penalizes late prediction more heavily)
    """
    diffs = y_pred - y_true
    penalties = np.where(
        diffs < 0,
        np.exp(-diffs / 13.0) - 1.0,
        np.exp(diffs / 10.0) - 1.0,
    )
    return float(np.sum(penalties))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    split_name: str = "test",
    late_life_threshold: float = MAINTENANCE_WARNING_THRESHOLD,
) -> Dict[str, Any]:
    """Calculate comprehensive regression and domain-specific prognostic metrics.
    
    Validates input validity, finite values, and non-empty arrays.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Shape mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)} elements."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty target and prediction arrays.")

    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains NaN or infinite values.")

    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred contains NaN or infinite values.")

    # Core Regression Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    # Handle single-sample or constant y_true for R2
    if len(y_true) > 1 and np.var(y_true) > 1e-9:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = 0.0

    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)
    median_ae = float(np.median(abs_errors))

    # Error quantiles
    quantiles = {
        "p25": float(np.percentile(abs_errors, 25)),
        "p50": float(np.percentile(abs_errors, 50)),
        "p75": float(np.percentile(abs_errors, 75)),
        "p90": float(np.percentile(abs_errors, 90)),
        "p95": float(np.percentile(abs_errors, 95)),
    }

    # Late-life evaluation (critical domain focus: true RUL <= threshold)
    late_mask = y_true <= late_life_threshold
    late_count = int(np.sum(late_mask))
    if late_count > 0:
        late_mae = float(mean_absolute_error(y_true[late_mask], y_pred[late_mask]))
        late_rmse = float(np.sqrt(mean_squared_error(y_true[late_mask], y_pred[late_mask])))
    else:
        late_mae = None
        late_rmse = None

    nasa_score = compute_nasa_scoring(y_true, y_pred)

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "split": split_name,
        "sample_count": len(y_true),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "median_ae": round(median_ae, 3),
        "nasa_score": round(nasa_score, 2),
        "max_positive_residual": round(float(np.max(residuals)), 3),
        "max_negative_residual": round(float(np.min(residuals)), 3),
        "error_quantiles": quantiles,
        "late_life": {
            "threshold": late_life_threshold,
            "sample_count": late_count,
            "late_life_mae": round(late_mae, 3) if late_mae is not None else None,
            "late_life_rmse": round(late_rmse, 3) if late_rmse is not None else None,
        },
    }

    return metrics


def save_evaluation_results(
    results: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    engine_ids: Optional[np.ndarray | list] = None,
    filename_prefix: str = "eval",
) -> Path:
    """Persist metrics JSON and prediction records to artifacts/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{filename_prefix}_metrics.json"

    # Save metrics JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save predictions npz
    npz_path = RESULTS_DIR / f"{filename_prefix}_predictions.npz"
    save_dict = {
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    }
    if engine_ids is not None:
        save_dict["engine_ids"] = np.asarray(engine_ids)

    np.savez_compressed(npz_path, **save_dict)
    return json_path
