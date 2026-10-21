"""Unified inference engine for single-engine inspections and CSV upload predictions."""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch

from src.config import (
    MAINTENANCE_WARNING_THRESHOLD,
    MAINTENANCE_CRITICAL_THRESHOLD,
    SEQUENCE_LENGTH,
)
from src.preprocessing import Preprocessor


def map_rul_to_maintenance_status(predicted_rul: float) -> Dict[str, Any]:
    """Map continuous predicted RUL to actionable maintenance status alerts.
    
    Thresholds:
        - Critical (<= 15 cycles): Immediate inspection and maintenance required.
        - Warning (16 - 30 cycles): Component approaching end-of-life; schedule maintenance window.
        - Normal (> 30 cycles): Routine operational condition monitoring.
    """
    raw_rul = float(predicted_rul)
    display_rul = max(0.0, round(raw_rul, 1))

    if raw_rul <= MAINTENANCE_CRITICAL_THRESHOLD:
        status = "CRITICAL"
        action = "Urgent maintenance action required immediately. Remove engine from scheduled operations for overhaul."
        badge_color = "danger"
        severity_rank = 3
    elif raw_rul <= MAINTENANCE_WARNING_THRESHOLD:
        status = "WARNING"
        action = "Plan maintenance inspection within next 10-15 cycles. Order replacement parts and prepare shop visit."
        badge_color = "warning"
        severity_rank = 2
    else:
        status = "NORMAL"
        action = "Engine operating within standard health parameters. Continue routine condition monitoring."
        badge_color = "success"
        severity_rank = 1

    return {
        "raw_rul": round(raw_rul, 2),
        "display_rul": display_rul,
        "status": status,
        "action": action,
        "badge_color": badge_color,
        "severity_rank": severity_rank,
        "critical_threshold": MAINTENANCE_CRITICAL_THRESHOLD,
        "warning_threshold": MAINTENANCE_WARNING_THRESHOLD,
        "disclaimer": "These alert thresholds represent illustrative engineering guidelines for academic demonstration, not FAA/EASA-certified airworthiness limits.",
    }


def predict_single_cycle(
    model: Any,
    feature_vector: np.ndarray,
    preprocessor: Optional[Preprocessor] = None,
    feature_cols: Optional[List[str]] = None,
) -> float:
    """Predict RUL for a single feature vector using tree-based models."""
    fv = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)

    if preprocessor is not None and feature_cols is not None:
        temp_df = pd.DataFrame(fv, columns=feature_cols)
        scaled_df = preprocessor.transform(temp_df)
        fv = scaled_df[feature_cols].to_numpy(dtype=np.float32)

    pred = float(model.predict(fv)[0])
    return pred


def predict_lstm_sequence(
    model: Any,
    sequence_matrix: np.ndarray,
    device: Optional[str] = None,
) -> float:
    """Predict RUL for a single (sequence_length, num_features) window using PyTorch LSTM."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seq = np.asarray(sequence_matrix, dtype=np.float32)

    if len(seq.shape) == 2:
        seq = np.expand_dims(seq, axis=0)  # (1, seq_len, features)

    tensor = torch.tensor(seq, dtype=torch.float32).to(dev)
    model.eval()
    with torch.no_grad():
        out = model(tensor).cpu().numpy().ravel()[0]
    return float(out)


def validate_uploaded_csv(
    df: pd.DataFrame,
    required_features: List[str],
    is_lstm: bool = False,
    sequence_length: int = SEQUENCE_LENGTH,
) -> Tuple[bool, List[str], pd.DataFrame]:
    """Validate user-uploaded CSV file for inference.
    
    Checks:
        1. Required columns present
        2. Numeric casting and validity (no NaN/Inf)
        3. Sorting and engine history checks
        4. LSTM sequence length check
    """
    errors: List[str] = []
    df_clean = df.copy()

    # Column presence check
    missing_cols = [c for c in required_features if c not in df_clean.columns]
    if missing_cols:
        errors.append(f"Uploaded CSV is missing required sensor/feature columns: {missing_cols}")
        return False, errors, df_clean

    # Numeric cast check
    for col in required_features:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col])
        except Exception as e:
            errors.append(f"Column '{col}' could not be converted to numbers: {e}")

    # Missing value check
    nan_count = df_clean[required_features].isna().sum().sum()
    if nan_count > 0:
        errors.append(f"Found {nan_count} missing (NaN) values in required columns.")

    # Infinite value check
    inf_count = np.isinf(df_clean[required_features].to_numpy()).sum()
    if inf_count > 0:
        errors.append(f"Found {inf_count} infinite values in required columns.")

    if errors:
        return False, errors, df_clean

    # Cycle / Engine check if available
    if "engine_id" in df_clean.columns and "cycle" in df_clean.columns:
        df_clean = df_clean.sort_values(by=["engine_id", "cycle"]).reset_index(drop=True)
    elif "cycle" in df_clean.columns:
        df_clean = df_clean.sort_values(by="cycle").reset_index(drop=True)

    # LSTM specific check
    if is_lstm:
        if len(df_clean) < sequence_length:
            errors.append(
                f"LSTM sequence prediction requires at least {sequence_length} chronological cycles, "
                f"but uploaded CSV has only {len(df_clean)} rows."
            )
            return False, errors, df_clean

    return True, errors, df_clean
