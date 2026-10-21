"""Unit tests for inference validation and maintenance alert mapping."""

import pytest
import numpy as np
import pandas as pd

from src.inference import (
    map_rul_to_maintenance_status,
    validate_uploaded_csv,
    predict_single_cycle,
)
from src.config import BASELINE_SELECTED_SENSORS


def test_maintenance_status_thresholds():
    """Verify maintenance alert mapping at boundary conditions."""
    # Critical: <= 15
    res_crit = map_rul_to_maintenance_status(10.0)
    assert res_crit["status"] == "CRITICAL"
    assert res_crit["badge_color"] == "danger"

    res_bound_15 = map_rul_to_maintenance_status(15.0)
    assert res_bound_15["status"] == "CRITICAL"

    # Warning: 16 to 30
    res_warn = map_rul_to_maintenance_status(25.0)
    assert res_warn["status"] == "WARNING"
    assert res_warn["badge_color"] == "warning"

    res_bound_30 = map_rul_to_maintenance_status(30.0)
    assert res_bound_30["status"] == "WARNING"

    # Normal: > 30
    res_norm = map_rul_to_maintenance_status(45.0)
    assert res_norm["status"] == "NORMAL"
    assert res_norm["badge_color"] == "success"

    # Negative RUL handling
    res_neg = map_rul_to_maintenance_status(-2.5)
    assert res_neg["display_rul"] == 0.0
    assert res_neg["raw_rul"] == -2.5
    assert res_neg["status"] == "CRITICAL"


def test_validate_uploaded_csv():
    """Verify validation rules for user-uploaded inference CSV."""
    required = ["cycle", "sensor_2", "sensor_3"]

    # Valid CSV
    valid_df = pd.DataFrame({
        "cycle": [1, 2],
        "sensor_2": [642.0, 642.5],
        "sensor_3": [1589.0, 1590.0],
    })
    is_valid, errors, _ = validate_uploaded_csv(valid_df, required)
    assert is_valid is True
    assert len(errors) == 0

    # Missing column
    bad_df = pd.DataFrame({"cycle": [1], "sensor_2": [642.0]})
    is_valid, errors, _ = validate_uploaded_csv(bad_df, required)
    assert is_valid is False
    assert any("missing required" in e for e in errors)

    # Missing values (NaN)
    nan_df = pd.DataFrame({
        "cycle": [1, 2],
        "sensor_2": [642.0, np.nan],
        "sensor_3": [1589.0, 1590.0],
    })
    is_valid, errors, _ = validate_uploaded_csv(nan_df, required)
    assert is_valid is False
    assert any("missing (NaN)" in e for e in errors)

    # LSTM sequence length check
    short_df = pd.DataFrame({
        "cycle": list(range(1, 10)),
        "sensor_2": [642.0] * 9,
        "sensor_3": [1589.0] * 9,
    })
    is_valid, errors, _ = validate_uploaded_csv(short_df, required, is_lstm=True, sequence_length=30)
    assert is_valid is False
    assert any("at least 30" in e for e in errors)
