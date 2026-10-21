"""Unit tests for engine-level splitting, scaling, and sequence building."""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import split_train_val_by_engine, Preprocessor, detect_constant_columns
from src.sequence_builder import build_all_sequences, build_test_final_sequences


def test_split_train_val_no_leakage():
    """Verify engine-aware split strictly separates engine IDs with zero cycle overlap."""
    records = []
    for eng in range(1, 11):
        for cyc in range(1, 21):
            records.append({"engine_id": eng, "cycle": cyc, "feat": float(eng * 10 + cyc)})

    df = pd.DataFrame(records)
    train_df, val_df, train_eids, val_eids = split_train_val_by_engine(df, val_ratio=0.2, seed=42)

    assert len(train_eids) == 8
    assert len(val_eids) == 2
    # Zero overlap
    assert set(train_eids).isdisjoint(set(val_eids))
    assert set(train_df["engine_id"].unique()).isdisjoint(set(val_df["engine_id"].unique()))


def test_preprocessor_fit_transform(tmp_path):
    """Verify preprocessor fits strictly on train and can be saved/loaded."""
    train_df = pd.DataFrame({
        "engine_id": [1, 1],
        "cycle": [1, 2],
        "s2": [10.0, 20.0],
        "s3": [100.0, 200.0],
    })
    test_df = pd.DataFrame({
        "engine_id": [2],
        "cycle": [1],
        "s2": [15.0],
        "s3": [150.0],
    })

    prep = Preprocessor(feature_cols=["s2", "s3"], scaler_type="minmax")
    train_scaled = prep.fit_transform(train_df)

    assert train_scaled["s2"].iloc[0] == 0.0
    assert train_scaled["s2"].iloc[1] == 1.0

    test_scaled = prep.transform(test_df)
    assert test_scaled["s2"].iloc[0] == 0.5
    assert test_scaled["s3"].iloc[0] == 0.5

    # Test serialization
    model_file = tmp_path / "prep.joblib"
    prep.save(model_file)
    loaded_prep = Preprocessor.load(model_file)
    assert loaded_prep.is_fitted
    assert loaded_prep.feature_cols == ["s2", "s3"]


def test_sequence_builder_engine_isolation():
    """Verify sliding sequences are strictly contained within individual engines."""
    records = []
    # Engine 1 has 5 cycles
    for c in range(1, 6):
        records.append({"engine_id": 1, "cycle": c, "s1": float(c), "RUL_clipped": 5.0 - c})
    # Engine 2 has 4 cycles
    for c in range(1, 5):
        records.append({"engine_id": 2, "cycle": c, "s1": float(c * 10), "RUL_clipped": 4.0 - c})

    df = pd.DataFrame(records)
    seq_len = 3

    X, y, meta = build_all_sequences(df, feature_cols=["s1"], sequence_length=seq_len, target_col="RUL_clipped")

    # Engine 1: 5 - 3 + 1 = 3 sequences
    # Engine 2: 4 - 3 + 1 = 2 sequences
    # Total = 5 sequences
    assert X.shape == (5, seq_len, 1)
    assert y.shape == (5,)

    # Verify target corresponds to the final cycle of the window
    # First sequence is Engine 1 cycles [1, 2, 3] -> final cycle 3, RUL_clipped is 2.0
    assert y[0] == 2.0
    assert meta[0]["engine_id"] == 1
    assert meta[0]["end_cycle"] == 3
