"""Unit tests for RUL target calculation and official test alignment."""

import pytest
import numpy as np
import pandas as pd

from src.rul_targets import compute_training_rul, align_official_test_rul


def test_compute_training_rul():
    """Verify run-to-failure linear and clipped RUL construction."""
    # Engine 1 has 10 cycles, Engine 2 has 20 cycles
    data = []
    for c in range(1, 11):
        data.append({"engine_id": 1, "cycle": c})
    for c in range(1, 21):
        data.append({"engine_id": 2, "cycle": c})

    df = pd.DataFrame(data)
    df_out, meta = compute_training_rul(df, rul_cap=8, binary_threshold=3)

    # Engine 1: max_cycle is 10
    # cycle 10: RUL = 0, RUL_clipped = 0, label_binary = 1
    # cycle 9: RUL = 1, RUL_clipped = 1, label_binary = 1
    # cycle 1: RUL = 9, RUL_clipped = 8, label_binary = 0
    e1_c10 = df_out[(df_out["engine_id"] == 1) & (df_out["cycle"] == 10)].iloc[0]
    assert e1_c10["RUL"] == 0
    assert e1_c10["RUL_clipped"] == 0
    assert e1_c10["label_binary"] == 1

    e1_c1 = df_out[(df_out["engine_id"] == 1) & (df_out["cycle"] == 1)].iloc[0]
    assert e1_c1["RUL"] == 9
    assert e1_c1["RUL_clipped"] == 8  # capped at 8
    assert e1_c1["label_binary"] == 0

    assert meta["total_engines"] == 2
    assert meta["total_records"] == 30


def test_align_official_test_rul():
    """Verify alignment of official test ground-truth values at the final observed cycle."""
    data = []
    # Test engine 1 observed up to cycle 15
    for c in range(1, 16):
        data.append({"engine_id": 1, "cycle": c})
    # Test engine 2 observed up to cycle 20
    for c in range(1, 21):
        data.append({"engine_id": 2, "cycle": c})

    df = pd.DataFrame(data)
    official_ruls = np.array([40.0, 15.0])  # Engine 1 has 40 cycles left, Engine 2 has 15 cycles left

    test_full, test_final, meta = align_official_test_rul(df, official_ruls, rul_cap=125)

    assert len(test_final) == 2
    assert test_final[test_final["engine_id"] == 1]["RUL"].iloc[0] == 40.0
    assert test_final[test_final["engine_id"] == 2]["RUL"].iloc[0] == 15.0

    # Check cycle-by-cycle reconstruction
    # For Engine 1 at cycle 1: RUL = 40 + (15 - 1) = 54
    e1_c1 = test_full[(test_full["engine_id"] == 1) & (test_full["cycle"] == 1)].iloc[0]
    assert e1_c1["RUL"] == 54.0
