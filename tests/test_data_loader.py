"""Unit tests for NASA C-MAPSS data loading and validation."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import load_raw_txt, load_official_rul, DataValidationError
from src.config import ALL_COLUMNS, INDEX_COLS, SENSOR_COLS


def test_txt_loading_and_schema(tmp_path):
    """Test standard 26-column whitespace delimited loading."""
    # Create synthetic C-MAPSS data
    rows = []
    for eng in [1, 2]:
        for cyc in range(1, 11):
            settings = [0.0, 0.0, 100.0]
            sensors = [float(i + cyc) for i in range(1, 22)]
            row = [eng, cyc] + settings + sensors
            rows.append(" ".join(str(x) for x in row))

    sample_file = tmp_path / "sample_CMAPSS.txt"
    sample_file.write_text("\n".join(rows), encoding="utf-8")

    df, report = load_raw_txt(sample_file, export_csv=False)

    assert len(df) == 20
    assert list(df.columns) == ALL_COLUMNS
    assert df["engine_id"].dtype == int
    assert df["cycle"].dtype == int
    assert df["sensor_1"].dtype == float
    assert report["num_engines"] == 2
    assert report["missing_values"] == 0
    assert report["duplicate_engine_cycle_pairs"] == 0


def test_malformed_columns_error(tmp_path):
    """Test that rows with missing or excess columns raise DataValidationError."""
    malformed_content = "1 1 0.0 0.0 100.0 518.67 641.82\n"  # only 7 columns
    sample_file = tmp_path / "bad_columns.txt"
    sample_file.write_text(malformed_content, encoding="utf-8")

    with pytest.raises(DataValidationError, match="columns"):
        load_raw_txt(sample_file, export_csv=False)


def test_duplicate_engine_cycle_error(tmp_path):
    """Test that duplicate (engine_id, cycle) pairs are detected and raise an error."""
    rows = []
    # Two identical records for engine 1, cycle 1
    row_str = "1 1 " + " ".join(["10.0"] * 24)
    sample_file = tmp_path / "duplicate_rows.txt"
    sample_file.write_text(f"{row_str}\n{row_str}\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="duplicate"):
        load_raw_txt(sample_file, export_csv=False)


def test_invalid_engine_id_error(tmp_path):
    """Test that zero or negative engine IDs raise validation errors."""
    row_str = "0 1 " + " ".join(["10.0"] * 24)
    sample_file = tmp_path / "invalid_engine.txt"
    sample_file.write_text(row_str, encoding="utf-8")

    with pytest.raises(DataValidationError, match="non-positive engine IDs"):
        load_raw_txt(sample_file, export_csv=False)


def test_load_official_rul_file(tmp_path):
    """Test loading and validating official RUL ground-truth vectors."""
    rul_file = tmp_path / "RUL_test.txt"
    rul_file.write_text("112\n98\n69\n82\n", encoding="utf-8")

    ruls, meta = load_official_rul(rul_file)
    assert len(ruls) == 4
    assert meta["min"] == 69.0
    assert meta["max"] == 112.0
    assert ruls[0] == 112.0
