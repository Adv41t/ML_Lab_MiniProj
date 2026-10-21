"""Data Pipeline & Preprocessing Overview - Minimal & Direct."""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.config import (
    BASELINE_SELECTED_SENSORS,
    CONSTANT_SENSORS_FD001,
    RUL_CAP,
    PROCESSED_DATA_DIR,
)
from src.rul_targets import compute_training_rul
from src.ui_helpers import apply_custom_theme, render_header

st.set_page_config(page_title="Data Preprocessing | ENGINE HEALTH AI", page_icon="🔄", layout="wide")
apply_custom_theme()

render_header(
    title="Data Pipeline & Preprocessing",
    subtitle="Overview of Data Ingestion, Integrity Verification, Feature Selection, and RUL Construction",
    tag="Data Engineering Pipeline",
)

train_csv_path = PROCESSED_DATA_DIR / "train_FD001.csv"
if not train_csv_path.exists():
    st.error("Processed training data not found. Please verify dataset files.")
    st.stop()

train_df = pd.read_csv(train_csv_path)

if "RUL_clipped" not in train_df.columns or "RUL" not in train_df.columns:
    train_df, _ = compute_training_rul(train_df, rul_cap=RUL_CAP)
    train_df.to_csv(train_csv_path, index=False)

# What was done and How - Minimal 6-Step Summary
st.markdown("### 🛠️ Preprocessing Pipeline: What Was Done & How")

st.markdown(
    """
    1. **Raw TXT Ingestion & Schema Assignment:**
       - **What:** Parsed whitespace-separated NASA C-MAPSS text files into structured tables.
       - **How:** Read using whitespace delimiter (`\\s+`) and mapped to the standard 26-column schema: `engine_id`, `cycle`, 3 operational settings, and 21 sensors.

    2. **Data Integrity & Quality Verification:**
       - **What:** Verified that the dataset contains no corrupted or malformed entries.
       - **How:** Checked and confirmed **0 missing (NaN) values**, **0 non-finite/infinite values**, and **0 duplicate `(engine_id, cycle)` pairs**. Ensured strictly chronological ordering.

    3. **RUL Target Construction (Piecewise Linear):**
       - **What:** Formulated the continuous Remaining Useful Life (RUL) regression target.
       - **How:** Computed $\\text{RUL} = \\text{max\\_cycle} - \\text{cycle}$ for each engine, capped at **125 cycles** ($\\min(\\text{RUL}, 125)$) to model the initial non-degrading operational plateau.

    4. **Constant Sensor Removal & Feature Selection:**
       - **What:** Pruned non-informative variables to reduce noise and dimensionality.
       - **How:** Eliminated 7 constant features with near-zero standard deviation: `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19`. Retained the 10 active degradation sensors (`sensor_2`, `3`, `4`, `7`, `8`, `9`, `15`, `17`, `20`, `21`) together with `cycle` (11 features total).

    5. **Data Leakage Prevention & Scaling:**
       - **What:** Ensured no training information contaminates validation or test evaluations.
       - **How:** Partitioned engines into 80 training and 20 validation engines (zero engine overlap). Fitted `MinMaxScaler` strictly on training engines, and applied the frozen transformation to validation and test data.

    6. **Sliding-Window Sequence Generation (LSTM):**
       - **What:** Formatted sequential time-series data for deep recurrent modeling.
       - **How:** Generated 30-cycle sliding windows ($W = 30$) strictly within individual engines. Aligned the RUL target with the final cycle of each window.
    """
)

st.markdown("---")

# Compact Summary Table
st.markdown("### 📊 Pipeline Summary Audit")
col_summary, col_preview = st.columns([1, 1.2])

with col_summary:
    summary_data = [
        {"Pipeline Step": "Raw Records Parsed", "Specification / Count": "20,631 records (26 columns)"},
        {"Pipeline Step": "Fleet Size", "Specification / Count": "100 run-to-failure engines"},
        {"Pipeline Step": "Missing / Corrupt Values", "Specification / Count": "0 (Verified)"},
        {"Pipeline Step": "Features Retained", "Specification / Count": "10 sensors + cycle (11 features)"},
        {"Pipeline Step": "Constant Sensors Dropped", "Specification / Count": "7 non-informative sensors"},
        {"Pipeline Step": "RUL Target Upper Cap", "Specification / Count": "125 cycles (Piecewise linear)"},
        {"Pipeline Step": "Train / Validation Split", "Specification / Count": "80 engines / 20 engines (No leakage)"},
        {"Pipeline Step": "LSTM Sequences (W=30)", "Specification / Count": "17,631 3D sequential tensors"},
    ]
    st.table(pd.DataFrame(summary_data))

with col_preview:
    st.markdown("**Processed Telemetry Preview (First 5 Rows with Targets):**")
    preview_cols = [c for c in ["engine_id", "cycle", "sensor_2", "sensor_3", "sensor_4", "sensor_7", "RUL_clipped"] if c in train_df.columns]
    st.dataframe(train_df[preview_cols].head(5), use_container_width=True)

    # Simple Direct Download Button
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    csv_bytes = train_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Clean Processed CSV (train_FD001.csv)",
        data=csv_bytes,
        file_name="train_FD001_processed.csv",
        mime="text/csv",
        use_container_width=True,
    )
