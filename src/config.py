"""Configuration settings and constants for ENGINE HEALTH AI."""

from pathlib import Path
from typing import List, Dict

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset filenames
TRAIN_TXT_NAME = "train_FD001.txt"
TEST_TXT_NAME = "test_FD001.txt"
RUL_TXT_NAME = "RUL_FD001.txt"

# Exact 26-column schema for NASA C-MAPSS
INDEX_COLS = ["engine_id", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Detailed sensor descriptions based on NASA technical documentation
SENSOR_METADATA: Dict[str, Dict[str, str]] = {
    "sensor_1": {"name": "T2", "desc": "Total temperature at fan inlet", "unit": "°R"},
    "sensor_2": {"name": "T24", "desc": "Total temperature at LPC outlet", "unit": "°R"},
    "sensor_3": {"name": "T30", "desc": "Total temperature at HPC outlet", "unit": "°R"},
    "sensor_4": {"name": "T50", "desc": "Total temperature at LPT outlet", "unit": "°R"},
    "sensor_5": {"name": "P2", "desc": "Pressure at fan inlet", "unit": "psia"},
    "sensor_6": {"name": "P15", "desc": "Total pressure in bypass-duct", "unit": "psia"},
    "sensor_7": {"name": "P30", "desc": "Total pressure at HPC outlet", "unit": "psia"},
    "sensor_8": {"name": "Nf", "desc": "Physical fan speed", "unit": "rpm"},
    "sensor_9": {"name": "Nc", "desc": "Physical core speed", "unit": "rpm"},
    "sensor_10": {"name": "epr", "desc": "Engine pressure ratio (P50/P2)", "unit": "--"},
    "sensor_11": {"name": "Ps30", "desc": "Static pressure at HPC outlet", "unit": "psia"},
    "sensor_12": {"name": "phi", "desc": "Ratio of fuel flow to Ps30", "unit": "pps/psi"},
    "sensor_13": {"name": "NRf", "desc": "Corrected fan speed", "unit": "rpm"},
    "sensor_14": {"name": "NRc", "desc": "Corrected core speed", "unit": "rpm"},
    "sensor_15": {"name": "BPR", "desc": "Bypass Ratio", "unit": "--"},
    "sensor_16": {"name": "farB", "desc": "Burner fuel-air ratio", "unit": "--"},
    "sensor_17": {"name": "htBleed", "desc": "Bleed Enthalpy", "unit": "--"},
    "sensor_18": {"name": "Nf_dmd", "desc": "Demanded fan speed", "unit": "rpm"},
    "sensor_19": {"name": "PCNfR_dmd", "desc": "Demanded corrected fan speed", "unit": "rpm"},
    "sensor_20": {"name": "W31", "desc": "HPT coolant bleed", "unit": "lbm/s"},
    "sensor_21": {"name": "W32", "desc": "LPT coolant bleed", "unit": "lbm/s"},
}

# Constant or near-constant sensors in FD001 (std < 1e-5)
CONSTANT_SENSORS_FD001 = [
    "setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

# Baseline selected sensors documented in Phase 1 report (10 sensors + cycle = 11 features)
BASELINE_SELECTED_SENSORS = [
    "cycle",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

# All informative sensors in FD001 (14 non-constant sensors + setting_1 + setting_2 + cycle)
ALL_INFORMATIVE_FEATURES = [
    "cycle",
    "setting_1",
    "setting_2",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

# RUL and maintenance thresholds
RUL_CAP = 125  # Piecewise linear upper cap (standard C-MAPSS benchmark)
MAINTENANCE_WARNING_THRESHOLD = 30  # Cycles remaining
MAINTENANCE_CRITICAL_THRESHOLD = 15  # Cycles remaining

# Data splitting & sequences
TRAIN_VAL_ENGINE_SPLIT_RATIO = 0.2  # 80 engines train, 20 engines val
SEQUENCE_LENGTH = 30  # LSTM window size
RANDOM_SEED = 42

# UI theme colors
THEME_COLORS = {
    "bg_dark": "#0B1120",
    "card_bg": "#1E293B",
    "card_border": "#334155",
    "primary": "#38BDF8",  # Sky cyan
    "accent": "#818CF8",   # Indigo
    "success": "#10B981",  # Emerald
    "warning": "#F59E0B",  # Amber
    "danger": "#EF4444",   # Crimson
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8",
}
