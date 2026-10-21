# ENGINE HEALTH AI
### ML-Based Condition Monitoring and Remaining Life Prediction of Turbofan Engines

---

## 1. Project Overview

**ENGINE HEALTH AI** is an end-to-end, academic machine learning framework and interactive Streamlit engineering dashboard developed for prognostics and health management (PHM) of commercial turbofan engines. Using the benchmark **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset, the system models multi-channel sensor degradation dynamics across successive operational flight cycles to accurately estimate Remaining Useful Life (RUL) and deliver proactive, safety-critical maintenance alerts.

The system incorporates data ingestion from raw whitespace-delimited telemetry files, strict schema verification, engine-isolated validation splitting to prevent data leakage, and comparative benchmarking of four machine learning architectures:
1. **Random Forest Regressor** (Ensemble Bootstrap Aggregation)
2. **XGBoost Regressor** (Extreme Gradient Boosting with Regularized Objective)
3. **LightGBM Regressor** (Histogram-Based Leaf-Wise Gradient Boosting)
4. **LSTM Regressor** (Deep Recurrent Neural Network with 30-Cycle Sliding Windows)

### Academic Context
- **Institution:** School of Computer Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education (MAHE)
- **Program:** B.Tech. in Computer Science & Engineering (AI & ML), V Semester
- **Project Team:**
  - Advait Balachandar (Reg No: 240962600) — Evaluation Framework, Explainability (SHAP), Integration, UI/UX
  - Madaka Shreyaj Naidu (Reg No: 240962594) — Temporal Sequence Engineering, PyTorch LSTM Deep Architecture
  - Dipen Batharaju (Reg No: 240962574) — Raw Data Ingestion, XGBoost & LightGBM Gradient Boosting Pipelines
- **Academic Year:** 2026

---

## 2. Key Objectives & Features

- **End-to-End Pipeline:** Seamless execution from raw space-delimited text records (`train_FD001.txt`, `test_FD001.txt`) to verified CSVs, feature scaling, 3D tensor sequence generation, model training, evaluation, and interactive inference.
- **Strict Data Leakage Prevention:** Engine-aware train/validation splitting ensures no engine's operating cycles ever span both partitions. All feature scalers are fitted strictly on training data.
- **Common Single-Source Evaluation Framework:** Standardized regression metrics (MAE, RMSE, $R^2$, Median AE, Error Quantiles) and domain-specific prognostics metrics (Late-Life MAE on engines with $RUL \le 30$ cycles, and the official NASA asymmetric penalty score).
- **Interactive Multi-Mode Inference:**
  - **Mode A (Fleet Inspection):** Select any test engine (1 to 100) to inspect its operational cycle history, predict RUL at its final observed cycle, review local SHAP explanations, and observe real-time maintenance recommendations.
  - **Mode B (CSV Telemetry Ingestion):** Upload custom CSV sensor readings; the application validates column schemas, numerical integrity, and sequence length requirements.
- **Historical Baseline Context:** Faithfully reproduces the Phase 1 Random Forest binary classifier (95.98% reproduced vs. 96.05% reported accuracy), while explicitly documenting why classification accuracy cannot be compared to continuous regression cycle errors.

---

## 3. Application Navigation & Portals

The Streamlit dashboard features a dark navy/slate engineering aesthetic (`#0B1120`, `#1E293B`, `#38BDF8`) with responsive navigation across 6 dedicated pages:

```
app.py (Landing Dashboard)
├── pages/1_Random_Forest.py     -> Random Forest Regressor & Phase 1 Baseline
├── pages/2_XGBoost.py           -> XGBoost Regressor & Regularized Gradient Boosting
├── pages/3_LightGBM.py          -> LightGBM Regressor & Fast Leaf-Wise Tree Growth
├── pages/4_LSTM.py              -> PyTorch LSTM Deep Sequential Network & Loss Curves
├── pages/5_Admin_Analytics.py   -> Benchmark Leaderboard, Multi-Model Overlays & Recommendation
└── pages/6_Data_Preprocessing.py -> Educational Data Engineering Pipeline & Live Audits
```

---

## 4. System Architecture

```
                    ┌───────────────────────────────────────────────┐
                    │ NASA C-MAPSS Raw Whitespace Telemetry Files   │
                    │   (train_FD001.txt, test_FD001.txt, RUL.txt)  │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Data Ingestion & Strict 26-Column Validation  │
                    │   (data_loader.py: check types, NaNs, dups)   │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Target Construction & Sensor Pruning          │
                    │   - RUL = max_cycle - cycle (Capped at 125)   │
                    │   - Drop 7 zero-variance sensors (std < 1e-5) │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Engine-Aware Split (80 Train / 20 Val Engines)│
                    │ MinMaxScaler fitted strictly on Training Set  │
                    └───────────────┬───────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│ Tabular Representation        │               │ 3D Sliding-Window Tensors     │
│ Shape: (N, 11 features)       │               │ Shape: (N, 30 cycles, 11 feat)│
└───────┬───────────────┬───────┘               └───────────────┬───────────────┘
        │               │                                       │
        ▼               ▼                                       ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐      ┌──────────────┐
│Random Forest │ │   XGBoost    │ │   LightGBM   │      │ PyTorch LSTM │
│  Regressor   │ │  Regressor   │ │  Regressor   │      │  Regressor   │
└───────┬──────┘ └──────┬───────┘ └──────┬───────┘      └──────┬───────┘
        │               │                │                     │
        └───────────────┼────────────────┴─────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Common Model Evaluation Framework (Single Source of Truth)                    │
│   - Metrics: MAE, RMSE, R², Late-Life MAE (≤30), NASA Asymmetric Score        │
│   - Evaluated on Official 100 Test Engines against RUL_FD001.txt              │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Streamlit Interactive Engineering Dashboard                                  │
│   - Landing Page Status KPIs, Primary Model Portals, Real-Time Inference      │
│   - Admin Analytics Leaderboard, Residual Plots, SHAP Local Explanations      │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Dataset Specifications & Ingestion Schema

The dataset used is **NASA C-MAPSS FD001**:
- **Operating Conditions:** Single operating condition (Sea Level)
- **Fault Modes:** High-Pressure Compressor (HPC) degradation
- **Fleet Size:** 100 training engines (run to complete failure), 100 test engines (truncated before failure)
- **Training Records:** 20,631 cycles
- **Test Records:** 13,096 cycles
- **Official Test Ground Truth:** `RUL_FD001.txt` containing exactly 100 integer values corresponding to the remaining operational cycles at the final recorded cycle of each test engine.

### Exact 26-Column Schema

| Index | Column Name | Type | Physical Interpretation | Status in FD001 |
|---|---|---|---|---|
| 1 | `engine_id` | int | Engine unit identifier (1 to 100) | Primary Key |
| 2 | `cycle` | int | Operational flight cycle count (time step) | Primary Key |
| 3 | `setting_1` | float | Flight altitude / resolver setting | Informative |
| 4 | `setting_2` | float | Mach flight speed setting | Informative |
| 5 | `setting_3` | float | Sea-level throttle condition (100.0) | Dropped (Constant) |
| 6 | `sensor_1` | float | Fan inlet total temperature ($T_2$) | Dropped (Constant) |
| 7 | `sensor_2` | float | LPC outlet total temperature ($T_{24}$) | Retained |
| 8 | `sensor_3` | float | HPC outlet total temperature ($T_{30}$) | Retained |
| 9 | `sensor_4` | float | LPT outlet total temperature ($T_{50}$) | Retained |
| 10 | `sensor_5` | float | Fan inlet pressure ($P_2$) | Dropped (Constant) |
| 11 | `sensor_6` | float | Bypass-duct total pressure ($P_{15}$) | Informative |
| 12 | `sensor_7` | float | HPC outlet total pressure ($P_{30}$) | Retained |
| 13 | `sensor_8` | float | Physical fan speed ($N_f$) | Retained |
| 14 | `sensor_9` | float | Physical core speed ($N_c$) | Retained |
| 15 | `sensor_10`| float | Engine pressure ratio EPR ($P_{50}/P_2$) | Dropped (Constant) |
| 16 | `sensor_11`| float | HPC outlet static pressure ($P_{s30}$) | Informative |
| 17 | `sensor_12`| float | Ratio of fuel flow to $P_{s30}$ ($\phi$) | Informative |
| 18 | `sensor_13`| float | Corrected fan speed ($NR_f$) | Informative |
| 19 | `sensor_14`| float | Corrected core speed ($NR_c$) | Informative |
| 20 | `sensor_15`| float | Bypass ratio ($BPR$) | Retained |
| 21 | `sensor_16`| float | Burner fuel-air ratio ($farB$) | Dropped (Constant) |
| 22 | `sensor_17`| float | Bleed enthalpy ($htBleed$) | Retained |
| 23 | `sensor_18`| float | Demanded fan speed ($Nf_{dmd}$) | Dropped (Constant) |
| 24 | `sensor_19`| float | Demanded corrected fan speed | Dropped (Constant) |
| 25 | `sensor_20`| float | HPT coolant bleed ($W_{31}$) | Retained |
| 26 | `sensor_21`| float | LPT coolant bleed ($W_{32}$) | Retained |

---

## 6. RUL Target Construction & Normalization

1. **Continuous Run-to-Failure Target:**
   For training engine $i$ at operating cycle $c$:
   $$\text{RUL}_{i, c} = C_{i, \max} - c$$
2. **Piecewise Linear Degradation Plateau (RUL Cap):**
   In modern gas turbine engines, measurable degradation does not begin immediately on cycle 1. Following benchmark literature (Heimes, 2008; Saxena et al., 2008), an upper bound is placed at 125 cycles:
   $$\text{RUL}_{i, c}^{\text{clipped}} = \min(\text{RUL}_{i, c}, 125)$$
3. **Official Test Ground-Truth Alignment:**
   Test engines are stopped prematurely before failure. Ground-truth values in `RUL_FD001.txt` define the true RUL at each test engine's maximum observed cycle $C_{i, \text{test\_max}}$. Models are evaluated against these 100 ground-truth targets.

---

## 7. Experimental Results & Comparative Leaderboard

All four models were trained, evaluated, and serialized in the local environment. Evaluation was conducted on the **official 100 test engines** against verified reference targets:

| Model | Category | Val MAE | Val RMSE | Test MAE | Test RMSE | Test $R^2$ | Late-Life MAE ($\le 30$) | NASA Score | Train Time | Artifact Size |
|---|---|---|---|---|---|---|---|---|---|---|
| **LSTM Regressor** | Deep Recurrent | **8.93** | **12.48** | **9.82** | **13.64** | **0.8842** | **2.30** | **324.33** | 41.3 s | 0.13 MB |
| **XGBoost Regressor** | Gradient Boosted | 10.01 | 15.11 | 11.99 | 16.87 | 0.8229 | 8.81 | 599.77 | 0.28 s | 0.41 MB |
| **LightGBM Regressor** | Leaf-Wise Boosted| 10.06 | 15.26 | 12.19 | 16.96 | 0.8209 | 9.25 | 669.19 | **0.12 s** | 0.41 MB |
| **Random Forest Regressor** | Bagging Ensemble| 10.34 | 15.35 | 12.55 | 17.47 | 0.8099 | 9.32 | 592.34 | 0.24 s | 35.06 MB |

---

## 8. Empirical Findings & Model Selection Policy

### Key Analytical Takeaways
1. **Superiority of Temporal Modeling (LSTM Winner):**
   The PyTorch LSTM model achieves the lowest Test MAE (**9.82 cycles**), representing an **18.0% relative improvement over XGBoost** and **21.7% over Random Forest**. Its performance in late-life operation ($RUL \le 30$) is particularly remarkable: **2.30 cycles MAE**, compared to ~9.0 cycles for tree models.
2. **Asymmetric Safety Risk (NASA Score):**
   In aviation maintenance, late predictions (predicting higher RUL than reality) risk unexpected catastrophic engine failure. The NASA scoring function penalizes late predictions exponentially. LSTM scores **324.33**, approximately half that of LightGBM (669.19), demonstrating that its errors are conservative rather than hazardous.
3. **Efficiency vs. Accuracy Trade-Off:**
   If deploying to resource-constrained microcontrollers, **LightGBM** provides the optimal operational profile: training in just **0.12 seconds** with a sub-millisecond inference latency (0.001 ms) and a compact 414 KB footprint.

---

## 9. Historical Baseline Distinction

In the Phase 1 Progress Report (*Engine_Health_ML_Progress_Report.pdf*), engine health was modeled as a binary classification problem:
- **Safe:** RUL > 30 cycles
- **Maintenance Needed:** RUL $\le$ 30 cycles
- Reported accuracy: **96.05%**; our reproduced accuracy: **95.98%**.

> [!WARNING]
> **Methodological Note:** Classification accuracy measures discrete binning performance on random train/test splits. It cannot and must not be compared to continuous regression cycle errors (MAE/RMSE). The final phase focuses strictly on RUL regression across all 4 architectures.

---

## 10. Quick Start & Execution Instructions

### Prerequisites
- Python 3.12 (or 3.10+)
- Windows, macOS, or Linux

### Installation

```bash
# Clone or navigate to the project directory
cd "c:\Users\advai\OneDrive\Desktop\ML Project"

# Install required dependencies
py -3.12 -m pip install -r requirements.txt
```

### Running Automated Test Suite

```bash
py -3.12 -m pytest -v
```
*(All 15 tests verify data loaders, target calculation, engine-aware splits, model persistence, and inference validation.)*

### Retraining Models (Optional)

```bash
py -3.12 -c "from src.experiment_manager import ExperimentManager; ExperimentManager().run_full_pipeline(force_retrain=True)"
```

### Launching the Streamlit Application

```bash
# Standard Streamlit command:
py -3.12 -m streamlit run app.py

# Or using the cross-platform launcher:
py -3.12 run_app.py
```

The application will open in your web browser at `http://localhost:8501`.

---

## 11. Maintenance Interpretation Protocol

Predicted RUL values are mapped to actionable operational alerts:
- 🚨 **CRITICAL (RUL $\le$ 15 cycles):** Immediate maintenance required. Withdraw engine from scheduled flights for teardown inspection.
- ⚠️ **WARNING (15 < RUL $\le$ 30 cycles):** Approaching end-of-life threshold. Schedule maintenance inspection within 10 cycles; stage replacement modules.
- ✅ **NORMAL (RUL > 30 cycles):** Operating within nominal baseline parameters. Continue routine condition monitoring.

*Disclaimer: These thresholds represent academic engineering demonstrations and do not constitute certified aviation airworthiness limits.*
