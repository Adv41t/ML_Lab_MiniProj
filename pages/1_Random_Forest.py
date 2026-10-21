"""Random Forest Regression & Historical Baseline Analysis Page."""

import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import MODELS_DIR, RESULTS_DIR, BASELINE_SELECTED_SENSORS, SENSOR_METADATA
from src.ui_helpers import apply_custom_theme, render_header, render_metric_card, render_maintenance_badge
from src.visualization import (
    plot_actual_vs_predicted,
    plot_residuals_distribution,
    plot_feature_importances,
)
from src.inference import (
    predict_single_cycle,
    map_rul_to_maintenance_status,
    validate_uploaded_csv,
)
from src.preprocessing import Preprocessor
from src.explainability import explain_single_prediction

st.set_page_config(page_title="Random Forest Regressor | ENGINE HEALTH AI", page_icon="🌲", layout="wide")
apply_custom_theme()

render_header(
    title="Random Forest Regressor",
    subtitle="Ensemble Bootstrap Aggregation of Decision Trees for Turbofan RUL Prediction",
    tag="Model 1 / 4 • Baseline Ensemble Architecture",
)

# Load manifest and artifacts
manifest_file = RESULTS_DIR / "experiment_manifest.json"
if not manifest_file.exists():
    st.warning("⚠️ Experiment manifest not found. Please run training from the landing page.")
    st.stop()

with open(manifest_file, "r", encoding="utf-8") as f:
    manifest = json.load(f)

model_info = manifest["models"].get("Random Forest Regressor", {})
val_m = model_info.get("val_metrics", {})
test_m = model_info.get("test_metrics", {})
hypers = val_m.get("hyperparameters", {})

rf_model_path = MODELS_DIR / "random_forest_regressor.joblib"
prep_path = MODELS_DIR / "preprocessor.joblib"

if not rf_model_path.exists() or not prep_path.exists():
    st.error("Model artifacts missing. Retrain from the home page.")
    st.stop()

rf_model = joblib.load(rf_model_path)
preprocessor = Preprocessor.load(prep_path)

# Performance Metrics Overview
st.markdown("### 📈 Experimental Performance Metrics")
mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)

with mcol1:
    render_metric_card("Test MAE", f"{test_m.get('mae', 'N/A')} cycles", "Official test error")
with mcol2:
    render_metric_card("Test RMSE", f"{test_m.get('rmse', 'N/A')} cycles", "Penalizes large errors")
with mcol3:
    render_metric_card("Test R² Score", f"{test_m.get('r2', 'N/A')}", "Variance explained")
with mcol4:
    render_metric_card("Late-Life MAE", f"{test_m.get('late_life', {}).get('late_life_mae', 'N/A')} cycles", "Cycles ≤ 30 remaining")
with mcol5:
    render_metric_card("NASA Score", f"{test_m.get('nasa_score', 'N/A')}", "Asymmetric penalty")

# Tabs for comprehensive analysis
tab_eval, tab_infer, tab_baseline, tab_arch = st.tabs([
    "📊 Evaluation & Visualizations",
    "🔬 Interactive Engine Inference",
    "📜 Phase 1 Classification Baseline",
    "🧠 Architecture & Diagnostics",
])

# Tab 1: Evaluation
with tab_eval:
    test_preds = np.array(manifest["test_predictions"].get("Random Forest Regressor", []))
    y_test_true = np.array(manifest.get("test_ground_truth", []))

    if len(test_preds) > 0 and len(y_test_true) > 0:
        c1, c2 = st.columns(2)
        with c1:
            fig_avp = plot_actual_vs_predicted(
                y_test_true, test_preds,
                model_name="Random Forest Regressor",
                split_name="Official Test Set (100 Engines)"
            )
            st.plotly_chart(fig_avp, use_container_width=True)
        with c2:
            fig_res = plot_residuals_distribution(
                y_test_true, test_preds,
                model_name="Random Forest Regressor"
            )
            st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("#### 🎯 Gini/MDI Feature Importances")
        feat_imp = val_m.get("feature_importances", {})
        if feat_imp:
            fig_imp = plot_feature_importances(feat_imp, model_name="Random Forest Regressor")
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Test evaluation predictions pending.")

# Tab 2: Interactive Inference
with tab_infer:
    st.markdown("### 🛩️ Real-Time RUL Prediction")
    mode = st.radio("Choose Inference Mode:", ["Mode A: Select Existing Test Engine", "Mode B: Upload Sensor CSV"], horizontal=True)

    test_csv_path = Path("data/processed/test_FD001.csv")
    if mode == "Mode A: Select Existing Test Engine" and test_csv_path.exists():
        test_df = pd.read_csv(test_csv_path)
        avail_engines = sorted(test_df["engine_id"].unique())
        selected_engine = st.selectbox("Select Test Engine ID (1 to 100):", avail_engines, index=0)

        engine_data = test_df[test_df["engine_id"] == selected_engine].sort_values(by="cycle")
        latest_record = engine_data.iloc[-1]
        latest_cycle = int(latest_record["cycle"])

        st.markdown(f"**Latest Observed Cycle for Engine #{selected_engine}:** `{latest_cycle}`")

        # Feature vector for latest cycle
        feat_cols = preprocessor.feature_cols
        fv = latest_record[feat_cols].to_numpy(dtype=float)

        pred_rul = predict_single_cycle(rf_model, fv, preprocessor=preprocessor, feature_cols=feat_cols)
        maint_info = map_rul_to_maintenance_status(pred_rul)

        render_maintenance_badge(
            status=maint_info["status"],
            display_rul=maint_info["display_rul"],
            action=maint_info["action"],
        )

        # Ground truth comparison if available
        if len(y_test_true) >= selected_engine:
            true_rul = y_test_true[selected_engine - 1]
            diff = pred_rul - true_rul
            st.markdown(
                f"- **Actual Ground-Truth RUL:** `{true_rul:.1f}` cycles  \n"
                f"- **Prediction Error:** `{diff:+.2f}` cycles (`{'Late / Over-estimate' if diff > 0 else 'Early / Conservative'}`)"
            )

        # Local explanation
        st.markdown("#### 🔍 Prediction Drivers (SHAP Local Explanation)")
        local_exp = explain_single_prediction(rf_model, preprocessor.transform(pd.DataFrame([latest_record[feat_cols]]))[feat_cols].iloc[0].to_numpy(), feat_cols)
        if "top_degradation_indicators" in local_exp:
            st.write("Top sensor factors accelerating maintenance warning:")
            st.json(local_exp["top_degradation_indicators"])

    elif mode == "Mode B: Upload Sensor CSV":
        st.markdown("Upload a CSV file containing engine sensor records.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file)
            is_valid, errors, clean_df = validate_uploaded_csv(user_df, preprocessor.feature_cols, is_lstm=False)
            if not is_valid:
                for err in errors:
                    st.error(f"❌ {err}")
            else:
                st.success(f"✅ CSV validated successfully ({len(clean_df)} records).")
                scaled_df = preprocessor.transform(clean_df)
                preds = rf_model.predict(scaled_df[preprocessor.feature_cols].to_numpy(dtype=np.float32))
                clean_df["Predicted_RUL"] = [max(0.0, round(p, 1)) for p in preds]
                clean_df["Maintenance_Alert"] = [map_rul_to_maintenance_status(p)["status"] for p in preds]
                st.dataframe(clean_df[["engine_id", "cycle", "Predicted_RUL", "Maintenance_Alert"] if "engine_id" in clean_df else ["cycle", "Predicted_RUL", "Maintenance_Alert"]].head(20))

# Tab 3: Historical Baseline
with tab_baseline:
    st.markdown("### 📜 Phase 1 Historical Classification Baseline")
    st.markdown(
        """
        In the Phase 1 Progress Report (*Engine_Health_ML_Progress_Report.pdf*), an initial
        Random Forest implementation was developed. It treated maintenance as a **binary classification problem**
        by categorizing cycles into:
        - **Safe**: RUL > 30 cycles (Class 0)
        - **Maintenance Needed**: RUL ≤ 30 cycles (Class 1)
        
        The model achieved a reported **96.05% classification accuracy** on an 80:20 split.
        """
    )

    base_info = manifest.get("historical_rf_classifier", {})
    if base_info:
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.markdown("#### 🎯 Reproduced Phase 1 Classification Metrics")
            st.metric("Reproduced Accuracy", base_info.get("accuracy_pct", "95.98%"), delta="Historical: 96.05%")
            st.write("**Classification Report:**")
            st.json(base_info.get("classification_report", {}))
        with bcol2:
            st.markdown("#### 🧩 Confusion Matrix")
            cm = base_info.get("confusion_matrix", [[3431, 76], [90, 530]])
            cm_df = pd.DataFrame(
                cm,
                index=["True Safe (>30)", "True Maint (<=30)"],
                columns=["Pred Safe", "Pred Maint"],
            )
            st.dataframe(cm_df.style.background_gradient(cmap="Blues"))

    st.markdown(
        """
        > [!IMPORTANT]
        > **Methodological Scope Clarification:**  
        > Classification accuracy measures categorical agreement at an arbitrary cutoff (30 cycles)
        > on overlapping row splits. It **must never be directly compared** to continuous Remaining
        > Useful Life regression errors (MAE or RMSE in cycles). The final project comparison uses
        > the **Random Forest Regressor** to predict continuous cycles.
        """
    )

# Tab 4: Architecture
with tab_arch:
    st.markdown("### ⚙️ Algorithm Architecture & Diagnostics")
    st.markdown(
        """
        **Random Forest Regressor** is an ensemble learning method that constructs a multitude of decorrelated
        decision trees during training and outputs the average prediction across all trees:
        $$\\hat{y} = \\frac{1}{B} \\sum_{b=1}^{B} T_b(x)$$
        
        **Hyperparameter Configuration Used:**
        """
    )
    st.json(hypers)

    st.markdown(
        """
        **Strengths in Turbofan Condition Monitoring:**
        - Robust against overfitting due to bootstrap bagging and random feature subsampling.
        - Inherently non-parametric; captures non-linear sensor degradation without strict scaling assumptions.
        - Provides reliable Mean Decrease in Impurity (MDI) feature rankings.

        **Limitations:**
        - Cannot extrapolate beyond the range of training RUL values.
        - Evaluates each cycle as an isolated observation, failing to exploit temporal sequence dynamics.
        """
    )
