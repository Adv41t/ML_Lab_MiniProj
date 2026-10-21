"""ENGINE HEALTH AI - Main Streamlit Landing Application."""

import streamlit as st
from src.ui_helpers import apply_custom_theme, render_header

st.set_page_config(
    page_title="ENGINE HEALTH AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_theme()

# Top Clean Header
render_header(
    title="ENGINE HEALTH AI",
    subtitle="ML-Based Condition Monitoring and Remaining Life Prediction of Turbofan Engines",
    tag="Manipal Institute of Technology • CSE (AI & ML) • ML Lab Mini-Project",
)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# 6 Navigation Cards (2 rows of 3 columns)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <span class="nav-badge" style="background: rgba(56, 189, 248, 0.15); color: #38BDF8;">Model 1 / 4</span>
                <div class="nav-title" style="margin-top: 10px;">🌲 Random Forest</div>
                <div class="nav-desc">
                    Bootstrap aggregation of decision trees for baseline RUL regression & historical classification baseline.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Random Forest →", key="btn_rf", use_container_width=True):
        st.switch_page("pages/1_Random_Forest.py")

with col2:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <span class="nav-badge" style="background: rgba(245, 158, 11, 0.15); color: #F59E0B;">Model 2 / 4</span>
                <div class="nav-title" style="margin-top: 10px;">⚡ XGBoost</div>
                <div class="nav-desc">
                    Extreme Gradient Boosting with regularized objective, second-order loss approximation, and tree pruning.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open XGBoost →", key="btn_xgb", use_container_width=True):
        st.switch_page("pages/2_XGBoost.py")

with col3:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <span class="nav-badge" style="background: rgba(16, 185, 129, 0.15); color: #10B981;">Model 3 / 4</span>
                <div class="nav-title" style="margin-top: 10px;">🍃 LightGBM</div>
                <div class="nav-desc">
                    High-efficiency gradient boosting using histogram binning and leaf-wise tree growth.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open LightGBM →", key="btn_lgb", use_container_width=True):
        st.switch_page("pages/3_LightGBM.py")

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <span class="nav-badge" style="background: rgba(192, 132, 252, 0.15); color: #C084FC;">Model 4 / 4</span>
                <div class="nav-title" style="margin-top: 10px;">🧠 LSTM Network</div>
                <div class="nav-desc">
                    Deep recurrent neural network with 30-cycle sliding windows capturing temporal degradation dynamics.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open LSTM →", key="btn_lstm", use_container_width=True):
        st.switch_page("pages/4_LSTM.py")

with col5:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <span class="nav-badge" style="background: rgba(239, 68, 68, 0.15); color: #EF4444;">Intelligence</span>
                <div class="nav-title" style="margin-top: 10px;">📊 Admin & Analytics</div>
                <div class="nav-desc">
                    Unified comparative benchmark comparing all 4 models on official test data, residual analysis, and recommendations.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Admin & Analytics →", key="btn_admin", use_container_width=True):
        st.switch_page("pages/5_Admin_Analytics.py")

with col6:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <span class="nav-badge" style="background: rgba(56, 189, 248, 0.15); color: #38BDF8;">Pipeline</span>
                <div class="nav-title" style="margin-top: 10px;">🔄 Data Preprocessing</div>
                <div class="nav-desc">
                    Minimal walkthrough of raw TXT parsing, RUL target clipping, feature selection, and sequence creation.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Data Preprocessing →", key="btn_prep", use_container_width=True):
        st.switch_page("pages/6_Data_Preprocessing.py")

# Minimal Clean Sidebar
with st.sidebar:
    st.markdown("### ✈️ ENGINE HEALTH AI")
    st.markdown(
        """
        **Academic ML Project**  
        *V Semester B.Tech. CSE (AI & ML)*  
        Manipal Institute of Technology, MAHE  
        
        **Team Members:**  
        - **Advait Balachandar** (240962600)  
        - **Madaka Shreyaj Naidu** (240962594)  
        - **Dipen Batharaju** (240962574)
        """
    )
