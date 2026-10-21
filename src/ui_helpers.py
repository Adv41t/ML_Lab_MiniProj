"""Reusable UI components, custom styling, and layout helpers for Streamlit."""

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

from src.config import THEME_COLORS


CUSTOM_CSS = f"""
<style>
    /* Dark Navy / Slate Engineering Dashboard Theme */
    .stApp {{
        background-color: #0B1120;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    
    /* Top Header Bar */
    .app-header {{
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }}
    .app-title {{
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #38BDF8;
        margin-bottom: 6px;
    }}
    .app-subtitle {{
        font-size: 1.05rem;
        font-weight: 400;
        color: #94A3B8;
        line-height: 1.5;
    }}
    .academic-tag {{
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Card Styling */
    .metric-card {{
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .metric-card:hover {{
        border-color: #38BDF8;
    }}
    .metric-label {{
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }}
    .metric-subtext {{
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 6px;
    }}

    /* Navigation Model Cards on Landing Page */
    .nav-card {{
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .nav-card:hover {{
        border-color: #38BDF8;
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.15);
    }}
    .nav-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 8px;
    }}
    .nav-desc {{
        font-size: 0.9rem;
        color: #94A3B8;
        line-height: 1.5;
        margin-bottom: 16px;
    }}
    .nav-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }}

    /* Maintenance Badges */
    .badge-critical {{
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }}
    .badge-warning {{
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }}
    .badge-normal {{
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }}

    /* Streamlit overrides */
    div[data-testid="stSidebar"] {{
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }}
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #CBD5E1;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background-color: #0F172A;
        border-bottom: 1px solid #334155;
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #94A3B8;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }}
</style>
"""


def apply_custom_theme():
    """Inject custom dark navy/slate CSS."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, tag: str = "ENGINE HEALTH AI — ACADEMIC ML FRAMEWORK"):
    """Render consistent executive engineering header."""
    st.markdown(
        f"""
        <div class="app-header">
            <div class="academic-tag">{tag}</div>
            <div class="app-title">{title}</div>
            <div class="app-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str = "", delta: Optional[str] = None):
    """Render a styled metric tile."""
    delta_html = f"<span style='color: #10B981; font-size: 0.85rem;'>{delta}</span>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value} {delta_html}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_maintenance_badge(status: str, display_rul: float, action: str):
    """Render high-contrast maintenance status recommendation banner."""
    if status == "CRITICAL":
        css_class = "badge-critical"
        icon = "🚨 CRITICAL ATTENTION REQUIRED"
        border_color = "#EF4444"
    elif status == "WARNING":
        css_class = "badge-warning"
        icon = "⚠️ MAINTENANCE WARNING"
        border_color = "#F59E0B"
    else:
        css_class = "badge-normal"
        icon = "✅ HEALTHY / ROUTINE MONITORING"
        border_color = "#10B981"

    st.markdown(
        f"""
        <div style="background: #1E293B; border-left: 6px solid {border_color}; border-radius: 8px; padding: 18px 22px; margin: 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span class="{css_class}">{icon}</span>
                <span style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC;">Estimated RUL: {display_rul:.1f} Cycles</span>
            </div>
            <div style="color: #E2E8F0; font-size: 0.95rem; margin-top: 6px;">
                <b>Maintenance Action:</b> {action}
            </div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 8px; font-style: italic;">
                ⚠️ Prototype threshold alert. Non-certified illustrative guideline for research condition monitoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
