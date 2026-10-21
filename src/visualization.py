"""Reusable Plotly visualizations with dark navy/slate engineering dashboard styling."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import THEME_COLORS


DARK_LAYOUT = dict(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#1E293B",
    font=dict(color="#F8FAFC", family="Inter, system-ui, sans-serif"),
    xaxis=dict(
        gridcolor="#334155",
        zerolinecolor="#475569",
        linecolor="#475569",
        tickfont=dict(color="#CBD5E1"),
        title_font=dict(color="#F8FAFC", size=13),
    ),
    yaxis=dict(
        gridcolor="#334155",
        zerolinecolor="#475569",
        linecolor="#475569",
        tickfont=dict(color="#CBD5E1"),
        title_font=dict(color="#F8FAFC", size=13),
    ),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.8)",
        bordercolor="#334155",
        borderwidth=1,
        font=dict(color="#E2E8F0"),
    ),
)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    split_name: str = "Test",
) -> go.Figure:
    """Scatter plot of actual vs predicted RUL with perfect prediction line and error bands."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    max_val = max(float(y_t.max()), float(y_p.max())) + 10
    residuals = y_p - y_t

    fig = go.Figure()

    # Identity line (Ideal y = x)
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        name="Perfect Prediction (y = x)",
        line=dict(color="#94A3B8", dash="dash", width=2),
    ))

    # Error envelope (+/- 15 cycles)
    fig.add_trace(go.Scatter(
        x=[0, max_val, max_val, 0],
        y=[15, max_val + 15, max_val - 15, -15],
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.08)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="±15 Cycles Tolerance",
        showlegend=True,
    ))

    # Actual points
    fig.add_trace(go.Scatter(
        x=y_t,
        y=y_p,
        mode="markers",
        name="Predictions",
        marker=dict(
            size=7,
            color=np.abs(residuals),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(text="|Residual|", font=dict(color="#F8FAFC")),
                tickfont=dict(color="#CBD5E1"),
                len=0.7,
            ),
            line=dict(width=1, color="#0F172A"),
        ),
        text=[f"Engine/Sample #{i+1}<br>Actual: {act:.1f}<br>Predicted: {pred:.1f}<br>Error: {res:+.1f}"
              for i, (act, pred, res) in enumerate(zip(y_t, y_p, residuals))],
        hoverinfo="text",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(
            text=f"<b>Actual vs. Predicted RUL</b> — {model_name} ({split_name})",
            font=dict(color="#38BDF8", size=16),
        ),
        xaxis_title="Ground-Truth RUL (Operating Cycles)",
        yaxis_title="Predicted RUL (Operating Cycles)",
    )
    return fig


def plot_residuals_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Histogram and KDE distribution of prediction residuals (Predicted - Actual)."""
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        name="Residuals",
        marker=dict(
            color="#38BDF8",
            line=dict(color="#0F172A", width=1),
        ),
        opacity=0.85,
    ))

    # Vertical zero line
    fig.add_vline(
        x=0,
        line=dict(color="#EF4444", dash="dash", width=2),
        annotation_text="Zero Residual",
        annotation_position="top left",
        annotation_font=dict(color="#EF4444"),
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(
            text=f"<b>Residual Error Distribution</b> — {model_name}",
            font=dict(color="#38BDF8", size=16),
        ),
        xaxis_title="Prediction Error: (Predicted - Actual) Cycles",
        yaxis_title="Sample Frequency",
        bargap=0.08,
    )
    return fig


def plot_feature_importances(
    importance_dict: Dict[str, float],
    model_name: str = "Model",
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart for top feature importances."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [item[0] for item in reversed(sorted_items)]
    values = [item[1] for item in reversed(sorted_items)]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(
            color=values,
            colorscale="Teal",
            line=dict(color="#0F172A", width=1),
        ),
        text=[f"{v:.4f}" for v in values],
        textposition="auto",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(
            text=f"<b>Top Feature Importances</b> — {model_name}",
            font=dict(color="#38BDF8", size=16),
        ),
        xaxis_title="Relative Feature Importance",
        yaxis_title="Sensor / Attribute",
    )
    return fig


def plot_training_loss_curves(history: Dict[str, List[float]]) -> go.Figure:
    """Training vs Validation Loss curves across epochs for LSTM."""
    epochs = history.get("epochs", list(range(1, len(history.get("train_loss", [])) + 1)))
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode="lines+markers",
        name="Train Loss (MSE)",
        line=dict(color="#38BDF8", width=2.5),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss,
        mode="lines+markers",
        name="Val Loss (MSE)",
        line=dict(color="#F59E0B", width=2.5),
        marker=dict(size=5),
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(
            text="<b>LSTM Training & Validation Loss History</b>",
            font=dict(color="#38BDF8", size=16),
        ),
        xaxis_title="Epoch",
        yaxis_title="Mean Squared Error Loss",
    )
    return fig


def plot_metric_comparison_bars(
    leaderboard_df: pd.DataFrame,
    metric_col: str = "Test MAE",
    title: str = "Model Comparison",
) -> go.Figure:
    """Comparative bar chart for leaderboard models."""
    df_sorted = leaderboard_df.sort_values(by=metric_col, ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted["Model"],
        y=df_sorted[metric_col],
        marker=dict(
            color=["#10B981" if i == 0 else "#38BDF8" for i in range(len(df_sorted))],
            line=dict(color="#0F172A", width=1),
        ),
        text=[f"{v:.2f}" for v in df_sorted[metric_col]],
        textposition="auto",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color="#38BDF8", size=16),
        ),
        xaxis_title="Machine Learning Model",
        yaxis_title=f"{metric_col} (Cycles)",
    )
    return fig


def plot_multi_model_overlay(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    title: str = "Comparison of Model Predictions Across Test Engines",
) -> go.Figure:
    """Multi-model prediction overlay across all 100 test engines."""
    fig = go.Figure()
    indices = np.arange(1, len(y_true) + 1)

    # Ground truth line
    fig.add_trace(go.Scatter(
        x=indices,
        y=y_true,
        mode="lines",
        name="Ground Truth RUL",
        line=dict(color="#FFFFFF", width=3),
    ))

    model_colors = {
        "Random Forest Regressor": "#38BDF8",
        "XGBoost Regressor": "#F59E0B",
        "LightGBM Regressor": "#10B981",
        "LSTM Regressor": "#C084FC",
    }

    for name, preds in predictions_dict.items():
        color = model_colors.get(name, "#818CF8")
        fig.add_trace(go.Scatter(
            x=indices,
            y=preds,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=1.5),
            marker=dict(size=4),
            opacity=0.85,
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color="#38BDF8", size=16),
        ),
        xaxis_title="Test Engine Index (1 to 100)",
        yaxis_title="Remaining Useful Life (Cycles)",
    )
    return fig


def plot_sensor_degradation_trajectory(
    engine_df: pd.DataFrame,
    sensor_cols: List[str],
    engine_id: int = 1,
) -> go.Figure:
    """Plot degradation trajectories of key sensors over cycles for a selected engine."""
    fig = go.Figure()
    colors = ["#38BDF8", "#F59E0B", "#10B981", "#EC4899", "#8B5CF6", "#06B6D4"]

    for idx, sensor in enumerate(sensor_cols):
        color = colors[idx % len(colors)]
        # Min-max scale locally for visual comparison on one axis
        series = engine_df[sensor]
        s_min, s_max = series.min(), series.max()
        norm_series = (series - s_min) / (s_max - s_min + 1e-8)

        fig.add_trace(go.Scatter(
            x=engine_df["cycle"],
            y=norm_series,
            mode="lines",
            name=f"{sensor} (norm)",
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(
            text=f"<b>Sensor Degradation Trajectory (Engine #{engine_id})</b>",
            font=dict(color="#38BDF8", size=16),
        ),
        xaxis_title="Operating Cycle",
        yaxis_title="Normalized Sensor Reading [0, 1]",
    )
    return fig
