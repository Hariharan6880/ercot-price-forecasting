"""ERCOT Electricity Price Forecasting Dashboard — Streamlit App."""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="ERCOT Price Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b2838 0%, #0f1923 100%);
        border-right: 1px solid #2d4a5e;
    }
    [data-testid="stSidebar"] * {
        color: #e0e6ed !important;
    }
    [data-testid="stSidebar"] label {
        color: #e0e6ed !important;
        font-size: 0.95rem;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #e0e6ed !important;
        font-weight: 500;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #00d4ff !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #c4d4e4 !important;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #162d4a 100%);
        border: 1px solid #2d5a8e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 150, 255, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 150, 255, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8ba4c4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-delta {
        font-size: 0.9rem;
        color: #00e676;
        margin-top: 4px;
    }
    .metric-delta.negative {
        color: #ff5252;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #00d4ff22, transparent);
        border-left: 4px solid #00d4ff;
        padding: 12px 20px;
        margin: 30px 0 20px 0;
        border-radius: 0 8px 8px 0;
    }
    .section-header h3 {
        margin: 0;
        color: #e0e6ed;
    }

    /* Divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, #7b2ff7, transparent);
        margin: 30px 0;
        border: none;
    }

    /* Table styling */
    .styled-table {
        background: #1e2d3d;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Global text visibility */
    .stApp p, .stApp li, .stApp span, .stApp td, .stApp th {
        color: #e0e6ed !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #f0f4f8 !important;
    }

    /* Markdown tables */
    .stApp table {
        color: #e0e6ed !important;
    }
    .stApp table th {
        color: #00d4ff !important;
        font-weight: 600;
        border-bottom: 1px solid #2d5a8e;
    }
    .stApp table td {
        color: #c4d4e4 !important;
        border-bottom: 1px solid #1e3a5f;
    }
    .stApp table tr:hover td {
        color: #ffffff !important;
        background: rgba(0, 212, 255, 0.05);
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        color: #e0e6ed !important;
    }

    /* Date input label */
    .stApp label {
        color: #e0e6ed !important;
    }

    /* Bold text in markdown */
    .stApp strong {
        color: #ffffff !important;
    }

    /* Hide default streamlit metric styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #162d4a 100%);
        border: 1px solid #2d5a8e;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0, 150, 255, 0.08);
    }
    [data-testid="stMetricValue"] {
        color: #00d4ff;
    }

    /* Plotly chart backgrounds */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 30px 0;
    }
    .hero h1 {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .hero p {
        color: #8ba4c4;
        font-size: 1.1rem;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #00d4ff33, #7b2ff733);
        border: 1px solid #00d4ff55;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: #00d4ff;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Plotly dark theme ──────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15, 25, 40, 0.8)",
    plot_bgcolor="rgba(15, 25, 40, 0.6)",
    font=dict(color="#c4d4e4"),
    margin=dict(l=40, r=20, t=50, b=40),
)

COLORS = {
    "primary": "#00d4ff",
    "secondary": "#7b2ff7",
    "success": "#00e676",
    "danger": "#ff5252",
    "warning": "#ffab00",
    "normal": "#00e676",
    "stressed": "#ffab00",
    "scarcity": "#ff5252",
    "muted": "#546e7a",
}


def styled_metric(label, value, delta=None, delta_color="positive"):
    delta_html = ""
    if delta:
        cls = "negative" if delta_color == "negative" else ""
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def section_header(title):
    st.markdown(f'<div class="section-header"><h3>{title}</h3></div>', unsafe_allow_html=True)


def gradient_divider():
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)



# ── Load Data ──────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_data/predictions.csv", parse_dates=["ObsTime"])
    with open("dashboard_data/metrics.json") as f:
        metrics = json.load(f)
    return df, metrics


df, metrics = load_data()
test = df[df["split"] == "test"].copy()
val = df[df["split"] == "val"].copy()

REGIME_NAMES = {0: "Normal", 1: "Stressed", 2: "Scarcity"}
REGIME_COLORS_MAP = {"Normal": COLORS["normal"], "Stressed": COLORS["warning"], "Scarcity": COLORS["danger"]}

# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding: 20px 0;">
            <div style="font-size: 3rem;">⚡</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #00d4ff; margin-top: 8px;">
                ERCOT Forecast
            </div>
            <div style="color: #8ba4c4; font-size: 0.85rem;">HB_NORTH Node - 2019 to 2026</div>
        </div>
    """, unsafe_allow_html=True)

    gradient_divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Forecast", "Spikes",
         "Regimes", "Errors", "Architecture"],
        label_visibility="collapsed",
    )

    if page in ["Forecast", "Spikes", "Errors"]:
        gradient_divider()
        st.markdown("**Date Range**")
        date_range = st.date_input(
            "Filter",
            value=(test["ObsTime"].min().date(), test["ObsTime"].max().date()),
            min_value=df["ObsTime"].min().date(),
            max_value=df["ObsTime"].max().date(),
            label_visibility="collapsed",
        )
        if len(date_range) == 2:
            mask = (df["ObsTime"].dt.date >= date_range[0]) & (df["ObsTime"].dt.date <= date_range[1])
            filtered = df[mask]
        else:
            filtered = test
    else:
        filtered = test

    gradient_divider()
    st.markdown("""
        <div style="text-align:center;">
            <span class="badge">XGBoost</span>
            <span class="badge">SHAP</span>
            <span class="badge">FastAPI</span>
            <span class="badge">33 Features</span>
        </div>
    """, unsafe_allow_html=True)

    gradient_divider()
    st.markdown("""
        <div style="text-align:center; padding: 10px 0;">
            <div style="font-weight: 600; color: #e0e6ed; font-size: 1rem;">Hariharan Balaji</div>
            <div style="color: #8ba4c4; font-size: 0.8rem; margin-top: 6px;">
                &#x1F4DE; 9087652203
            </div>
            <div style="color: #8ba4c4; font-size: 0.8rem; margin-top: 4px;">
                &#x2709; hariharan2002.br@gmail.com
            </div>
        </div>
    """, unsafe_allow_html=True)


# PAGE: Overview
if page == "Overview":
    st.markdown("""
        <div class="hero">
            <h1>Electricity Price Forecasting</h1>
            <p>Multi-model XGBoost system predicting real-time LMP at ERCOT HB_NORTH</p>
        </div>
    """, unsafe_allow_html=True)

    gradient_divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        styled_metric("Mean Absolute Error", f"${metrics['xgb_mae']}/MWh", "Test Set (19,230 hrs)")
    with col2:
        styled_metric("Median Error", f"${metrics['xgb_median_ae']}/MWh", "50th percentile")
    with col3:
        styled_metric("RMSE", f"${metrics['xgb_rmse']}/MWh")
    with col4:
        styled_metric("Features", "33", "from 28 raw columns")

    gradient_divider()
    section_header("Model vs Baselines")

    lr_mae = metrics.get("linear_regression_mae", 16.21)
    naive_mae = metrics.get("naive_lag1_mae", 9.18)
    hourly_mae = metrics.get("hourly_mean_mae", 26.03)
    ridge_mae = metrics.get("ridge_regression_mae", 15.64)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            styled_metric("vs Linear Reg", f"-{(lr_mae - metrics['xgb_mae'])/lr_mae*100:.0f}%",
                         f"${lr_mae:.1f} to ${metrics['xgb_mae']}")
        with col_b:
            styled_metric("vs Naive (lag-1)", f"-{(naive_mae - metrics['xgb_mae'])/naive_mae*100:.0f}%",
                         f"${naive_mae:.1f} to ${metrics['xgb_mae']}")
        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            styled_metric("vs Hourly Mean", f"-{(hourly_mae - metrics['xgb_mae'])/hourly_mae*100:.0f}%",
                         f"${hourly_mae:.1f} to ${metrics['xgb_mae']}")
        with col_b:
            styled_metric("vs Ridge", f"-{(ridge_mae - metrics['xgb_mae'])/ridge_mae*100:.0f}%",
                         f"${ridge_mae:.1f} to ${metrics['xgb_mae']}")

    with col2:
        models_list = ["Hourly Mean", "Linear Reg", "Ridge", "Naive (lag-1)", "XGBoost"]
        maes = [hourly_mae, lr_mae, ridge_mae, naive_mae, metrics["xgb_mae"]]
        colors = [COLORS["muted"]] * 4 + [COLORS["primary"]]

        fig = go.Figure(go.Bar(
            x=models_list, y=maes,
            marker_color=colors,
            marker_line=dict(width=0),
            text=[f"${m:.1f}" for m in maes],
            textposition="outside",
            textfont=dict(color="#c4d4e4", size=13),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="MAE Comparison ($/MWh)", font=dict(size=16)),
            yaxis_title="MAE ($/MWh)",
            height=380,
            showlegend=False,
        )
        fig.update_yaxes(gridcolor="rgba(100,150,200,0.1)")
        st.plotly_chart(fig, use_container_width=True)

    gradient_divider()
    section_header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        styled_metric("Observations", f"{len(df):,}")
    with col2:
        styled_metric("Time Span", "7 Years")
    with col3:
        styled_metric("Training", f"{len(df[df['split']=='train']):,} hrs")
    with col4:
        styled_metric("Test", f"{len(test):,} hrs")

    st.markdown("")
    monthly_avg = df.groupby(df["ObsTime"].dt.to_period("M")).agg(
        price=("RT_LMP", "mean"),
        time=("ObsTime", "first"),
    ).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_avg["time"], y=monthly_avg["price"],
        mode="lines", fill="tozeroy",
        line=dict(color=COLORS["primary"], width=2),
        fillcolor="rgba(0, 212, 255, 0.1)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Monthly Average RT_LMP (Full Dataset)",
        yaxis_title="$/MWh",
        height=250,
    )
    fig.update_yaxes(gridcolor="rgba(100,150,200,0.1)")
    fig.update_xaxes(gridcolor="rgba(100,150,200,0.1)")
    st.plotly_chart(fig, use_container_width=True)


# PAGE: Forecast
elif page == "Forecast":
    st.markdown("""
        <div class="hero">
            <h1>Price Forecast</h1>
            <p>Actual vs Predicted RT_LMP</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        styled_metric("MAE (filtered)", f"${filtered['abs_error'].mean():.2f}")
    with col2:
        styled_metric("Median AE", f"${filtered['abs_error'].median():.2f}")
    with col3:
        styled_metric("Max Error", f"${filtered['abs_error'].max():.0f}")
    with col4:
        styled_metric("Hours", f"{len(filtered):,}")

    gradient_divider()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(go.Scatter(
        x=filtered["ObsTime"], y=filtered["RT_LMP"],
        mode="lines", name="Actual",
        line=dict(color=COLORS["primary"], width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=filtered["ObsTime"], y=filtered["predicted_price"],
        mode="lines", name="Predicted",
        line=dict(color=COLORS["secondary"], width=1.5),
    ), row=1, col=1)

    error_colors = [COLORS["danger"] if e > 0 else COLORS["success"]
                    for e in filtered["prediction_error"]]
    fig.add_trace(go.Bar(
        x=filtered["ObsTime"], y=filtered["prediction_error"],
        name="Error", marker_color=error_colors, opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=600,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="#2d5a8e"),
    )
    fig.update_yaxes(title_text="$/MWh", gridcolor="rgba(100,150,200,0.1)", row=1, col=1)
    fig.update_yaxes(title_text="Error", gridcolor="rgba(100,150,200,0.1)", row=2, col=1)
    fig.update_xaxes(gridcolor="rgba(100,150,200,0.1)")
    st.plotly_chart(fig, use_container_width=True)


# PAGE: Spike Detection
elif page == "Spikes":
    st.markdown("""
        <div class="hero">
            <h1>Spike Detection</h1>
            <p>Binary classifier for price events > $300/MWh</p>
        </div>
    """, unsafe_allow_html=True)

    spikes_actual = filtered["is_spike"].sum()
    spikes_predicted = filtered["spike_predicted"].sum()
    true_pos = ((filtered["is_spike"] == 1) & (filtered["spike_predicted"] == 1)).sum()
    recall = true_pos / spikes_actual if spikes_actual > 0 else 0
    precision = true_pos / spikes_predicted if spikes_predicted > 0 else 0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        styled_metric("Actual Spikes", f"{spikes_actual}")
    with col2:
        styled_metric("Detected", f"{true_pos}")
    with col3:
        styled_metric("Recall", f"{recall:.0%}")
    with col4:
        styled_metric("Precision", f"{precision:.0%}")
    with col5:
        styled_metric("F2 Score", f"{f2:.3f}")

    gradient_divider()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["ObsTime"], y=filtered["spike_probability"],
        mode="lines", name="Spike Probability",
        line=dict(color=COLORS["secondary"], width=1.5),
        fill="tozeroy", fillcolor="rgba(123, 47, 247, 0.1)",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Threshold", annotation_font_color=COLORS["warning"])

    spikes = filtered[filtered["is_spike"] == 1]
    fig.add_trace(go.Scatter(
        x=spikes["ObsTime"], y=spikes["spike_probability"],
        mode="markers", name="Actual Spikes",
        marker=dict(color=COLORS["danger"], size=10, symbol="triangle-up",
                    line=dict(width=1, color="white")),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Spike Probability Over Time",
        yaxis_title="P(spike)",
        height=400,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )
    fig.update_yaxes(gridcolor="rgba(100,150,200,0.1)")
    fig.update_xaxes(gridcolor="rgba(100,150,200,0.1)")
    st.plotly_chart(fig, use_container_width=True)

    if spikes_actual > 0:
        section_header("Spike Events")
        spike_df = filtered[filtered["is_spike"] == 1][
            ["ObsTime", "RT_LMP", "predicted_price", "spike_probability", "spike_predicted", "reserve_margin"]
        ].sort_values("RT_LMP", ascending=False).head(15).copy()
        spike_df.columns = ["Time", "Actual ($)", "Predicted ($)", "P(spike)", "Detected", "Reserve Margin"]
        spike_df["Detected"] = spike_df["Detected"].map({1: "Yes", 0: "No"})
        st.dataframe(spike_df, use_container_width=True, hide_index=True)


# PAGE: Regime Analysis
elif page == "Regimes":
    st.markdown("""
        <div class="hero">
            <h1>Regime Analysis</h1>
            <p>Three distinct market states with different pricing mechanisms</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card" style="border-color: {COLORS['normal']};">
                <div style="font-size:2rem;">&#x1F7E2;</div>
                <div class="metric-value" style="background: {COLORS['normal']}; -webkit-background-clip: text;">Normal</div>
                <div class="metric-label">Reserve Margin > 5,000 MW</div>
                <div style="color: #8ba4c4; margin-top:8px;">$12-50/MWh typical</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card" style="border-color: {COLORS['warning']};">
                <div style="font-size:2rem;">&#x1F7E1;</div>
                <div class="metric-value" style="background: {COLORS['warning']}; -webkit-background-clip: text;">Stressed</div>
                <div class="metric-label">Reserve Margin 1,000-5,000 MW</div>
                <div style="color: #8ba4c4; margin-top:8px;">$50-300/MWh typical</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card" style="border-color: {COLORS['danger']};">
                <div style="font-size:2rem;">&#x1F534;</div>
                <div class="metric-value" style="background: {COLORS['danger']}; -webkit-background-clip: text;">Scarcity</div>
                <div class="metric-label">Reserve Margin < 1,000 MW</div>
                <div style="color: #8ba4c4; margin-top:8px;">$300-9,000/MWh typical</div>
            </div>
        """, unsafe_allow_html=True)

    gradient_divider()

    col1, col2 = st.columns(2)
    with col1:
        regime_counts = test["regime"].value_counts().sort_index()
        fig = go.Figure(go.Pie(
            labels=[REGIME_NAMES[i] for i in regime_counts.index],
            values=regime_counts.values,
            marker_colors=[COLORS["normal"], COLORS["warning"], COLORS["danger"]],
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(size=13),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Test Set Distribution",
            height=380,
            showlegend=False,
        )
        fig.add_annotation(text=f"{len(test):,}", x=0.5, y=0.55,
                          font=dict(size=20, color=COLORS["primary"]), showarrow=False)
        fig.add_annotation(text="hours", x=0.5, y=0.42,
                          font=dict(size=12, color="#8ba4c4"), showarrow=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        regime_mae = test.groupby("regime")["abs_error"].mean()
        fig = go.Figure(go.Bar(
            x=[REGIME_NAMES[i] for i in regime_mae.index],
            y=regime_mae.values,
            marker_color=[COLORS["normal"], COLORS["warning"], COLORS["danger"]],
            text=[f"${v:.1f}" for v in regime_mae.values],
            textposition="outside",
            textfont=dict(color="#c4d4e4", size=13),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="MAE by Regime ($/MWh)",
            yaxis_title="MAE",
            height=380,
            showlegend=False,
        )
        fig.update_yaxes(gridcolor="rgba(100,150,200,0.1)")
        st.plotly_chart(fig, use_container_width=True)

    correct = (test["regime"] == test["predicted_regime"]).mean()
    col1, col2, col3 = st.columns(3)
    with col1:
        styled_metric("Classifier Accuracy", f"{correct:.1%}")
    with col2:
        styled_metric("Normal F1", "1.00")
    with col3:
        styled_metric("Scarcity Recall", "100%")


# PAGE: Error Analysis
elif page == "Errors":
    st.markdown("""
        <div class="hero">
            <h1>Error Analysis</h1>
            <p>Understanding where and why the model fails</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Histogram(
            x=filtered["prediction_error"],
            nbinsx=120,
            marker_color=COLORS["primary"],
            opacity=0.7,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color=COLORS["danger"])
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Error Distribution",
            xaxis_title="Error ($/MWh)",
            yaxis_title="Count",
            height=350,
        )
        fig.update_yaxes(gridcolor="rgba(100,150,200,0.1)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Scatter(
            x=filtered["RT_LMP"], y=filtered["predicted_price"],
            mode="markers",
            marker=dict(color=COLORS["secondary"], opacity=0.3, size=4),
        ))
        max_v = max(filtered["RT_LMP"].max(), filtered["predicted_price"].max())
        fig.add_trace(go.Scatter(
            x=[-50, max_v], y=[-50, max_v],
            mode="lines", line=dict(dash="dash", color=COLORS["danger"], width=2),
            name="Perfect",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Predicted vs Actual",
            xaxis_title="Actual ($/MWh)",
            yaxis_title="Predicted ($/MWh)",
            height=350,
            showlegend=False,
        )
        fig.update_yaxes(gridcolor="rgba(100,150,200,0.1)")
        fig.update_xaxes(gridcolor="rgba(100,150,200,0.1)")
        st.plotly_chart(fig, use_container_width=True)

    gradient_divider()
    section_header("Error Patterns")

    col1, col2 = st.columns(2)
    with col1:
        hourly_err = filtered.copy()
        hourly_err["hour"] = hourly_err["ObsTime"].dt.hour
        hourly_mae = hourly_err.groupby("hour")["abs_error"].mean()
        fig = go.Figure(go.Scatter(
            x=hourly_mae.index, y=hourly_mae.values,
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=6, color=COLORS["primary"]),
            fill="tozeroy", fillcolor="rgba(0, 212, 255, 0.1)",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, title="MAE by Hour of Day",
            xaxis_title="Hour", yaxis_title="MAE ($/MWh)", height=320,
        )
        fig.update_yaxes(gridcolor="rgba(100,150,200,0.1)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        monthly_err = filtered.copy()
        monthly_err["month"] = monthly_err["ObsTime"].dt.month
        monthly_mae = monthly_err.groupby("month")["abs_error"].mean()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = go.Figure(go.Bar(
            x=[month_names[i-1] for i in monthly_mae.index],
            y=monthly_mae.values,
            marker_color=[COLORS["primary"] if v < 15 else COLORS["warning"] if v < 30 else COLORS["danger"]
                         for v in monthly_mae.values],
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, title="MAE by Month",
            xaxis_title="Month", yaxis_title="MAE ($/MWh)", height=320,
        )
        fig.update_yaxes(gridcolor="rgba(100,150,200,0.1)")
        st.plotly_chart(fig, use_container_width=True)

    gradient_divider()
    section_header("Top 10 Worst Predictions")
    worst = filtered.nlargest(10, "abs_error")[
        ["ObsTime", "RT_LMP", "predicted_price", "prediction_error", "regime", "reserve_margin"]
    ].copy()
    worst["regime"] = worst["regime"].map(REGIME_NAMES)
    worst.columns = ["Time", "Actual ($)", "Predicted ($)", "Error ($)", "Regime", "Reserve Margin"]
    st.dataframe(worst, use_container_width=True, hide_index=True)


# PAGE: Architecture
elif page == "Architecture":
    st.markdown("""
        <div class="hero">
            <h1>Model Architecture</h1>
            <p>Multi-model regime-aware prediction system</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f22, #162d4a44); border: 1px solid #2d5a8e;
                border-radius: 12px; padding: 30px; font-family: monospace; line-height: 1.8; color: #c4d4e4;">
    <pre style="color: #c4d4e4; font-size: 0.9rem;">
    RAW ERCOT DATA (28 columns, 63,000 hourly observations)
                                 |
                                 v
    ETL: Gap fill (0.19%) - Flag censored prices - Drop reactive vars
                                 |
                                 v
    FEATURES: 33 engineered features
    Scarcity_Proximity - Load_Stress - Dynamic_Stress
    Hours_to_Scarcity - Exhaustion_Rate - Cyclical Time
                                 |
              +------------------+------------------+
              v                  v                  v
    +----------------+  +----------------+  +------------------+
    | REGRESSOR      |  | SPIKE CLF      |  | REGIME CLF       |
    | MAE: $7.58     |  | Recall: 69%    |  | Accuracy: ~100%  |
    | XGBoost        |  | F2: 0.52       |  | 3-class          |
    +----------------+  +----------------+  +--------+---------+
                                                     |
                                                     v
                                    +----------------------------+
                                    | REGIME-SPECIFIC REGRESSORS |
                                    | Normal - Stressed - Scarce |
                                    +-------------+--------------+
                                                  |
              +-----------------------------------+
              v
    ENSEMBLE: Weighted by regime probability -> MAE: $7.57/MWh
                                 |
                                 v
    FastAPI: /predict - /predict/batch - /health
    </pre>
    </div>
    """, unsafe_allow_html=True)

    gradient_divider()

    col1, col2 = st.columns(2)
    with col1:
        section_header("Key Design Decisions")
        st.markdown("""
        | Decision | Rationale |
        |----------|-----------|
        | **arcsinh transform** | Handles negatives, no shift needed |
        | **Temporal split** | No future leakage (train < val < test) |
        | **Drop reactive vars** | Generation & Responsive_Load are endogenous |
        | **Early stopping** | Optimal at ~230 trees |
        | **scale_pos_weight = 30** | Class imbalance for 0.3% spike rate |
        | **Squared error** | More stable on arcsinh scale |
        """)

    with col2:
        section_header("Top Features (SHAP)")
        st.markdown("""
        | Rank | Feature | Type |
        |------|---------|------|
        | 1 | `Price_lag_1` | Autoregressive |
        | 2 | `Dynamic_Stress` | Engineered |
        | 3 | `Load_Stress` | Engineered |
        | 4 | `Ramp` | Raw |
        | 5 | `Hours_to_Scarcity` | Engineered |
        | 6 | `Scarcity_Proximity` | Engineered |
        | 7 | `System_Load` | Raw |
        | 8 | `Wind_change_3hr` | Lag/diff |
        """)
