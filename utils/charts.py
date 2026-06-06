"""
Visualization Module.

Plotly-based interactive charts for the Streamlit app.
All charts follow a consistent style with professional formatting.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {
    "primary": "#1a365d",
    "secondary": "#2c5282",
    "accent": "#c05621",
    "positive": "#e24b4a",
    "negative": "#378add",
    "neutral": "#718096",
    "light_bg": "#f7fafc",
    "pfe_fill": "rgba(226, 75, 74, 0.15)",
    "ee_fill": "rgba(55, 138, 221, 0.15)",
    "path_color": "rgba(55, 138, 221, 0.08)",
    "grid": "rgba(0,0,0,0.06)",
}

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, Arial, sans-serif", size=12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=30, t=40, b=50),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def format_eur(val: float) -> str:
    """Format a value in EUR with appropriate suffix."""
    if abs(val) >= 1e6:
        return f"€{val/1e6:,.1f}M"
    elif abs(val) >= 1e3:
        return f"€{val/1e3:,.0f}k"
    else:
        return f"€{val:,.0f}"


def plot_price_paths(prices: np.ndarray, dt: float, n_display: int = 100,
                     forward_curve: np.ndarray = None) -> go.Figure:
    """Plot simulated price paths with forward curve overlay."""
    n_paths, total_steps = prices.shape
    time = np.arange(total_steps) * dt
    
    fig = go.Figure()
    
    # Sample paths
    indices = np.random.choice(n_paths, min(n_display, n_paths), replace=False)
    for idx in indices:
        fig.add_trace(go.Scatter(
            x=time, y=prices[idx], mode="lines",
            line=dict(color=COLORS["path_color"], width=0.5),
            showlegend=False, hoverinfo="skip",
        ))
    
    # Mean path
    mean_price = np.mean(prices, axis=0)
    fig.add_trace(go.Scatter(
        x=time, y=mean_price, mode="lines",
        line=dict(color=COLORS["secondary"], width=2),
        name="Mean path",
    ))
    
    # Percentile bands
    p5 = np.percentile(prices, 5, axis=0)
    p95 = np.percentile(prices, 95, axis=0)
    fig.add_trace(go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor="rgba(44, 82, 130, 0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th-95th percentile",
    ))
    
    # Forward curve
    if forward_curve is not None:
        fig.add_trace(go.Scatter(
            x=time, y=forward_curve, mode="lines",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
            name="Initial forward curve",
        ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Simulated power price paths",
        xaxis_title="Time (years)",
        yaxis_title="Price (EUR/MWh)",
    )
    return fig


def plot_price_distribution(prices: np.ndarray, time_idx: int,
                            dt: float) -> go.Figure:
    """Plot price distribution at a specific time point."""
    values = prices[:, time_idx]
    t_years = time_idx * dt
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values, nbinsx=80,
        marker_color=COLORS["secondary"],
        opacity=0.7, name="Simulated prices",
    ))
    
    fig.add_vline(x=np.mean(values), line_dash="dash",
                  line_color=COLORS["accent"],
                  annotation_text=f"Mean: €{np.mean(values):.1f}/MWh")
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Price distribution at t = {t_years:.1f} years",
        xaxis_title="Price (EUR/MWh)",
        yaxis_title="Frequency",
    )
    return fig


def plot_exposure_profile(profile, confidence_level: float,
                          profile_net=None) -> go.Figure:
    """Plot the full exposure profile (EE, PFE, with optional collateral)."""
    time = profile.time_grid
    
    fig = go.Figure()
    
    # PFE (gross)
    fig.add_trace(go.Scatter(
        x=time, y=profile.potential_future_exposure, mode="lines",
        line=dict(color=COLORS["positive"], width=2.5),
        name=f"PFE ({confidence_level*100:.0f}%)",
    ))
    
    # PFE fill
    fig.add_trace(go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([profile.potential_future_exposure,
                          np.zeros_like(time)[::-1]]),
        fill="toself", fillcolor=COLORS["pfe_fill"],
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    
    # EE (gross)
    fig.add_trace(go.Scatter(
        x=time, y=profile.expected_exposure, mode="lines",
        line=dict(color=COLORS["negative"], width=2),
        name="Expected Exposure (EE)",
    ))
    
    # Mean MtM
    fig.add_trace(go.Scatter(
        x=time, y=profile.mean_mtm, mode="lines",
        line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
        name="Mean MtM",
    ))
    
    # Collateralized profiles
    if profile_net is not None:
        fig.add_trace(go.Scatter(
            x=time, y=profile_net.potential_future_exposure, mode="lines",
            line=dict(color=COLORS["positive"], width=2, dash="dash"),
            name=f"PFE (collateralized)",
        ))
        fig.add_trace(go.Scatter(
            x=time, y=profile_net.expected_exposure, mode="lines",
            line=dict(color=COLORS["negative"], width=1.5, dash="dash"),
            name="EE (collateralized)",
        ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Counterparty exposure profile",
        xaxis_title="Time (years)",
        yaxis_title="Exposure (EUR)",
    )
    fig.update_yaxes(tickformat=",.0f")
    return fig


def plot_exposure_distribution(mtm_values: np.ndarray, time_idx: int,
                               dt: float, fixed_price: float) -> go.Figure:
    """Plot MtM distribution at a specific time point."""
    values = mtm_values[:, time_idx]
    t_years = time_idx * dt
    
    fig = go.Figure()
    
    # Separate positive and negative
    fig.add_trace(go.Histogram(
        x=values[values >= 0], nbinsx=60,
        marker_color=COLORS["positive"], opacity=0.6,
        name="Positive exposure (credit risk)",
    ))
    fig.add_trace(go.Histogram(
        x=values[values < 0], nbinsx=60,
        marker_color=COLORS["negative"], opacity=0.6,
        name="Negative exposure (owe counterparty)",
    ))
    
    fig.add_vline(x=0, line_color="black", line_width=1)
    
    pct_positive = np.mean(values > 0) * 100
    fig.add_annotation(
        x=0.95, y=0.95, xref="paper", yref="paper",
        text=f"{pct_positive:.1f}% of paths have positive exposure",
        showarrow=False, font=dict(size=11),
        bgcolor="rgba(255,255,255,0.8)",
    )
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"MtM distribution at t = {t_years:.1f} years",
        xaxis_title="Mark-to-Market (EUR)",
        yaxis_title="Frequency",
        barmode="overlay",
    )
    fig.update_xaxes(tickformat=",.0f")
    return fig


def plot_sensitivity(result) -> go.Figure:
    """Plot sensitivity analysis results."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Peak PFE", "EPE"))
    
    # Peak PFE
    fig.add_trace(go.Scatter(
        x=result.parameter_values, y=result.peak_pfe, mode="lines+markers",
        line=dict(color=COLORS["positive"], width=2),
        marker=dict(size=6), name="Gross",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=result.parameter_values, y=result.peak_pfe_collateralized,
        mode="lines+markers",
        line=dict(color=COLORS["positive"], width=2, dash="dash"),
        marker=dict(size=6), name="Collateralized",
    ), row=1, col=1)
    
    # EPE
    fig.add_trace(go.Scatter(
        x=result.parameter_values, y=result.epe, mode="lines+markers",
        line=dict(color=COLORS["negative"], width=2),
        marker=dict(size=6), name="Gross", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=result.parameter_values, y=result.epe_collateralized,
        mode="lines+markers",
        line=dict(color=COLORS["negative"], width=2, dash="dash"),
        marker=dict(size=6), name="Collateralized", showlegend=False,
    ), row=1, col=2)
    
    # Format parameter name for display
    display_name = result.parameter_name.replace("_", " ").title()
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Sensitivity to {display_name}",
        height=400,
    )
    fig.update_xaxes(title_text=display_name, row=1, col=1)
    fig.update_xaxes(title_text=display_name, row=1, col=2)
    fig.update_yaxes(title_text="EUR", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="EUR", tickformat=",.0f", row=1, col=2)
    return fig


def plot_convergence(convergence_data: dict) -> go.Figure:
    """Plot PFE convergence analysis."""
    paths = convergence_data["path_counts"]
    pfe = convergence_data["peak_pfe"]
    se = convergence_data["std_errors"]
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Peak PFE convergence",
                                       "Standard error"))
    
    fig.add_trace(go.Scatter(
        x=paths, y=pfe, mode="lines+markers",
        line=dict(color=COLORS["positive"], width=2),
        marker=dict(size=6), name="Peak PFE",
    ), row=1, col=1)
    
    # Reference line (last value)
    fig.add_hline(y=pfe[-1], line_dash="dash", line_color=COLORS["neutral"],
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=paths, y=se, mode="lines+markers",
        line=dict(color=COLORS["secondary"], width=2),
        marker=dict(size=6), name="Std Error",
    ), row=1, col=2)
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Monte Carlo convergence analysis",
        height=400,
    )
    fig.update_xaxes(title_text="Number of paths", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Number of paths", type="log", row=1, col=2)
    fig.update_yaxes(title_text="EUR", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="EUR", tickformat=",.0f", row=1, col=2)
    return fig


def plot_qq(diagnostics: dict) -> go.Figure:
    """Plot QQ diagram for distribution diagnostics."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=diagnostics["theoretical_quantiles"],
        y=diagnostics["empirical_quantiles"],
        mode="markers",
        marker=dict(color=COLORS["secondary"], size=4, opacity=0.6),
        name="QQ points",
    ))
    
    # 45-degree line
    all_vals = np.concatenate([diagnostics["theoretical_quantiles"],
                               diagnostics["empirical_quantiles"]])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(color=COLORS["positive"], dash="dash"),
        name="Normal reference",
    ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="QQ plot (MtM vs Normal distribution)",
        xaxis_title="Theoretical quantiles",
        yaxis_title="Empirical quantiles",
        height=400,
    )
    return fig
