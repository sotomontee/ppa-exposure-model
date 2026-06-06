import streamlit as st
import numpy as np
import pandas as pd
import time

from config import (
    PPA_DEFAULTS, PRICE_MODEL_DEFAULTS, SIMULATION_DEFAULTS,
    RISK_DEFAULTS,
)
from models.price_model import (PriceModelParams, simulate_prices, generate_forward_curve,
                                generate_sample_market_curve)
from models.ppa import PPAContract, calculate_mtm_matrix
from models.exposure import compute_exposure_profile, compute_exposure_at_time
from analysis.sensitivity import sensitivity_sweep
from analysis.validation import convergence_analysis, distribution_diagnostics
from ai.commentary import generate_risk_commentary, prepare_metrics_for_ai
from utils.charts import (
    plot_price_paths, plot_price_distribution, plot_exposure_profile,
    plot_exposure_distribution, plot_sensitivity, plot_convergence,
    plot_qq, format_eur,
)


st.set_page_config(
    page_title="PPA Exposure Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f7fafc;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #2c5282;
    }
    .metric-card.warn {
        border-left-color: #c05621;
    }
    .metric-card.danger {
        border-left-color: #e24b4a;
    }
    .metric-label {
        font-size: 13px;
        color: #718096;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #1a365d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("⚡ PPA Exposure Model")
    st.caption("Schwartz One-Factor Monte Carlo Engine")
    st.divider()

    with st.expander("📋 PPA Contract", expanded=True):
        fixed_price = st.number_input(
            "Fixed price (EUR/MWh)", value=PPA_DEFAULTS["fixed_price"],
            min_value=10.0, max_value=200.0, step=1.0,
        )
        capacity_mw = st.number_input(
            "Capacity (MW)", value=PPA_DEFAULTS["capacity_mw"],
            min_value=1.0, max_value=1000.0, step=10.0,
        )
        capacity_factor = st.slider(
            "Capacity factor", min_value=0.1, max_value=0.95,
            value=PPA_DEFAULTS["capacity_factor"], step=0.05,
            help="Expected average generation as fraction of nameplate capacity. "
                 "Typical: 0.25-0.35 onshore wind, 0.40-0.55 offshore wind, 0.10-0.20 solar.",
        )
        tenor_years = st.selectbox(
            "Tenor (years)", options=[3, 5, 7, 10, 15],
            index=1,
        )

    with st.expander("📈 Price Model", expanded=True):
        spot_price = st.number_input(
            "Current spot (EUR/MWh)", value=PRICE_MODEL_DEFAULTS["spot_price"],
            min_value=5.0, max_value=200.0, step=1.0,
        )
        long_run_mean = st.number_input(
            "Long-run mean (EUR/MWh)", value=PRICE_MODEL_DEFAULTS["long_run_mean"],
            min_value=5.0, max_value=200.0, step=1.0,
            help="Equilibrium price level the model reverts to.",
        )
        volatility = st.slider(
            "Volatility (σ)", min_value=0.10, max_value=0.80,
            value=PRICE_MODEL_DEFAULTS["volatility"], step=0.05,
            help="Annualized log-normal volatility.",
        )
        mean_reversion = st.slider(
            "Mean reversion (κ)", min_value=0.1, max_value=5.0,
            value=PRICE_MODEL_DEFAULTS["mean_reversion"], step=0.1,
            help="Speed at which prices revert to the long-run mean. "
                 "Higher = faster reversion. Typical: 1.0-3.0 for power/gas.",
        )
        seasonality_amplitude = st.slider(
            "Seasonality amplitude", min_value=0.0, max_value=0.40,
            value=PRICE_MODEL_DEFAULTS["seasonality_amplitude"], step=0.05,
            help="Amplitude of seasonal component in log-price. "
                 "0.15 = ±15% seasonal swing.",
        )

    with st.expander("📉 Forward Curve", expanded=True):
        curve_mode = st.radio(
            "Forward curve source",
            options=["Model-implied", "Sample market curve", "Upload CSV"],
            help="**Model-implied**: generated from price model parameters. "
                 "**Sample market curve**: realistic European baseload shape. "
                 "**Upload CSV**: provide your own forward curve.",
        )

        market_curve = None

        if curve_mode == "Sample market curve":
            sample_base = st.number_input(
                "Base price (EUR/MWh)", value=48.0,
                min_value=10.0, max_value=200.0, step=1.0,
                help="Starting price for the sample Nord Pool-style curve.",
            )
            n_steps_curve = int(tenor_years * 12)
            market_curve = generate_sample_market_curve(
                n_steps_curve, 1/12, sample_base
            )
            st.caption(f"Sample curve: €{market_curve[0]:.1f} → €{market_curve[-1]:.1f}/MWh")

        elif curve_mode == "Upload CSV":
            st.caption(
                "Upload a CSV with columns: **month** (1, 2, ...) and "
                "**price** (EUR/MWh). One row per monthly delivery period."
            )
            uploaded_file = st.file_uploader(
                "Forward curve CSV", type=["csv"],
                help="Monthly forward prices for the PPA tenor.",
            )
            if uploaded_file is not None:
                try:
                    df_curve = pd.read_csv(uploaded_file)
                    if "price" in df_curve.columns:
                        market_curve = df_curve["price"].values
                        if len(market_curve) > 0 and market_curve[0] != spot_price:
                            market_curve = np.concatenate([[spot_price], market_curve])
                        st.success(f"Loaded {len(market_curve)} price points.")
                    else:
                        st.error("CSV must have a 'price' column.")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            else:
                st.info("Upload a CSV file to use a custom forward curve.")

    with st.expander("⚙️ Simulation", expanded=False):
        n_paths = st.select_slider(
            "Number of paths",
            options=[1000, 2000, 5000, 10000, 20000, 50000],
            value=SIMULATION_DEFAULTS["n_paths"],
        )
        seed = st.number_input(
            "Random seed", value=SIMULATION_DEFAULTS["random_seed"],
            min_value=0, max_value=99999, step=1,
        )

    with st.expander("📊 Risk Parameters", expanded=False):
        confidence_level = st.select_slider(
            "PFE confidence level",
            options=[0.95, 0.975, 0.99],
            value=RISK_DEFAULTS["confidence_level"],
            format_func=lambda x: f"{x*100:.1f}%",
        )
        discount_rate = st.slider(
            "Discount rate", min_value=0.0, max_value=0.10,
            value=RISK_DEFAULTS["discount_rate"], step=0.005,
            format="%.3f",
        )

    with st.expander("🤖 AI Commentary", expanded=False):
        api_key = st.text_input(
            "Anthropic API key", type="password",
            help="Required for AI-generated risk commentary. "
                 "Your key is not stored.",
        )
        if not api_key:
            try:
                api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            except Exception:
                api_key = ""

    st.divider()
    run_button = st.button("▶ Run simulation", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False)
def run_simulation(
    _fixed_price, _capacity_mw, _capacity_factor, _tenor_years,
    _spot_price, _long_run_mean, _volatility, _mean_reversion,
    _seasonality_amplitude, _n_paths, _seed,
    _confidence_level, _discount_rate,
    _market_curve_tuple=None,
):
    mkt_curve = np.array(_market_curve_tuple) if _market_curve_tuple is not None else None

    params = PriceModelParams(
        spot_price=_spot_price,
        long_run_mean=_long_run_mean,
        volatility=_volatility,
        mean_reversion=_mean_reversion,
        seasonality_amplitude=_seasonality_amplitude,
        market_forward_curve=mkt_curve,
    )
    contract = PPAContract(
        fixed_price=_fixed_price,
        capacity_mw=_capacity_mw,
        capacity_factor=_capacity_factor,
        tenor_years=_tenor_years,
    )

    dt = 1.0 / 12
    n_steps = int(_tenor_years * 12)

    prices = simulate_prices(params, _n_paths, n_steps, dt, _seed)
    fwd_curve = generate_forward_curve(params, n_steps, dt)
    mtm = calculate_mtm_matrix(prices, contract, params, _discount_rate, dt)
    profile = compute_exposure_profile(mtm, dt, _confidence_level)

    return {
        "prices": prices,
        "fwd_curve": fwd_curve,
        "mtm": mtm,
        "profile": profile,
        "params": params,
        "contract": contract,
        "dt": dt,
        "n_steps": n_steps,
    }


if run_button or "results" not in st.session_state:
    with st.spinner("Running Monte Carlo simulation..."):
        start = time.time()
        mkt_tuple = tuple(market_curve) if market_curve is not None else None
        results = run_simulation(
            fixed_price, capacity_mw, capacity_factor, tenor_years,
            spot_price, long_run_mean, volatility, mean_reversion,
            seasonality_amplitude, n_paths, seed,
            confidence_level, discount_rate,
            mkt_tuple,
        )
        elapsed = time.time() - start
        st.session_state["results"] = results
        st.session_state["elapsed"] = elapsed

if "results" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run Simulation**.")
    st.stop()

results = st.session_state["results"]
elapsed = st.session_state.get("elapsed", 0)
profile = results["profile"]


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary",
    "📈 Price Simulation",
    "🎯 Exposure Profile",
    "🔬 Sensitivity Analysis",
    "✅ Model Validation",
    "📖 Methodology",
])

with tab1:
    st.header("Executive summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Peak PFE", format_eur(profile.peak_pfe),
                   help=f"At t = {profile.peak_pfe_time:.1f} years")
    with col2:
        st.metric("EPE", format_eur(profile.epe))
    with col3:
        st.metric("Effective EPE", format_eur(profile.effective_epe))
    with col4:
        st.metric("Peak PFE Time", f"{profile.peak_pfe_time:.1f} years")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Contract parameters")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Type | Fixed-for-Float PPA |
        | Fixed price | €{fixed_price:.1f}/MWh |
        | Capacity | {capacity_mw:.0f} MW |
        | Capacity factor | {capacity_factor:.0%} |
        | Tenor | {tenor_years} years |
        | Annual volume | {capacity_mw * capacity_factor * 8766:,.0f} MWh |
        """)

    with col2:
        st.subheader("Model parameters")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Forward curve | {curve_mode} |
        | Current spot | €{spot_price:.1f}/MWh |
        | Long-run mean | €{long_run_mean:.1f}/MWh |
        | Volatility (σ) | {volatility:.0%} |
        | Mean reversion (κ) | {mean_reversion:.1f} |
        | Seasonality | ±{seasonality_amplitude:.0%} |
        | Paths | {n_paths:,} |
        | Runtime | {elapsed:.1f}s |
        """)

    st.divider()

    st.subheader("🤖 AI risk commentary")
    if api_key:
        if st.button("Generate commentary", key="ai_btn"):
            with st.spinner("Generating AI analysis..."):
                metrics = prepare_metrics_for_ai(
                    profile, None,
                    vars(results["params"]),
                    vars(results["contract"]),
                    {},
                )
                commentary = generate_risk_commentary(metrics, api_key)
                st.markdown(commentary)
    else:
        st.info(
            "Enter your Anthropic API key in the sidebar under "
            "'AI Commentary' to enable AI-generated risk analysis."
        )


with tab2:
    st.header("Price simulation")

    fig_paths = plot_price_paths(
        results["prices"], results["dt"], 100, results["fwd_curve"]
    )
    st.plotly_chart(fig_paths, use_container_width=True)

    st.subheader("Price distribution at time point")
    dt = results["dt"]
    n_steps = results["n_steps"]
    time_point = st.slider(
        "Select time (years)", 0.0, float(tenor_years),
        value=float(tenor_years) / 2, step=float(dt),
        key="price_time",
    )
    time_idx = min(int(time_point / dt), n_steps)

    fig_dist = plot_price_distribution(results["prices"], time_idx, dt)
    st.plotly_chart(fig_dist, use_container_width=True)

    prices_t = results["prices"][:, time_idx]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"€{np.mean(prices_t):.1f}/MWh")
    col2.metric("Std Dev", f"€{np.std(prices_t):.1f}/MWh")
    col3.metric("5th pctl", f"€{np.percentile(prices_t, 5):.1f}/MWh")
    col4.metric("95th pctl", f"€{np.percentile(prices_t, 95):.1f}/MWh")


with tab3:
    st.header("Exposure profile")

    fig_exp = plot_exposure_profile(profile, confidence_level)
    st.plotly_chart(fig_exp, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Key metrics**")
        st.markdown(f"- Peak PFE: {format_eur(profile.peak_pfe)}")
        st.markdown(f"- Peak time: {profile.peak_pfe_time:.1f} years")
        st.markdown(f"- EPE: {format_eur(profile.epe)}")
        st.markdown(f"- Eff. EPE: {format_eur(profile.effective_epe)}")
    with col2:
        st.markdown("**Interpretation**")
        st.markdown(
            f"The maximum potential credit exposure under the "
            f"{confidence_level*100:.0f}% confidence level is "
            f"{format_eur(profile.peak_pfe)}, occurring at "
            f"t = {profile.peak_pfe_time:.1f} years. This represents "
            f"the counterparty credit limit required for this PPA."
        )

    st.divider()
    st.subheader("MtM distribution at time point")

    time_point_exp = st.slider(
        "Select time (years)", 0.0, float(tenor_years),
        value=profile.peak_pfe_time, step=float(dt),
        key="exp_time",
    )
    time_idx_exp = min(int(time_point_exp / dt), n_steps)

    fig_mtm_dist = plot_exposure_distribution(
        results["mtm"], time_idx_exp, dt, fixed_price
    )
    st.plotly_chart(fig_mtm_dist, use_container_width=True)

    dist_stats = compute_exposure_at_time(results["mtm"], time_idx_exp)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean MtM", format_eur(dist_stats["mean"]))
    col2.metric("EE", format_eur(dist_stats["ee"]))
    col3.metric(f"PFE {confidence_level*100:.0f}%", format_eur(dist_stats["pfe_99"]))
    col4.metric("% Positive", f"{dist_stats['pct_positive']:.1f}%")


with tab4:
    st.header("Sensitivity analysis")
    st.caption("Impact of key parameters on exposure metrics. "
               "Uses reduced path count (2,000) for speed.")

    sens_param = st.selectbox(
        "Parameter to sweep",
        options=["volatility", "mean_reversion", "fixed_price"],
        format_func=lambda x: {
            "volatility": "Volatility (σ)",
            "mean_reversion": "Mean reversion (κ)",
            "fixed_price": "Fixed price (EUR/MWh)",
        }[x],
    )

    sweep_ranges = {
        "volatility": np.arange(0.10, 0.85, 0.05),
        "mean_reversion": np.arange(0.2, 5.2, 0.2),
        "fixed_price": np.arange(30, 85, 5.0),
    }

    if st.button("Run sensitivity", key="sens_btn"):
        with st.spinner(f"Running sensitivity sweep on {sens_param}..."):
            from models.collateral import CSAParams
            dummy_csa = CSAParams(threshold=1e18, mta=1e18, independent_amount=0,
                                  margin_period_of_risk=0)
            sens_result = sensitivity_sweep(
                results["params"], results["contract"], dummy_csa,
                sens_param, sweep_ranges[sens_param],
                2000, results["n_steps"], 1.0/12,
                discount_rate, confidence_level, seed,
            )
            fig_sens = plot_sensitivity(sens_result)
            st.plotly_chart(fig_sens, use_container_width=True)

            pfe_range = sens_result.peak_pfe
            st.caption(
                f"Peak PFE ranges from {format_eur(np.min(pfe_range))} to "
                f"{format_eur(np.max(pfe_range))} across the sweep range "
                f"({np.max(pfe_range)/max(np.min(pfe_range),1):.1f}x)."
            )


with tab5:
    st.header("Model validation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Convergence analysis")
        st.caption("Does PFE stabilize as we add more simulation paths?")

        if st.button("Run convergence test", key="conv_btn"):
            with st.spinner("Running convergence analysis (this may take a moment)..."):
                conv_data = convergence_analysis(
                    results["params"], results["contract"],
                    discount_rate, results["dt"], confidence_level,
                    path_counts=[500, 1000, 2000, 5000, 10000, 20000],
                    seed=seed,
                )
                fig_conv = plot_convergence(conv_data)
                st.plotly_chart(fig_conv, use_container_width=True)

    with col2:
        st.subheader("Distribution diagnostics")
        st.caption("Is the MtM distribution well-behaved?")

        diag_time = st.slider(
            "Time point (years)", 0.0, float(tenor_years),
            value=profile.peak_pfe_time, step=float(dt),
            key="diag_time",
        )
        diag_idx = min(int(diag_time / dt), n_steps)

        if st.button("Run diagnostics", key="diag_btn"):
            try:
                diag = distribution_diagnostics(results["mtm"], diag_idx)

                st.markdown(f"""
                | Statistic | Value |
                |---|---|
                | Mean | {format_eur(diag['mean'])} |
                | Std Dev | {format_eur(diag['std'])} |
                | Skewness | {diag['skewness']:.3f} |
                | Excess Kurtosis | {diag['excess_kurtosis']:.3f} |
                | Jarque-Bera | {diag['jb_statistic']:.1f} |
                """)

                fig_qq = plot_qq(diag)
                st.plotly_chart(fig_qq, use_container_width=True)
            except ImportError:
                st.warning("scipy is required for distribution diagnostics.")


with tab6:
    st.header("Methodology")

    st.markdown("""
    ### 1. What this model does

    This tool estimates the **counterparty credit exposure** arising from a
    fixed-for-float Power Purchase Agreement (PPA) over its full lifetime.
    It answers: *"If this counterparty defaults at any point during the
    contract, how much could we lose?"*

    The model simulates thousands of possible future power price scenarios
    using Monte Carlo simulation, revalues the PPA under each scenario at
    monthly intervals, and extracts the exposure profile — the range of
    potential credit losses over time.

    ---

    ### 2. Price dynamics

    Power prices don't behave like equities — they tend to **revert to an
    equilibrium level** driven by supply and demand fundamentals, and they
    exhibit **seasonal patterns** (higher in winter, lower in summer in
    European markets).

    The model captures both features using the Schwartz one-factor framework,
    a standard approach in energy risk:

    $$dX_t = \\kappa(\\mu(t) - X_t)\\,dt + \\sigma\\,dW_t$$

    where $X_t = \\ln(S_t)$, $\\kappa$ controls how fast prices revert,
    $\\mu(t)$ is the seasonal equilibrium level, and $\\sigma$ is volatility.
    The simulation uses monthly time steps with 10,000 paths by default.

    ---

    ### 3. Forward curve

    The model supports two modes:

    **Model-implied** — the forward curve is derived analytically from the
    price model parameters. Useful for scenario analysis and stress testing.

    **Market-anchored** — the user provides an observed forward curve (e.g.,
    from Nord Pool or EEX), and the simulation generates stochastic deviations
    around it while ensuring that the expected simulated price at each tenor
    matches the market forward:

    $$\\mathbb{E}[S_t] = F_{mkt}(0, t) \\quad \\forall \\, t$$

    This is the standard production approach — it separates the market's
    view of fair value (the forward curve) from the model's role (generating
    the uncertainty around it).

    ---

    ### 4. PPA valuation

    The PPA is modeled from the **generator's perspective** (fixed price
    receiver). At each time step, the Mark-to-Market is the present value
    of all remaining cash flows:

    $$\\text{MtM}(t) = \\sum_{i: t_i > t} (K - F(t, t_i)) \\cdot Q_i \\cdot DF(t, t_i)$$

    where $K$ is the fixed price, $F(t, t_i)$ is the model-implied forward
    (consistent with information at time $t$, not realized future prices),
    $Q_i$ is the delivered volume, and $DF$ is the discount factor.

    Positive MtM means the counterparty owes us — this is where credit
    exposure exists. Negative MtM means we owe them — no credit risk.

    ---

    ### 5. Exposure metrics

    | Metric | What it tells you | Used for |
    |--------|-------------------|----------|
    | **EE** (Expected Exposure) | Average credit exposure at each point in time | Portfolio monitoring |
    | **PFE** (Potential Future Exposure) | Worst-case exposure at a given confidence level (e.g., 99%) | Credit limit setting |
    | **EPE** (Expected Positive Exposure) | Time-weighted average of EE over the contract life | CVA calculation |
    | **Effective EPE** | Non-decreasing version of EPE | Regulatory capital (Basel III / SA-CCR) |
    | **Peak PFE** | Maximum PFE across all time horizons | Credit limit allocation |

    The **Peak PFE** is the key output for credit risk management — it represents
    the maximum counterparty credit limit required to support this PPA.

    ---

    ### 6. Known limitations and future extensions

    | Limitation | Impact | Potential extension |
    |------------|--------|---------------------|
    | Single stochastic factor | Cannot capture decorrelation between short and long-term prices | Schwartz-Smith two-factor model |
    | No price spikes (jumps) | Underestimates short-horizon tail risk | Jump-diffusion (Merton / Kou) |
    | Constant volatility | Vol is assumed flat across tenors and time | Stochastic volatility or term structure of vol |
    | Deterministic generation | Capacity factor is fixed; real wind/solar output is stochastic | Correlated volume-price simulation |

    These are documented as areas for incremental extension. The current
    prototype provides a methodologically sound foundation.
    """)
