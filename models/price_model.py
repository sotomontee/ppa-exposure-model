"""
Schwartz One-Factor Mean-Reverting Price Model with Seasonality.

Supports two modes:
    1. Model-implied forward curve (synthetic, from parameters)
    2. Market-anchored simulation (user-provided forward curve)

In market-anchored mode, the simulation generates stochastic deviations
around the observed forward curve, ensuring that the expected simulated
price at each tenor matches the market forward. This is the standard
approach in production risk systems.

Model specification (Ornstein-Uhlenbeck in log-space):

    dX_t = kappa * (mu(t) - X_t) * dt + sigma * dW_t

    where X_t = ln(S_t) and mu(t) = mu_base + A * sin(2*pi*(t - phi))

References:
    Schwartz, E.S. (1997). "The Stochastic Behavior of Commodity Prices:
    Implications for Valuation and Hedging." Journal of Finance, 52(3).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PriceModelParams:
    """Parameters for the Schwartz one-factor price model."""
    spot_price: float = 50.0
    long_run_mean: float = 52.0
    volatility: float = 0.35
    mean_reversion: float = 1.5
    seasonality_amplitude: float = 0.15
    seasonality_phase: float = 0.0
    market_forward_curve: Optional[np.ndarray] = field(default=None, repr=False)


def seasonal_mean(t: float, mu_base: float, amplitude: float, phase: float) -> float:
    """
    Compute the seasonal long-run mean at time t.

    mu(t) = mu_base + A * sin(2*pi*(t - phi))

    Phase convention: phi=0 means peak in Q1 (winter), which is typical
    for European power/gas markets.
    """
    return mu_base + amplitude * np.sin(2 * np.pi * (t - phase))


def simulate_prices(params: PriceModelParams, n_paths: int, n_steps: int,
                    dt: float, seed: int = 42) -> np.ndarray:
    """
    Simulate price paths using Euler-Maruyama discretization.

    If a market forward curve is provided, the simulation is anchored to it:
    simulated prices fluctuate around the market curve rather than the
    model-implied curve. This ensures E[S_t] ≈ F_market(0, t).

    Args:
        params: Model parameters (including optional market_forward_curve)
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps
        dt: Time step size in years (e.g., 1/12 for monthly)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (n_paths, n_steps + 1) including initial price
    """
    rng = np.random.default_rng(seed)

    kappa = params.mean_reversion
    sigma = params.volatility
    mu_base = np.log(params.long_run_mean)

    log_prices = np.zeros((n_paths, n_steps + 1))
    log_prices[:, 0] = np.log(params.spot_price)

    # Pre-generate all random shocks
    dW = rng.standard_normal((n_paths, n_steps))

    if params.market_forward_curve is not None:
        # ── Market-anchored mode ──
        # Simulate mean-reverting deviations around the market curve.
        # Let Y_t = X_t - ln(F_market(0,t)) be the deviation from the
        # market forward. Y follows:
        #   dY_t = -kappa * Y_t * dt + sigma * dW_t
        # Then S_t = F_market(0,t) * exp(Y_t - var(Y_t)/2)
        # The variance correction ensures E[S_t] = F_market(0,t).

        mkt_curve = params.market_forward_curve
        # Ensure curve covers all steps
        if len(mkt_curve) < n_steps + 1:
            # Extend with last value
            extension = np.full(n_steps + 1 - len(mkt_curve), mkt_curve[-1])
            mkt_curve = np.concatenate([mkt_curve, extension])

        log_mkt = np.log(mkt_curve[:n_steps + 1])

        # Simulate deviation process Y_t (starts at 0)
        Y = np.zeros((n_paths, n_steps + 1))

        for t_idx in range(n_steps):
            Y[:, t_idx + 1] = (
                Y[:, t_idx]
                - kappa * Y[:, t_idx] * dt
                + sigma * np.sqrt(dt) * dW[:, t_idx]
            )

        # Variance of Y at each time step (analytical)
        times = np.arange(n_steps + 1) * dt
        var_Y = np.where(
            times > 0,
            (sigma ** 2) / (2 * kappa) * (1 - np.exp(-2 * kappa * times)),
            0.0,
        )

        # Anchored prices: F_market * exp(Y - var/2)
        # The -var/2 is a convexity correction ensuring E[exp(Y)] = exp(var/2)
        # so E[S_t] = F_market * exp(var/2) * exp(-var/2) = F_market
        log_prices = log_mkt[np.newaxis, :] + Y - var_Y[np.newaxis, :] / 2

    else:
        # ── Model-implied mode ──
        for t_idx in range(n_steps):
            t_years = t_idx * dt
            mu_t = seasonal_mean(t_years, mu_base, params.seasonality_amplitude,
                                 params.seasonality_phase)

            log_prices[:, t_idx + 1] = (
                log_prices[:, t_idx]
                + kappa * (mu_t - log_prices[:, t_idx]) * dt
                + sigma * np.sqrt(dt) * dW[:, t_idx]
            )

    return np.exp(log_prices)


def implied_forward(s_t: np.ndarray, t: float, T: float,
                    params: PriceModelParams) -> np.ndarray:
    """
    Compute the model-implied forward price F(t,T) given current spot S_t.

    If a market forward curve is provided, uses it as the base and adjusts
    for the deviation from it at time t. Otherwise, uses the analytical
    Schwartz formula.

    Under the Schwartz model (no market curve):
        F(t,T) = exp(m(t,T) + v(t,T)/2)

    where:
        m(t,T) = e^{-kappa*tau} * ln(S_t) + (1 - e^{-kappa*tau}) * mu(T)
        v(t,T) = sigma^2 / (2*kappa) * (1 - e^{-2*kappa*tau})
        tau = T - t

    Args:
        s_t: Current spot price(s), scalar or array
        t: Current time in years
        T: Forward delivery time in years
        params: Model parameters

    Returns:
        Forward price(s), same shape as s_t
    """
    kappa = params.mean_reversion
    sigma = params.volatility
    tau = T - t

    if tau <= 0:
        return s_t

    decay = np.exp(-kappa * tau)
    v = (sigma ** 2) / (2 * kappa) * (1 - np.exp(-2 * kappa * tau))

    if params.market_forward_curve is not None:
        # Market-anchored forward pricing:
        # The forward at time t for delivery at T, given that the spot
        # at time t deviates from the market curve by Y_t = ln(S_t) - ln(F_mkt(0,t)):
        #
        # F(t,T) = F_mkt(0,T) * exp(decay * Y_t + v/2 - v_t_to_T_correction)
        #
        # Simplified: use the mean-reverting deviation from market curve
        mkt_curve = params.market_forward_curve
        dt_approx = 1.0 / 12  # Assume monthly
        t_idx = min(int(round(t / dt_approx)), len(mkt_curve) - 1)
        T_idx = min(int(round(T / dt_approx)), len(mkt_curve) - 1)

        log_mkt_t = np.log(mkt_curve[t_idx])
        log_mkt_T = np.log(mkt_curve[T_idx])

        # Deviation of current spot from market curve at time t
        Y_t = np.log(s_t) - log_mkt_t

        # Forward = F_mkt(0,T) * exp(decay * Y_t)
        # The decay factor means deviations from the curve revert
        log_fwd = log_mkt_T + decay * Y_t
        return np.exp(log_fwd)

    else:
        mu_base = np.log(params.long_run_mean)
        mu_T = seasonal_mean(T, mu_base, params.seasonality_amplitude,
                             params.seasonality_phase)

        m = decay * np.log(s_t) + (1 - decay) * mu_T
        return np.exp(m + v / 2)


def generate_forward_curve(params: PriceModelParams, n_steps: int,
                           dt: float) -> np.ndarray:
    """
    Generate the forward curve F(0, T) for all delivery periods.

    If a market forward curve is provided, returns it directly.
    Otherwise, generates the model-implied curve.

    Args:
        params: Model parameters
        n_steps: Number of future delivery periods
        dt: Time step in years

    Returns:
        np.ndarray: Shape (n_steps + 1,) forward curve including spot
    """
    if params.market_forward_curve is not None:
        mkt = params.market_forward_curve
        if len(mkt) >= n_steps + 1:
            return mkt[:n_steps + 1]
        else:
            extension = np.full(n_steps + 1 - len(mkt), mkt[-1])
            return np.concatenate([mkt, extension])

    curve = np.zeros(n_steps + 1)
    curve[0] = params.spot_price

    for i in range(1, n_steps + 1):
        T = i * dt
        curve[i] = implied_forward(
            np.array([params.spot_price]), 0.0, T, params
        )[0]

    return curve


def generate_sample_market_curve(n_steps: int, dt: float,
                                  base_price: float = 48.0) -> np.ndarray:
    """
    Generate a realistic sample European baseload power forward curve.

    Based on typical Nord Pool / EEX forward curve shapes:
    - Slight contango in the first 1-2 years
    - Seasonal pattern (winter premium)
    - Gradual flattening at longer tenors

    Args:
        n_steps: Number of periods
        dt: Time step in years
        base_price: Starting spot price

    Returns:
        np.ndarray: Shape (n_steps + 1,) realistic forward curve
    """
    curve = np.zeros(n_steps + 1)
    curve[0] = base_price

    for i in range(1, n_steps + 1):
        t = i * dt
        # Contango component (flattening toward long-run level)
        contango = 5.0 * (1 - np.exp(-0.5 * t))
        # Seasonal component (winter premium)
        seasonal = 4.0 * np.sin(2 * np.pi * (t - 0.0))
        # Small upward drift for green premium
        drift = 0.8 * t
        curve[i] = base_price + contango + seasonal + drift

    return curve
