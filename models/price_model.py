import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PriceModelParams:
    spot_price: float = 50.0
    long_run_mean: float = 52.0
    volatility: float = 0.35
    mean_reversion: float = 1.5
    seasonality_amplitude: float = 0.15
    seasonality_phase: float = 0.0
    market_forward_curve: Optional[np.ndarray] = field(default=None, repr=False)


def seasonal_mean(t, mu_base, amplitude, phase):
    return mu_base + amplitude * np.sin(2 * np.pi * (t - phase))


def simulate_prices(params, n_paths, n_steps, dt, seed=42):
    rng = np.random.default_rng(seed)
    kappa = params.mean_reversion
    sigma = params.volatility
    mu_base = np.log(params.long_run_mean)

    log_prices = np.zeros((n_paths, n_steps + 1))
    log_prices[:, 0] = np.log(params.spot_price)
    dW = rng.standard_normal((n_paths, n_steps))

    if params.market_forward_curve is not None:
        mkt_curve = params.market_forward_curve
        if len(mkt_curve) < n_steps + 1:
            extension = np.full(n_steps + 1 - len(mkt_curve), mkt_curve[-1])
            mkt_curve = np.concatenate([mkt_curve, extension])

        log_mkt = np.log(mkt_curve[:n_steps + 1])
        Y = np.zeros((n_paths, n_steps + 1))

        for t_idx in range(n_steps):
            Y[:, t_idx + 1] = (
                Y[:, t_idx]
                - kappa * Y[:, t_idx] * dt
                + sigma * np.sqrt(dt) * dW[:, t_idx]
            )

        times = np.arange(n_steps + 1) * dt
        var_Y = np.where(
            times > 0,
            (sigma ** 2) / (2 * kappa) * (1 - np.exp(-2 * kappa * times)),
            0.0,
        )
        # Convexity correction ensures E[S_t] = F_mkt(0,t)
        log_prices = log_mkt[np.newaxis, :] + Y - var_Y[np.newaxis, :] / 2

    else:
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


def implied_forward(s_t, t, T, params):
    kappa = params.mean_reversion
    sigma = params.volatility
    tau = T - t

    if tau <= 0:
        return s_t

    decay = np.exp(-kappa * tau)
    v = (sigma ** 2) / (2 * kappa) * (1 - np.exp(-2 * kappa * tau))

    if params.market_forward_curve is not None:
        mkt_curve = params.market_forward_curve
        dt_approx = 1.0 / 12
        t_idx = min(int(round(t / dt_approx)), len(mkt_curve) - 1)
        T_idx = min(int(round(T / dt_approx)), len(mkt_curve) - 1)

        log_mkt_t = np.log(mkt_curve[t_idx])
        log_mkt_T = np.log(mkt_curve[T_idx])
        Y_t = np.log(s_t) - log_mkt_t
        log_fwd = log_mkt_T + decay * Y_t
        return np.exp(log_fwd)

    else:
        mu_base = np.log(params.long_run_mean)
        mu_T = seasonal_mean(T, mu_base, params.seasonality_amplitude,
                             params.seasonality_phase)
        m = decay * np.log(s_t) + (1 - decay) * mu_T
        return np.exp(m + v / 2)


def generate_forward_curve(params, n_steps, dt):
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
        curve[i] = implied_forward(np.array([params.spot_price]), 0.0, T, params)[0]
    return curve


def generate_sample_market_curve(n_steps, dt, base_price=48.0):
    curve = np.zeros(n_steps + 1)
    curve[0] = base_price
    for i in range(1, n_steps + 1):
        t = i * dt
        contango = 5.0 * (1 - np.exp(-0.5 * t))
        seasonal = 4.0 * np.sin(2 * np.pi * (t - 0.0))
        drift = 0.8 * t
        curve[i] = base_price + contango + seasonal + drift
    return curve
