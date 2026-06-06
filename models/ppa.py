import numpy as np
from dataclasses import dataclass
from models.price_model import PriceModelParams, implied_forward


@dataclass
class PPAContract:
    fixed_price: float = 55.0
    capacity_mw: float = 100.0
    capacity_factor: float = 0.45
    tenor_years: int = 5


def calculate_period_volumes(contract, n_steps, dt):
    hours_per_period = dt * 365.25 * 24
    volume = contract.capacity_mw * contract.capacity_factor * hours_per_period
    return np.full(n_steps, volume)


def calculate_discount_factors(discount_rate, n_steps, dt):
    times = np.arange(1, n_steps + 1) * dt
    return np.exp(-discount_rate * times)


def calculate_mtm_matrix(prices, contract, params, discount_rate, dt):
    n_paths, total_steps = prices.shape
    n_steps = total_steps - 1

    volumes = calculate_period_volumes(contract, n_steps, dt)
    mtm = np.zeros((n_paths, total_steps))

    for t_idx in range(total_steps):
        t_years = t_idx * dt
        s_t = prices[:, t_idx]
        pv_remaining = np.zeros(n_paths)

        for i_idx in range(t_idx, n_steps):
            t_i = (i_idx + 1) * dt
            fwd = implied_forward(s_t, t_years, t_i, params)
            df_ti = np.exp(-discount_rate * (t_i - t_years))
            pv_remaining += (contract.fixed_price - fwd) * volumes[i_idx] * df_ti

        mtm[:, t_idx] = pv_remaining

    return mtm
