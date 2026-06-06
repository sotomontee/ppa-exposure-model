"""
PPA (Power Purchase Agreement) Contract Valuation.

Structure: Fixed-for-Float PPA
    - Generator (e.g., Ørsted) receives fixed price K per MWh
    - Generator pays floating market price (or equivalently, the PPA settles
      the difference between fixed and floating)
    - Net settlement per period: (K - F_t) * Volume

Mark-to-Market at time t (generator perspective):
    MtM(t) = sum_{i>t} (K - F(t, t_i)) * Q_i * DF(t, t_i)

    where:
        K      = fixed price (EUR/MWh)
        F(t,i) = model-implied forward price at time t for delivery at t_i
        Q_i    = delivered volume in period i (MWh)
        DF     = discount factor

Exposure interpretation:
    MtM > 0 => Generator is owed money (counterparty credit exposure)
    MtM < 0 => Generator owes money (no counterparty exposure)
"""

import numpy as np
from dataclasses import dataclass
from models.price_model import PriceModelParams, implied_forward


@dataclass
class PPAContract:
    """Definition of a fixed-for-float PPA contract."""
    fixed_price: float = 55.0       # EUR/MWh
    capacity_mw: float = 100.0      # MW nameplate capacity
    capacity_factor: float = 0.45   # Expected capacity factor
    tenor_years: int = 5            # Contract duration in years


def calculate_period_volumes(contract: PPAContract, n_steps: int,
                             dt: float) -> np.ndarray:
    """
    Calculate delivered energy volume (MWh) per settlement period.
    
    Volume = Capacity (MW) * Capacity Factor * Hours in Period
    
    Args:
        contract: PPA contract parameters
        n_steps: Number of settlement periods
        dt: Time step in years
        
    Returns:
        np.ndarray: Shape (n_steps,) volume in MWh per period
    """
    hours_per_period = dt * 365.25 * 24
    volume = contract.capacity_mw * contract.capacity_factor * hours_per_period
    return np.full(n_steps, volume)


def calculate_discount_factors(discount_rate: float, n_steps: int,
                               dt: float) -> np.ndarray:
    """
    Calculate continuous discount factors DF(0, t_i) = exp(-r * t_i).
    
    Args:
        discount_rate: Annualized risk-free rate
        n_steps: Number of periods
        dt: Time step in years
        
    Returns:
        np.ndarray: Shape (n_steps,) discount factors for each period
    """
    times = np.arange(1, n_steps + 1) * dt
    return np.exp(-discount_rate * times)


def calculate_mtm_matrix(
    prices: np.ndarray,
    contract: PPAContract,
    params: PriceModelParams,
    discount_rate: float,
    dt: float,
) -> np.ndarray:
    """
    Calculate the Mark-to-Market of the PPA at each time step for each path.
    
    At time step t, the MtM is computed using model-implied forward prices
    (not realized future prices), ensuring the valuation is consistent with
    information available at time t.
    
    MtM(t) = sum_{i=t+1}^{N} (K - F(t, t_i)) * Q_i * DF_relative(t, t_i)
    
    Args:
        prices: Shape (n_paths, n_steps+1) simulated spot price paths
        contract: PPA contract definition
        params: Price model parameters (for forward calculation)
        discount_rate: Risk-free rate for discounting
        dt: Time step in years
        
    Returns:
        np.ndarray: Shape (n_paths, n_steps+1) MtM at each time step
    """
    n_paths, total_steps = prices.shape
    n_steps = total_steps - 1
    
    volumes = calculate_period_volumes(contract, n_steps, dt)
    abs_df = calculate_discount_factors(discount_rate, n_steps, dt)
    
    mtm = np.zeros((n_paths, total_steps))
    
    for t_idx in range(total_steps):
        t_years = t_idx * dt
        s_t = prices[:, t_idx]  # Current spot on each path
        
        pv_remaining = np.zeros(n_paths)
        
        for i_idx in range(t_idx, n_steps):
            t_i = (i_idx + 1) * dt  # Delivery time of period i+1
            
            # Model-implied forward price for delivery at t_i
            fwd = implied_forward(s_t, t_years, t_i, params)
            
            # Relative discount factor from t to t_i
            df_ti = np.exp(-discount_rate * (t_i - t_years))
            
            # Cash flow: (K - F(t, t_i)) * Volume * DF
            pv_remaining += (contract.fixed_price - fwd) * volumes[i_idx] * df_ti
        
        mtm[:, t_idx] = pv_remaining
    
    return mtm
