"""
Model Validation Module.

Provides convergence analysis and distribution diagnostics to verify
that the Monte Carlo simulation produces stable, reliable results.

Analyses:
    1. Convergence: PFE stability as number of paths increases
    2. Distribution diagnostics: normality tests, QQ data, moments
    3. Variance reduction assessment
"""

import numpy as np
from models.price_model import PriceModelParams, simulate_prices
from models.ppa import PPAContract, calculate_mtm_matrix
from models.exposure import compute_exposure_profile


def convergence_analysis(
    params: PriceModelParams,
    contract: PPAContract,
    discount_rate: float,
    dt: float,
    confidence_level: float,
    path_counts: list = None,
    seed: int = 42,
) -> dict:
    """
    Assess PFE convergence by running simulations with increasing path counts.
    
    Args:
        params: Price model parameters
        contract: PPA contract
        discount_rate: Risk-free rate
        dt: Time step in years
        confidence_level: PFE confidence level
        path_counts: List of path counts to test
        seed: Base random seed
        
    Returns:
        dict with convergence data
    """
    if path_counts is None:
        path_counts = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    
    n_steps = int(contract.tenor_years / dt)
    
    peak_pfe_values = []
    epe_values = []
    std_errors = []
    
    for n_paths in path_counts:
        prices = simulate_prices(params, n_paths, n_steps, dt, seed)
        mtm = calculate_mtm_matrix(prices, contract, params, discount_rate, dt)
        profile = compute_exposure_profile(mtm, dt, confidence_level)
        
        peak_pfe_values.append(profile.peak_pfe)
        epe_values.append(profile.epe)
        
        # Standard error of the mean exposure (at peak PFE time)
        peak_idx = np.argmax(profile.potential_future_exposure)
        positive_exp = np.maximum(mtm[:, peak_idx], 0)
        se = np.std(positive_exp) / np.sqrt(n_paths)
        std_errors.append(se)
    
    return {
        "path_counts": path_counts,
        "peak_pfe": peak_pfe_values,
        "epe": epe_values,
        "std_errors": std_errors,
    }


def distribution_diagnostics(
    mtm: np.ndarray,
    time_idx: int,
) -> dict:
    """
    Compute distribution diagnostics for MtM at a given time step.
    
    Args:
        mtm: Shape (n_paths, n_steps+1)
        time_idx: Time step index to analyze
        
    Returns:
        dict with distribution statistics and QQ plot data
    """
    values = mtm[:, time_idx]
    
    # Basic moments
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    
    # Skewness and kurtosis
    if std > 0:
        skewness = np.mean(((values - mean) / std) ** 3)
        kurtosis = np.mean(((values - mean) / std) ** 4) - 3  # Excess kurtosis
    else:
        skewness = 0.0
        kurtosis = 0.0
    
    # QQ plot data (theoretical normal quantiles vs empirical)
    sorted_values = np.sort(values)
    n_points = min(100, n)
    indices = np.linspace(0, n - 1, n_points).astype(int)
    empirical = sorted_values[indices]
    
    from scipy import stats
    theoretical_quantiles = stats.norm.ppf(
        np.linspace(0.01, 0.99, n_points), loc=mean, scale=std
    )
    
    # Jarque-Bera test for normality
    jb_stat = (n / 6) * (skewness ** 2 + (kurtosis ** 2) / 4)
    
    return {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
        "jb_statistic": jb_stat,
        "empirical_quantiles": empirical,
        "theoretical_quantiles": theoretical_quantiles,
        "histogram_values": values,
    }
