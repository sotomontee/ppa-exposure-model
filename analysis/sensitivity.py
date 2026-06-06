"""
Sensitivity Analysis Module.

Runs the exposure model across parameter ranges to assess the impact
of key risk drivers on PFE and EPE.

Analyses:
    1. Volatility sensitivity
    2. Mean-reversion speed sensitivity
    3. Fixed price level sensitivity
    4. CSA threshold sensitivity
    5. Combined parameter grid (for heatmap)
"""

import numpy as np
from dataclasses import dataclass
from models.price_model import PriceModelParams, simulate_prices
from models.ppa import PPAContract, calculate_mtm_matrix
from models.exposure import compute_exposure_profile
from models.collateral import CSAParams, compute_collateral_held, compute_net_exposure


@dataclass
class SensitivityResult:
    """Result of a single sensitivity sweep."""
    parameter_name: str
    parameter_values: np.ndarray
    peak_pfe: np.ndarray
    epe: np.ndarray
    peak_pfe_collateralized: np.ndarray
    epe_collateralized: np.ndarray


def run_single_scenario(
    params: PriceModelParams,
    contract: PPAContract,
    csa: CSAParams,
    n_paths: int,
    n_steps: int,
    dt: float,
    discount_rate: float,
    confidence_level: float,
    seed: int = 42,
) -> dict:
    """Run a single scenario and return key metrics."""
    prices = simulate_prices(params, n_paths, n_steps, dt, seed)
    mtm = calculate_mtm_matrix(prices, contract, params, discount_rate, dt)
    
    # Gross exposure
    profile_gross = compute_exposure_profile(mtm, dt, confidence_level)
    
    # Collateralized exposure
    collateral = compute_collateral_held(mtm, csa, dt)
    net_exp = compute_net_exposure(mtm, collateral)
    
    # Create a "net MtM" for exposure computation
    # Net MtM = MtM - Collateral (can still be negative)
    net_mtm = mtm - collateral
    profile_net = compute_exposure_profile(net_mtm, dt, confidence_level)
    
    return {
        "peak_pfe": profile_gross.peak_pfe,
        "epe": profile_gross.epe,
        "peak_pfe_coll": profile_net.peak_pfe,
        "epe_coll": profile_net.epe,
    }


def sensitivity_sweep(
    base_params: PriceModelParams,
    contract: PPAContract,
    csa: CSAParams,
    param_name: str,
    param_values: np.ndarray,
    n_paths: int,
    n_steps: int,
    dt: float,
    discount_rate: float,
    confidence_level: float,
    seed: int = 42,
) -> SensitivityResult:
    """
    Sweep a single parameter across a range and compute exposure metrics.
    
    Args:
        base_params: Base price model parameters
        contract: PPA contract
        csa: CSA parameters
        param_name: Name of parameter to sweep (must be a field of 
                    PriceModelParams, PPAContract, or CSAParams)
        param_values: Array of values to sweep
        ... simulation settings ...
        
    Returns:
        SensitivityResult with metrics for each parameter value
    """
    n_vals = len(param_values)
    peak_pfe = np.zeros(n_vals)
    epe = np.zeros(n_vals)
    peak_pfe_coll = np.zeros(n_vals)
    epe_coll = np.zeros(n_vals)
    
    for i, val in enumerate(param_values):
        # Create copies and modify the relevant parameter
        p = PriceModelParams(
            spot_price=base_params.spot_price,
            long_run_mean=base_params.long_run_mean,
            volatility=base_params.volatility,
            mean_reversion=base_params.mean_reversion,
            seasonality_amplitude=base_params.seasonality_amplitude,
            seasonality_phase=base_params.seasonality_phase,
        )
        c = PPAContract(
            fixed_price=contract.fixed_price,
            capacity_mw=contract.capacity_mw,
            capacity_factor=contract.capacity_factor,
            tenor_years=contract.tenor_years,
        )
        csa_copy = CSAParams(
            threshold=csa.threshold,
            mta=csa.mta,
            independent_amount=csa.independent_amount,
            margin_period_of_risk=csa.margin_period_of_risk,
            rounding=csa.rounding,
        )
        
        # Set the parameter on the right object
        if hasattr(p, param_name):
            setattr(p, param_name, val)
        elif hasattr(c, param_name):
            setattr(c, param_name, val)
        elif hasattr(csa_copy, param_name):
            setattr(csa_copy, param_name, val)
        
        result = run_single_scenario(
            p, c, csa_copy, n_paths, n_steps, dt,
            discount_rate, confidence_level, seed
        )
        
        peak_pfe[i] = result["peak_pfe"]
        epe[i] = result["epe"]
        peak_pfe_coll[i] = result["peak_pfe_coll"]
        epe_coll[i] = result["epe_coll"]
    
    return SensitivityResult(
        parameter_name=param_name,
        parameter_values=param_values,
        peak_pfe=peak_pfe,
        epe=epe,
        peak_pfe_collateralized=peak_pfe_coll,
        epe_collateralized=epe_coll,
    )
