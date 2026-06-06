"""
Counterparty Exposure Metrics.

Computed from the MtM matrix produced by the PPA valuation module.

Metrics:
    - Positive Exposure (PE):   max(MtM, 0) — what the counterparty owes us
    - Negative Exposure (NE):   min(MtM, 0) — what we owe the counterparty
    - Expected Exposure (EE):   E[PE(t)] — average positive exposure at time t
    - Expected Negative Exp.:   E[NE(t)] — average negative exposure at time t
    - PFE (Potential Future):   Quantile_alpha[PE(t)] — tail exposure
    - EPE (Expected Positive):  Time-weighted average of EE
    - Effective EPE:            Non-decreasing EPE (regulatory definition)
    - Peak PFE:                 max over t of PFE(t)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ExposureProfile:
    """Container for all exposure metrics over time."""
    time_grid: np.ndarray           # Time points in years
    expected_exposure: np.ndarray   # EE(t)
    potential_future_exposure: np.ndarray  # PFE(t)
    expected_negative_exposure: np.ndarray  # ENE(t)
    pfe_lower: np.ndarray           # Lower PFE band (e.g., 1st percentile)
    median_exposure: np.ndarray     # Median exposure
    epe: float                      # Expected Positive Exposure (scalar)
    effective_epe: float            # Effective EPE (scalar)
    peak_pfe: float                 # Peak PFE
    peak_pfe_time: float            # Time of peak PFE
    mean_mtm: np.ndarray            # Mean MtM (can be negative)


def compute_exposure_profile(
    mtm: np.ndarray,
    dt: float,
    confidence_level: float = 0.99,
) -> ExposureProfile:
    """
    Compute the full exposure profile from a MtM matrix.
    
    Args:
        mtm: Shape (n_paths, n_steps+1) Mark-to-Market values
        dt: Time step in years
        confidence_level: Confidence level for PFE (e.g., 0.99)
        
    Returns:
        ExposureProfile with all metrics
    """
    n_paths, total_steps = mtm.shape
    
    # Time grid
    time_grid = np.arange(total_steps) * dt
    
    # Positive and negative exposure
    positive_exp = np.maximum(mtm, 0)
    negative_exp = np.minimum(mtm, 0)
    
    # Expected Exposure: E[max(MtM, 0)] at each time step
    ee = np.mean(positive_exp, axis=0)
    
    # Expected Negative Exposure: E[min(MtM, 0)]
    ene = np.mean(negative_exp, axis=0)
    
    # Mean MtM (can be negative)
    mean_mtm = np.mean(mtm, axis=0)
    
    # PFE at specified confidence level
    pfe = np.percentile(positive_exp, confidence_level * 100, axis=0)
    
    # Lower band (complement percentile on negative side)
    lower_pct = (1 - confidence_level) * 100
    pfe_lower = np.percentile(mtm, lower_pct, axis=0)
    
    # Median
    median_exp = np.median(mtm, axis=0)
    
    # EPE: time-weighted average of EE (excluding t=0)
    if total_steps > 1:
        epe = np.mean(ee[1:])
    else:
        epe = 0.0
    
    # Effective EPE: non-decreasing version of EPE up to each time
    eff_ee = np.copy(ee)
    for i in range(1, total_steps):
        eff_ee[i] = max(eff_ee[i], eff_ee[i - 1])
    effective_epe = np.mean(eff_ee[1:]) if total_steps > 1 else 0.0
    
    # Peak PFE
    peak_pfe = np.max(pfe)
    peak_pfe_idx = np.argmax(pfe)
    peak_pfe_time = peak_pfe_idx * dt
    
    return ExposureProfile(
        time_grid=time_grid,
        expected_exposure=ee,
        potential_future_exposure=pfe,
        expected_negative_exposure=ene,
        pfe_lower=pfe_lower,
        median_exposure=median_exp,
        epe=epe,
        effective_epe=effective_epe,
        peak_pfe=peak_pfe,
        peak_pfe_time=peak_pfe_time,
        mean_mtm=mean_mtm,
    )


def compute_exposure_at_time(
    mtm: np.ndarray,
    time_idx: int,
) -> dict:
    """
    Compute the full exposure distribution at a specific time step.
    Useful for histogram / distribution analysis.
    
    Args:
        mtm: Shape (n_paths, n_steps+1)
        time_idx: Index of the time step
        
    Returns:
        dict with distribution statistics
    """
    values = mtm[:, time_idx]
    positive = np.maximum(values, 0)
    
    return {
        "values": values,
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "median": np.median(values),
        "pct_positive": np.mean(values > 0) * 100,
        "pct_1": np.percentile(values, 1),
        "pct_5": np.percentile(values, 5),
        "pct_25": np.percentile(values, 25),
        "pct_75": np.percentile(values, 75),
        "pct_95": np.percentile(values, 95),
        "pct_99": np.percentile(values, 99),
        "ee": np.mean(positive),
        "pfe_99": np.percentile(positive, 99),
        "pfe_975": np.percentile(positive, 97.5),
    }
