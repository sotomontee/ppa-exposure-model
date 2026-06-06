"""
CSA (Credit Support Annex) Collateral Modeling.

Models the impact of collateral agreements on counterparty exposure.

CSA Parameters:
    - Threshold (H):        MtM level below which no collateral is required
    - Minimum Transfer Amount (MTA): Minimum amount for a margin call
    - Independent Amount (IA):       Additional collateral buffer
    - Margin Period of Risk (MPOR):  Days between last margin call and close-out
    - Rounding:             Rounding convention for margin calls

Collateral call logic:
    Collateral_required = max(MtM - H - IA_received, 0)
    Collateral_call = Collateral_required if Collateral_required >= MTA else 0
    
Net exposure (after collateral):
    Net_Exposure = max(MtM - Collateral_held, 0)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CSAParams:
    """Credit Support Annex parameters."""
    threshold: float = 5_000_000
    mta: float = 500_000
    independent_amount: float = 2_000_000
    margin_period_of_risk: int = 10  # days
    rounding: float = 100_000


def apply_rounding(amount: np.ndarray, rounding: float) -> np.ndarray:
    """Round amounts to the nearest rounding increment."""
    if rounding <= 0:
        return amount
    return np.round(amount / rounding) * rounding


def compute_collateral_held(
    mtm: np.ndarray,
    csa: CSAParams,
    dt: float,
) -> np.ndarray:
    """
    Compute collateral held at each time step, accounting for CSA terms
    and the margin period of risk.
    
    The MPOR introduces a delay: collateral at time t reflects the margin
    call based on MtM at time (t - MPOR). This means during the MPOR window,
    the exposure is uncollateralized.
    
    Args:
        mtm: Shape (n_paths, n_steps+1) MtM values
        csa: CSA parameters
        dt: Time step in years
        
    Returns:
        np.ndarray: Shape (n_paths, n_steps+1) collateral held
    """
    n_paths, total_steps = mtm.shape
    collateral = np.zeros_like(mtm)
    
    # Convert MPOR from days to time steps (approximate)
    days_per_step = dt * 365.25
    mpor_steps = max(1, int(np.ceil(csa.margin_period_of_risk / days_per_step)))
    
    for t in range(total_steps):
        # Collateral is based on MtM from (t - mpor_steps) ago
        ref_t = max(0, t - mpor_steps)
        ref_mtm = mtm[:, ref_t]
        
        # Collateral required (only when MtM exceeds threshold)
        required = np.maximum(ref_mtm - csa.threshold, 0)
        
        # Add independent amount (always posted regardless of MtM)
        required = required + csa.independent_amount
        
        # Apply MTA filter
        required = np.where(required >= csa.mta, required, 0.0)
        
        # Apply rounding
        required = apply_rounding(required, csa.rounding)
        
        collateral[:, t] = required
    
    return collateral


def compute_net_exposure(
    mtm: np.ndarray,
    collateral: np.ndarray,
) -> np.ndarray:
    """
    Compute net exposure after collateral.
    
    Net Exposure = max(MtM - Collateral, 0)
    
    Args:
        mtm: Shape (n_paths, n_steps+1) gross MtM
        collateral: Shape (n_paths, n_steps+1) collateral held
        
    Returns:
        np.ndarray: Shape (n_paths, n_steps+1) net exposure
    """
    return np.maximum(mtm - collateral, 0)
