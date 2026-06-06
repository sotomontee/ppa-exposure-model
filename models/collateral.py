import numpy as np
from dataclasses import dataclass


@dataclass
class CSAParams:
    threshold: float = 5_000_000
    mta: float = 500_000
    independent_amount: float = 2_000_000
    margin_period_of_risk: int = 10
    rounding: float = 100_000


def apply_rounding(amount, rounding):
    if rounding <= 0:
        return amount
    return np.round(amount / rounding) * rounding


def compute_collateral_held(mtm, csa, dt):
    n_paths, total_steps = mtm.shape
    collateral = np.zeros_like(mtm)

    days_per_step = dt * 365.25
    mpor_steps = max(1, int(np.ceil(csa.margin_period_of_risk / days_per_step)))

    for t in range(total_steps):
        ref_t = max(0, t - mpor_steps)
        ref_mtm = mtm[:, ref_t]

        required = np.maximum(ref_mtm - csa.threshold, 0)
        required = required + csa.independent_amount
        required = np.where(required >= csa.mta, required, 0.0)
        required = apply_rounding(required, csa.rounding)

        collateral[:, t] = required

    return collateral


def compute_net_exposure(mtm, collateral):
    return np.maximum(mtm - collateral, 0)
