import numpy as np
from dataclasses import dataclass


@dataclass
class ExposureProfile:
    time_grid: np.ndarray
    expected_exposure: np.ndarray
    potential_future_exposure: np.ndarray
    expected_negative_exposure: np.ndarray
    pfe_lower: np.ndarray
    median_exposure: np.ndarray
    epe: float
    effective_epe: float
    peak_pfe: float
    peak_pfe_time: float
    mean_mtm: np.ndarray


def compute_exposure_profile(mtm, dt, confidence_level=0.99):
    n_paths, total_steps = mtm.shape
    time_grid = np.arange(total_steps) * dt

    positive_exp = np.maximum(mtm, 0)
    negative_exp = np.minimum(mtm, 0)

    ee = np.mean(positive_exp, axis=0)
    ene = np.mean(negative_exp, axis=0)
    mean_mtm = np.mean(mtm, axis=0)
    pfe = np.percentile(positive_exp, confidence_level * 100, axis=0)
    pfe_lower = np.percentile(mtm, (1 - confidence_level) * 100, axis=0)
    median_exp = np.median(mtm, axis=0)

    epe = np.mean(ee[1:]) if total_steps > 1 else 0.0

    eff_ee = np.copy(ee)
    for i in range(1, total_steps):
        eff_ee[i] = max(eff_ee[i], eff_ee[i - 1])
    effective_epe = np.mean(eff_ee[1:]) if total_steps > 1 else 0.0

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


def compute_exposure_at_time(mtm, time_idx):
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
