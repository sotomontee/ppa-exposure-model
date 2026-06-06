import numpy as np
from models.price_model import PriceModelParams, simulate_prices
from models.ppa import PPAContract, calculate_mtm_matrix
from models.exposure import compute_exposure_profile


def convergence_analysis(params, contract, discount_rate, dt, confidence_level,
                         path_counts=None, seed=42):
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


def distribution_diagnostics(mtm, time_idx):
    values = mtm[:, time_idx]
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)

    if std > 0:
        skewness = np.mean(((values - mean) / std) ** 3)
        kurtosis = np.mean(((values - mean) / std) ** 4) - 3
    else:
        skewness = 0.0
        kurtosis = 0.0

    sorted_values = np.sort(values)
    n_points = min(100, n)
    indices = np.linspace(0, n - 1, n_points).astype(int)
    empirical = sorted_values[indices]

    from scipy import stats
    theoretical_quantiles = stats.norm.ppf(
        np.linspace(0.01, 0.99, n_points), loc=mean, scale=std
    )

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
