PPA_DEFAULTS = {
    "fixed_price": 55.0,
    "capacity_mw": 100.0,
    "capacity_factor": 0.45,
    "tenor_years": 5,
}

PRICE_MODEL_DEFAULTS = {
    "spot_price": 50.0,
    "long_run_mean": 52.0,
    "volatility": 0.35,
    "mean_reversion": 1.5,
    "seasonality_amplitude": 0.15,
    "seasonality_phase": 0.0,
}

SIMULATION_DEFAULTS = {
    "n_paths": 10000,
    "time_steps_per_year": 12,
    "random_seed": 42,
}

CSA_DEFAULTS = {
    "threshold": 5_000_000,
    "mta": 500_000,
    "independent_amount": 2_000_000,
    "margin_period_of_risk": 10,
    "rounding": 100_000,
}

RISK_DEFAULTS = {
    "confidence_level": 0.99,
    "discount_rate": 0.03,
}
