# PPA Exposure Model

Monte Carlo counterparty exposure model for a fixed-for-float Power Purchase Agreement (PPA), built on the Schwartz one-factor mean-reverting price framework with seasonality.

## Features

- **Price simulation**: Schwartz one-factor model with mean-reversion and seasonality (Euler-Maruyama discretization)
- **PPA valuation**: Mark-to-Market using model-implied forward curves, consistent with information available at each time step
- **Exposure metrics**: EE, PFE, EPE, Effective EPE with configurable confidence levels
- **Collateral modeling**: CSA terms (threshold, MTA, Independent Amount, MPOR) with net exposure computation
- **Sensitivity analysis**: Parameter sweeps for volatility, mean-reversion, fixed price, and CSA threshold
- **Model validation**: Convergence analysis and distribution diagnostics (QQ plot, Jarque-Bera)
- **AI risk commentary**: Anthropic Claude integration for automated natural-language risk analysis

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

For AI commentary, add your Anthropic API key:
- **Local**: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your key
- **Streamlit Cloud**: add `ANTHROPIC_API_KEY` in Settings > Secrets

## Project structure

```
├── app.py                  # Streamlit application
├── config.py               # Default parameters
├── models/
│   ├── price_model.py      # Schwartz one-factor price simulation
│   ├── ppa.py              # PPA contract valuation
│   ├── exposure.py         # Exposure metrics (EE, PFE, EPE)
│   └── collateral.py       # CSA / collateral modeling
├── analysis/
│   ├── sensitivity.py      # Parameter sensitivity sweeps
│   └── validation.py       # Convergence & distribution diagnostics
├── ai/
│   └── commentary.py       # Claude AI risk commentary
├── utils/
│   └── charts.py           # Plotly visualizations
├── requirements.txt
└── .streamlit/
    └── config.toml         # Streamlit theme
```

## Model specification

See the Methodology tab in the app or the accompanying methodology document for the full mathematical specification.

## Assumptions and limitations

1. Single stochastic factor (no multi-factor term structure dynamics)
2. No jump-diffusion (power price spikes not explicitly modeled)
3. Constant volatility assumption
4. Deterministic generation volumes (capacity factor fixed)
5. Simplified collateral mechanics

These are documented in the app and methodology PDF as areas for future development.
