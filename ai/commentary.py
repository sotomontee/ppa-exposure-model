import json
from typing import Optional


def generate_risk_commentary(metrics, api_key=None):
    if not api_key:
        return (
            "AI commentary requires an Anthropic API key. "
            "Add your key in the sidebar to enable this feature."
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        return "anthropic package not installed. Run: pip install anthropic"
    except Exception as e:
        return f"Error initializing API client: {str(e)}"

    data_summary = json.dumps(metrics, indent=2, default=str)

    prompt = f"""You are a senior quantitative risk analyst at a major European energy company.
Analyze the following PPA (Power Purchase Agreement) exposure profile and provide a concise
risk commentary suitable for senior management and internal audit.

The PPA is a fixed-for-float contract from the generator's perspective. Positive exposure
means the counterparty owes us money (credit risk). The model uses a Schwartz one-factor
mean-reverting framework with seasonality.

EXPOSURE METRICS:
{data_summary}

Provide your analysis in the following structure:
1. **Executive Summary** (2-3 sentences on overall risk profile)
2. **Key Risk Drivers** (what parameters most affect the exposure)
3. **Risk Considerations** (any concerns, model limitations, or recommendations)

Keep the tone professional and analytical. Use specific numbers from the data.
Be concise — this is a management summary, not a research paper."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error generating commentary: {str(e)}"


def prepare_metrics_for_ai(profile_gross, profile_net, params_dict, contract_dict, csa_dict):
    result = {
        "contract": {
            "type": "Fixed-for-Float PPA",
            "fixed_price_eur_mwh": contract_dict.get("fixed_price"),
            "capacity_mw": contract_dict.get("capacity_mw"),
            "capacity_factor": contract_dict.get("capacity_factor"),
            "tenor_years": contract_dict.get("tenor_years"),
        },
        "price_model": {
            "current_spot_eur_mwh": params_dict.get("spot_price"),
            "long_run_mean_eur_mwh": params_dict.get("long_run_mean"),
            "annual_volatility": params_dict.get("volatility"),
            "mean_reversion_speed": params_dict.get("mean_reversion"),
        },
        "gross_exposure": {
            "peak_pfe_eur": round(profile_gross.peak_pfe, 0),
            "peak_pfe_time_years": round(profile_gross.peak_pfe_time, 2),
            "epe_eur": round(profile_gross.epe, 0),
            "effective_epe_eur": round(profile_gross.effective_epe, 0),
        },
    }

    if profile_net is not None:
        result["collateralized_exposure"] = {
            "peak_pfe_eur": round(profile_net.peak_pfe, 0),
            "epe_eur": round(profile_net.epe, 0),
            "pfe_reduction_pct": round(
                (1 - profile_net.peak_pfe / max(profile_gross.peak_pfe, 1)) * 100, 1
            ),
        }

    return result
