"""
Synthetic credit scoring data generator.

Generates realistic credit data with configurable size and a logistic
model for binary default labels.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_credit_data(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic credit scoring data.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw features, engineered features, and target column.
    """
    rng = np.random.RandomState(random_state)

    # --- raw features ---
    age = rng.normal(loc=40, scale=12, size=n_samples).clip(18, 80).astype(int)
    income = rng.lognormal(mean=10.5, sigma=0.6, size=n_samples).round(2)
    debt = (income * rng.uniform(0.0, 0.8, size=n_samples)).round(2)
    loan_amount = rng.lognormal(mean=9.5, sigma=0.8, size=n_samples).round(2)
    credit_history_years = (
        rng.gamma(shape=3, scale=3, size=n_samples).clip(0, 40).round(1)
    )
    num_late_payments = rng.poisson(lam=1.5, size=n_samples)
    num_credit_lines = rng.poisson(lam=4, size=n_samples).clip(0, 20)
    employment_years = (
        rng.gamma(shape=3, scale=3, size=n_samples).clip(0, 45).round(1)
    )

    # --- engineered feature ---
    debt_to_income_ratio = np.where(income > 0, debt / income, 0.0)

    # --- inject a small amount of NaN (~2 %) for realism ---
    def _inject_nan(arr: np.ndarray, frac: float = 0.02) -> np.ndarray:
        arr = arr.astype(float)
        mask = rng.rand(len(arr)) < frac
        arr[mask] = np.nan
        return arr

    income = _inject_nan(income)
    credit_history_years = _inject_nan(credit_history_years)
    employment_years = _inject_nan(employment_years)

    # --- logistic model for labels ---
    z = (
        -3.0
        + 0.02 * (age - 40)
        + 0.5 * np.log1p(np.nan_to_num(income, nan=np.nanmedian(income))) * -0.3
        + 0.8 * debt_to_income_ratio
        + 0.0001 * loan_amount
        - 0.1 * np.nan_to_num(credit_history_years, nan=np.nanmedian(credit_history_years))
        + 0.3 * num_late_payments
        - 0.05 * num_credit_lines
        - 0.08 * np.nan_to_num(employment_years, nan=np.nanmedian(employment_years))
    )
    prob_default = 1 / (1 + np.exp(-z))
    # add noise so it isn't perfectly separable
    prob_default = np.clip(prob_default + rng.normal(0, 0.1, n_samples), 0.01, 0.99)
    default = (rng.rand(n_samples) < prob_default).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "debt": debt,
            "loan_amount": loan_amount,
            "credit_history_years": credit_history_years,
            "num_late_payments": num_late_payments,
            "num_credit_lines": num_credit_lines,
            "employment_years": employment_years,
            "debt_to_income_ratio": debt_to_income_ratio.round(4),
            "default": default,
        }
    )
    return df


if __name__ == "__main__":
    data = generate_credit_data()
    print(f"Generated {len(data)} samples")
    print(f"Default rate: {data['default'].mean():.2%}")
    print(data.head())
    print(data.describe())
