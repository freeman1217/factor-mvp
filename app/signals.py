import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "z_revenue_growth": 0.20,
    "z_earnings_quality": 0.20,
    "z_low_volatility": 0.20,
    "z_trust_buy_ratio": 0.20,
    "z_estimated_yield": 0.20
}


def generate_signals(df, weights=None):
    """
    Generate composite factor scores and cross-sectional ranks.

    Stocks with missing factor values are excluded.
    """

    df = df.copy()

    weights = weights or DEFAULT_WEIGHTS

    total_weight = sum(weights.values())

    if total_weight <= 0:
        raise ValueError("Total factor weight must be positive.")

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Factor weights cannot be negative.")

    factor_columns = list(weights.keys())

    missing_columns = set(factor_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing factor columns: {missing_columns}"
        )

    df["signal_score"] = np.nan

    valid = df[factor_columns].notna().all(axis=1)

    normalized_weights = {
        key: value / total_weight
        for key, value in weights.items()
    }

    score = sum(
        df.loc[valid, col] * weight
        for col, weight in normalized_weights.items()
    )

    df.loc[valid, "signal_score"] = score

    # Cross-sectional ranking: 1 is the best stock.
    df["signal_rank"] = (
        df.groupby("date")["signal_score"]
        .rank(ascending=False, method="first")
    )

    return df

