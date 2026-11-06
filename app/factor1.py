import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FACTOR_COLUMNS = [
    "revenue_growth",
    "earnings_quality",
    "low_volatility",
    "trust_buy_ratio",
    "estimated_yield"
]


def calculate_raw_factors(df):
    """Calculate five investment factors."""

    df = df.copy()
    df = df.sort_values(["ticker", "date"])

    # 1. Revenue growth (YoY)
    df["factor_revenue_growth"] = df["revenue_growth"]

    # 2. Earnings quality: operating cash flow / net income
    net_income = df["net_income"].replace(0, np.nan)

    df["factor_earnings_quality"] = (
        df["operating_cash_flow"] / net_income
    )

    # 3. Low volatility: negative 60-day return volatility
    df["daily_return"] = (
        df.groupby("ticker")["close"].pct_change()
    )

    volatility = (
        df.groupby("ticker")["daily_return"]
        .transform(
            lambda x: x.rolling(
                window=60,
                min_periods=40
            ).std()
        )
    )

    # Lower volatility receives a higher factor score.
    df["factor_low_volatility"] = -volatility

    # 4. Institutional buying ratio
    # Required fields:
    # trust_net_buy, volume
    volume = df["volume"].replace(0, np.nan)

    df["factor_trust_buy_ratio"] = (
        df["trust_net_buy"] / volume
    )

    # 5. Estimated dividend yield
    # estimated_annual_dividend must be point-in-time data.
    df["factor_estimated_yield"] = (
        df["estimated_annual_dividend"] /
        df["close"].replace(0, np.nan)
    )

    return df


def cross_sectional_zscore(df):
    """
    Calculate cross-sectional Z-scores for each date.

    Missing values remain NaN.
    StandardScaler is fitted separately for each date.
    """

    df = df.copy()

    raw_columns = [
        "factor_revenue_growth",
        "factor_earnings_quality",
        "factor_low_volatility",
        "factor_trust_buy_ratio",
        "factor_estimated_yield"
    ]

    for col in raw_columns:
        df[col.replace("factor_", "z_")] = np.nan

    def normalize(group):
        values = group[raw_columns]

        # Only complete observations are used.
        valid = values.notna().all(axis=1)

        if valid.sum() < 2:
            return group

        scaler = StandardScaler()

        scaled = scaler.fit_transform(
            values.loc[valid]
        )

        for i, col in enumerate(raw_columns):
            output_col = col.replace("factor_", "z_")

            group.loc[valid, output_col] = scaled[:, i]

        return group

    df = (
        df.groupby("date", group_keys=False)
        .apply(normalize)
    )

    return df


def build_factors(df):
    """Main factor pipeline."""

    df = calculate_raw_factors(df)
    df = cross_sectional_zscore(df)

    return df

