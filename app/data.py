import pandas as pd
import numpy as np


def load_csv(path, date_cols=None):
    """Load CSV and parse date columns."""
    df = pd.read_csv(path)

    if date_cols:
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def prepare_fundamentals(fundamentals):
    """
    Calculate quarterly YoY revenue growth.

    Required columns:
        ticker, period_end, published_at,
        revenue, net_income, operating_cash_flow
    """
    df = fundamentals.copy()

    df["period_end"] = pd.to_datetime(df["period_end"])
    df["published_at"] = pd.to_datetime(df["published_at"])

    df = df.sort_values(["ticker", "period_end"])

    # Same fiscal quarter from previous year.
    df["revenue_prev_year"] = (
        df.groupby("ticker")["revenue"].shift(4)
    )

    df["revenue_growth"] = (
        df["revenue"] / df["revenue_prev_year"].replace(0, np.nan) - 1
    )

    return df


def point_in_time_merge(prices, fundamentals):
    """
    Merge the latest publicly available financial statement
    into each trading date.

    Only records with published_at <= trading date are allowed.
    """
    prices = prices.copy()
    fundamentals = fundamentals.copy()

    prices["date"] = pd.to_datetime(prices["date"])
    fundamentals["published_at"] = pd.to_datetime(
        fundamentals["published_at"]
    )

    prices = prices.dropna(subset=["date", "ticker"])
    fundamentals = fundamentals.dropna(
        subset=["published_at", "ticker"]
    )

    prices = prices.sort_values(["date", "ticker"])
    fundamentals = fundamentals.sort_values(
        ["published_at", "ticker"]
    )

    result = pd.merge_asof(
        prices,
        fundamentals,
        left_on="date",
        right_on="published_at",
        by="ticker",
        direction="backward",
        allow_exact_matches=True
    )

    # Defensive validation against look-ahead bias.
    invalid = (
        result["published_at"].notna()
        & (result["published_at"] > result["date"])
    )

    if invalid.any():
        raise ValueError("Look-ahead bias detected in fundamentals!")

    return result


def merge_point_in_time_data(prices, data, date_col="published_at"):
    """
    Merge another point-in-time dataset, such as
    institutional trading or dividend forecasts.
    """
    left = prices.copy()
    right = data.copy()

    left["date"] = pd.to_datetime(left["date"])
    right[date_col] = pd.to_datetime(right[date_col])

    left = left.sort_values(["date", "ticker"])
    right = right.sort_values([date_col, "ticker"])

    return pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on=date_col,
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_source")
    )


def load_market_data(
    price_path,
    fundamental_path,
    institutional_path,
    dividend_path
):
    """Load and combine all datasets."""

    prices = load_csv(price_path, ["date"])

    fundamentals = load_csv(
        fundamental_path,
        ["period_end", "published_at"]
    )

    institutional = load_csv(
        institutional_path,
        ["published_at"]
    )

    dividends = load_csv(
        dividend_path,
        ["published_at"]
    )

    # Calculate financial factors before expanding to daily data.
    fundamentals = prepare_fundamentals(fundamentals)

    df = point_in_time_merge(prices, fundamentals)

    df = merge_point_in_time_data(
        df,
        institutional,
        "published_at"
    )

    df = merge_point_in_time_data(
        df,
        dividends,
        "published_at"
    )

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df

