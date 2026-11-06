import pandas as pd
import numpy as np

def get_rebalance_dates(df):
    """Get the last available trading day of each month."""

    dates = pd.to_datetime(df["date"].drop_duplicates())

    calendar = pd.DataFrame({"date": dates})

    calendar["month"] = calendar["date"].dt.to_period("M")

    rebalance_dates = (
        calendar.groupby("month")["date"]
        .max()
        .sort_values()
        .tolist()
    )

    return rebalance_dates

def build_portfolio(
    df,
    top_n=20,
    min_score=None
):
    """
    Monthly equal-weight portfolio.
    Signals are generated using data available at each rebalance date.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    rebalance_dates = get_rebalance_dates(df)

    portfolios = []

    for date in rebalance_dates:

        snapshot = df[df["date"] == date].copy()

        snapshot = snapshot.dropna(
            subset=["signal_score"]
        )

        if min_score is not None:
            snapshot = snapshot[
                snapshot["signal_score"] >= min_score
            ]

        snapshot = snapshot.sort_values(
            "signal_score",
            ascending=False
        )

        selected = snapshot.head(top_n).copy()

        if selected.empty:
            continue

        # Equal-weight allocation.
        selected["weight"] = 1.0 / len(selected)

        selected["rebalance_date"] = date

        portfolios.append(
            selected[
                [
                    "rebalance_date",
                    "date",
                    "ticker",
                    "signal_score",
                    "signal_rank",
                    "close",
                    "weight"
                ]
            ]
        )

    if not portfolios:
        return pd.DataFrame(
            columns=[
                "rebalance_date",
                "date",
                "ticker",
                "signal_score",
                "signal_rank",
                "close",
                "weight"
            ]
        )

    result = pd.concat(
        portfolios,
        ignore_index=True
    )

    return result


def get_target_weights(portfolio):
    """Convert portfolio holdings to target weights."""

    if portfolio.empty:
        return pd.DataFrame()

    return portfolio.pivot_table(
        index="rebalance_date",
        columns="ticker",
        values="weight",
        aggfunc="sum",
        fill_value=0.0
    ).sort_index()

