import numpy as np
import pandas as pd


class BacktestEngine:

    def __init__(
        self,
        initial_capital=1_000_000,
        commission=0.001425,
        transaction_tax=0.003
    ):

        self.initial_capital = initial_capital
        self.commission = commission
        self.transaction_tax = transaction_tax

    def calculate_returns(self, prices):
        """Calculate daily close-to-close returns."""

        prices = prices.copy()

        prices["date"] = pd.to_datetime(prices["date"])

        prices = prices.sort_values(
            ["ticker", "date"]
        )

        prices["daily_return"] = (
            prices.groupby("ticker")["close"]
            .pct_change()
        )

        return prices

    def run(self, prices, portfolio):
        """
        Execute monthly portfolio backtest.

        The target portfolio formed at the previous
        rebalance close earns the next trading interval's
        close-to-close return.

        Transaction costs:
            Buy  = commission
            Sell = commission + transaction tax
        """

        if portfolio.empty:
            raise ValueError("Portfolio is empty.")

        prices = self.calculate_returns(prices)

        prices = prices.dropna(
            subset=["daily_return"]
        )

        returns = prices.pivot(
            index="date",
            columns="ticker",
            values="daily_return"
        ).sort_index()

        targets = (
            portfolio.pivot_table(
                index="rebalance_date",
                columns="ticker",
                values="weight",
                aggfunc="sum",
                fill_value=0.0
            ).sort_index()
        )

        all_dates = returns.index.union(
            targets.index
        ).sort_values()

        returns = returns.reindex(
            all_dates
        ).fillna(0.0)

        targets = targets.reindex(
            all_dates
        )

        tickers = returns.columns.union(
            targets.columns
        )

        returns = returns.reindex(
            columns=tickers,
            fill_value=0.0
        )

        targets = targets.reindex(
            columns=tickers,
            fill_value=0.0
        )

        current_weights = pd.Series(
            0.0,
            index=tickers
        )

        nav = float(self.initial_capital)

        results = []

        rebalance_dates = set(targets.dropna(how="all").index)

        for date in all_dates:

            # Rebalance at today's close.
            # Previous holdings earned today's return.
            daily_returns = returns.loc[date].fillna(0.0)

            gross_return = float(
                (current_weights * daily_returns).sum()
            )

            # Update portfolio weights after market movement.
            updated_weights = (
                current_weights * (1.0 + daily_returns)
            )

            total_value = updated_weights.sum()

            if total_value > 0:
                updated_weights = updated_weights / total_value

            else:
                updated_weights[:] = 0.0

            # Execute target portfolio at today's close.
            buy_turnover = 0.0
            sell_turnover = 0.0

            commission_cost = 0.0
            tax_cost = 0.0

            if date in rebalance_dates:

                target_weights = targets.loc[date].fillna(0.0)

                target_total = target_weights.sum()

                if target_total > 1.0 + 1e-8:
                    raise ValueError(
                        f"Portfolio weights exceed 100% on {date}"
                    )

                weight_change = (
                    target_weights - updated_weights
                )

                buy_turnover = float(
                    weight_change.clip(lower=0).sum()
                )

                sell_turnover = float(
                    -weight_change.clip(upper=0).sum()
                )

                commission_cost = (
                    buy_turnover + sell_turnover
                ) * self.commission

                tax_cost = (
                    sell_turnover * self.transaction_tax
                )

                total_cost = (
                    commission_cost + tax_cost
                )

                # Deduct transaction costs from NAV.
                nav *= (1.0 + gross_return)

                nav *= (1.0 - total_cost)

                current_weights = target_weights.copy()

            else:

                nav *= (1.0 + gross_return)

                current_weights = updated_weights

                total_cost = 0.0

            net_return = (
                nav / results[-1]["nav"] - 1.0
                if results
                else nav / self.initial_capital - 1.0
            )

            results.append({
                "date": date,
                "nav": nav,
                "gross_return": gross_return,
                "net_return": net_return,
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "commission_cost": commission_cost,
                "tax_cost": tax_cost,
                "total_cost": total_cost
            })

        result = pd.DataFrame(results).set_index("date")

        return result


def calculate_performance(backtest):
    """Calculate performance statistics."""

    if backtest.empty:
        return {}

    nav = backtest["nav"]
    returns = backtest["net_return"]

    total_return = (
        nav.iloc[-1] / nav.iloc[0] - 1
    )

    trading_days = max(len(nav) - 1, 1)

    annual_return = (
        (nav.iloc[-1] / nav.iloc[0])
        ** (252 / trading_days) - 1
    )

    annual_volatility = (
        returns.std() * np.sqrt(252)
    )

    sharpe_ratio = (
        annual_return / annual_volatility
        if annual_volatility > 0
        else np.nan
    )

    drawdown = (
        nav / nav.cummax() - 1
    )

    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

