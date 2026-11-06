import os
import pandas as pd

from data import load_market_data
from factors import build_factors
from signals import generate_signals
from portfolio import build_portfolio
from backtest import BacktestEngine, calculate_performance


DATA_DIR = "data"
OUTPUT_DIR = "output"


def run_pipeline():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Taiwan Stock Quantitative Investment System")
    print("=" * 60)

    # 1. DATA
    print("\n[1/6] Loading market data...")

    df = load_market_data(
        price_path=f"{DATA_DIR}/prices.csv",
        fundamental_path=f"{DATA_DIR}/fundamentals.csv",
        institutional_path=f"{DATA_DIR}/institutional.csv",
        dividend_path=f"{DATA_DIR}/dividends.csv"
    )

    print(f"Loaded {len(df)} records.")

    # 2. FACTORS
    print("\n[2/6] Calculating factors...")

    df = build_factors(df)

    # 3. SIGNALS
    print("\n[3/6] Generating signals...")

    df = generate_signals(df)

    # 4. PORTFOLIO
    print("\n[4/6] Building portfolio...")

    portfolio = build_portfolio(
        df,
        top_n=20
    )

    if portfolio.empty:
        raise ValueError(
            "No portfolio holdings generated."
        )

    # 5. BACKTEST
    print("\n[5/6] Running backtest...")

    engine = BacktestEngine(
        initial_capital=1_000_000,
        commission=0.001425,
        transaction_tax=0.003
    )

    backtest = engine.run(
        prices=df[["date", "ticker", "close"]],
        portfolio=portfolio
    )

    performance = calculate_performance(backtest)

    # 6. REPORT
    print("\n[6/6] Generating report...")

    df.to_csv(
        f"{OUTPUT_DIR}/factor_data.csv",
        index=False
    )

    portfolio.to_csv(
        f"{OUTPUT_DIR}/portfolio.csv",
        index=False
    )

    backtest.to_csv(
        f"{OUTPUT_DIR}/backtest.csv"
    )

    report = pd.DataFrame(
        performance.items(),
        columns=["Metric", "Value"]
    )

    report.to_csv(
        f"{OUTPUT_DIR}/performance.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("BACKTEST PERFORMANCE")
    print("=" * 60)

    for metric, value in performance.items():

        print(
            f"{metric:25s}: {value:.4%}"
            if pd.notna(value)
            else f"{metric:25s}: N/A"
        )

    print("\nPipeline completed successfully.")

    return {
        "data": df,
        "portfolio": portfolio,
        "backtest": backtest,
        "performance": performance
    }


if __name__ == "__main__":
    run_pipeline()

