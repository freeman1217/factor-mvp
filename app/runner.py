import os
import pandas as pd
from app_data import load_market_data
from factors import build_factors
from signals import generate_signals
from portfolio import build_portfolio
from backtest import BacktestEngine, calculate_performance


def run_pipeline(
    price_path="data/prices.csv",
    fundamental_path="data/fundamentals.csv",
    institutional_path="data/institutional.csv",
    dividend_path="data/dividends.csv",
    top_n=20,
    initial_capital=1_000_000
):
    """台股智慧因子投資系統：主執行管線"""
    
    print("🚀 [1/5] 載入並串接 Point-in-Time 市場與財報資料...")
    df = load_market_data(
        price_path=price_path,
        fundamental_path=fundamental_path,
        institutional_path=institutional_path,
        dividend_path=dividend_path
    )

    print("📊 [2/5] 計算五大因子與橫斷面 Z-score...")
    df = build_factors(df)

    print("🎯 [3/5] 生成多因子合成分數與股票排名...")
    df = generate_signals(df)

    print("💼 [4/5] 建構月調倉等權重投資組合...")
    portfolio = build_portfolio(df, top_n=top_n)

    print("📈 [5/5] 執行歷史回測與交易成本核算...")
    engine = BacktestEngine(initial_capital=initial_capital)
    backtest_result = engine.run(df, portfolio)

    # 計算統計績效
    stats = calculate_performance(backtest_result)

    print("\n" + "=" * 40)
    print("🏆 【台股智慧因子策略 - 樣本外績效報告】")
    print("=" * 40)
    for k, v in stats.items():
        if "Return" in k or "Volatility" in k or "Drawdown" in k:
            print(f" - {k:<18}: {v:>8.2%}")
        else:
            print(f" - {k:<18}: {v:>8.2f}")
    print("=" * 40 + "\n")

    return backtest_result, stats


if __name__ == "__main__":
    # 若資料檔存在則直接執行回測
    run_pipeline()
