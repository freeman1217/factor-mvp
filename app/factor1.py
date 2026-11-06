import numpy as np
import pandas as pd

FACTOR_COLUMNS = [
    "revenue_growth",
    "earnings_quality",
    "low_volatility",
    "trust_buy_ratio",
    "estimated_yield"
]

def calculate_raw_factors(df):
    """計算五大基礎因子"""
    df = df.copy()
    df = df.sort_values(["ticker", "date"])

    # 1. 營收成長 (Revenue growth)
    df["factor_revenue_growth"] = df["revenue_growth"]

    # 2. 盈餘品質 (Earnings quality)
    # 【修正】分母必須取絕對值，避免「雙負號」得出高分的地雷
    net_income_abs = df["net_income"].abs().replace(0, np.nan)
    df["factor_earnings_quality"] = df["operating_cash_flow"] / net_income_abs

    # 3. 低波動 (Low volatility)
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    df["factor_low_volatility"] = -df.groupby("ticker")["daily_return"].transform(
        lambda x: x.rolling(window=60, min_periods=40).std()
    )

    # 4. 投信買超佔比 (Institutional buying ratio)
    # 【修正】依據 US-104 規格，計算過去 5 日累計買超，並除以發行總張數(或成交量)
    trust_net_buy_5d = df.groupby("ticker")["trust_net_buy"].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    # 若資料庫有 shares_outstanding 就優先用，否則退回使用 volume
    base_shares = df.get("shares_outstanding", df["volume"]).replace(0, np.nan)
    df["factor_trust_buy_ratio"] = trust_net_buy_5d / base_shares

    # 5. 預估殖利率 (Estimated dividend yield)
    df["factor_estimated_yield"] = (
        df["estimated_annual_dividend"] / df["close"].replace(0, np.nan)
    )

    return df


def cross_sectional_zscore(df):
    """
    計算橫斷面 Z-score (Cross-sectional Z-score)
    """
    df = df.copy()
    
    raw_columns = [f"factor_{col}" for col in FACTOR_COLUMNS]
    z_columns = [f"z_{col}" for col in FACTOR_COLUMNS]

    # 【極致優化】移除 sklearn StandardScaler 與繁瑣的 apply 迴圈
    # 1. Pandas 的 .transform 會針對每個 factor 獨立運算，解決「連坐法」遺失資料的問題。
    # 2. (x - mean) / std 自動忽略 NaN，運算速度大幅提升。
    df[z_columns] = df.groupby("date")[raw_columns].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1)
    )

    return df


def build_factors(df):
    """因子計算主流程"""
    df = calculate_raw_factors(df)
    df = cross_sectional_zscore(df)
    
    return df
