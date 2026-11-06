import pandas as pd
import numpy as np

def prepare_fundamentals(fundamentals):
    """計算 YoY 營收成長率"""
    return (fundamentals
        .drop_duplicates(subset=["ticker", "period_end"], keep="last")
        .sort_values(["ticker", "period_end"])
        .assign(
            revenue_prev_year=lambda x: x.groupby("ticker")["revenue"].shift(4),
            revenue_growth=lambda x: (x["revenue"] / x["revenue_prev_year"].replace(0, np.nan)) - 1
        )
    )

def merge_point_in_time(left, right, date_col="published_at", suffix=""):
    """安全串接 Point-in-Time 資料，並嚴格檢查前視偏誤"""
    # 清理與排序 (防 NaN 與重複鍵)
    left = left.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"])
    right = right.dropna(subset=[date_col, "ticker"]) \
                 .sort_values([date_col, "ticker"]) \
                 .drop_duplicates(subset=[date_col, "ticker"], keep="last")

    # 執行 merge_asof
    result = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on=date_col,
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", suffix)
    )

    # 統一執行前視偏誤防呆檢查
    # 注意：如果多次 merge，右表的日期欄位可能會被加上 suffix
    check_col = f"{date_col}{suffix}" if f"{date_col}{suffix}" in result.columns else date_col
    invalid = result[check_col].notna() & (result[check_col] > result["date"])
    if invalid.any():
        raise ValueError(f"檢測到前視偏誤 (Look-ahead bias)！來源後綴：{suffix}")

    return result

def load_market_data(price_path, fundamental_path, institutional_path, dividend_path):
    """載入並整併所有市場與基本面資料"""
    
    # 1. 讀取並直接解析日期 (取代原本的 load_csv)
    prices = pd.read_csv(price_path, parse_dates=["date"])
    fundamentals = pd.read_csv(fundamental_path, parse_dates=["period_end", "published_at"])
    institutional = pd.read_csv(institutional_path, parse_dates=["published_at"])
    dividends = pd.read_csv(dividend_path, parse_dates=["published_at"])

    # 2. 基本面預處理
    fundamentals = prepare_fundamentals(fundamentals)

    # 3. 統一呼叫同一個 merge 函式，依序串接
    df = merge_point_in_time(prices, fundamentals, suffix="_fund")
    df = merge_point_in_time(df, institutional, suffix="_inst")
    df = merge_point_in_time(df, dividends, suffix="_div")

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
