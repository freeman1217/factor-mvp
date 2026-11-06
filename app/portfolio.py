import numpy as np
import pandas as pd


def get_rebalance_dates(df):
    """取得資料集內每個月最後一個交易日"""
    dates = pd.to_datetime(df["date"]).drop_duplicates().sort_values()
    return dates.groupby(dates.dt.to_period("M")).max().tolist()


def build_portfolio(df, top_n=20, min_score=None):
    """
    月調倉等權重投資組合建構（全向量化優化版）
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. 批次篩選出所有調倉日且訊號非 NaN 的資料
    rebalance_dates = get_rebalance_dates(df)
    reb_df = df[df["date"].isin(rebalance_dates)].dropna(subset=["signal_score"])

    # 2. 分數門檻過濾
    if min_score is not None:
        reb_df = reb_df[reb_df["signal_score"] >= min_score]

    # 3. 向量化選股：依日期與分數排序後，取每期前 top_n 檔
    portfolio = (
        reb_df.sort_values(["date", "signal_score"], ascending=[True, False])
        .groupby("date", as_index=False)
        .head(top_n)
        .copy()
    )

    # 若全無選出標的，回傳空的 DataFrame
    if portfolio.empty:
        target_cols = ["rebalance_date", "date", "ticker", "signal_score", "signal_rank", "close", "weight"]
        return pd.DataFrame(columns=[c for c in target_cols if c in df.columns or c in ["rebalance_date", "weight"]])

    # 4. 計算等權重（依每期實際入選檔數平分 1.0）
    portfolio["weight"] = 1.0 / portfolio.groupby("date")["ticker"].transform("count")
    portfolio["rebalance_date"] = portfolio["date"]

    # 動態選取存在的欄位，防範 KeyError
    target_cols = ["rebalance_date", "date", "ticker", "signal_score", "signal_rank", "close", "weight"]
    valid_cols = [col for col in target_cols if col in portfolio.columns]

    return portfolio[valid_cols].reset_index(drop=True)


def get_target_weights(portfolio):
    """
    將投資組合明細轉為目標權重矩陣 (Rebalance Date x Ticker)
    """
    if portfolio.empty:
        return pd.DataFrame()

    return portfolio.pivot_table(
        index="rebalance_date",
        columns="ticker",
        values="weight",
        aggfunc="sum",
        fill_value=0.0
    ).sort_index()
