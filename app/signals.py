import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "z_revenue_growth": 0.20,
    "z_earnings_quality": 0.20,
    "z_low_volatility": 0.20,
    "z_trust_buy_ratio": 0.20,
    "z_estimated_yield": 0.20,
}

def generate_signals(df, weights=None, min_factors=1):
    """
    計算多因子合成分數 (Signal Score) 與每日橫斷面排名 (Signal Rank)
    """
    df = df.copy()
    weights = weights or DEFAULT_WEIGHTS

    # 1. 權重檢查與正規化
    weight_s = pd.Series(weights)
    if (weight_s < 0).any() or weight_s.sum() <= 0:
        raise ValueError("權重必須為非負數且總和需大於 0。")

    missing_cols = set(weight_s.index) - set(df.columns)
    if missing_cols:
        raise ValueError(f"缺少因子欄位: {missing_cols}")

    norm_weights = weight_s / weight_s.sum()
    factor_cols = list(norm_weights.index)

    # 2. 合成分數計算 (向量化矩陣運算)
    # 【修正】缺值因子補 0 (Z-score 0 代表中性/市場平均)，避免單一因子缺失導致整檔股票被剔除
    valid_count = df[factor_cols].notna().sum(axis=1)
    weighted_score = df[factor_cols].fillna(0).mul(norm_weights).sum(axis=1)

    # 只有當有效因子數量 >= min_factors 時才保留分數
    df["signal_score"] = np.where(valid_count >= min_factors, weighted_score, np.nan)

    # 3. 每日橫斷面排名 (1 代表最優股票)
    # 【修正】改用 method="min"，避免 method="first" 依資料庫順序造成的隨機偏誤
    df["signal_rank"] = (
        df.groupby("date")["signal_score"]
        .rank(ascending=False, method="min")
    )

    return df
