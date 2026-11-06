from behave import given, when, then
import pandas as pd
import numpy as np

# 載入你寫好的主程式模組 (若放在 app 資料夾下，請改為 from app.signals import ...)
from signals import generate_signals 

@given(u'有一份包含股票 A (資料完整) 與股票 B (缺少殖利率因子) 的資料集')
def step_impl(context):
    # 建立測試用的假資料
    context.df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
        "ticker": ["Stock_A", "Stock_B"],
        "z_revenue_growth": [1.0, 0.5],
        "z_earnings_quality": [0.8, -0.2],
        "z_low_volatility": [0.5, 0.1],
        "z_trust_buy_ratio": [1.2, 0.8],
        "z_estimated_yield": [0.9, np.nan]  # 刻意讓 B 缺少殖利率因子
    })

@when(u'系統以預設權重執行 generate_signals')
def step_impl(context):
    # 將假資料丟進你寫好的 generate_signals 函式中執行
    context.result = generate_signals(context.df)

@then(u'股票 A 必須獲得有效的分數')
def step_impl(context):
    score_a = context.result.loc[context.result["ticker"] == "Stock_A", "signal_score"].iloc[0]
    
    # 斷言 (Assert)：如果 score_a 是 NaN，測試就會報錯（亮紅燈）
    assert not np.isnan(score_a), "錯誤：股票 A 的分數不應為 NaN"

@then(u'股票 B 也必須獲得有效分數，不應被剔除為 NaN')
def step_impl(context):
    score_b = context.result.loc[context.result["ticker"] == "Stock_B", "signal_score"].iloc[0]
    
    # 驗證之前修復的靜默 Bug：缺單一因子不能被無情淘汰
    assert not np.isnan(score_b), "錯誤：股票 B 缺單一因子，不應被剔除"
