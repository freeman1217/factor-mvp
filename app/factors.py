"""月營收年增率因子（示範）。對應 requirements/US-101。"""
import pandas as pd


def revenue_yoy(revenue: pd.DataFrame, asof: str) -> pd.DataFrame:
    """回傳每檔股票在 asof 當天「已公布」的最新月營收年增率。

    revenue 欄位：stock_id, month (YYYY-MM), revenue, publish_date
    回傳欄位：stock_id, month_used, yoy（缺值為 NaN，不補 0）
    """
    asof_ts = pd.Timestamp(asof)
    df = revenue.copy()
    df["publish_date"] = pd.to_datetime(df["publish_date"])

    # 對應 Scenario: 公布前使用上一個月營收（只用公布日 <= asof 的資料）
    visible = df[df["publish_date"] <= asof_ts]

    rows = []
    for stock_id, g in visible.groupby("stock_id"):
        latest = g.sort_values("month").iloc[-1]
        month_used = latest["month"]
        last_year = str(pd.Period(month_used, "M") - 12)
        # 去年同月資料也必須是 asof 前已公布（此處以 visible 篩選後的資料查找）
        prev = g[g["month"] == last_year]
        if prev.empty:
            # 對應 Scenario: 缺少去年同月資料時為缺值（不補 0）
            yoy = float("nan")
        else:
            yoy = latest["revenue"] / prev.iloc[0]["revenue"] - 1
        rows.append({"stock_id": stock_id, "month_used": month_used, "yoy": yoy})
    return pd.DataFrame(rows, columns=["stock_id", "month_used", "yoy"])
