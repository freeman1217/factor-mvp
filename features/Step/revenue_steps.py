"""US-101 的步驟定義。每個步驟對應 features/revenue_yoy_pit.feature。"""
import math

import pandas as pd
from behave import given, when, then

from app.factors import revenue_yoy


@given('公司 "{stock}" 的月營收資料如下')
def step_table(context, stock):
    context.rows = [
        {"stock_id": stock, "month": r["month"], "revenue": float(r["revenue"]),
         "publish_date": r["publish_date"]}
        for r in context.table
    ]


@given('公司 "{stock}" 只有 {month} 的營收 {revenue:d}，公布日為 "{pub}"')
def step_single(context, stock, month, revenue, pub):
    context.rows.append({"stock_id": stock, "month": month,
                         "revenue": float(revenue), "publish_date": pub})


@when('系統在 "{asof}" 計算月營收年增率')
def step_compute(context, asof):
    context.result = revenue_yoy(pd.DataFrame(context.rows), asof).set_index("stock_id")


@then('公司 "{stock}" 使用的營收月份為 "{month}"')
def step_month(context, stock, month):
    assert context.result.loc[stock, "month_used"] == month


@then('公司 "{stock}" 的年增率為 {pct:d}%')
def step_yoy(context, stock, pct):
    assert abs(context.result.loc[stock, "yoy"] - pct / 100) < 1e-9


@then('公司 "{stock}" 的年增率為缺值')
def step_nan(context, stock):
    assert math.isnan(context.result.loc[stock, "yoy"])


@then('公司 "{stock}" 的年增率不是 0')
def step_not_zero(context, stock):
    assert context.result.loc[stock, "yoy"] != 0
