# =====================================================
# SMART PRODUCTION & PROFIT ANALYTICS
# Role: Senior Data Analyst / Data Science
# Goal: Profit Optimization, No Loss, No Returns
# Author: Ismoil Murotaliev
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Smart Production Analytics",
    layout="wide"
)

# =====================================================
# 1. DATA UPLOAD
# =====================================================

st.sidebar.header("📂 Excel fayllarni yuklash")

orders_file = st.sidebar.file_uploader("Buyurtmalar (zakaz.xlsx)", type="xlsx")
returns_file = st.sidebar.file_uploader("Qaytishlar (returns.xlsx)", type="xlsx")

if not orders_file or not returns_file:
    st.info("Analizni boshlash uchun ikkala faylni yuklang")
    st.stop()

orders = pd.read_excel(orders_file)
returns = pd.read_excel(returns_file)

# =====================================================
# 2. DATE & TIME PREPARATION (XATOSIZ)
# =====================================================

for df in [orders, returns]:
    df["Период"] = pd.to_datetime(
        df["Период"],
        errors="coerce",
        dayfirst=True
    )
    df.dropna(subset=["Период"], inplace=True)
    df["date"] = df["Период"].dt.date
    df["hour"] = df["Период"].dt.hour

returns_only = returns[returns["Возрат количество"].notna()]

# =====================================================
# 3. GLOBAL FILTERS
# =====================================================

st.sidebar.header("⏱ Filtrlar")

min_date = orders["date"].min()
max_date = orders["date"].max()

date_range = st.sidebar.date_input(
    "Sana oralig‘i",
    [min_date, max_date]
)

orders_f = orders[
    (orders["date"] >= date_range[0]) &
    (orders["date"] <= date_range[1])
]

returns_f = returns_only[
    (returns_only["date"] >= date_range[0]) &
    (returns_only["date"] <= date_range[1])
]

# =====================================================
# 4. DAILY PRODUCT PERFORMANCE (DESCRIPTIVE)
# =====================================================

st.header("📦 Mahsulotlar bo‘yicha KUNLIK NATIJA")

daily_product = (
    orders_f
    .groupby(["date", "Номенклатура"])
    .agg(
        sold_qty=("Количество", "sum"),
        sold_sum=("Сумма", "sum")
    )
    .reset_index()
)

daily_returns = (
    returns_f
    .groupby(["date", "Номенклатура"])
    .agg(
        return_qty=("Возрат количество", "sum"),
        return_sum=("Возврат сумма", "sum")
    )
    .reset_index()
)

daily_product = daily_product.merge(
    daily_returns,
    on=["date", "Номенклатура"],
    how="left"
).fillna(0)

daily_product["net_result"] = (
    daily_product["sold_sum"] - daily_product["return_sum"]
)

st.dataframe(daily_product)

# =====================================================
# 5. PREDICTIVE: RETURN RISK PER PRODUCT
# =====================================================

st.header("🔮 Ertangi qaytish RISK bashorati")

predictions = []

for product, df_p in daily_product.groupby("Номенклатура"):
    if len(df_p) < 3:
        continue

    df_p = df_p.sort_values("date")
    df_p["day_index"] = range(len(df_p))

    X = df_p[["day_index"]]
    y = df_p["return_qty"]

    model = LinearRegression()
    model.fit(X, y)

    next_day_risk = model.predict([[df_p["day_index"].max() + 1]])[0]

    predictions.append({
        "Номенклатура": product,
        "expected_return_qty": max(0, round(next_day_risk, 2))
    })

risk_forecast = pd.DataFrame(predictions)

st.dataframe(risk_forecast)

# =====================================================
# 6. PRESCRIPTIVE: SAFE PRODUCTION & SALES PLAN
# =====================================================

st.header("🛠 QANDAY QILSA ZARAR BO‘LMAYDI (REJA)")

safe_plan = daily_product.merge(
    risk_forecast,
    on="Номенклатура",
    how="left"
).fillna(0)

# Prescriptive formula (CORE LOGIC)
safe_plan["recommended_production_qty"] = (
    safe_plan["sold_qty"] -
    (safe_plan["expected_return_qty"] * 1.5)
).clip(lower=0)

safe_plan["comment"] = np.where(
    safe_plan["expected_return_qty"] > 0,
    "Ishlab chiqarishni kamaytir",
    "Xavfsiz ishlab chiqarish"
)

st.dataframe(
    safe_plan[[
        "date",
        "Номенклатура",
        "sold_qty",
        "return_qty",
        "expected_return_qty",
        "recommended_production_qty",
        "comment"
    ]]
)

# =====================================================
# 7. DAILY PROFIT GUARANTEE CONTROL
# =====================================================

st.header("✅ Har kun foyda bilan yopish nazorati")

daily_summary = (
    daily_product
    .groupby("date")
    .agg(
        sales=("sold_sum", "sum"),
        returns=("return_sum", "sum")
    )
    .reset_index()
)

daily_summary["risk_reserve"] = daily_summary["sales"] * 0.05

daily_summary["safe_profit"] = (
    daily_summary["sales"] -
    daily_summary["returns"] -
    daily_summary["risk_reserve"]
)

daily_summary["status"] = np.where(
    daily_summary["safe_profit"] > 0,
    "FOYDA ✅",
    "REJANI O‘ZGARTIR ⚠️"
)

st.dataframe(daily_summary)

# =====================================================
# 8. MANAGEMENT CONCLUSION
# =====================================================

st.success(
    "Bu tizim mahsulotni yomon demaydi — "
    "qanday ishlab chiqarsa ZARAR BO‘LMASLIGINI aytadi."
)
