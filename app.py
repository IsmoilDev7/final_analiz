# =====================================================
# SMART DAILY PRODUCTION & PROFIT ANALYTICS
# Focus: How to avoid loss, not who is bad
# Level: Senior Data Analyst / Data Science
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Smart Production Analytics",
    layout="wide"
)

# =====================================================
# 1. DATA UPLOAD
# =====================================================

st.sidebar.header("📂 Excel fayllarni yuklash")

orders_file = st.sidebar.file_uploader(
    "Buyurtmalar (zakaz.xlsx)", type="xlsx"
)
returns_file = st.sidebar.file_uploader(
    "Qaytishlar (returns.xlsx)", type="xlsx"
)

if not orders_file or not returns_file:
    st.info("Analizni boshlash uchun ikkala Excel faylni yuklang")
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

returns_only = returns[returns["Возрат количество"].notna()]

# =====================================================
# 3. DATE FILTER (1 KUNLIK HAM XATOSIZ)
# =====================================================

st.sidebar.header("⏱ Sana filtri")

min_date = orders["date"].min()
max_date = orders["date"].max()

date_range = st.sidebar.date_input(
    "Sana oralig‘i",
    value=[min_date, max_date]
)

# MUHIM: 1 kun tanlansa ham ishlaydi
if isinstance(date_range, list) and len(date_range) == 1:
    start_date = end_date = date_range[0]
else:
    start_date, end_date = date_range

orders_f = orders[
    (orders["date"] >= start_date) &
    (orders["date"] <= end_date)
]

returns_f = returns_only[
    (returns_only["date"] >= start_date) &
    (returns_only["date"] <= end_date)
]

# =====================================================
# 4. 1️⃣ DAILY PRODUCT RESULT (FAQAT RAQAMLAR)
# =====================================================

st.header("1️⃣ Mahsulotlar bo‘yicha KUNLIK NATIJA (Raqamlar)")

daily_sales = (
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

daily = daily_sales.merge(
    daily_returns,
    on=["date", "Номенклатура"],
    how="left"
).fillna(0)

daily["net_qty"] = daily["sold_qty"] - daily["return_qty"]
daily["net_sum"] = daily["sold_sum"] - daily["return_sum"]

st.dataframe(
    daily.rename(columns={
        "sold_qty": "Sotilgan (dona)",
        "return_qty": "Qaytgan (dona)",
        "net_qty": "Sof sotilgan (dona)",
        "sold_sum": "Sotuv summasi",
        "return_sum": "Qaytish summasi",
        "net_sum": "Sof tushum"
    })
)

# =====================================================
# 5. 2️⃣ TOMORROW RETURN RISK (SON BILAN)
# =====================================================

st.header("2️⃣ Ertangi qaytish RISK bashorati (dona)")

risk_rows = []

for product, df_p in daily.groupby("Номенклатура"):
    if len(df_p) < 2:
        continue

    df_p = df_p.sort_values("date")
    df_p["day_index"] = range(len(df_p))

    X = df_p[["day_index"]]
    y = df_p["return_qty"]

    model = LinearRegression()
    model.fit(X, y)

    next_day_return = model.predict(
        [[df_p["day_index"].max() + 1]]
    )[0]

    risk_rows.append({
        "Номенклатура": product,
        "Ertaga kutilayotgan qaytish (dona)": round(
            max(0, next_day_return), 2
        )
    })

risk_forecast = pd.DataFrame(risk_rows)
st.dataframe(risk_forecast)

# =====================================================
# 6. 3️⃣ HOW TO AVOID LOSS (PRESCRIPTIVE PLAN)
# =====================================================

st.header("3️⃣ QANDAY QILSA ZARAR BO‘LMAYDI (REJA)")

plan = daily.merge(
    risk_forecast,
    on="Номенклатура",
    how="left"
).fillna(0)

plan["Tavsiya etilgan ishlab chiqarish (dona)"] = (
    plan["sold_qty"] -
    (plan["Ertaga kutilayotgan qaytish (dona)"] * 1.5)
).clip(lower=0)

st.dataframe(
    plan[[
        "date",
        "Номенклатура",
        "sold_qty",
        "return_qty",
        "Ertaga kutilayotgan qaytish (dona)",
        "Tavsiya etilgan ishlab chiqarish (dona)"
    ]].rename(columns={
        "sold_qty": "Bugun sotilgan (dona)",
        "return_qty": "Bugun qaytgan (dona)"
    })
)

# =====================================================
# 7. 4️⃣ DAILY PROFIT GUARANTEE CONTROL
# =====================================================

st.header("4️⃣ Har kun foyda bilan yopish NAZORATI")

daily_profit = (
    daily
    .groupby("date")
    .agg(
        total_sales_sum=("sold_sum", "sum"),
        total_return_sum=("return_sum", "sum")
    )
    .reset_index()
)

daily_profit["risk_reserve_5_percent"] = (
    daily_profit["total_sales_sum"] * 0.05
)

daily_profit["safe_profit"] = (
    daily_profit["total_sales_sum"] -
    daily_profit["total_return_sum"] -
    daily_profit["risk_reserve_5_percent"]
)

st.dataframe(
    daily_profit.rename(columns={
        "total_sales_sum": "Kunlik sotuv summasi",
        "total_return_sum": "Kunlik qaytish summasi",
        "risk_reserve_5_percent": "Risk rezerv (5%)",
        "safe_profit": "Xavfsiz sof foyda"
    })
)

# =====================================================
# 8. FINAL MESSAGE
# =====================================================

st.success(
    "Bu tizim mahsulotni yomon demaydi. "
    "U ishlab chiqarishni shunday rejalaydiki – ZARAR BO‘LMAYDI."
)
