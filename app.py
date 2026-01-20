# ============================================
# STREAMLIT SALES / RETURNS ANALYTICS SYSTEM
# Author: Ismoil Murotaliev
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Sales & Returns Analytics",
    layout="wide"
)

# ============================================
# 1. DATA LOADING
# ============================================

st.sidebar.header("📂 Ma'lumot yuklash")

orders_file = st.sidebar.file_uploader(
    "Buyurtmalar (zakaz.xlsx)", type=["xlsx"]
)
returns_file = st.sidebar.file_uploader(
    "Qaytishlar (returns.xlsx)", type=["xlsx"]
)

if not orders_file or not returns_file:
    st.warning("Iltimos, ikkala Excel faylni yuklang")
    st.stop()

orders = pd.read_excel(orders_file)
returns = pd.read_excel(returns_file)

# ============================================
# 2. DATE & TIME CLEANING (MUHIM QISM)
# ============================================

orders["Период"] = pd.to_datetime(
    orders["Период"],
    errors="coerce",
    dayfirst=True
)

returns["Период"] = pd.to_datetime(
    returns["Период"],
    errors="coerce",
    dayfirst=True
)

# NULL vaqtlarni olib tashlash
orders = orders.dropna(subset=["Период"])
returns = returns.dropna(subset=["Период"])

# Kun va soat
orders["date"] = orders["Период"].dt.date
orders["hour"] = orders["Период"].dt.hour

returns["date"] = returns["Период"].dt.date
returns["hour"] = returns["Период"].dt.hour

# ============================================
# 3. GLOBAL FILTERLAR
# ============================================

st.sidebar.header("⏱ Filtrlar")

min_date = min(orders["date"])
max_date = max(orders["date"])

date_range = st.sidebar.date_input(
    "Sana oralig'i",
    [min_date, max_date]
)

hour_range = st.sidebar.slider(
    "Soat oralig'i",
    0, 23, (0, 23)
)

orders_f = orders[
    (orders["date"] >= date_range[0]) &
    (orders["date"] <= date_range[1]) &
    (orders["hour"] >= hour_range[0]) &
    (orders["hour"] <= hour_range[1])
]

returns_f = returns[
    (returns["date"] >= date_range[0]) &
    (returns["date"] <= date_range[1]) &
    (returns["hour"] >= hour_range[0]) &
    (returns["hour"] <= hour_range[1])
]

returns_only = returns_f[returns_f["Возрат количество"].notna()]

# ============================================
# 4. ANALYSIS #1 – DAILY NET RESULT
# ============================================

st.header("📅 Kunlik sof natija")

daily_sales = orders_f.groupby("date").agg(
    sales_sum=("Сумма", "sum")
).reset_index()

daily_returns = returns_only.groupby("date").agg(
    return_sum=("Возврат сумма", "sum")
).reset_index()

daily = daily_sales.merge(
    daily_returns, how="left", on="date"
).fillna(0)

daily["net_result"] = daily["sales_sum"] - daily["return_sum"]

st.dataframe(daily)

# ============================================
# 5. ANALYSIS #2 – PRODUCT PROFITABILITY
# ============================================

st.header("📦 Mahsulotlar bo‘yicha zarar")

product_returns = returns_only.groupby("Номенклатура").agg(
    return_qty=("Возрат количество", "sum"),
    return_sum=("Возврат сумма", "sum")
).reset_index()

product_sales = orders_f.groupby("Номенклатура").agg(
    sales_qty=("Количество", "sum"),
    sales_sum=("Сумма", "sum")
).reset_index()

product = product_sales.merge(
    product_returns, how="left", on="Номенклатура"
).fillna(0)

product["return_ratio_%"] = (
    product["return_qty"] / product["sales_qty"]
) * 100

st.dataframe(product.sort_values("return_ratio_%", ascending=False))

# ============================================
# 6. ANALYSIS #3 – STOP LIST
# ============================================

st.header("⛔ STOP-LIST mahsulotlar")

stop_list = product[
    product["return_ratio_%"] > 7
][["Номенклатура", "return_ratio_%"]]

st.dataframe(stop_list)

# ============================================
# 7. ANALYSIS #4 – CLIENT RISK
# ============================================

st.header("🏢 Kampaniyalar bo‘yicha zarar")

client_returns = returns_only.groupby("Контрагент").agg(
    return_sum=("Возврат сумма", "sum"),
    return_count=("Возрат количество", "count")
).reset_index()

client_returns["risk"] = np.where(
    client_returns["return_sum"] > 50000,
    "ZARARLI", "YAXSHI"
)

st.dataframe(client_returns.sort_values("return_sum", ascending=False))

# ============================================
# 8. ANALYSIS #5 – CLIENT × PRODUCT MATRIX
# ============================================

st.header("📊 Kampaniya × Mahsulot matritsasi")

matrix = returns_only.pivot_table(
    index="Контрагент",
    columns="Номенклатура",
    values="Возрат количество",
    aggfunc="sum",
    fill_value=0
)

st.dataframe(matrix)

# ============================================
# 9. ANALYSIS #6 – HOURLY RISK
# ============================================

st.header("⏰ Soatlik qaytish xavfi")

hourly = returns_only.groupby("hour").agg(
    return_qty=("Возрат количество", "sum")
).reset_index()

fig, ax = plt.subplots()
ax.bar(hourly["hour"], hourly["return_qty"])
ax.set_xlabel("Soat")
ax.set_ylabel("Qaytish miqdori")
st.pyplot(fig)

# ============================================
# 10. ANALYSIS #7 – DAILY RETURN TREND (ML)
# ============================================

st.header("📈 Qaytish trend prognozi")

trend = returns_only.groupby("date").agg(
    return_sum=("Возврат сумма", "sum")
).reset_index()

trend["day_index"] = range(len(trend))

X = trend[["day_index"]]
y = trend["return_sum"]

model = LinearRegression()
model.fit(X, y)

trend["prediction"] = model.predict(X)

fig2, ax2 = plt.subplots()
ax2.plot(trend["date"], trend["return_sum"], label="Real")
ax2.plot(trend["date"], trend["prediction"], label="Trend")
ax2.legend()
st.pyplot(fig2)

# ============================================
# 11. FINAL CONCLUSION
# ============================================

st.success("Analiz yakunlandi. Qarorlarni qabul qilishga tayyor 🚀")
