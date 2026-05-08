import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 云端部署：直接读取已上传的小样本文件
DATA_FILE = "prices.csv"

# ==================== 缓存函数 ====================
@st.cache_data
def load_all_data():
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    return df

# ==================== 页面配置 ====================
st.set_page_config(page_title="Multi-Asset Analyzer", layout="wide")
st.title("📊 Multi-Asset Efficient Frontier & Risk Parity Analyzer")
st.markdown("Select assets, adjust parameters, then click **Run Analysis**.")

# 加载全部数据
prices = load_all_data()
all_tickers = prices.columns.tolist()
max_data_date = prices.index.max().date()

# ==================== 侧边栏 ====================
st.sidebar.header("⚙️ Parameters")

# 下拉多选（全量股票，可搜索）
selected_tickers = st.sidebar.multiselect(
    "Select assets (type to search, 3-10 recommended)",
    options=all_tickers,
    default=all_tickers[:3] if len(all_tickers) >= 3 else all_tickers
)

# 日期范围
default_start = (pd.Timestamp(max_data_date) - pd.DateOffset(years=1)).date()
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=max_data_date)

rf_input = st.sidebar.number_input("Risk-free Rate (annual %)", value=3.0, step=0.5)
rf = rf_input / 100.0
num_portfolios = st.sidebar.slider("Monte Carlo simulations", 1000, 20000, 1000, 1000)

# 运行按钮
run_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

if not run_btn:
    st.info("👆 Select assets and parameters, then click **Run Analysis**.")
    st.stop()

if len(selected_tickers) < 2:
    st.warning("Need at least 2 assets.")
    st.stop()

# 过滤数据
df = prices.loc[start_date:end_date, selected_tickers].dropna()
returns = df.pct_change().dropna()
mu = returns.mean() * 252
cov = returns.cov() * 252

# ==================== 1. 归一化净值走势 ====================
st.subheader("📋 Normalized Price Trends (Base=1)")
norm_df = df / df.iloc[0]
st.line_chart(norm_df)

# ==================== 2. 年化统计 ====================
st.subheader("📈 Annualized Return & Risk")
stats = pd.DataFrame({
    "Annual Return %": mu * 100,
    "Annual Vol %": returns.std() * np.sqrt(252) * 100,
    "Sharpe Ratio": (mu - rf) / (returns.std() * np.sqrt(252))
})
st.dataframe(stats.style.highlight_max(axis=0))

# ==================== 3. 相关性热力图（带数值） ====================
st.subheader("🔗 Correlation Heatmap")
corr = returns.corr()
fig_corr, ax_corr = plt.subplots(figsize=(max(6, len(corr)*1.2), max(5, len(corr)*1.0)))
im = ax_corr.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
fig_corr.colorbar(im, ax=ax_corr, shrink=0.8)
ax_corr.set_xticks(range(len(corr.columns)))
ax_corr.set_yticks(range(len(corr.columns)))
ax_corr.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
ax_corr.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.iloc[i, j]
        text_color = 'white' if abs(val) > 0.5 else 'black'
        ax_corr.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=8)
ax_corr.set_title("Asset Correlation Matrix")
st.pyplot(fig_corr)

# ==================== 4. 有效前沿 ====================
st.subheader("🚀 Efficient Frontier")
n = len(selected_tickers)
np.random.seed(42)
weights = np.random.dirichlet(np.ones(n), num_portfolios)
port_returns = weights @ mu
port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov, weights))
port_sharpes = (port_returns - rf) / port_vols

max_idx = np.argmax(port_sharpes)
min_idx = np.argmin(port_vols)

fig_ef, ax_ef = plt.subplots()
sc = ax_ef.scatter(port_vols, port_returns, c=port_sharpes, cmap='viridis', alpha=0.5)
ax_ef.scatter(port_vols[max_idx], port_returns[max_idx], marker='*', color='red', s=200, label='Max Sharpe')
ax_ef.scatter(port_vols[min_idx], port_returns[min_idx], marker='*', color='blue', s=200, label='Min Vol')

def risk_parity_weights(cov):
    def rc_objective(w):
        sigma = np.sqrt(w @ cov @ w)
        rc = w * (cov @ w) / sigma
        return np.sum((rc - sigma/n)**2)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0,1) for _ in range(n)]
    res = minimize(rc_objective, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

rp_w = risk_parity_weights(cov.values)
rp_ret = rp_w @ mu
rp_vol = np.sqrt(rp_w @ cov.values @ rp_w)
rp_sharpe = (rp_ret - rf) / rp_vol
ax_ef.scatter(rp_vol, rp_ret, marker='*', color='green', s=200, label='Risk Parity')
ax_ef.set_xlabel('Volatility')
ax_ef.set_ylabel('Return')
ax_ef.set_title('Efficient Frontier')
fig_ef.colorbar(sc, label='Sharpe Ratio')
ax_ef.legend()
st.pyplot(fig_ef)

# ==================== 5. 组合对比 ====================
st.subheader("📊 Optimal Portfolios Comparison")
max_w = weights[max_idx]
min_w = weights[min_idx]
ew_w = np.ones(n) / n

def port_stats(w):
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe

df_comp = pd.DataFrame({
    'Max Sharpe': port_stats(max_w),
    'Min Vol': port_stats(min_w),
    'Risk Parity': port_stats(rp_w),
    'Equal Weight': port_stats(ew_w)
}, index=['Return', 'Volatility', 'Sharpe'])
st.dataframe(df_comp.style.highlight_max(axis=1))

st.subheader("💼 Optimal Weights")
weights_df = pd.DataFrame({
    'Max Sharpe': max_w,
    'Min Vol': min_w,
    'Risk Parity': rp_w,
    'Equal Weight': ew_w
}, index=selected_tickers)
st.dataframe(weights_df.style.format("{:.2%}"))

# ==================== 6. 净值回测 ====================
st.subheader("📈 Portfolio Cumulative Returns")
daily_ret = returns.values
cum_ret = pd.DataFrame({
    'Max Sharpe': (1 + daily_ret @ max_w).cumprod(),
    'Min Vol': (1 + daily_ret @ min_w).cumprod(),
    'Risk Parity': (1 + daily_ret @ rp_w).cumprod(),
    'Equal Weight': (1 + daily_ret @ ew_w).cumprod(),
}, index=returns.index)
st.line_chart(cum_ret)
