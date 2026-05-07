import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ⚠️ 替换为你的原始数据文件完整路径
DATA_FILE = r"C:\Users\lenovo\Desktop\二层桌面\因子库\沪深收盘价2014.2.20-2026.2.11"

# 一些常见的 A 股股票/ETF 代码，方便下拉快速选择（只显示你本地数据中存在者）
COMMON_TICKERS = [
    "000001.XSHE", "000002.XSHE", "000858.XSHE", "000651.XSHE",
    "600519.XSHG", "601318.XSHG", "600036.XSHG", "600276.XSHG",
    "300750.XSHE", "002415.XSHE", "688981.XSHG",
    "510050.XSHG", "510300.XSHG", "510500.XSHG", "159915.XSHE", "159919.XSHE"
]

# ==================== 缓存 ====================
@st.cache_data
def get_tickers_and_max_date():
    df_header = pd.read_csv(DATA_FILE, index_col=0, nrows=0)
    tickers = df_header.columns.tolist()
    dates = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True).index
    max_date = dates.max().date()
    return tickers, max_date

@st.cache_data
def load_prices(selected_tickers, start_date, end_date):
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True, usecols=['date'] + selected_tickers)
    df = df.loc[start_date:end_date].dropna()
    return df

# ==================== 页面 ====================
st.set_page_config(page_title="Multi-Asset Analyzer", layout="wide")
st.title("📊 Multi-Asset Efficient Frontier & Risk Parity Analyzer")
st.markdown("Use dropdown to pick common assets, or type codes manually. Click **Run Analysis** to compute.")

all_tickers, max_data_date = get_tickers_and_max_date()

# 筛选出存在于本地的常用代码，作为下拉选项
common_options = [t for t in COMMON_TICKERS if t in all_tickers]

# ==================== 侧边栏 ====================
st.sidebar.header("⚙️ Parameters")

# --- 方式一：下拉多选（从常用列表中点选）---
st.sidebar.subheader("📋 Quick Select (common assets)")
dropdown_selected = st.sidebar.multiselect(
    "Click to select from common list",
    options=common_options,
    default=[],
    help="Select by clicking (avoid typing in this box)."
)

# --- 方式二：文本输入（自由输入代码）---
st.sidebar.subheader("✏️ Manual Input")
manual_input = st.sidebar.text_area(
    "Enter stock codes (one per line, or comma/space)",
    height=100,
    placeholder="e.g.:\n000001.XSHE\n000858.XSHE\n600519.XSHG",
    key="manual_input"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    parse_btn = st.button("🔍 Parse Input", use_container_width=True)
with col2:
    clear_btn = st.button("🗑 Clear All", use_container_width=True)

if clear_btn:
    st.session_state.manual_input = ""
    st.session_state.parsed_manual = []
    st.rerun()

if "parsed_manual" not in st.session_state:
    st.session_state.parsed_manual = []

if parse_btn and manual_input.strip():
    raw = manual_input.replace(',', ' ').split()
    raw = [c.strip() for c in raw if c.strip()]
    valid = [c for c in raw if c in all_tickers]
    invalid = [c for c in raw if c not in all_tickers]
    st.session_state.parsed_manual = valid
    if invalid:
        st.sidebar.warning(f"Ignored invalid: {', '.join(invalid)}")
    if not valid:
        st.sidebar.error("No valid codes found!")
    # 不再 rerun，让下方显示结果

# 合并两个来源，去重
combined = list(dict.fromkeys(list(dropdown_selected) + st.session_state.parsed_manual))
st.sidebar.success(f"✅ Selected: {len(combined)} assets")
if combined:
    st.sidebar.write(", ".join(combined[:20]) + (" ..." if len(combined)>20 else ""))

# 其他参数
default_start = (pd.Timestamp(max_data_date) - pd.DateOffset(years=1)).date()
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=max_data_date)
rf_input = st.sidebar.number_input("Risk-free Rate (annual %)", value=3.0, step=0.5)
rf = rf_input / 100.0
num_portfolios = st.sidebar.slider("Monte Carlo simulations", 1000, 20000, 1000, 1000)

# 运行按钮
run_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

if not run_btn:
    st.info("👆 Select or enter assets, then click **Run Analysis**.")
    st.stop()

if len(combined) < 2:
    st.warning("Need at least 2 valid assets.")
    st.stop()

selected_tickers = combined
df = load_prices(selected_tickers, start_date, end_date)
returns = df.pct_change().dropna()
mu = returns.mean() * 252
cov = returns.cov() * 252

# ==================== 分析模块 ====================
st.subheader("📋 Closing Price Trends")
st.line_chart(df)
# ==================== 1. Normalized Price Trends (Base=1) ====================
st.subheader("📋 Normalized Price Trends (Base=1)")
norm_df = df / df.iloc[0]   # 所有资产起点归一化为 1
st.line_chart(norm_df)
st.subheader("📈 Annualized Return & Risk")
stats = pd.DataFrame({
    "Annual Return %": mu * 100,
    "Annual Vol %": returns.std() * np.sqrt(252) * 100,
    "Sharpe Ratio": (mu - rf) / (returns.std() * np.sqrt(252))
})
st.dataframe(stats.style.highlight_max(axis=0))

st.subheader("🔗 Correlation Heatmap")
corr = returns.corr()
fig_corr, ax_corr = plt.subplots(figsize=(max(6, len(corr)*1.2), max(5, len(corr)*1.0)))
# 用 imshow 代替 matshow，便于控制
im = ax_corr.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
fig_corr.colorbar(im, ax=ax_corr, shrink=0.8)

# 设置刻度
ax_corr.set_xticks(range(len(corr.columns)))
ax_corr.set_yticks(range(len(corr.columns)))
ax_corr.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
ax_corr.set_yticklabels(corr.columns, fontsize=9)

# 在每个格子添加数值
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.iloc[i, j]
        text_color = 'white' if abs(val) > 0.5 else 'black'
        ax_corr.text(j, i, f"{val:.2f}", ha='center', va='center',
                     color=text_color, fontsize=8)

ax_corr.set_title("Asset Correlation Matrix")
st.pyplot(fig_corr)

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

st.subheader("📈 Portfolio Cumulative Returns")
daily_ret = returns.values
cum_ret = pd.DataFrame({
    'Max Sharpe': (1 + daily_ret @ max_w).cumprod(),
    'Min Vol': (1 + daily_ret @ min_w).cumprod(),
    'Risk Parity': (1 + daily_ret @ rp_w).cumprod(),
    'Equal Weight': (1 + daily_ret @ ew_w).cumprod(),
}, index=returns.index)
st.line_chart(cum_ret)