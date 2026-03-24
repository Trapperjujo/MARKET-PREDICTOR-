"""
Stock Market Predictor — LSTM + Prophet + Linear Regression
============================================================
Interactive Streamlit dashboard for stock price forecasting.
DISCLAIMER: For educational purposes only. NOT financial advice.

Usage:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(79,195,247,0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero h1 { color: #4fc3f7; font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero p  { color: #b0bec5; font-size: 0.95rem; margin: 0.4rem 0 0; }

/* Metric card */
.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(79,195,247,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-card .label { color: #78909c; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .value { color: #4fc3f7; font-size: 1.55rem; font-weight: 700; }
.metric-card .delta-pos { color: #66bb6a; font-size: 0.85rem; }
.metric-card .delta-neg { color: #ef5350; font-size: 0.85rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a, #1a1a2e);
    border-right: 1px solid rgba(79,195,247,0.15);
}

/* Disclaimer banner */
.disclaimer {
    background: rgba(239,83,80,0.1);
    border: 1px solid rgba(239,83,80,0.3);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: #ef9a9a;
    font-size: 0.78rem;
    text-align: center;
    margin-top: 0.5rem;
}

/* Divider */
hr { border-color: rgba(79,195,247,0.15); }

/* Streamlit overrides */
div.stButton > button {
    background: linear-gradient(135deg, #0f4c81, #1565c0);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #1565c0, #1976d2);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(21,101,192,0.4);
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        return df
    return df[["Close", "Volume"]].copy()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    df["MA_10"]  = close.rolling(10).mean()
    df["MA_20"]  = close.rolling(20).mean()
    df["MA_50"]  = close.rolling(50).mean()

    # User's added Moving Averages
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()

    # 4. Generate crossover signals
    df['Signal'] = 0
    df.loc[(df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1)), 'Signal'] = 1  # Buy
    df.loc[(df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1)), 'Signal'] = -1 # Sell

    roll_std       = close.rolling(20).std()
    df["BB_upper"] = df["MA_20"] + 2 * roll_std
    df["BB_lower"] = df["MA_20"] - 2 * roll_std

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12         = close.ewm(span=12, adjust=False).mean()
    ema26         = close.ewm(span=26, adjust=False).mean()
    df["MACD"]         = ema12 - ema26
    df["MACD_signal"]  = df["MACD"].ewm(span=9, adjust=False).mean()

    df["Return"]     = close.pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()
    for lag in [1, 2, 3, 5]:
        df[f"Lag_{lag}"] = close.shift(lag)
    return df


FEATURE_COLS = [
    "MA_10", "MA_20", "MA_50",
    "SMA_20", "SMA_50", "EMA_20", "Signal",
    "BB_upper", "BB_lower",
    "RSI", "MACD", "MACD_signal",
    "Return", "Volatility",
    "Lag_1", "Lag_2", "Lag_3", "Lag_5",
]


def future_business_dates(last_date, n):
    dates, cur = [], pd.Timestamp(last_date)
    while len(dates) < n:
        cur += pd.Timedelta(days=1)
        if cur.weekday() < 5:
            dates.append(cur)
    return dates


# ── Linear Regression Model ──────────────────────────────────────────────────
def run_linear(df_clean, forecast_days):
    X = df_clean[FEATURE_COLS].values
    y = df_clean["Close"].values
    split = int(len(X) * 0.80)

    sx, sy = MinMaxScaler(), MinMaxScaler()
    Xtr = sx.fit_transform(X[:split])
    Xte = sx.transform(X[split:])
    ytr = sy.fit_transform(y[:split].reshape(-1, 1)).ravel()

    m = LinearRegression().fit(Xtr, ytr)
    y_pred_test = sy.inverse_transform(m.predict(Xte).reshape(-1, 1)).ravel()
    mape = mean_absolute_percentage_error(y[split:], y_pred_test) * 100

    # Future forecast
    last_row = sx.transform(df_clean[FEATURE_COLS].iloc[-1:].values)
    preds = []
    cur = last_row.copy()
    for _ in range(forecast_days):
        p_s = m.predict(cur)
        p   = sy.inverse_transform(p_s.reshape(-1, 1))[0, 0]
        preds.append(p)
        nxt = cur.copy()
        # Shift lag columns (indices 10-13)
        nxt[0, 13] = nxt[0, 12]
        nxt[0, 12] = nxt[0, 11]
        nxt[0, 11] = nxt[0, 10]
        nxt[0, 10] = p_s[0]
        cur = nxt

    return y[split:], y_pred_test, mape, split, preds


# ── LSTM Model ───────────────────────────────────────────────────────────────
def run_lstm(df_clean, forecast_days, lookback=60, epochs=25):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        st.error("TensorFlow not installed. Run: pip install tensorflow")
        return None

    prices = df_clean["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X_all, y_all = [], []
    for i in range(lookback, len(scaled)):
        X_all.append(scaled[i - lookback:i, 0])
        y_all.append(scaled[i, 0])
    if len(X_all) < 10:
        st.error(f"Dataset too small ({len(scaled)} days) for a {lookback}-day lookback window. Increase historical range or decrease lookback.")
        return None

    X_all = np.array(X_all)[..., np.newaxis]
    y_all = np.array(y_all)

    split = int(len(X_all) * 0.80)
    Xtr, Xte = X_all[:split], X_all[split:]
    ytr, yte = y_all[:split], y_all[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    
    val_split = 0.1 if len(Xtr) >= 10 else 0.0
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True)] if val_split > 0 else []

    with st.spinner("🧠 Training LSTM model…"):
        model.fit(Xtr, ytr, epochs=epochs, batch_size=32,
                  validation_split=val_split, callbacks=callbacks, verbose=0)

    y_pred_test = scaler.inverse_transform(model.predict(Xte)).ravel()
    y_test      = scaler.inverse_transform(yte.reshape(-1, 1)).ravel()
    mape        = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    # Iterative future forecast
    seq = scaled[-lookback:].reshape(1, lookback, 1)
    fut_preds = []
    for _ in range(forecast_days):
        p = model.predict(seq, verbose=0)[0, 0]
        fut_preds.append(scaler.inverse_transform([[p]])[0, 0])
        seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)

    return y_test, y_pred_test, mape, lookback + split, fut_preds


# ── Prophet Model ────────────────────────────────────────────────────────────
def run_prophet(df_clean, forecast_days):
    try:
        from prophet import Prophet
    except ImportError:
        st.error("Prophet not installed. Run: pip install prophet")
        return None

    pdf = df_clean["Close"].reset_index()[["Date", "Close"]].copy()
    # Strip timezone from Date (Prophet doesn't accept tz-aware)
    pdf["ds"] = pd.to_datetime(pdf["Date"]).dt.tz_localize(None)
    pdf["y"]  = pdf["Close"].values
    pdf = pdf[["ds", "y"]]

    split_idx = int(len(pdf) * 0.80)
    train_df  = pdf.iloc[:split_idx]
    test_df   = pdf.iloc[split_idx:]

    with st.spinner("🔮 Training Prophet model…"):
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.15,
        )
        m.fit(train_df)

    # In-sample test predictions
    test_forecast = m.predict(test_df[["ds"]])
    y_test      = test_df["y"].values
    y_pred_test = test_forecast["yhat"].values
    mape        = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    # Future forecast
    future = m.make_future_dataframe(periods=forecast_days, freq="B")
    forecast = m.predict(future)
    fut = forecast.iloc[-forecast_days:][["ds", "yhat", "yhat_lower", "yhat_upper"]]

    return y_test, y_pred_test, mape, split_idx, fut


# ── Plotting helpers ──────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
ACCENT   = "#4fc3f7"
GREEN    = "#56d364"
RED      = "#f85149"
ORANGE   = "#f9a825"
PURPLE   = "#ce93d8"
PINK     = "#f48fb1"
GRAY     = "#8b949e"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color="#c9d1d9", family="Inter, sans-serif", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#30363d",
                borderwidth=1, font=dict(size=11)),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#21262d", showgrid=True, linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", showgrid=True, linecolor="#30363d"),
)


def make_price_chart(df, ticker, model_name, test_dates, y_pred_test,
                     fut_dates, fut_preds, fut_lower=None, fut_upper=None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.23, 0.22],
                        vertical_spacing=0.05)

    dates = df.index.to_list()

    # Bollinger bands
    fig.add_trace(go.Scatter(x=dates, y=df["BB_upper"], line=dict(width=0),
                             showlegend=False, name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df["BB_lower"], fill="tonexty",
                             fillcolor="rgba(79,195,247,0.07)",
                             line=dict(width=0), name="Bollinger Band"), row=1, col=1)

    # Actual price
    fig.add_trace(go.Scatter(x=dates, y=df["Close"], line=dict(color="#e6edf3", width=2),
                             name="Actual Price"), row=1, col=1)

    # MAs
    fig.add_trace(go.Scatter(x=dates, y=df["MA_10"],
                             line=dict(color=ORANGE, width=1, dash="dot"), name="MA 10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df["MA_20"],
                             line=dict(color=RED, width=1, dash="dot"), name="MA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df["MA_50"],
                             line=dict(color=GREEN, width=1, dash="dot"), name="MA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df["EMA_20"],
                             line=dict(color="#ffff00", width=1.5, dash="dash"), name="EMA 20"), row=1, col=1)

    # Crossover Signals
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['SMA_20'],
                             mode='markers', marker=dict(color=GREEN, symbol='triangle-up', size=12),
                             name='Buy Signal'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['SMA_20'],
                             mode='markers', marker=dict(color=RED, symbol='triangle-down', size=12),
                             name='Sell Signal'), row=1, col=1)

    # Test predictions
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred_test,
                             line=dict(color=PURPLE, width=1.5), name=f"{model_name} (test)"), row=1, col=1)

    # Confidence interval for Prophet
    if fut_lower is not None and fut_upper is not None:
        fig.add_trace(go.Scatter(x=list(fut_dates) + list(fut_dates)[::-1],
                                 y=list(fut_upper) + list(fut_lower)[::-1],
                                 fill="toself", fillcolor="rgba(79,195,247,0.12)",
                                 line=dict(width=0), name="Confidence Interval"), row=1, col=1)

    # Future forecast
    fig.add_trace(go.Scatter(x=list(fut_dates), y=list(fut_preds),
                             line=dict(color=ACCENT, width=2.5, dash="dash"),
                             marker=dict(size=6, color=ACCENT),
                             name=f"Forecast"), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=dates, y=df["RSI"],
                             line=dict(color="#ff8a65", width=1.4), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=RED,   line_width=0.8, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=GREEN, line_width=0.8, row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=RED,   opacity=0.04, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=GREEN, opacity=0.04, row=2, col=1)

    # MACD histogram
    macd_hist = df["MACD"] - df["MACD_signal"]
    colors_hist = [GREEN if v >= 0 else RED for v in macd_hist]
    fig.add_trace(go.Bar(x=dates, y=macd_hist, marker_color=colors_hist,
                         name="MACD Hist", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df["MACD"],
                             line=dict(color=ACCENT, width=1.2), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df["MACD_signal"],
                             line=dict(color=PINK, width=1.2), name="Signal"), row=3, col=1)

    # Row labels
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1,
                     title_font=dict(color=GRAY, size=11))
    fig.update_yaxes(title_text="RSI", row=2, col=1,
                     range=[0, 100], title_font=dict(color=GRAY, size=11))
    fig.update_yaxes(title_text="MACD", row=3, col=1,
                     title_font=dict(color=GRAY, size=11))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"<b>{ticker}</b> — {model_name} Forecast",
                   font=dict(size=18, color=ACCENT)),
        height=680,
    )
    return fig


def make_forecast_table(fut_dates, fut_preds, last_price, n_show=14):
    rows = []
    for d, p in zip(list(fut_dates)[:n_show], list(fut_preds)[:n_show]):
        change = p - last_price
        pct    = (change / last_price) * 100
        rows.append({
            "Date": d.strftime("%a %b %d") if hasattr(d, "strftime") else str(d)[:10],
            "Forecast Price": f"${p:.2f}",
            "Δ from Today": f"{'▲' if change >= 0 else '▼'} ${abs(change):.2f}",
            "% Change": f"{pct:+.2f}%",
            "_up": change >= 0,
        })
    return pd.DataFrame(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL",
                           help="e.g. AAPL, TSLA, NVDA, MSFT, SPY").upper().strip()

    period = st.selectbox("Historical Data Range",
                          ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    model_choice = st.radio("Prediction Model",
                            ["LSTM (Neural Network)", "Prophet (Facebook)", "Linear Regression"])

    forecast_days = st.slider("Forecast Days", min_value=5, max_value=90, value=30, step=5)

    if model_choice == "LSTM (Neural Network)":
        lookback     = st.slider("LSTM Lookback Window", 20, 120, 60, 10)
        lstm_epochs  = st.slider("Max Training Epochs", 10, 100, 30, 10)
    else:
        lookback, lstm_epochs = 60, 30

    st.markdown("---")
    run_btn = st.button("🚀 Run Prediction")

    st.markdown("""
<div class="disclaimer">
⚠️ <b>DISCLAIMER:</b> Educational purposes only.<br>NOT financial advice. Always do your own research.
</div>
""", unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📈 AI Stock Predictor</h1>
  <p>LSTM · Prophet · Technical Analysis — Powered by real-time market data</p>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    # Landing state
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="label">LSTM</div>
            <div class="value">🧠</div>
            <div class="delta-pos">Deep Learning</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="label">Prophet</div>
            <div class="value">🔮</div>
            <div class="delta-pos">Time-Series AI</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="label">Technical</div>
            <div class="value">📊</div>
            <div class="delta-pos">RSI · MACD · BB</div>
        </div>""", unsafe_allow_html=True)

    st.info("👈 **Configure your settings in the sidebar and press** *Run Prediction* **to start.**")
    st.stop()

# ── Fetch data ────────────────────────────────────────────────────────────────
with st.spinner(f"📡 Fetching {ticker} data…"):
    raw = fetch_data(ticker, period)

if raw.empty:
    st.error(f"❌ Could not fetch data for **{ticker}**. Check the symbol and try again.")
    st.stop()

df = add_indicators(raw.copy()).dropna()

if len(df) < 10:
    st.warning("⚠️ Not enough data points available after calculating technical indicators (which require up to 50 days of warmup). Please increase your 'Historical Data Range' in the sidebar to at least 3mo or 6mo.")
    st.stop()

# Summary metrics
last_price  = df["Close"].iloc[-1]
prev_price  = df["Close"].iloc[-2]
day_change  = last_price - prev_price
day_pct     = (day_change / prev_price) * 100
week_ago    = df["Close"].iloc[-6] if len(df) > 6 else prev_price
week_pct    = ((last_price - week_ago) / week_ago) * 100
rsi_now     = df["RSI"].iloc[-1]

c1, c2, c3, c4, c5 = st.columns(5)
def metric_html(label, val, delta=None):
    delta_cls = "delta-pos" if delta and delta >= 0 else "delta-neg"
    delta_str = ""
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        delta_str = f'<div class="{delta_cls}">{arrow} {abs(delta):.2f}%</div>'
    return f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{val}</div>
        {delta_str}
    </div>"""

with c1: st.markdown(metric_html("Current Price", f"${last_price:.2f}", day_pct), unsafe_allow_html=True)
with c2: st.markdown(metric_html("Day Change", f"{'▲' if day_change>=0 else '▼'} ${abs(day_change):.2f}", day_pct), unsafe_allow_html=True)
with c3: st.markdown(metric_html("Week Change", f"{week_pct:+.2f}%"), unsafe_allow_html=True)
with c4: st.markdown(metric_html("RSI (14)", f"{rsi_now:.1f}",
                                  None if 30 < rsi_now < 70 else (1 if rsi_now <= 30 else -1)), unsafe_allow_html=True)
with c5: st.markdown(metric_html("Data Points", f"{len(df):,}"), unsafe_allow_html=True)

sma20_now = df["SMA_20"].iloc[-1]
sma50_now = df["SMA_50"].iloc[-1]
ema20_now = df["EMA_20"].iloc[-1]
crossovers = df[df["Signal"] != 0]
last_signal = "None"
if not crossovers.empty:
    last_sig_val = crossovers["Signal"].iloc[-1]
    last_signal = "🟢 Buy" if last_sig_val == 1 else "🔴 Sell"
    
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1: st.markdown(metric_html("SMA 20", f"${sma20_now:.2f}"), unsafe_allow_html=True)
with mc2: st.markdown(metric_html("SMA 50", f"${sma50_now:.2f}"), unsafe_allow_html=True)
with mc3: st.markdown(metric_html("EMA 20", f"${ema20_now:.2f}"), unsafe_allow_html=True)
with mc4: st.markdown(metric_html("Last Crossover", last_signal), unsafe_allow_html=True)

st.markdown("---")

# ── Run selected model ────────────────────────────────────────────────────────
result = None

if model_choice == "Linear Regression":
    y_test, y_pred_test, mape, split_idx, fut_preds_list = run_linear(df, forecast_days)
    test_dates  = df.index[split_idx:]
    fut_dates   = future_business_dates(df.index[-1], forecast_days)
    model_name  = "Linear Regression"
    result = (y_test, y_pred_test, mape, test_dates, fut_dates, fut_preds_list, None, None)

elif model_choice == "LSTM (Neural Network)":
    res = run_lstm(df, forecast_days, lookback=lookback, epochs=lstm_epochs)
    if res:
        y_test, y_pred_test, mape, split_idx, fut_preds_list = res
        test_dates = df.index[split_idx:]
        fut_dates  = future_business_dates(df.index[-1], forecast_days)
        model_name = "LSTM"
        result = (y_test, y_pred_test, mape, test_dates, fut_dates, fut_preds_list, None, None)

elif model_choice == "Prophet (Facebook)":
    res = run_prophet(df, forecast_days)
    if res:
        y_test, y_pred_test, mape, split_idx, fut_df = res
        test_dates = df.index[split_idx:]
        fut_dates  = fut_df["ds"].values
        fut_preds_list = fut_df["yhat"].values
        fut_lower = fut_df["yhat_lower"].values
        fut_upper = fut_df["yhat_upper"].values
        model_name = "Prophet"
        result = (y_test, y_pred_test, mape, test_dates, fut_dates, fut_preds_list, fut_lower, fut_upper)

if result is None:
    st.stop()

y_test, y_pred_test, mape, test_dates, fut_dates, fut_preds_list, fut_lower, fut_upper = result

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader(f"🔮 {model_name} Forecast for **{ticker}**")
fig = make_price_chart(df, ticker, model_name, test_dates, y_pred_test,
                       fut_dates, fut_preds_list, fut_lower, fut_upper)
st.plotly_chart(fig, use_container_width=True)

# ── Model accuracy ────────────────────────────────────────────────────────────
acc_col, info_col = st.columns([1, 2])
with acc_col:
    accuracy = max(0, 100 - mape)
    color    = GREEN if accuracy >= 90 else (ORANGE if accuracy >= 75 else RED)
    st.markdown(f"""<div class="metric-card">
        <div class="label">Backtest Accuracy</div>
        <div class="value" style="color:{color}">{accuracy:.1f}%</div>
        <div class="delta-pos">Test MAPE: {mape:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with info_col:
    end_pred = fut_preds_list[-1] if hasattr(fut_preds_list, '__len__') else float(fut_preds_list[-1])
    total_change = end_pred - last_price
    total_pct    = (total_change / last_price) * 100
    direction    = "📈 Bullish" if total_change >= 0 else "📉 Bearish"
    st.markdown(f"""
**{direction} outlook** over the next **{forecast_days} trading days**

- **Start:** ${last_price:.2f}   → **End forecast:** ${end_pred:.2f}
- **Expected move:** {'▲' if total_change>=0 else '▼'} ${abs(total_change):.2f} ({total_pct:+.2f}%)
- **Model:** {model_name}  |  **Lookback period:** {period}
""")

st.markdown("---")

# ── Forecast table ────────────────────────────────────────────────────────────
st.subheader(f"📅 {min(14, forecast_days)}-Day Price Forecast Table")
table_df = make_forecast_table(fut_dates, fut_preds_list, last_price)

def highlight_row(row):
    color = "color: #56d364" if row["_up"] else "color: #f85149"
    return [color if c in ("Δ from Today", "% Change") else "color: #c9d1d9"
            for c in table_df.columns]

display_df = table_df.drop(columns=["_up"])
st.dataframe(
    display_df.style.apply(
        lambda row: ["" if col == "Date" else
                     ("color: #56d364;" if "▲" in str(row.get(col, "")) or "+" in str(row.get(col, "")) else
                      "color: #f85149;" if "▼" in str(row.get(col, "")) or (
                              "%" in str(row.get(col, "")) and "-" in str(row.get(col, ""))) else "")
                     for col in display_df.columns],
        axis=1,
    ),
    use_container_width=True,
    height=min(450, (len(display_df) + 1) * 35 + 10),
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:0.78rem; margin-top:2rem;">
    AI Stock Predictor · Data via Yahoo Finance · Educational use only · Not financial advice<br>
    Built with Streamlit, TensorFlow, Prophet, Plotly
</div>
""", unsafe_allow_html=True)
