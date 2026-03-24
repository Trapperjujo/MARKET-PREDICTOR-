"""
Stock Market Predictor
======================
Uses real historical data + technical indicators + ML to forecast short-term price trends.
DISCLAIMER: For educational purposes only. NOT financial advice.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURATION  (edit these)
# ─────────────────────────────────────────────
TICKER       = "NVDA"    # Stock symbol  (e.g. AAPL, TSLA, MSFT, NVDA)
PERIOD       = "6mo"     # Historical data range: 1mo 3mo 6mo 1y 2y 5y
FORECAST_DAYS = 365      # How many trading days ahead to predict
# ─────────────────────────────────────────────


def add_indicators(df):
    """Add technical indicators as features."""
    close = df["Close"]

    # Moving averages
    df["MA_10"]  = close.rolling(10).mean()
    df["MA_20"]  = close.rolling(20).mean()
    df["MA_50"]  = close.rolling(50).mean()

    # Bollinger Bands (20-day)
    rolling_std   = close.rolling(20).std()
    df["BB_upper"] = df["MA_20"] + 2 * rolling_std
    df["BB_lower"] = df["MA_20"] - 2 * rolling_std

    # RSI (Relative Strength Index)
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12      = close.ewm(span=12, adjust=False).mean()
    ema26      = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Daily return & volatility
    df["Return"]     = close.pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"Lag_{lag}"] = close.shift(lag)

    return df


def build_features(df):
    """Select and prepare feature matrix."""
    feature_cols = [
        "MA_10", "MA_20", "MA_50",
        "BB_upper", "BB_lower",
        "RSI", "MACD", "MACD_signal",
        "Return", "Volatility",
        "Lag_1", "Lag_2", "Lag_3", "Lag_5",
    ]
    df = df.dropna()
    X = df[feature_cols].values
    y = df["Close"].values
    return X, y, df


def predict_future(model, scaler_X, scaler_y, df, n_days):
    """Extrapolate n_days into the future using the trained model."""
    feature_cols = [
        "MA_10", "MA_20", "MA_50",
        "BB_upper", "BB_lower",
        "RSI", "MACD", "MACD_signal",
        "Return", "Volatility",
        "Lag_1", "Lag_2", "Lag_3", "Lag_5",
    ]
    last_row  = df[feature_cols].iloc[-1].values.reshape(1, -1)
    last_row_scaled = scaler_X.transform(last_row)

    predictions = []
    current_features = last_row_scaled.copy()

    for _ in range(n_days):
        pred_scaled = model.predict(current_features)
        pred_price  = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        predictions.append(pred_price)

        # Shift lag features (crude approximation for multi-step)
        new_features = current_features.copy()
        # Update lag indices: Lag_1 → col 10, Lag_2 → 11, Lag_3 → 12, Lag_5 → 13
        new_features[0, 13] = new_features[0, 12]   # lag5  ← lag3 approx
        new_features[0, 12] = new_features[0, 11]   # lag3  ← lag2
        new_features[0, 11] = new_features[0, 10]   # lag2  ← lag1
        new_features[0, 10] = pred_scaled[0]         # lag1  ← new pred

        current_features = new_features

    return predictions


def make_future_dates(last_date, n_days):
    """Generate n_days of future business days from last_date."""
    dates = []
    current = pd.Timestamp(last_date)
    while len(dates) < n_days:
        current += pd.Timedelta(days=1)
        if current.weekday() < 5:   # Mon–Fri only
            dates.append(current)
    return dates


def plot_results(ticker, df, train_end_idx, y_test, y_pred_test, future_dates, future_preds):
    fig = plt.figure(figsize=(16, 12), facecolor="#0d0d0d")
    fig.suptitle(f"  {ticker} — Stock Predictor  ",
                 fontsize=20, fontweight="bold", color="white",
                 x=0.5, y=0.98, va="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#4fc3f7", linewidth=2))

    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                  left=0.07, right=0.97, top=0.91, bottom=0.07)

    # ── Panel 1: Price + MA + Bollinger + Forecast ──────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#111122")
    dates_all  = df.index
    close_all  = df["Close"].values

    ax1.fill_between(dates_all, df["BB_lower"], df["BB_upper"],
                     alpha=0.15, color="#4fc3f7", label="Bollinger Band")
    ax1.plot(dates_all, close_all,   color="#ffffff",  lw=1.5, label="Actual Price")
    ax1.plot(dates_all, df["MA_10"], color="#f9a825",  lw=1,   linestyle="--", label="MA 10")
    ax1.plot(dates_all, df["MA_20"], color="#ef5350",  lw=1,   linestyle="--", label="MA 20")
    ax1.plot(dates_all, df["MA_50"], color="#66bb6a",  lw=1,   linestyle="--", label="MA 50")

    # Test predictions
    test_dates = df.index[train_end_idx:]
    ax1.plot(test_dates, y_pred_test, color="#ce93d8", lw=1.5, linestyle="-", label="ML Predicted")

    # Future forecast
    ax1.plot(future_dates, future_preds, color="#4fc3f7", lw=2,
             linestyle="--", marker="o", markersize=4, label=f"Forecast (+{FORECAST_DAYS}d)")
    ax1.axvline(x=df.index[-1], color="#555", linestyle=":", lw=1)

    ax1.set_title("Price History + Bollinger Bands + ML Forecast", color="white", fontsize=11, pad=6)
    ax1.tick_params(colors="gray")
    ax1.spines[["top","right","left","bottom"]].set_edgecolor("#333")
    ax1.set_facecolor("#111122")
    ax1.yaxis.label.set_color("white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    leg = ax1.legend(loc="upper left", fontsize=8, fancybox=True, framealpha=0.3,
                     labelcolor="white", facecolor="#111")
    ax1.set_ylabel("Price (USD)", color="gray", fontsize=9)

    # ── Panel 2: RSI ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#111122")
    ax2.plot(dates_all, df["RSI"], color="#ff8a65", lw=1.4)
    ax2.axhline(70, color="#ef5350", linestyle="--", lw=0.8, alpha=0.7)
    ax2.axhline(30, color="#66bb6a", linestyle="--", lw=0.8, alpha=0.7)
    ax2.fill_between(dates_all, df["RSI"], 70,
                     where=(df["RSI"] >= 70), alpha=0.25, color="#ef5350")
    ax2.fill_between(dates_all, df["RSI"], 30,
                     where=(df["RSI"] <= 30), alpha=0.25, color="#66bb6a")
    ax2.set_ylim(0, 100)
    ax2.set_title("RSI (14)", color="white", fontsize=10, pad=4)
    ax2.tick_params(colors="gray")
    ax2.spines[["top","right","left","bottom"]].set_edgecolor("#333")
    ax2.text(dates_all[5], 72, "Overbought", color="#ef5350", fontsize=7)
    ax2.text(dates_all[5], 23, "Oversold",   color="#66bb6a", fontsize=7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # ── Panel 3: MACD ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#111122")
    ax3.plot(dates_all, df["MACD"],        color="#4fc3f7", lw=1.2, label="MACD")
    ax3.plot(dates_all, df["MACD_signal"], color="#f48fb1", lw=1.2, label="Signal")
    macd_hist = df["MACD"] - df["MACD_signal"]
    ax3.bar(dates_all, macd_hist, color=np.where(macd_hist >= 0, "#66bb6a", "#ef5350"),
            alpha=0.5, width=1)
    ax3.axhline(0, color="#555", lw=0.8)
    ax3.set_title("MACD", color="white", fontsize=10, pad=4)
    ax3.tick_params(colors="gray")
    ax3.spines[["top","right","left","bottom"]].set_edgecolor("#333")
    leg3 = ax3.legend(fontsize=8, fancybox=True, framealpha=0.3,
                      labelcolor="white", facecolor="#111")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # ── Panel 4: Prediction table ────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor("#0d0d0d")
    ax4.axis("off")

    last_actual = df["Close"].iloc[-1]
    rows = []
    for i, (d, p) in enumerate(zip(future_dates, future_preds)):
        change   = p - last_actual
        pct      = (change / last_actual) * 100
        direction = "▲" if change >= 0 else "▼"
        rows.append([
            d.strftime("%a %b %d"),
            f"${p:.2f}",
            f"{direction} ${abs(change):.2f}",
            f"{pct:+.2f}%"
        ])

    table = ax4.table(
        cellText=rows,
        colLabels=["Date", "Predicted Price", "Change from Today", "% Change"],
        cellLoc="center", loc="center",
        bbox=[0.05, 0, 0.90, 1.0]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if r == 0:
            cell.set_facecolor("#1a1a2e")
            cell.set_text_props(color="#4fc3f7", fontweight="bold")
        else:
            cell.set_facecolor("#111122")
            # Color code direction
            if c == 2 or c == 3:
                text = cell.get_text().get_text()
                cell.set_text_props(color="#66bb6a" if "▲" in text or "+" in text else "#ef5350")
            else:
                cell.set_text_props(color="white")

    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    accuracy_note = (
        f"Model: Linear Regression  |  Test MAPE: {mape:.1f}%  |  "
        f"Current Price: ${last_actual:.2f}  |  "
        f"⚠️  NOT financial advice — predictions are probabilistic estimates"
    )
    fig.text(0.5, 0.01, accuracy_note, ha="center", fontsize=8.5, color="#888",
             style="italic")

    plt.savefig(f"{ticker}_prediction.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n✅  Chart saved as {ticker}_prediction.png")
    plt.show()


def main():
    ticker = TICKER.upper()
    print(f"\n📈  Fetching data for {ticker}...")

    stock = yf.Ticker(ticker)
    df    = stock.history(period=PERIOD)

    if df.empty:
        print(f"❌  Could not fetch data for '{ticker}'. Check the symbol and try again.")
        sys.exit(1)

    df = df[["Close", "Volume"]].copy()
    df = add_indicators(df)

    X, y, df = build_features(df)

    # 80/20 train-test split
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s  = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Train
    print(f"🤖  Training model...")
    model = LinearRegression()
    model.fit(X_train_s, y_train_s)

    # Evaluate
    y_pred_s   = model.predict(X_test_s)
    y_pred_test = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    print(f"📊  Test MAPE: {mape:.1f}%")

    # Forecast
    future_preds = predict_future(model, scaler_X, scaler_y, df, FORECAST_DAYS)
    future_dates = make_future_dates(df.index[-1], FORECAST_DAYS)

    print(f"\n🔮  {ticker} — {FORECAST_DAYS}-Day Forecast:")
    last_price = df["Close"].iloc[-1]
    print(f"    Current price: ${last_price:.2f}\n")
    for d, p in zip(future_dates, future_preds):
        arrow = "▲" if p >= last_price else "▼"
        print(f"    {d.strftime('%a %b %d')}: ${p:.2f}  {arrow} ({((p-last_price)/last_price)*100:+.2f}%)")

    # Plot
    plot_results(ticker, df, split, y_test, y_pred_test, future_dates, future_preds)


if __name__ == "__main__":
    main()
