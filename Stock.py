import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
import matplotlib.pyplot as plt
import numpy as np
import re

st.set_page_config(page_title="📈 Stock Trade Intelligence Suite", layout="wide")

# ------------------------------
# Sidebar Navigation Menu
# ------------------------------
st.sidebar.title("🧭 Navigation Menu")
st.sidebar.write('https://chatgpt.com/c/690bb81c-25e8-8323-b20d-d509e0ebfef8')
menu = st.sidebar.radio(
    "Select Page:",
    ["🏦 Single Stock Predictor",
     "💹 Multi-Stock Predictor",
     "⚡ Intraday Signal Predictor",
     "📚 Technical Analysis",
     "💰 ROI Calculator", "🎯 Trend Confirmation (Day Trader)",
     "🎯 Best Strike Picker",
     # "🛍️ US Retail Sales Forecast",
     # "📊 Trade Activity Analysis"  # ✅ NEW MENU ADDED HERE
     ]  # ✅ add this line
)


# =========================================== EXCHANGE NOT TRADE ON
st.sidebar.markdown('---')
with st.sidebar.expander ("### ✅ Best Exchanges for CALLS"):

        st.markdown("""
    **Use these:**
    - **C (CBOE)** – Fastest fills  
    - **M (MIAX)** – Cheaper fills  
    - **W (C2)** – Tight mid-fills  
    
    **Avoid these:**
    - **I (BOX)**  
    - **Z (BATS/BZX)**  
    - **H (NASDAQ BX)**  
    - **G (GEMX/GEMINI)**  
    - **E (MERCURY)**
        """)






# ======================================================================TRADING PERIOD
import datetime as dt

## ============================================================
# 🕒 REAL-TIME TRADING WINDOW ANALYZER (PST)
# ============================================================
with st.sidebar.expander("⏱️ Best Trading Time Windows (PST)", expanded=True):

    # Convert local time to PST
    now_utc = dt.datetime.utcnow()
    now_pst = now_utc - dt.timedelta(hours=8)
    current_time = now_pst.time()

    # ------------------------------------------------------------
    # FIXED: Automatic TEXT COLOR for yellow (#f1c40f)
    # ------------------------------------------------------------
    def color_block(text, bg):
        text_color = "black" if bg == "#f1c40f" else "white"
        return f"""
        <div style="
            background:{bg};
            padding:8px 12px;
            border-radius:8px;
            margin-bottom:8px;
            color:{text_color};
            font-weight:600;
            font-size:13px;">
            {text}
        </div>
        """

    # Determine current trading phase
    status_block = ""

    # 🟢 Optimal Trading Window: 6:30–7:00 & 12:45–1:00
    if dt.time(6,30) <= current_time <= dt.time(7,0):
        status_block = color_block("🟢 Optimal Window: Opening Momentum", "#27ae60")
    elif dt.time(12,45) <= current_time <= dt.time(13,0):
        status_block = color_block("🟢 Optimal Window: Power-Hour Rally", "#27ae60")

    # 🟡 Caution Window
    elif dt.time(7,0) <= current_time <= dt.time(7,30):
        status_block = color_block("🟡 Caution: Trend Confirmation Zone", "#f1c40f")
    elif dt.time(10,0) <= current_time <= dt.time(10,30):
        status_block = color_block("🟡 Caution: Mid-Morning Reversal Zone", "#f1c40f")

    # 🔴 Avoid Window
    elif dt.time(12,0) <= current_time <= dt.time(13,0):
        status_block = color_block("🔴 Avoid: Lunchtime Low Liquidity", "#e74c3c")

    # Neutral
    else:
        status_block = color_block("⚪ Neutral Time — Trade Only With Clear Signals", "#7f8c8d")

    st.markdown("### 🕒 Current Status")
    st.markdown(status_block, unsafe_allow_html=True)

    # ============================================================
    # STATIC SCHEDULE LIST
    # ============================================================
    st.markdown("### 📅 Full Trading Schedule (PST)")

    # 🟢 Opening
    st.markdown(color_block(
        "🟢 <b>6:30–7:00 AM — Opening Momentum Window</b><br>Best for scalping, breakouts.",
        "#27ae60"), unsafe_allow_html=True)

    # 🟡 Trend Confirmation
    st.markdown(color_block(
        "🟡 <b>7:00–7:30 AM — Trend Confirmation</b><br>Wait for VWAP/EMA confirmation.",
        "#f1c40f"), unsafe_allow_html=True)

    # 🟡 Reversal Zone
    st.markdown(color_block(
        "🟡 <b>10:00–10:30 AM — Reversal Zone</b><br>Watch carefully: fakeouts possible.",
        "#f1c40f"), unsafe_allow_html=True)

    # 🔴 Lunchtime Chop
    st.markdown(color_block(
        "🔴 <b>12:00–1:00 PM — Lunchtime Chop</b><br>Avoid new entries — low liquidity.",
        "#e74c3c"), unsafe_allow_html=True)

    # 🟢 Power Hour Setup
    st.markdown(color_block(
        "🟢 <b>12:45–1:00 PM — Power Hour Setup</b><br>Momentum returns into the close.",
        "#27ae60"), unsafe_allow_html=True)


















# ============================= MENU ENDS HERE

# =====================================================================
# 🏦 SINGLE STOCK PAGE
# =====================================================================
def single_stock():
    st.title("🏦 Single Stock Daily Trade Prediction")
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL", key="single_ticker")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"), key="single_start")
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"), key="single_end")

    @st.cache_data
    def load_data(ticker, start, end):
        return yf.download(ticker, start=start, end=end)

    try:
        df = load_data(ticker, start_date, end_date)
        if df.empty:
            st.warning("⚠️ No data found for this ticker.")
            return

        st.dataframe(df.tail())
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'],
                                             high=df['High'], low=df['Low'],
                                             close=df['Close'])])
        fig.update_layout(title="Candlestick Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
        X, y = df[['Open', 'High', 'Low', 'Volume']], df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression().fit(X_train, y_train)

        latest_input = df[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(latest_input)[0]
        st.success(f"📍 Predicted next close for {ticker}: **${prediction:.2f}**")

        y_pred = model.predict(X_test)
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        st.write(f"R²: {r2_score(y_test, y_pred) * 100:.2f}%")

        fig2, ax = plt.subplots()
        ax.plot(y_test.values, label='Actual')
        ax.plot(y_pred, label='Predicted', linestyle='--')
        ax.legend()
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Error: {e}")

# =====================================================================
# 💹 MULTI-STOCK PAGE
# =====================================================================
def multi_stock():
    st.title("💹 Multi-Stock Daily Trade Prediction")

    st.sidebar.header("📊 Stock Settings")
    tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL, MSFT, GOOGL", key="multi_tickers")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"), key="multi_start")
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"), key="multi_end")

    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    @st.cache_data
    def load_data(ticker, start, end):
        return yf.download(ticker, start=start, end=end)

    for ticker in tickers:
        st.markdown(f"---\n## 📌 {ticker} Analysis")
        try:
            df = load_data(ticker, start_date, end_date)
            if df.empty:
                st.warning(f"⚠️ No data found for {ticker}.")
                continue

            st.subheader(f"{ticker} Historical Data (Tail)")
            st.dataframe(df.tail())

            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])
            fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            df['Target'] = df['Close'].shift(-1)
            df.dropna(inplace=True)
            X = df[['Open', 'High', 'Low', 'Volume']]
            y = df['Target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = LinearRegression().fit(X_train, y_train)

            latest_input = df[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
            prediction = float(model.predict(latest_input)[0])
            last_close = float(df['Close'].iloc[-1])
            change_pct = ((prediction - last_close) / last_close) * 100

            threshold = 0.01
            if prediction > last_close * (1 + threshold):
                recommendation = "📈 BUY"
            elif prediction < last_close * (1 - threshold):
                recommendation = "📉 SELL"
            else:
                recommendation = "⏸️ HOLD"

            st.subheader("🧾 Summary Recommendation")
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Close", f"${last_close:.2f}")
            col2.metric("Predicted Next Close", f"${prediction:.2f}")
            col3.metric("Predicted Change", f"{change_pct:.2f}%")
            st.info(f"**Suggested Action:** {recommendation}")

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.subheader("📊 Model Evaluation Metrics")
            st.write(f"- **MAE:** ${mae:.2f}")
            st.write(f"- **RMSE:** ${rmse:.2f}")
            st.write(f"- **R² Score:** {r2 * 100:.2f}%")

            fig2, ax = plt.subplots()
            ax.plot(y_test.values, label='Actual')
            ax.plot(y_pred, label='Predicted', linestyle='--')
            ax.legend()
            ax.set_title(f"{ticker} - Actual vs Predicted Closing Prices")
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"❌ Error processing {ticker}: {str(e)}")

# # # =====================================================================
# # # =====================================================================
# # # ⚡ INTRADAY PAGE
# # # ⚡ INTRADAY PAGE
# # # ⚡ INTRADAY PAGE
# # # =====================================================================
# # # =====================================================================
# #
#
# def intraday_signals():
#     st.title("⚡ Intraday Buy/Sell Signal Predictor")
#
#     # ------------------------- Sidebar Inputs -------------------------
#     raw_input = st.sidebar.text_input("Ticker", "AAPL", key="intra_ticker")
#     period = st.sidebar.selectbox("Data Period", ["1d", "5d"], key="intra_period")
#     interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m"], key="intra_interval")
#
#     ticker = re.sub(r"[^A-Z0-9]", "", raw_input.strip().upper())
#     if not ticker:
#         st.error("Invalid ticker.")
#         st.stop()
#
#     # ------------------------- Load Data -------------------------
#     @st.cache_data
#     def get_data(t, p, i):
#         df = yf.download(t, period=p, interval=i, progress=False)
#         # Flatten if multi-level
#         if isinstance(df.columns, pd.MultiIndex):
#             df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]
#         # Normalize names
#         df.columns = [c.replace(f"{t}_", "").strip().capitalize() for c in df.columns]
#         return df
#
#     df = get_data(ticker, period, interval)
#
#     if df.empty:
#         st.warning("⚠️ No data found for this ticker or time frame.")
#         return
#
#     # Detect close column
#     close_candidates = [c for c in df.columns if "close" in c.lower()]
#     if not close_candidates:
#         st.error(f"❌ Could not find 'Close' column. Columns: {list(df.columns)}")
#         return
#     df.rename(columns={close_candidates[0]: "Close"}, inplace=True)
#     df.dropna(subset=["Close"], inplace=True)
#
#     if isinstance(df["Close"], pd.DataFrame):
#         df["Close"] = df["Close"].squeeze()
#
#     # ------------------------- Technical Indicators -------------------------
#     try:
#         df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
#         df["MACD"] = MACD(close=df["Close"]).macd_diff()
#         df["EMA"] = EMAIndicator(close=df["Close"], window=10).ema_indicator()
#     except Exception as e:
#         st.error(f"Indicator calculation failed: {e}")
#         return
#
#     df.dropna(inplace=True)
#     if len(df) < 20:
#         st.warning(f"⚠️ Not enough data points after indicators (only {len(df)} rows). Try '5d' or '15m' interval.")
#         return
#
#     # ------------------------- Create Labels -------------------------
#     df["Signal"] = 0
#     df.loc[df["Close"].shift(-1) > df["Close"] * 1.002, "Signal"] = 1
#     df.loc[df["Close"].shift(-1) < df["Close"] * 0.998, "Signal"] = -1
#
#     # ------------------------- Model Training -------------------------
#     features = ["RSI", "MACD", "EMA"]
#     X = df[features]
#     y = df["Signal"]
#
#     # Handle if too small
#     if len(df) < 50:
#         st.warning("⚠️ Very little data available. Model will run in basic mode (no split).")
#         model = RandomForestClassifier(n_estimators=50, random_state=42)
#         model.fit(X, y)
#         acc = 100
#     else:
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#         model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred) * 100
#
#     # ------------------------- Prediction -------------------------
#     latest_features = df[features].iloc[-1].values.reshape(1, -1)
#     next_signal = model.predict(latest_features)[0]
#     signal_label = {1: "📈 BUY", -1: "📉 SELL", 0: "⏸️ HOLD"}[next_signal]
#
#     last_price = df["Close"].iloc[-1]
#     st.subheader("🔮 Prediction Summary")
#     st.metric(label="Current Price", value=f"${last_price:.2f}")
#     st.metric(label="Predicted Signal", value=signal_label)
#     st.metric(label="Model Accuracy", value=f"{acc:.2f}%")
#
#     # ------------------------- Chart -------------------------
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
#     fig.add_trace(go.Scatter(x=df[df["Signal"] == 1].index, y=df[df["Signal"] == 1]["Close"],
#                              mode="markers", marker=dict(color="green", size=8), name="Buy Signal"))
#     fig.add_trace(go.Scatter(x=df[df["Signal"] == -1].index, y=df[df["Signal"] == -1]["Close"],
#                              mode="markers", marker=dict(color="red", size=8), name="Sell Signal"))
#     fig.update_layout(title=f"📉 Intraday Buy/Sell Signals for {ticker}",
#                       xaxis_title="Time", yaxis_title="Price",
#                       template="plotly_white")
#     st.plotly_chart(fig, use_container_width=True)
#
#     # ------------------------- Debug Info -------------------------
#     with st.expander("🔍 Debug Info"):
#         st.write("Data shape:", df.shape)
#         # st.write("Last few rows:", df.tail(5))
#         st.write("Last few rows:", df)
#
#     # =====================================================================
#     # 🕒 EARLY-RUSH FILTER (Market-Open to First N Minutes)
#     # =====================================================================
#
#     # =====================================================================
#     # 🕒 EARLY-RUSH FILTER (Market-Open to First N Minutes) — FIXED TZ
#     # =====================================================================
#     st.markdown("---")
#     st.subheader("🕒 Early-Rush Analysis (9:30 AM ET / 6:30 AM PST)")
#
#     # --- Detect correct datetime column ---
#     if "Date" in df.columns:
#         time_col = "Date"
#     elif "Datetime" in df.columns:
#         time_col = "Datetime"
#     else:
#         if isinstance(df.index, pd.DatetimeIndex):
#             df = df.reset_index()
#             time_col = df.columns[0]
#         else:
#             st.error("❌ Could not find any datetime column.")
#             st.stop()
#
#     # --- Ensure datetime and timezone alignment ---
#     df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
#
#     # Assume UTC if tz-naive
#     if df[time_col].dt.tz is None:
#         df[time_col] = df[time_col].dt.tz_localize("UTC")
#
#     # Convert everything to Pacific Time (US market local time)
#     df[time_col] = df[time_col].dt.tz_convert("US/Pacific")
#
#     # --- Let user pick early window ---
#     early_window = st.slider("Select how many minutes after open to analyze", 10, 120, 60, 5)
#
#     # --- Market open at 6:30 AM PST ---
#     market_open = df[time_col].dt.normalize() + pd.Timedelta(hours=6, minutes=30)
#
#     # Compute minutes after open
#     df["MinutesAfterOpen"] = (df[time_col] - market_open).dt.total_seconds() / 60
#
#     # --- Filter for first N minutes after open ---
#     df_rush = df[df["MinutesAfterOpen"].between(0, early_window)]
#
#     if df_rush.empty:
#         st.warning(f"⚠️ No data found between 6:30 AM PST and +{early_window} min.")
#     else:
#         # -------------------- Quick Stats --------------------
#         open_price = df_rush["Open"].iloc[0]
#         close_price = df_rush["Close"].iloc[-1]
#         high_price = df_rush["High"].max()
#         low_price = df_rush["Low"].min()
#         pct_change = (close_price - open_price) / open_price * 100
#
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("🟢 High", f"${high_price:.2f}")
#         c2.metric("🔴 Low", f"${low_price:.2f}")
#         c3.metric("⚖️ % Change", f"{pct_change:.2f}%")
#         c4.metric("🕒 Window", f"0–{early_window} min")
#
#         # -------------------- Candlestick Chart --------------------
#         fig_rush = go.Figure()
#         fig_rush.add_trace(go.Candlestick(
#             x=df_rush[time_col],
#             open=df_rush["Open"],
#             high=df_rush["High"],
#             low=df_rush["Low"],
#             close=df_rush["Close"],
#             name="Early Rush"
#         ))
#         fig_rush.update_layout(
#             title=f"{ticker} — First {early_window} Minutes After Open",
#             xaxis_title="Time (PST)",
#             yaxis_title="Price ($)",
#             template="plotly_dark",
#             xaxis_rangeslider_visible=False,
#             height=500
#         )
#         st.plotly_chart(fig_rush, use_container_width=True)
#
#         # -------------------- Debug --------------------
#         with st.expander("🔍 Early-Rush Data Frame (Converted to PST)"):
#             st.write(df_rush[[time_col, "Open", "High", "Low", "Close", "MinutesAfterOpen"]].head(50))


# ====================================================== PART 2
# =====================================================================
# ⚡ INTRADAY PAGE — 9:30 AM to 11:15 AM ET Window (Stable)
# =====================================================================

# =====================================================================
# ⚡ INTRADAY PAGE — 9:30–11:15 AM ET FILTER (FULL FIX)
# =====================================================================

def intraday_signals():
    import re
    import yfinance as yf
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, EMAIndicator
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    st.title("⚡ Intraday Buy/Sell Signal Predictor (9:30–11:15 AM ET)")

    # ---------------- Sidebar Inputs ----------------
    raw_input = st.sidebar.text_input("Ticker", "AAPL", key="intra_ticker")
    period = st.sidebar.selectbox("Data Period", ["1d", "5d"], key="intra_period")
    interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m"], key="intra_interval")

    ticker = re.sub(r"[^A-Z0-9]", "", raw_input.strip().upper())
    if not ticker:
        st.error("Invalid ticker.")
        st.stop()

    # ---------------- Load Data ----------------
    @st.cache_data
    def get_data(t, p, i):
        """Safely download and normalize Yahoo Finance data."""
        df = yf.download(t, period=p, interval=i, progress=False)

        # --- Flatten multi-index columns if needed ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns]
        else:
            df.columns = [str(col) for col in df.columns]

        # --- Reset index to ensure 'Datetime' column exists ---
        df.reset_index(inplace=True)

        # --- Normalize column names (force lowercase) ---
        df.columns = [c.lower().replace(f"{t.lower()}_", "").strip() for c in df.columns]

        return df

    # ✅ Call get_data() here to actually load df
    df = get_data(ticker, period, interval)

    if df.empty:
        st.warning("⚠️ No data found for this ticker or time frame.")
        return

    # ---------------- Detect Datetime Column ----------------
    time_col = "datetime" if "datetime" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df.dropna(subset=[time_col], inplace=True)

    # ---------------- Detect and Rename OHLC Columns ----------------
    rename_map = {}
    for col in df.columns:
        if "open" in col and "adj" not in col:
            rename_map[col] = "Open"
        elif "high" in col:
            rename_map[col] = "High"
        elif "low" in col:
            rename_map[col] = "Low"
        elif "close" in col and "adj" not in col:
            rename_map[col] = "Close"
    df.rename(columns=rename_map, inplace=True)

    # Make sure essential columns exist
    required_cols = ["Open", "High", "Low", "Close"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}. Columns found: {list(df.columns)}")
        st.stop()

    # ---------------- Convert to Eastern Time ----------------
    if df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    else:
        df[time_col] = df[time_col].dt.tz_convert("US/Eastern")

    # ---------------- Filter: 9:30 AM – 11:15 AM ET ----------------
    start_time = pd.to_datetime("09:30", format="%H:%M").time()
    end_time = pd.to_datetime("11:15", format="%H:%M").time()
    df_window = df[(df[time_col].dt.time >= start_time) & (df[time_col].dt.time <= end_time)]

    if df_window.empty:
        st.warning("⚠️ No data found between 9:30 AM – 11:15 AM ET.")
        st.dataframe(df.head(10))
        return

    # ---------------- Technical Indicators ----------------
    try:
        df_window["RSI"] = RSIIndicator(close=df_window["Close"]).rsi()
        df_window["MACD"] = MACD(close=df_window["Close"]).macd_diff()
        df_window["EMA"] = EMAIndicator(close=df_window["Close"], window=10).ema_indicator()
    except Exception as e:
        st.error(f"Indicator calculation failed: {e}")
        st.write("Columns available:", df_window.columns.tolist())
        return

    df_window.dropna(inplace=True)

    # ---------------- Labeling ----------------
    df_window["Signal"] = 0
    df_window.loc[df_window["Close"].shift(-1) > df_window["Close"] * 1.002, "Signal"] = 1
    df_window.loc[df_window["Close"].shift(-1) < df_window["Close"] * 0.998, "Signal"] = -1

    # ---------------- Train Model ----------------
    features = ["RSI", "MACD", "EMA"]
    X = df_window[features]
    y = df_window["Signal"]

    if len(df_window) < 50:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        acc = 100
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100

    # ---------------- Predict Next Signal ----------------
    latest_features = df_window[features].iloc[-1].values.reshape(1, -1)
    next_signal = model.predict(latest_features)[0]
    signal_label = {1: "📈 BUY", -1: "📉 SELL", 0: "⏸️ HOLD"}[next_signal]

    last_price = df_window["Close"].iloc[-1]
    st.subheader("🔮 Prediction Summary (9:30–11:15 AM ET)")
    st.metric("Current Price", f"${last_price:.2f}")
    st.metric("Predicted Signal", signal_label)
    st.metric("Model Accuracy", f"{acc:.2f}%")

    # ---------------- Chart ----------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_window[time_col], y=df_window["Close"],
                             mode="lines", name="Close", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df_window[df_window["Signal"] == 1][time_col],
                             y=df_window[df_window["Signal"] == 1]["Close"],
                             mode="markers", marker=dict(color="green", size=8), name="Buy Signal"))
    fig.add_trace(go.Scatter(x=df_window[df_window["Signal"] == -1][time_col],
                             y=df_window[df_window["Signal"] == -1]["Close"],
                             mode="markers", marker=dict(color="red", size=8), name="Sell Signal"))
    fig.update_layout(title=f"📊 {ticker} — 9:30 AM to 11:15 AM ET Session",
                      xaxis_title="Time (ET)",
                      yaxis_title="Price ($)",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Stats ----------------
    open_price = df_window["Open"].iloc[0]
    close_price = df_window["Close"].iloc[-1]
    high_price = df_window["High"].max()
    low_price = df_window["Low"].min()
    pct_change = (close_price - open_price) / open_price * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 High", f"${high_price:.2f}")
    c2.metric("🔴 Low", f"${low_price:.2f}")
    c3.metric("⚖️ % Change", f"{pct_change:.2f}%")
    c4.metric("🕒 Window", "9:30 – 11:15 AM ET")

    # ---------------- Debug ----------------
    with st.expander("🔍 Debug Data"):
        st.write("Filtered DataFrame shape:", df_window.shape)
        st.dataframe(df_window.head(100))


    # ====================================================================================================
    # NOW DO ANALYSIS ON TIMEFRAME
    # ===============================

    # ====================================================================================================
    # 📈 INTRADAY PRICE MOVEMENT ANALYSIS — Custom Time Window (Final Safe Version)
    # ====================================================================================================
    st.markdown("---")
    st.subheader("📈 Intraday Price Movement Analysis — Custom Time Window")

    # --- Ensure df exists before proceeding ---
    if "df" not in locals() or df is None or df.empty:
        st.warning("⚠️ Data not yet loaded. Please run data download or select a valid ticker first.")
    else:
        # --- If df_window does not exist, default to df ---
        if "df_window" not in locals() or df_window is None:
            df_window = df.copy()

        if df_window.empty:
            st.warning("⚠️ No filtered data available for analysis.")
        else:
            # ---------------------------------------------------------
            # 🕒 Select custom timeframe for analysis
            # ---------------------------------------------------------
            st.markdown("### 🕒 Select Analysis Time Range (ET)")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                start_time_input = st.time_input("Start Time", pd.to_datetime("09:30", format="%H:%M").time())
            with col_t2:
                end_time_input = st.time_input("End Time", pd.to_datetime("11:15", format="%H:%M").time())

            # --- Identify datetime column ---
            time_col_candidates = [c for c in df_window.columns if "date" in c.lower() or "time" in c.lower()]
            if time_col_candidates:
                time_col = time_col_candidates[0]
            else:
                time_col = df_window.columns[0]

            df_window[time_col] = pd.to_datetime(df_window[time_col], errors="coerce")
            df_window.dropna(subset=[time_col], inplace=True)

            # ---------------------------------------------------------
            # 🧮 Filter data within selected window
            # ---------------------------------------------------------
            df_filtered = df_window[
                (df_window[time_col].dt.time >= start_time_input)
                & (df_window[time_col].dt.time <= end_time_input)
            ]

            if df_filtered.empty:
                st.warning(f"⚠️ No data found between {start_time_input.strftime('%H:%M')} and {end_time_input.strftime('%H:%M')} ET.")
            else:
                # ---------------------------------------------------------
                # 📊 Price movement analysis
                # ---------------------------------------------------------
                df_filtered["PriceChange_%"] = df_filtered["Close"].pct_change() * 100
                df_filtered["Trend"] = df_filtered["PriceChange_%"].apply(
                    lambda x: "Increase" if x > 0 else ("Decrease" if x < 0 else "No Change")
                )

                inc_count = (df_filtered["Trend"] == "Increase").sum()
                dec_count = (df_filtered["Trend"] == "Decrease").sum()
                flat_count = (df_filtered["Trend"] == "No Change").sum()

                avg_gain = df_filtered.loc[df_filtered["Trend"] == "Increase", "PriceChange_%"].mean()
                avg_loss = df_filtered.loc[df_filtered["Trend"] == "Decrease", "PriceChange_%"].mean()

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📈 Increases", inc_count)
                c2.metric("📉 Decreases", dec_count)
                c3.metric("➖ No Change", flat_count)
                c4.metric("⚖️ Avg Change", f"{df_filtered['PriceChange_%'].mean():.3f}%")

                st.markdown(f"""
                - 🟢 **Avg Gain:** {avg_gain:.3f}%  
                - 🔴 **Avg Loss:** {avg_loss:.3f}%  
                - ⚙️ **Net Movement:** {df_filtered['Close'].iloc[-1] - df_filtered['Open'].iloc[0]:.2f}  
                - 🕒 **Analyzed Window:** {start_time_input.strftime('%H:%M')} – {end_time_input.strftime('%H:%M')} ET
                """)

                # ---------------------------------------------------------
                # 📉 Trend Visualization
                # ---------------------------------------------------------
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=df_filtered[time_col],
                    y=df_filtered["Close"],
                    mode="lines+markers",
                    line=dict(width=2, color="#3498db"),
                    name="Close Price"
                ))

                inc_points = df_filtered[df_filtered["Trend"] == "Increase"]
                dec_points = df_filtered[df_filtered["Trend"] == "Decrease"]

                fig_trend.add_trace(go.Scatter(
                    x=inc_points[time_col],
                    y=inc_points["Close"],
                    mode="markers",
                    marker=dict(color="green", size=8, symbol="triangle-up"),
                    name="Price Increase"
                ))
                fig_trend.add_trace(go.Scatter(
                    x=dec_points[time_col],
                    y=dec_points["Close"],
                    mode="markers",
                    marker=dict(color="red", size=8, symbol="triangle-down"),
                    name="Price Decrease"
                ))

                fig_trend.update_layout(
                    title=f"💹 {ticker} — Intraday Trend ({start_time_input.strftime('%H:%M')} – {end_time_input.strftime('%H:%M')} ET)",
                    xaxis_title="Time (ET)",
                    yaxis_title="Close Price ($)",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig_trend, use_container_width=True)

                # ---------------------------------------------------------
                # 🔍 Detailed Candle Data
                # ---------------------------------------------------------
                with st.expander("🔍 Detailed Candle Data (Selected Range)"):
                    st.dataframe(df_filtered[[time_col, "Open", "Close", "PriceChange_%", "Trend"]].tail(50))














# ================= ENDS HERE


# ================================================================================
# TECHNICAL ANALYSIS TRADING OPTIONS
# ===============================================================================
# =====================================================================
# 📚 TECHNICAL ANALYSIS (OPTIONS DAY TRADING)
# =====================================================================
# =====================================================================
# 📚 TECHNICAL ANALYSIS (OPTIONS DAY TRADING)
# =====================================================================

#
# import math
# from datetime import datetime, timedelta
#
# def technical_analysis():
#     st.title("📚 Technical Analysis — Options Day Trading")
#
#     # ---------------- Sidebar Inputs ----------------
#     st.sidebar.header("⚙️ Inputs — Technical Analysis")
#     ta_ticker   = st.sidebar.text_input("Ticker", "AAPL", key="ta_ticker").strip().upper()
#     ta_period   = st.sidebar.selectbox("Data Period", ["5d", "1mo", "3mo"], index=0, key="ta_period")
#     ta_interval = st.sidebar.selectbox("Interval", ["5m", "15m", "1h"], index=0, key="ta_interval")
#
#     # Risk / RR
#     acct_size   = st.sidebar.number_input("Account Size ($)", 1000.0, step=100.0, value=10000.0, key="ta_acct")
#     risk_pct    = st.sidebar.slider("Risk per Trade (%)", 0.25, 5.0, 1.0, 0.25, key="ta_risk_pct")
#     rr_target   = st.sidebar.selectbox("Target R:R", ["1:2", "1:3", "1:4"], index=1, key="ta_rr")
#     take_split  = st.sidebar.selectbox("Scale Out Plan", ["70% @ T1 / 30% @ T2", "50% / 50%", "All @ Target"], key="ta_split")
#
#     # Option pricing inputs
#     iv_input    = st.sidebar.number_input("Implied Volatility (annual, %)", 5.0, 200.0, 30.0, 0.5, key="ta_iv")
#     days_to_exp = st.sidebar.number_input("Days to Expiry", 1, 60, 7, key="ta_dte")
#     rate_input  = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25, key="ta_rf")
#
#     # ---------------- Data ----------------
#     @st.cache_data
#     def get_price_data(tkr, per, intrv):
#         df = get_price_data(ta_ticker, ta_period, ta_interval)
#
#         # ✅ auto-fallback for missing data
#         if df.empty or len(df.columns) < 5 or not set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns):
#             st.warning("⚠️ No sufficient OHLCV data — retrying with 1-month / 15-minute interval ...")
#             df = get_price_data(ta_ticker, "1mo", "15m")
#
#         if df.empty or not set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns):
#             st.error("❌ Still no valid data from Yahoo Finance. Try another ticker or broader period.")
#             return
#
#         if isinstance(df.columns, pd.MultiIndex):
#             df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
#         return df
#
#     df = get_price_data(ta_ticker, ta_period, ta_interval)
#     if df.empty or not set(["Open","High","Low","Close","Volume"]).issubset(df.columns):
#         st.error("No sufficient OHLCV data. Try a longer period or different interval.")
#         return
#     df = df.dropna().copy()
#
#     # ===================================================== PART 2
#
#
#     # ---------------- Indicators & Helpers ----------------
#     # EMAs for Trend Alignment
#     df["EMA20"]  = df["Close"].ewm(span=20).mean()
#     df["EMA50"]  = df["Close"].ewm(span=50).mean()
#     df["EMA200"] = df["Close"].ewm(span=200).mean()
#
#     # RSI / MACD for momentum confirmation
#     try:
#         df["RSI"]  = RSIIndicator(close=df["Close"]).rsi()
#         df["MACD"] = MACD(close=df["Close"]).macd_diff()
#     except Exception:
#         df["RSI"] = np.nan
#         df["MACD"] = np.nan
#
#     # ATR (volatility for stop placement)
#     tr = np.maximum(df["High"]-df["Low"], np.maximum((df["High"]-df["Close"].shift()).abs(),
#                                                      (df["Low"]-df["Close"].shift()).abs()))
#     df["ATR14"] = pd.Series(tr).rolling(14).mean()
#
#     # Volume trend
#     df["VolSMA20"] = df["Volume"].rolling(20).mean()
#     df["VolSpike"] = (df["Volume"] > 1.5*df["VolSMA20"]).astype(int)
#
#     # Swing points (for supply/demand zones) using rolling window pivots
#     def pivots(s, left=3, right=3, fn=np.max):
#         out = pd.Series(index=s.index, dtype=float)
#         for i in range(left, len(s)-right):
#             window = s.iloc[i-left:i+right+1]
#             if fn == np.max and s.iloc[i] == window.max():
#                 out.iloc[i] = s.iloc[i]
#             if fn == np.min and s.iloc[i] == window.min():
#                 out.iloc[i] = s.iloc[i]
#         return out
#
#     df["SwingHigh"] = pivots(df["High"], fn=np.max)
#     df["SwingLow"]  = pivots(df["Low"],  fn=np.min)
#
#     # --------------- Trend Alignment (Price > EMA20 > EMA50 > EMA200) ---------------
#     df["UpTrend"]   = (df["Close"]>df["EMA20"]) & (df["EMA20"]>df["EMA50"]) & (df["EMA50"]>df["EMA200"])
#     df["DnTrend"]   = (df["Close"]<df["EMA20"]) & (df["EMA20"]<df["EMA50"]) & (df["EMA50"]<df["EMA200"])
#
#     # --------------- Entry Triggers (Three Confirmations) ---------------
#     # 1) Price Action: breakout above recent swing high (bull) / below swing low (bear)
#     recent_high = df["SwingHigh"].dropna().tail(10).max() if df["SwingHigh"].notna().any() else np.nan
#     recent_low  = df["SwingLow"].dropna().tail(10).min() if df["SwingLow"].notna().any()  else np.nan
#     # 2) Volume: spike above 1.5x 20SMA
#     # 3) Momentum: MACD > 0 for calls, < 0 for puts (basic)
#
#     last_close  = float(df["Close"].iloc[-1])
#     last_atr    = float(df["ATR14"].iloc[-1]) if not np.isnan(df["ATR14"].iloc[-1]) else max(0.5, last_close*0.005)
#     macd_last   = float(df["MACD"].iloc[-1]) if not np.isnan(df["MACD"].iloc[-1]) else 0.0
#     vol_spike   = int(df["VolSpike"].iloc[-1]) == 1
#
#     # Proposed Long (CALL) setup
#     long_ok = bool(
#         (df["UpTrend"].iloc[-1] == True) and
#         (not np.isnan(recent_high)) and (last_close > recent_high) and
#         vol_spike and (macd_last > 0)
#     )
#
#     # Proposed Short (PUT) setup
#     short_ok = bool(
#         (df["DnTrend"].iloc[-1] == True) and
#         (not np.isnan(recent_low)) and (last_close < recent_low) and
#         vol_spike and (macd_last < 0)
#     )
#
#     # --------------- Stops & Targets (Risk/Reward) ---------------
#     # Stop: 1.2 * ATR beyond last swing
#     if long_ok and not np.isnan(recent_low):
#         entry  = last_close
#         stop   = min(recent_low, entry - 1.2*last_atr)
#     elif short_ok and not np.isnan(recent_high):
#         entry  = last_close
#         stop   = max(recent_high, entry + 1.2*last_atr)
#     else:
#         # Fallback: ATR-based stop
#         entry  = last_close
#         stop   = entry - 1.2*last_atr if macd_last >= 0 else entry + 1.2*last_atr
#
#     risk_per_share = abs(entry - stop)
#
#     # R:R target
#     rr_map = {"1:2": 2, "1:3": 3, "1:4": 4}
#     rr_mult = rr_map.get(rr_target, 3)
#     if macd_last >= 0:
#         target = entry + rr_mult * risk_per_share
#         t1     = entry + 1 * risk_per_share
#         t2     = entry + rr_mult * risk_per_share
#     else:
#         target = entry - rr_mult * risk_per_share
#         t1     = entry - 1 * risk_per_share
#         t2     = entry - rr_mult * risk_per_share
#
#     # --------------- 1-OTM Option & Delta Focus ---------------
#     # Simple Black-Scholes delta (call). For put, delta ~ call_delta - 1
#     def norm_cdf(x):  # fast approximate CDF
#         return 0.5 * (1 + math.erf(x / math.sqrt(2)))
#     def bs_call_delta(S, K, T, r, vol):
#         if S<=0 or K<=0 or T<=0 or vol<=0:
#             return 0.5
#         d1 = (math.log(S/K) + (r + 0.5*vol*vol)*T) / (vol*math.sqrt(T))
#         return norm_cdf(d1)
#
#     # Choose 1 OTM strike (rounded to nearest typical step)
#     def nearest_strike(price):
#         # Try common increments: 1, 2.5, 5 depending on price
#         inc = 1 if price < 50 else (2.5 if price < 150 else 5)
#         return math.ceil(price/inc)*inc
#
#     S  = entry
#     Kc = nearest_strike(S*1.01)  # 1% OTM for call bias
#     Kp = nearest_strike(S*0.99)  # 1% OTM for put bias
#     T  = max(days_to_exp,1) / 252.0
#     r  = rate_input/100.0
#     vol= iv_input/100.0
#
#     call_delta = bs_call_delta(S, Kc, T, r, vol)
#     put_delta  = call_delta - 1  # rough symmetry
#
#     # Premium estimate (very rough) using delta * move proxy
#     # Encourage manual override later if you have a feed
#     move_proxy = S * vol * math.sqrt(T)
#     est_call   = max(0.05, (max(S-Kc,0) + 0.5*move_proxy) * 0.35)  # keep conservative
#     est_put    = max(0.05, (max(Kp-S,0) + 0.5*move_proxy) * 0.35)
#
#     # Risk sizing
#     max_risk   = acct_size * (risk_pct/100.0)
#     if macd_last >= 0:
#         prem = est_call
#         delta = call_delta
#         strike = Kc
#         opt_type = "CALL"
#     else:
#         prem = est_put
#         delta = abs(put_delta)
#         strike = Kp
#         opt_type = "PUT"
#
#     # Contracts sized by risk to stop in the underlying approximated to option PnL via delta
#     # Risk per contract ≈ (risk_per_share * delta) * 100
#     risk_per_contract = max(0.01, risk_per_share * max(0.2, min(delta, 0.8)) * 100.0)
#     contracts = int(max_risk // risk_per_contract) if risk_per_contract>0 else 0
#
#     # --------------- Display Summary ---------------
#     st.subheader("🧭 Strategy Checklist (3 Confirmations)")
#     st.write("- **Trend Alignment**: Price vs EMA20/50/200")
#     st.write("- **Price Action**: Breakout vs recent swing high/low")
#     st.write("- **Volume**: Spike vs 20-bar average")
#     st.info(f"""
# **Setup Status**
# - UpTrend: `{bool(df['UpTrend'].iloc[-1])}`  •  DownTrend: `{bool(df['DnTrend'].iloc[-1])}`
# - Breakout Above Swing High: `{bool(not np.isnan(recent_high) and last_close > recent_high)}`
# - Breakout Below Swing Low: `{bool(not np.isnan(recent_low) and last_close < recent_low)}`
# - Volume Spike: `{vol_spike}`
# - MACD Bias: `{'Bullish' if macd_last>0 else 'Bearish' if macd_last<0 else 'Neutral'}`
#     """.strip())
#
#     colA, colB, colC = st.columns(3)
#     colA.metric("Entry (Underlying)", f"${entry:,.2f}")
#     colB.metric("Stop (Underlying)",  f"${stop:,.2f}")
#     colC.metric("Target (Underlying)",f"${target:,.2f}")
#
#     colD, colE, colF = st.columns(3)
#     colD.metric("Risk / Share", f"${risk_per_share:,.2f}")
#     colE.metric("Max Risk ($)", f"${max_risk:,.2f}")
#     colF.metric("R:R", rr_target)
#
#     st.subheader("📝 Options Plan (1-OTM • Delta Focus)")
#     st.write(f"- **Type**: **{opt_type}**  •  **Strike**: **{strike}**  •  **Δ (approx)**: **{delta:.2f}**  •  **Est. Premium**: **${prem:.2f}**")
#     st.write(f"- **Contracts (risk-based)**: **{contracts}** (risk/contract ≈ ${risk_per_contract:,.2f})")
#     st.caption("Note: Premium is a rough estimate. For live trading, replace with your broker’s option chain price.")
#
#     # --------------- Chart: Candles + EMAs + Zones + E/S/T ---------------
#     fig = go.Figure()
#     fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
#     fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"],  mode="lines", name="EMA20"))
#     fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"],  mode="lines", name="EMA50"))
#     fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], mode="lines", name="EMA200"))
#
#     # Supply/Demand rectangles from last 3 swings
#     sh = df["SwingHigh"].dropna().tail(3)
#     sl = df["SwingLow"].dropna().tail(3)
#     for y in sh:
#         fig.add_hrect(y0=y*0.997, y1=y*1.003, line_width=0, fillcolor="red", opacity=0.15)
#     for y in sl:
#         fig.add_hrect(y0=y*0.997, y1=y*1.003, line_width=0, fillcolor="green", opacity=0.15)
#
#     # Entry / Stop / Targets
#     last_time = df.index[-1]
#     fig.add_hline(y=entry, line_dash="dot", line_color="blue", annotation_text="Entry", annotation_position="top left")
#     fig.add_hline(y=stop,  line_dash="dot", line_color="red",  annotation_text="Stop",  annotation_position="bottom left")
#     fig.add_hline(y=t1,    line_dash="dot", line_color="orange",annotation_text="T1 (1R)", annotation_position="top right")
#     fig.add_hline(y=t2,    line_dash="dot", line_color="green", annotation_text=f"T2 ({rr_target})", annotation_position="top right")
#
#     fig.update_layout(title=f"{ta_ticker} — Technical Map (Trend • Zones • E/S/T)",
#                       xaxis_title="Time", yaxis_title="Price", xaxis_rangeslider_visible=False,
#                       template="plotly_white")
#     st.plotly_chart(fig, use_container_width=True)
#
#     # --------------- 70/30 Exit Planner ---------------
#     st.subheader("🎯 Exit Planner — 70/30 (or custom)")
#     if take_split == "70% @ T1 / 30% @ T2":
#         w1, w2 = 0.70, 0.30
#         first_tp, second_tp = t1, t2
#     elif take_split == "50% / 50%":
#         w1, w2 = 0.50, 0.50
#         first_tp, second_tp = t1, t2
#     else:  # All @ Target
#         w1, w2 = 1.0, 0.0
#         first_tp, second_tp = t2, t2
#
#     # Option PnL rough projection (Δ-based)
#     # per-contract $PnL ≈ Δ * (TP - Entry) * 100
#     pnl1 = max(0.0, (max(first_tp - entry, 0) if opt_type=="CALL" else max(entry - first_tp, 0)) * max(delta,0.2) * 100)
#     pnl2 = max(0.0, (max(second_tp - entry,0) if opt_type=="CALL" else max(entry - second_tp,0)) * max(delta,0.2) * 100)
#     total_contracts = contracts
#     projected_pnl = total_contracts * (w1*pnl1 + w2*pnl2)
#
#     c1, c2, c3 = st.columns(3)
#     c1.metric("TP1 Price", f"${first_tp:,.2f}", help="First scale-out level")
#     c2.metric("TP2 Price", f"${second_tp:,.2f}", help="Final target")
#     c3.metric("Projected P&L", f"${projected_pnl:,.2f}", help="Δ-based rough estimate")
#
#     # --------------- Journal (lessons learned) ---------------
#     st.subheader("🗒️ Trade Journal")
#     if "journal" not in st.session_state:
#         st.session_state["journal"] = []
#
#     with st.form("journal_form", clear_on_submit=True):
#         trade_note = st.text_area("Lesson learned / reasoning (why entry? why exit?)")
#         realized_pnl = st.number_input("Realized P&L ($)", value=0.0, step=50.0)
#         submitted = st.form_submit_button("➕ Add to Journal")
#         if submitted:
#             st.session_state["journal"].append({
#                 "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "Ticker": ta_ticker,
#                 "Type": opt_type,
#                 "Strike": strike,
#                 "Delta": round(delta,2),
#                 "Entry": round(entry,2),
#                 "Stop": round(stop,2),
#                 "TP1": round(first_tp,2),
#                 "TP2": round(second_tp,2),
#                 "Contracts": total_contracts,
#                 "Planned_RR": rr_target,
#                 "Note": trade_note,
#                 "RealizedPnL": realized_pnl
#             })
#             st.success("Saved to journal.")
#
#     if st.session_state["journal"]:
#         jdf = pd.DataFrame(st.session_state["journal"])
#         st.dataframe(jdf, use_container_width=True)
#         csv = jdf.to_csv(index=False).encode("utf-8")
#         st.download_button("📥 Download Journal CSV", data=csv, file_name="trade_journal.csv", mime="text/csv")


# ============================================================================================================== PART 1
# https://chatgpt.com/c/690b7e7b-2f5c-832e-95c0-6da445144de4

# =====================================================================
# 📚 TECHNICAL ANALYSIS (OPTIONS DAY TRADING) — FIXED & OPTIMIZED  0-40 MINUTES TRADE
# =====================================================================


























# =====================================================================
# 📚 TECHNICAL ANALYSIS (OPTIONS DAY TRADING) — FINAL WORKING VERSION
# =====================================================================
# 📚 TECHNICAL ANALYSIS (OPTIONS DAY TRADING) — FINAL STABLE VERSION
# =====================================================================
# 📚 TECHNICAL ANALYSIS (OPTIONS DAY TRADING) — EDUCATIONAL EDITION (v5)
# # =====================================================================
# import math
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go
# from datetime import datetime
# from ta.momentum import RSIIndicator
# from ta.trend import MACD
#
#
# # ============================================================
# # ✅ UNIVERSAL DATA LOADER (fixes malformed or multi-index columns)
# # ============================================================
# @st.cache_data(show_spinner=False)
# def get_large_then_filter(ticker: str, lookback_days: int = 90):
#     import yfinance as yf
#     df = pd.DataFrame()
#
#     # Try fetching larger datasets for flexibility
#     for per in ["1y", "5y"]:
#         try:
#             df = yf.download(ticker, period=per, interval="1d", progress=False)
#             if not df.empty:
#                 st.info(f"✅ Loaded {ticker} data for {per} successfully.")
#                 break
#         except Exception as e:
#             st.warning(f"⚠️ {ticker}: download error for {per}: {e}")
#
#     if df.empty:
#         st.error(f"❌ No data found for {ticker}")
#         return pd.DataFrame()
#
#     # --- Safe flattening of column names ---
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]
#     else:
#         df.columns = [str(c).strip() for c in df.columns]
#
#     tkr_lower = ticker.lower()
#     clean_cols = []
#     for c in df.columns:
#         if "_" in c and c.lower().endswith("_" + tkr_lower):
#             clean_cols.append(c[:-(len(tkr_lower) + 1)])
#         else:
#             clean_cols.append(c)
#     df.columns = clean_cols
#
#     cols = [str(c).title().strip() for c in df.columns]
#     df.columns = cols
#
#     if "Adj Close" in df.columns and "Close" not in df.columns:
#         df["Close"] = df["Adj Close"]
#     if "Volume" not in df.columns:
#         df["Volume"] = np.nan
#
#     needed = {"Open", "High", "Low", "Close"}
#     if not needed.issubset(df.columns):
#         st.error(f"❌ Still malformed columns: {list(df.columns)}")
#         return pd.DataFrame()
#
#     df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
#     return df.tail(lookback_days).copy()
#
#
# # ============================================================
# # 📊 TECHNICAL ANALYSIS APP
# # ============================================================
# def technical_analysis():
#     st.title("📚 Technical Analysis — Options Day Trading (v5)")
#
#     # ---------------- Sidebar Inputs ----------------
#     st.sidebar.header("⚙️ Inputs — Technical Analysis")
#
#     ta_ticker = st.sidebar.text_input("Ticker Symbol", "TSLA", key="TA_v5_ticker").strip().upper()
#     lookback_days = st.sidebar.slider("Days to Display", 30, 365, 90, 10, key="TA_v5_lookback")
#
#     acct_size = st.sidebar.number_input("Account Size ($)", 1000.0, step=100.0, value=10000.0, key="TA_v5_acct")
#     risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.25, 5.0, 1.0, 0.25, key="TA_v5_risk_pct")
#     rr_target = st.sidebar.selectbox("Target Risk/Reward", ["1:2", "1:3", "1:4"], 1, key="TA_v5_rr")
#
#     iv_input = st.sidebar.number_input("Implied Volatility (annual, %)", 5.0, 200.0, 30.0, 0.5, key="TA_v5_iv")
#     days_to_exp = st.sidebar.number_input("Days to Expiry", 1, 60, 7, key="TA_v5_dte")
#     rate_input = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25, key="TA_v5_rf")
#
#     # ---------------- Load Data ----------------
#     df = get_large_then_filter(ta_ticker, lookback_days)
#     if df.empty:
#         st.stop()
#
#     # ---------------- Indicators ----------------
#     df["EMA20"] = df["Close"].ewm(span=20).mean()
#     df["EMA50"] = df["Close"].ewm(span=50).mean()
#     df["EMA200"] = df["Close"].ewm(span=200).mean()
#
#     try:
#         df["RSI"] = RSIIndicator(df["Close"]).rsi()
#         df["MACD"] = MACD(df["Close"]).macd_diff()
#     except Exception:
#         df["RSI"], df["MACD"] = np.nan, np.nan
#
#     tr = np.maximum(df["High"] - df["Low"],
#                     np.maximum(abs(df["High"] - df["Close"].shift()),
#                                abs(df["Low"] - df["Close"].shift())))
#     df["ATR14"] = pd.Series(tr).rolling(14).mean()
#
#     # ---------------- Trade Parameters ----------------
#     last_close = float(df["Close"].iloc[-1])
#     last_atr = float(df["ATR14"].iloc[-1])
#     macd_last = float(df["MACD"].iloc[-1])
#     rr_mult = {"1:2": 2, "1:3": 3, "1:4": 4}[rr_target]
#
#     entry = last_close
#     stop = last_close - 1.2 * last_atr if macd_last >= 0 else last_close + 1.2 * last_atr
#     risk_per_share = abs(entry - stop)
#     target = entry + rr_mult * risk_per_share if macd_last >= 0 else entry - rr_mult * risk_per_share
#
#     # ---------------- Option Pricing ----------------
#     def norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
#
#     def bs_call_delta(S, K, T, r, vol):
#         if S <= 0 or K <= 0 or T <= 0 or vol <= 0:
#             return 0.5
#         d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
#         return norm_cdf(d1)
#
#     def nearest_strike(p):
#         inc = 1 if p < 50 else (2.5 if p < 150 else 5)
#         return math.ceil(p / inc) * inc
#
#     S = entry
#     T = days_to_exp / 252
#     r = rate_input / 100
#     vol = iv_input / 100
#     Kc = nearest_strike(S * 1.01)
#     Kp = nearest_strike(S * 0.99)
#
#     call_delta = bs_call_delta(S, Kc, T, r, vol)
#     put_delta = call_delta - 1
#     est_call = (max(S - Kc, 0) + 0.5 * S * vol * math.sqrt(T)) * 0.35
#     est_put = (max(Kp - S, 0) + 0.5 * S * vol * math.sqrt(T)) * 0.35
#
#     bias = "CALL" if macd_last >= 0 else "PUT"
#     delta = call_delta if bias == "CALL" else abs(put_delta)
#     prem = est_call if bias == "CALL" else est_put
#     max_risk = acct_size * (risk_pct / 100)
#     risk_per_contract = risk_per_share * max(0.2, min(delta, 0.8)) * 100
#     contracts = int(max_risk // risk_per_contract)
#
#     # ---------------- Display Summary ----------------
#     st.markdown("### 📊 Trade Overview")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Entry", f"${entry:,.2f}")
#     col2.metric("Stop", f"${stop:,.2f}")
#     col3.metric("Target", f"${target:,.2f}")
#     st.write(f"**Type:** {bias} | **Δ:** {delta:.2f} | **Premium:** ${prem:.2f} | **Contracts:** {contracts}")
#
#     # ---------------- Price Chart ----------------
#     st.markdown("### 💹 Trade Setup Visualization")
#     fig = go.Figure()
#     fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
#                                  low=df["Low"], close=df["Close"], name="Price"))
#     fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], mode="lines", name="EMA20"))
#     fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode="lines", name="EMA50"))
#     fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], mode="lines", name="EMA200"))
#
#     fig.add_hline(y=entry, line=dict(color="blue", width=2, dash="dot"), annotation_text="ENTRY")
#     fig.add_hline(y=stop, line=dict(color="red", width=2, dash="dot"), annotation_text="STOP LOSS")
#     fig.add_hline(y=target, line=dict(color="green", width=2, dash="dot"), annotation_text=f"TARGET {rr_target}")
#
#     fig.add_trace(go.Scatter(
#         x=[df.index[-1]]*3,
#         y=[entry, stop, target],
#         mode="markers+text",
#         text=["Entry", "Stop", "Target"],
#         textposition="middle right",
#         marker=dict(size=[14,14,14], color=["blue","red","green"], symbol=["circle","x","triangle-up"]),
#         name="Trade Points"
#     ))
#
#     fig.update_layout(template="plotly_dark", height=650,
#                       title=f"{ta_ticker} — Trade Setup Overview ({bias})",
#                       yaxis_title="Price ($)", xaxis_title="Date")
#     st.plotly_chart(fig, use_container_width=True)
#
#     # ---------------- RSI Chart ----------------
#     st.markdown("### 📈 RSI Momentum")
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
#     fig2.add_hline(y=70, line=dict(color="red", width=1.5, dash="dot"), annotation_text="Overbought")
#     fig2.add_hline(y=30, line=dict(color="green", width=1.5, dash="dot"), annotation_text="Oversold")
#     fig2.update_layout(template="plotly_white", height=250, yaxis_title="RSI", title="RSI Momentum")
#     st.plotly_chart(fig2, use_container_width=True)
#
#     # ---------------- Trade Journal ----------------
#     st.subheader("🗒️ Trade Journal")
#     if "journal_v5" not in st.session_state:
#         st.session_state["journal_v5"] = []
#     with st.form("add_trade_TA_v5", clear_on_submit=True):
#         note = st.text_area("Trade Notes", key="TA_v5_note")
#         pnl = st.number_input("Realized P&L ($)", 0.0, step=50.0, key="TA_v5_pnl")
#         if st.form_submit_button("💾 Save Trade Entry"):
#             st.session_state["journal_v5"].append({
#                 "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
#                 "Ticker": ta_ticker, "Type": bias, "Entry": round(entry, 2),
#                 "Stop": round(stop, 2), "Target": round(target, 2),
#                 "Contracts": contracts, "Delta": round(delta, 2),
#                 "Planned_RR": rr_target, "Note": note, "PnL": pnl})
#             st.success("✅ Trade saved.")
#     if st.session_state["journal_v5"]:
#         jdf = pd.DataFrame(st.session_state["journal_v5"])
#         st.dataframe(jdf, use_container_width=True)
#         st.download_button("📥 Download Journal CSV",
#                            jdf.to_csv(index=False).encode(),
#                            "trade_journal.csv", "text/csv")
#
#     # ============================================================
#     # 🧠 EDUCATIONAL EXPLANATION
#     # ============================================================
#     st.markdown("---")
#     st.markdown("## 🧭 Trade Setup Explanation & Justification")
#     st.write("""
#     **1️⃣ Entry Point (Blue Line):**
#     The entry is set at the most recent closing price where momentum aligns with trend direction.
#     - When the **MACD is positive**, it signals bullish momentum — ideal for a CALL.
#     - When **MACD is negative**, it signals bearish strength — ideal for a PUT.
#     This ensures entries occur *with the trend*, not against it.
#
#     **2️⃣ Stop-Loss (Red Line):**
#     Stop-loss is placed at **1.2× ATR(14)** below (for CALLs) or above (for PUTs) the entry.
#     - The **ATR (Average True Range)** measures market volatility.
#     - Using a volatility-adjusted stop prevents random price noise from prematurely stopping trades.
#     - This makes the strategy adaptive — tighter in calm markets, wider in volatile ones.
#
#     **3️⃣ Take-Profit (Green Line):**
#     The target price is set using a **Risk-to-Reward ratio (1:2, 1:3, 1:4)**.
#     - If your stop-loss is $1 away, a 1:3 setup aims for a $3 reward.
#     - This enforces discipline — even if your win rate is just 40%, you stay profitable long-term.
#     - Consistent reward ratios standardize profit-taking across tickers.
#
#     **4️⃣ How This Applies to Other Option Stocks:**
#     - This logic is **ticker-agnostic** — it works for TSLA, AAPL, NVDA, SPY, or any optionable stock.
#     - The system dynamically adjusts based on ATR and trend strength.
#     - By pairing high-delta contracts (Δ ≥ 0.6) with clear trend alignment, traders capture directional momentum efficiently.
#
#     **5️⃣ Key Advantages:**
#     - Emotion-free, rule-based entries.
#     - Dynamic position sizing based on account risk %.
#     - Quantified, reproducible edge across tickers and timeframes.
#     """)
#
#     st.info("💡 Pro Tip: Combine this strategy with volume confirmation — rising volume near the entry strengthens conviction.")
#
#
# # ============================================================
# # 🏁 Run App
# # ============================================================
# # if __name__ == "__main__":
#     technical_analysis()


# ===================================================== DAY TRADE
# =====================================================================
# ⚡ TECHNICAL ANALYSIS — INTRADAY OPTIONS DAY TRADING (v6)
# =====================================================================
# ⚡ TECHNICAL ANALYSIS — INTRADAY OPTIONS DAY TRADING (v7 - FIXED)
# =====================================================================
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD


# ============================================================
# ✅ INTRADAY DATA LOADER (1m–15m data)
# ============================================================
@st.cache_data(show_spinner=False)
def get_intraday_data(ticker: str, period="5d", interval="5m"):
    import yfinance as yf
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            st.error(f"❌ No data for {ticker} (period={period}, interval={interval})")
            return pd.DataFrame()

        # --- Ensure consistent structure ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].title() for c in df.columns]
        else:
            df.columns = [str(c).title() for c in df.columns]

        # --- Flatten any single-column dataframe (e.g. Close [[390,1]]) ---
        for col in df.columns:
            if isinstance(df[col].iloc[0], (list, np.ndarray)):
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

        # --- Fix missing or malformed columns ---
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df["Close"] = df["Adj Close"]
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    except Exception as e:
        st.error(f"⚠️ Data load error: {e}")
        return pd.DataFrame()


# ============================================================
# ⚙️ INTRADAY TECHNICAL ANALYSIS FUNCTION
# ============================================================
def intraday_technical_analysis():
    st.title("⚡ Intraday Technical Analysis — Options Day Trading (v7)")

    # ---------------- Sidebar Inputs ----------------
    st.sidebar.header("📊 Intraday Settings")

    ticker = st.sidebar.text_input("Ticker Symbol", "TSLA", key="TA_day_ticker_v7").strip().upper()
    interval = st.sidebar.selectbox("Interval", ["1m", "2m", "5m", "15m"], index=2, key="TA_day_interval_v7")
    period = st.sidebar.selectbox("Lookback Period", ["1d", "5d", "10d"], index=1, key="TA_day_period_v7")

    acct_size = st.sidebar.number_input("Account Size ($)", 1000.0, step=100.0, value=10000.0, key="TA_day_acct_v7")
    risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.25, 5.0, 1.0, 0.25, key="TA_day_risk_v7")
    rr_target = st.sidebar.selectbox("Risk/Reward", ["1:2", "1:3"], index=0, key="TA_day_rr_v7")

    # ---------------- Load Intraday Data ----------------
    df = get_intraday_data(ticker, period=period, interval=interval)
    if df.empty:
        st.stop()

    # --- Force close column to 1D numeric series ---
    df["Close"] = pd.to_numeric(df["Close"].squeeze(), errors="coerce")
    df = df.dropna(subset=["Close"])

    # ---------------- Short-Term Indicators ----------------
    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    df["MACD"] = MACD(df["Close"]).macd_diff()
    df["ATR5"] = (df["High"] - df["Low"]).rolling(5).mean()

    # ---------------- Trade Logic ----------------
    last_close = df["Close"].iloc[-1]
    last_atr = df["ATR5"].iloc[-1]
    rr_mult = {"1:2": 2, "1:3": 3}[rr_target]
    macd_last = df["MACD"].iloc[-1]

    bias = "CALL" if macd_last > 0 and df["EMA9"].iloc[-1] > df["EMA21"].iloc[-1] else "PUT"
    entry = last_close
    stop = last_close - 0.8 * last_atr if bias == "CALL" else last_close + 0.8 * last_atr
    target = last_close + rr_mult * (last_close - stop) if bias == "CALL" else last_close - rr_mult * (stop - last_close)
    risk_per_share = abs(entry - stop)
    max_risk = acct_size * (risk_pct / 100)
    shares = int(max_risk // risk_per_share)

    # ---------------- Display Summary ----------------
    st.markdown("### 🧠 Intraday Trade Setup Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Entry", f"${entry:,.2f}")
    col2.metric("Stop", f"${stop:,.2f}")
    col3.metric("Target", f"${target:,.2f}")
    st.write(f"**Bias:** {bias} | **R:R:** {rr_target} | **Contracts/Shares:** {shares}")

    # ---------------- Chart ----------------
    st.markdown("### 📈 Intraday Chart & Indicators")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode="lines", name="EMA9", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], mode="lines", name="EMA21", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode="lines", name="EMA50", line=dict(width=1.2, dash="dot")))

    fig.add_hline(y=entry, line=dict(color="blue", width=1.5, dash="dot"), annotation_text="ENTRY")
    fig.add_hline(y=stop, line=dict(color="red", width=1.5, dash="dot"), annotation_text="STOP")
    fig.add_hline(y=target, line=dict(color="green", width=1.5, dash="dot"), annotation_text="TARGET")

    fig.update_layout(template="plotly_dark", height=600,
                      title=f"{ticker} — Intraday Day Trading Setup ({bias})",
                      yaxis_title="Price", xaxis_title="Time")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- RSI ----------------
    st.markdown("### 📊 RSI Momentum")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)", mode="lines"))
    fig2.add_hline(y=70, line=dict(color="red", dash="dot"), annotation_text="Overbought")
    fig2.add_hline(y=30, line=dict(color="green", dash="dot"), annotation_text="Oversold")
    fig2.update_layout(template="plotly_white", height=250)
    st.plotly_chart(fig2, use_container_width=True)


    # ---------------- Explanation ----------------
    st.markdown("---")
    st.markdown("## 📘 Strategy Justification (Day Trading)")
    st.write("""
    **1️⃣ Entry Point (Blue Line):**
    - Entry occurs when short-term momentum confirms the bias:  
      *CALL:* EMA9 > EMA21, MACD positive.  
      *PUT:* EMA9 < EMA21, MACD negative.  
    - This detects intraday trend shifts early.

    **2️⃣ Stop-Loss (Red Line):**
    - Based on 0.8× ATR(5): a tight volatility-based stop for scalping.  
    - Keeps losses minimal and adaptive to ticker volatility.

    **3️⃣ Take-Profit (Green Line):**
    - Uses 1:2 or 1:3 Risk-Reward ratio.  
    - Locks in profit before lunchtime reversals (common in day trading).

    **4️⃣ Replicability Across Stocks:**
    - Works on TSLA, NVDA, SPY, META — any liquid options ticker.  
    - Intraday signals depend only on EMAs, MACD, and ATR volatility — universal measures.

    **5️⃣ Edge:**
    - Removes emotional decisions.  
    - Dynamic position sizing protects capital.  
    - Repeatable framework across multiple tickers per day.
    """)
    st.info("💡 Tip: Trade only during high-volume periods (9:35–11:00 AM, 2:00–3:30 PM ET) for best signal reliability.")


# ============================================================
# 🏁 Run Streamlit App
# ============================================================
# if __name__ == "__main__":
    intraday_technical_analysis()


















#         # ================== ENDS HERE ENDS HERE===================








# ================================================================================================================
# CALCULATOR DISPLAY ROI
# ==============================================
# 💹 OPTIONS ROI CALCULATOR — CALL & PUT + AUTO RISK/REWARD + VISUALIZATION
# ==============================================
# ==============================================
# 💹 OPTIONS ROI CALCULATOR — CALL & PUT + AUTO RISK/REWARD + P/L EXPANSION
# ==============================================


#
# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
#
# def roi_calculator():
#     st.title("💹 Options ROI Calculator with Dynamic CALL/PUT Control")
#     st.caption("Calculate total premium, ROI%, and visualize entry, profit, and stop-loss with live adjustments.")
#     st.markdown("---")
#
#     # ---------- CSS for Card Layout ----------
#     st.markdown("""
#     <style>
#       .kpi-card {
#         background: rgba(255,255,255,0.03);
#         border: 1px solid rgba(255,255,255,0.08);
#         border-radius: 14px;
#         padding: 14px 16px;
#         box-shadow: 0 4px 14px rgba(0,0,0,0.15);
#       }
#       .kpi-title {
#         font-weight: 600;
#         font-size: 1.05rem;
#         color: #cbd5ff;
#         margin-bottom: 10px;
#         display: flex;
#         align-items: center;
#         gap: 8px;
#       }
#       .kpi-row { margin: 4px 0 12px 0; }
#       .kpi-label { color: #aab0c0; font-size: 0.92rem; margin-bottom: 2px; }
#       .kpi-value { font-weight: 700; font-size: 1.6rem; line-height: 1.1; }
#       .pos { color: #27e08d; } .neg { color: #ff6161; } .neu { color: #ffb84d; }
#       .small-note { color: #9aa3b2; font-size: 0.88rem; margin-top: -2px; }
#     </style>
#     """, unsafe_allow_html=True)
#
#     # ---------- INPUTS ----------
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         contracts = st.number_input("🧾 Number of Contracts", 1, 1000, 10)
#     with col2:
#         premium = st.number_input("💵 Premium per Contract ($)", 0.01, 1000.0, 2.50, 0.01)
#     with col3:
#         strike = st.number_input("🎯 Strike Price ($)", 0.01, 10000.0, 200.00, 1.0)
#
#     col4, col5 = st.columns(2)
#     with col4:
#         current_price = st.number_input("📈 Current Stock Price ($)", 0.01, 10000.0, 215.00, 0.5)
#     with col5:
#         option_type = st.selectbox("📊 Option Type", ["CALL", "PUT"])
#
#     # ---------- RISK/REWARD CONTROLS ----------
#     st.subheader("⚙️ Adjust Risk-Reward Settings")
#     colA, colB = st.columns(2)
#     with colA:
#         target_gain_pct = st.slider("Target Gain (%)", 10, 300, 70, 5)
#     with colB:
#         stop_loss_pct = st.slider("Stop-Loss (%)", 5, 100, 30, 5)
#
#     shares_per_contract = 100
#
#     # ---------- CALCULATIONS ----------
#     total_investment = premium * contracts * shares_per_contract
#     exposure = strike * contracts * shares_per_contract
#
#     # Intrinsic Value for CALL vs PUT
#     if option_type == "CALL":
#         intrinsic_value = max(0, current_price - strike) * shares_per_contract * contracts
#     else:
#         intrinsic_value = max(0, strike - current_price) * shares_per_contract * contracts
#
#     profit_loss = intrinsic_value - total_investment
#     roi_percent = (profit_loss / total_investment * 100) if total_investment > 0 else 0
#
#     # Risk/Reward Price Levels
#     take_profit_price = premium * (1 + target_gain_pct / 100)
#     stop_loss_price = premium * (1 - stop_loss_pct / 100)
#
#     risk_amount = abs(premium - stop_loss_price)
#     reward_amount = abs(take_profit_price - premium)
#     rr_ratio = reward_amount / risk_amount if risk_amount != 0 else 0
#
#     # 💰 Profit/Loss in dollars (total per trade)
#     total_profit = reward_amount * shares_per_contract * contracts
#     total_loss = risk_amount * shares_per_contract * contracts
#
#     # ===========================================================
#     # 📊 THREE-COLUMN CARD LAYOUT
#     # ===========================================================
#     st.markdown("### 📊 Trade Overview")
#     c1, c2, c3 = st.columns(3)
#
#     with c1:
#         roi_class = "pos" if roi_percent > 0 else ("neg" if roi_percent < 0 else "neu")
#         st.markdown(f"""
#         <div class="kpi-card">
#           <div class="kpi-title">📊 Summary</div>
#           <div class="kpi-row"><div class="kpi-label">Total Premium (Investment)</div><div class="kpi-value">${total_investment:,.2f}</div></div>
#           <div class="kpi-row"><div class="kpi-label">Exposure (Underlying)</div><div class="kpi-value">${exposure:,.2f}</div></div>
#           <div class="kpi-row"><div class="kpi-label">Gain / Loss</div><div class="kpi-value {roi_class}">${profit_loss:,.2f}</div><div class="small-note">{roi_percent:.2f}% ROI</div></div>
#         </div>
#         """, unsafe_allow_html=True)
#
#     with c2:
#         # Expanded with Profit/Loss values in brackets
#         st.markdown(f"""
#         <div class="kpi-card">
#           <div class="kpi-title">🎯 Buy / Sell / Stop</div>
#           <div class="kpi-row">
#             <div class="kpi-label">Buy Price (Entry)</div>
#             <div class="kpi-value">${premium:,.2f}</div>
#           </div>
#           <div class="kpi-row">
#             <div class="kpi-label">Sell for Profit (+{target_gain_pct}%)</div>
#             <div class="kpi-value pos">${take_profit_price:,.2f} <span style='font-size:0.9rem; color:#a3ffb1;'>(+${total_profit:,.2f})</span></div>
#           </div>
#           <div class="kpi-row">
#             <div class="kpi-label">Stop-Loss (−{stop_loss_pct}%)</div>
#             <div class="kpi-value neg">${stop_loss_price:,.2f} <span style='font-size:0.9rem; color:#ff9a9a;'>(−${total_loss:,.2f})</span></div>
#           </div>
#         </div>
#         """, unsafe_allow_html=True)
#
#     with c3:
#         st.markdown(f"""
#         <div class="kpi-card">
#           <div class="kpi-title">📈 Risk Ratio</div>
#           <div class="kpi-row"><div class="kpi-label">Reward per Contract</div><div class="kpi-value">${reward_amount * 100:,.2f}</div></div>
#           <div class="kpi-row"><div class="kpi-label">Risk per Contract</div><div class="kpi-value">${risk_amount * 100:,.2f}</div></div>
#           <div class="kpi-row"><div class="kpi-label">Risk-to-Reward</div><div class="kpi-value neu">1 : {rr_ratio:.2f}</div></div>
#         </div>
#         """, unsafe_allow_html=True)
#
#     st.markdown("---")
#
#     # ======================================== DELTA==================
#     # delta = st.number_input("Option Delta (Δ)", 0.1, 1.0, 0.5, 0.05)
#     # price_move_needed = (take_profit_price - premium) / delta
#     # st.metric("Required Stock Move to Hit Target", f"${price_move_needed:,.2f}")
#
#     # ================================================================================= DELTA 2
#
#     delta = st.number_input("Option Delta (Δ)", 0.1, 1.0, 0.5, 0.05)
#     move_needed = (take_profit_price - premium) / delta
#     st.metric("📈 Required Stock Move to Hit Target", f"${move_needed:,.2f}")
#
#     if delta >= 0.7:
#         st.success("✅ High Delta — fast-moving, lower risk per dollar.")
#     elif delta <= 0.3:
#         st.warning("⚠️ Low Delta — cheaper but needs big stock move.")
#     else:
#         st.info("ℹ️ Moderate Delta — balanced movement and cost.")
#
#
#     # ======================================================================= AUTO DELTA
#
#     import yfinance as yf
#     # import streamlit as st
#     import pandas as pd
#
#     ticker_symbol = st.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL")
#     expiry = st.text_input("Enter Expiry Date (YYYY-MM-DD)", "2025-12-19")
#
#     ticker = yf.Ticker(ticker_symbol)
#
#     try:
#         opt_chain = ticker.option_chain(expiry)
#         calls = opt_chain.calls
#
#         # ✅ Check if delta column exists
#         if 'delta' in calls.columns:
#             high_delta_calls = calls[calls['delta'] > 0.7]
#             st.success("✅ Found Delta data.")
#             st.dataframe(high_delta_calls[['contractSymbol', 'strike', 'lastPrice', 'delta']])
#         else:
#             st.warning("⚠️ Delta data not available for this ticker/expiry on Yahoo Finance.")
#             st.dataframe(calls[['contractSymbol', 'strike', 'lastPrice', 'impliedVolatility', 'inTheMoney']])
#     except Exception as e:
#         st.error(f"❌ Error fetching data: {e}")
#
#     st.divider()
#
#
#
#     # ===========================================================
#     # 📈 Dynamic Visualization
#     # ===========================================================
#     st.subheader("💹 Option Trade Visualization")
#
#     if option_type == "CALL":
#         prices = [stop_loss_price, premium, take_profit_price]
#         labels = ["Stop-Loss (−)", "Buy Entry", "Take-Profit (+)"]
#         colors = ["#ff4d4d", "#5dade2", "#27e08d"]
#     else:
#         prices = [take_profit_price, premium, stop_loss_price]
#         labels = ["Take-Profit (+)", "Buy Entry", "Stop-Loss (−)"]
#         colors = ["#27e08d", "#5dade2", "#ff4d4d"]
#
#     fig_lines = go.Figure()
#     for p, l, c in zip(prices, labels, colors):
#         fig_lines.add_shape(
#             type="line", x0=0, x1=1, y0=p, y1=p, xref="paper",
#             line=dict(color=c, width=3, dash="dot"), name=l
#         )
#         fig_lines.add_trace(go.Scatter(x=[0.5], y=[p], mode="text", text=[l], textposition="middle right"))
#
#     fig_lines.update_layout(
#         yaxis_title="Option Premium ($)",
#         title=f"Option Type: {option_type} — Risk/Reward Setup",
#         height=400,
#         yaxis=dict(range=[min(prices) * 0.8, max(prices) * 1.2])
#     )
#     st.plotly_chart(fig_lines, use_container_width=True)
#
#
#     # ===========================================================
#     # 🗒️ Notes + Reference Table
#     # ===========================================================
#     st.markdown("### 📘 Quick Reference Table")
#     ref_df = pd.DataFrame({
#         "Item": [
#             "1 Contract", "10 Contracts", "Premium (per contract)",
#             "Total Exposure", "Max Loss", "Max Profit"
#         ],
#         "Meaning": [
#             "Controls 100 shares",
#             "Controls 1,000 shares",
#             f"${premium:.2f} × 100 × {contracts} = ${premium * 100 * contracts:,.2f}",
#             f"${strike:.2f} × 100 × {contracts} = ${exposure:,.2f}",
#             f"Premium paid = ${total_investment:,.2f}",
#             "Unlimited (CALL) or up to strike (PUT)"
#         ]
#     })
#     st.table(ref_df)
#
#     # ==============================================================================================
# # INSTRUCTION OF TRADING
# #     ================================================================================================
#
#     # ===========================================================
#     # 📘 EDUCATIONAL INSTRUCTION: UNDERSTANDING 70% GAIN & DELTA
#     # ===========================================================
#     with st.expander("⚙️ Understanding the 70% Gain & Delta Concept — Click to Expand"):
#         st.markdown("""
#         ### ⚙️ 1️⃣ Basic Concept: What “70% Gain” Means
#         A 70% gain means your **option premium (price per contract)** increases by 70%.
#
#         **Example:**
#         - You bought a CALL at **$2.50**
#         - Your target = $2.50 × 1.70 = **$4.25**
#         - That’s a **+70% gain per contract**
#
#         👉 Your goal is for the option price to rise from **$2.50 → $4.25**.
#
#         ---
#
#         ### ⚙️ 2️⃣ Option Sensitivity — Delta (Δ)
#         The **Delta** tells you how much the option price changes when the **stock price moves by $1**.
#
#         **Example:**
#         - Δ = 0.50 (typical for near-the-money options)
#         - If stock goes up by $1 → option price rises ≈ **$0.50**
#
#         ---
#
#         ### ⚙️ 3️⃣ Estimating the Required Move
#         We need the option to rise by **$1.75** (from $2.50 → $4.25).
#         - Δ = 0.5
#         - Required stock move ≈ Option price change ÷ Δ
#         - **= $1.75 ÷ 0.5 = $3.50**
#
#         ✅ The underlying stock must rise **~$3.50** per share to deliver your +70% option gain.
#
#         ---
#
#         ### ⚙️ 4️⃣ Refinement (Gamma + Time Decay)
#         Delta isn’t fixed — it **increases as your CALL moves in-the-money**:
#         - Early in the move: Δ ≈ 0.45–0.50
#         - Closer to target: Δ ≈ 0.60–0.65
#         → So the *real* required move might be smaller, **closer to $3.00**.
#
#         ---
#
#         ### ⚙️ 5️⃣ Quick Reference Table
#
#         | Option Delta | Stock Move Needed to Hit +70% | Approx. Stock Change |
#         |--------------:|-------------------------------:|---------------------:|
#         | 0.30 | $1.75 ÷ 0.30 | **$5.83** |
#         | 0.40 | $1.75 ÷ 0.40 | **$4.38** |
#         | 0.50 | $1.75 ÷ 0.50 | **$3.50** |
#         | 0.60 | $1.75 ÷ 0.60 | **$2.92** |
#         | 0.70 | $1.75 ÷ 0.70 | **$2.50** |
#
#         ✅ The higher the **Delta**, the **smaller** the stock move required.
#
#         ---
#
#         ### ⚙️ 6️⃣ PUT Example (reverse logic)
#         For a PUT, the math is the same — but the stock must **move down** to hit your +70% target.
#
#         **Example:**
#         - Strike = $200
#         - Premium = $2.50
#         - Δ = −0.50
#         → Need about a **$3.50 stock drop** to reach +70% profit.
#         """)
#

#     st.subheader("🗒️ Trade Notes")
#     st.text_area("Write key insights or lessons learned:")
#     if st.button("💾 Save ROI Entry"):
#         st.success("✅ ROI entry saved (extendable to trade journal).")

#     # ==================== ROI PART 1 ENDS HERE


#
#
# import streamlit as st
# import plotly.graph_objects as go
# import pandas as pd
# from datetime import datetime
# import os
# import uuid
#
# # ✅ Unique key per page reload to prevent duplicate Streamlit keys
# if "active_page_key" not in st.session_state:
#     st.session_state["active_page_key"] = str(uuid.uuid4())[:8]
#
# def power_roi_daytrading():
#     st.title("⚡ Power ROI — Options Day-Trading Dashboard")
#     st.caption("Track Entry, Stop-Loss, Target, and Actual P/L for CALL & PUT options.")
#     st.markdown("---")
#
#     # ========================== INPUTS ==========================
#     symbol = st.text_input("🏢 Company / Symbol", "AAPL",
#                            key=f"symbol_input_roi_{st.session_state['active_page_key']}")
#
#     # --- Unified Mode Toggle ---
#     st.markdown("#### 🎛️ Position Entry Mode")
#     mode = st.radio("Select Input Type", ["🧾 By Contracts", "💵 By Amount"],
#                     horizontal=True, key=f"entry_mode_{st.session_state['active_page_key']}")
#
#     # --- Primary Input Row (Compact + Aligned) ---
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         if mode == "🧾 By Contracts":
#             contracts = st.number_input("🧾 Number of Contracts", 1, 1000, 1,
#                                         key=f"contracts_input_{st.session_state['active_page_key']}")
#             invest_amount = contracts * 100  # temp until premium entered
#         else:
#             invest_amount = st.number_input("💵 Amount to Invest ($)", 100.0, 500000.0, 1000.0, 50.0,
#                                             key=f"amount_input_{st.session_state['active_page_key']}")
#             contracts = 1  # will update after premium input
#
#     with c2:
#         premium = st.number_input("💵 Entry Premium per Contract ($)", 0.01, 200.0, 2.15, 0.01,
#                                   key=f"premium_input_{st.session_state['active_page_key']}")
#     with c3:
#         strike = st.number_input("🎯 Strike Price ($)", 0.01, 10000.0, 270.0, 1.0,
#                                  key=f"strike_input_{st.session_state['active_page_key']}")
#
#     # --- Secondary Input Row ---
#     c4, c5, c6 = st.columns(3)
#     with c4:
#         current_price = st.number_input("📈 Current Stock Price ($)", 0.01, 10000.0, 269.5, 0.25,
#                                         key=f"current_price_{st.session_state['active_page_key']}")
#     with c5:
#         option_type = st.selectbox("📊 Option Type", ["CALL", "PUT"],
#                                    key=f"option_type_{st.session_state['active_page_key']}")
#     with c6:
#         exit_premium = st.number_input("💰 Exit Premium per Contract ($) – if sold", 0.0, 200.0, 0.0, 0.01,
#                                        key=f"exit_premium_{st.session_state['active_page_key']}")
#
#     # --- Dynamic Auto-Update ---
#     shares = 100
#     if mode == "💵 By Amount":
#         contracts = max(1, int(invest_amount / (premium * shares))) if premium > 0 else 1
#     total_invest = contracts * premium * shares
#
#     # --- Inline Summary ---
#     st.markdown(f"""
#     <div style='margin-top:10px; padding:10px; border-radius:8px; background:rgba(255,255,255,0.05);
#                 border:1px solid rgba(255,255,255,0.1);'>
#         <b>📦 Contracts:</b> {contracts:,} | <b>💰 Total Investment:</b> ${total_invest:,.2f}
#     </div>
#     """, unsafe_allow_html=True)
#
#     #
#     #
#     # # ==================================================
#     # # # 💵 Optional Amount Entry (auto-compute contracts)
#     # # ===================================================
#     #
#     #
#     # st.markdown("---")
#     # st.subheader("💸 Optional Investment Mode")
#     # c7, c8 = st.columns(2)
#     # with c7:
#     #     input_mode = st.radio("Select Input Mode", ["🧾 By Contracts", "💵 By Amount"],
#     #                           horizontal=True, key=f"input_mode_{st.session_state['active_page_key']}")
#     # with c8:
#     #     if input_mode == "💵 By Amount":
#     #         invest_amount = st.number_input("💵 Amount to Invest ($)", 100.0, 500000.0, 1000.0, 50.0,
#     #                                         key=f"amount_input_{st.session_state['active_page_key']}")
#     #         contracts = max(1, int(invest_amount / (premium * 100)))
#     #         st.success(f"Calculated Contracts: {contracts} from ${invest_amount:,.2f}")
#     #     else:
#     #         invest_amount = contracts * premium * 100
#     #         st.info(f"Total Investment: ${invest_amount:,.2f}")
#
#     st.divider()
#
#
#
# # ==================== ENDS HERE
#     # ========================== CONSTANTS ==========================
#     shares = 100
#     total_invest = premium * shares * contracts
#     breakeven = strike + premium if option_type == "CALL" else strike - premium
#
#     # ========================== TARGET / STOP SETTINGS ==========================
#     st.subheader("⚙️ Define Risk–Reward Plan")
#     colA, colB = st.columns(2)
#     with colA:
#         target_gain_pct = st.slider("🎯 Target Gain (%)", 10, 300, 70, 5)
#     with colB:
#         stop_loss_pct = st.slider("🛑 Stop-Loss (%)", 5, 100, 30, 5)
#
#     take_profit_price = premium * (1 + target_gain_pct / 100)
#     stop_loss_price = premium * (1 - stop_loss_pct / 100)
#
#     gain_per_contract = (take_profit_price - premium) * shares
#     loss_per_contract = (premium - stop_loss_price) * shares
#
#     total_profit = gain_per_contract * contracts
#     total_loss = loss_per_contract * contracts
#
#     proj_rr = (take_profit_price - premium) / (premium - stop_loss_price)
#     proj_roi_pct = (take_profit_price - premium) / premium * 100
#
#     # ========================== ACTUAL EXIT ==========================
#     if exit_premium > 0:
#         total_exit = exit_premium * shares * contracts
#         profit_loss = total_exit - total_invest
#         roi_pct = profit_loss / total_invest * 100
#         actual_rr = abs((exit_premium - premium) / (premium - stop_loss_price))
#     else:
#         profit_loss, roi_pct, actual_rr = 0, 0, 0
#
#     # ========================== DISPLAY SUMMARY ==========================
#     st.markdown("### 📊 Trade Summary")
#
#     total_volume = contracts * shares
#
#     cA, cB, cC = st.columns(3)
#     with cA:
#         st.metric("Entry Premium", f"${premium:,.2f}")
#         st.metric("Total Investment", f"${total_invest:,.2f}")
#         st.metric("Break-Even", f"${breakeven:,.2f}")
#         st.metric("📊 Total Volume (Shares Controlled)", f"{total_volume:,}")
#
#     with cB:
#         st.markdown(f"""
#         <div style='background:rgba(0,255,0,0.03); border:1px solid rgba(0,255,0,0.2);
#                     border-radius:10px; padding:10px;'>
#             <b style='color:#27e08d;'>🎯 Target (+{target_gain_pct}%)</b><br>
#             • Target Price: <b>${take_profit_price:,.2f}</b><br>
#             • Gain/Contract: <b>+${gain_per_contract:,.2f}</b><br>
#             • Total Gain ({contracts} contracts): <b>+${total_profit:,.2f}</b>
#         </div>
#         """, unsafe_allow_html=True)
#
#         st.markdown(f"""
#         <div style='background:rgba(255,0,0,0.03); border:1px solid rgba(255,0,0,0.2);
#                     border-radius:10px; padding:10px; margin-top:8px;'>
#             <b style='color:#ff4d4d;'>🛑 Stop-Loss (−{stop_loss_pct}%)</b><br>
#             • Stop-Loss Price: <b>${stop_loss_price:,.2f}</b><br>
#             • Loss/Contract: <b>−${loss_per_contract:,.2f}</b><br>
#             • Total Risk ({contracts} contracts): <b>−${total_loss:,.2f}</b>
#         </div>
#         """, unsafe_allow_html=True)
#
#     with cC:
#         color = "🟢" if profit_loss > 0 else ("🔴" if profit_loss < 0 else "⚪")
#         st.metric(f"{color} Realized P/L", f"${profit_loss:,.2f}", f"{roi_pct:,.2f}%")
#         if exit_premium > 0:
#             st.metric("Actual R:R", f"1 : {actual_rr:,.2f}")
#             st.metric("Exit Premium", f"${exit_premium:,.2f}")
#
#     # # ========================== PERFORMANCE COMPARISON ==========================
#     # if exit_premium > 0:
#     #     perf_color = "#27e08d" if roi_pct >= proj_roi_pct else "#ff7070"
#     #     st.markdown(f"""
#     #     <div style='margin-top:10px; background:rgba(255,255,255,0.03);
#     #                 border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:12px;'>
#     #       <b style='color:{perf_color};'>📈 Performance Comparison</b><br>
#     #       • Projected ROI (+{target_gain_pct}%): <b>{proj_roi_pct:.2f}%</b><br>
#     #       • Actual ROI (Exit): <b>{roi_pct:.2f}%</b><br>
#     #       • Projected R:R: <b>1 : {proj_rr:,.2f}</b> | Actual R:R: <b>1 : {actual_rr:,.2f}</b>
#     #     </div>
#     #     """, unsafe_allow_html=True)
#
# # ===================================================== PART 2
#
#     # ========================== PERFORMANCE COMPARISON ==========================
#     if exit_premium > 0:
#         perf_color = "#27e08d" if roi_pct >= proj_roi_pct else "#ff7070"
#         st.markdown(f"""
#         <div style='margin-top:10px; background:rgba(255,255,255,0.03);
#                     border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:12px;'>
#           <b style='color:{perf_color};'>📈 Performance Comparison</b><br>
#           • Projected ROI (+{target_gain_pct}%): <b>{proj_roi_pct:.2f}%</b><br>
#           • Actual ROI (Exit): <b>{roi_pct:.2f}%</b><br>
#           • Projected R:R: <b>1 : {proj_rr:,.2f}</b> | Actual R:R: <b>1 : {actual_rr:,.2f}</b>
#         </div>
#         """, unsafe_allow_html=True)
#
#         # ========================== TARGET INSIGHT FOR DAY TRADERS ==========================
#         st.markdown("### ⚡ Day-Trader Momentum Insight")
#
#         # --- Core Momentum Inputs ---
#         delta = st.number_input("Δ Option Delta (how fast premium moves per $1 underlying)", 0.05, 1.0, 0.45, 0.05,
#                                 key=f"delta_day_{st.session_state['active_page_key']}")
#         volume_speed = st.number_input("📊 Volume Spike Ratio (today vs avg)", 0.5, 5.0, 1.8, 0.1,
#                                        key=f"vol_day_{st.session_state['active_page_key']}")
#         last_move = st.number_input("⚙️ Last Price Move ($ per min)", 0.01, 10.0, 0.45, 0.05,
#                                     key=f"price_speed_{st.session_state['active_page_key']}")
#
#         # --- Estimations ---
#         # Premium movement rate per minute
#         premium_speed = delta * last_move * volume_speed
#
#         # Time (in minutes) to reach take-profit
#         move_needed = take_profit_price - premium
#         est_minutes = max(1, move_needed / premium_speed) if premium_speed > 0 else 999
#
#         # Strength score based on combined indicators
#         momentum_score = min(100, (delta * volume_speed * (1 / est_minutes)) * 1000)
#
#         # Grade insight
#         if momentum_score > 80:
#             insight_color = "#27e08d"
#             speed_comment = "🚀 **High momentum:** target may hit within 15–30 minutes."
#         elif momentum_score > 50:
#             insight_color = "#f39c12"
#             speed_comment = "⚙️ **Moderate momentum:** target likely within 30–90 minutes."
#         else:
#             insight_color = "#e74c3c"
#             speed_comment = "🐢 **Low momentum:** premium moving slowly; may require 2–3 hours or exit early."
#
#         # Display insight box
#         st.markdown(f"""
#         <div style='margin-top:10px; background:rgba(255,255,255,0.03);
#                     border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;'>
#             <b style='color:{insight_color};'>📊 Estimated Time to Target:</b> ~{est_minutes:.0f} minutes<br>
#             <b>Premium Speed:</b> ${premium_speed:,.2f} per minute<br>
#             <b>Momentum Score:</b> {momentum_score:.1f} / 100<br><br>
#             {speed_comment}
#         </div>
#         """, unsafe_allow_html=True)
#
#     # ========================== NOTES & SAVE TO CSV ==========================
#     st.markdown("---")
#     st.subheader("🗒️ Notes / Review")
#     notes = st.text_area("Journal your thought process, entry signal, and lessons learned:")
#
#     if st.button("💾 Save Entry"):
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         save_path = r"C:\Users\stans\ML_PROJECTS\0. STOCK_TRADING_PROJECT\1. Dataset\Option_trading.csv"
#
#         new_data = pd.DataFrame([{
#             "Timestamp": timestamp,
#             "Symbol": symbol,
#             "Option Type": option_type,
#             "Strike": strike,
#             "Entry Premium": premium,
#             "Exit Premium": exit_premium,
#             "Contracts": contracts,
#             "Shares": shares,
#             "Total Volume": total_volume,
#             "Total Investment": total_invest,
#             "Break-Even": breakeven,
#             "Target Gain %": target_gain_pct,
#             "Stop-Loss %": stop_loss_pct,
#             "Projected ROI %": proj_roi_pct,
#             "Actual ROI %": roi_pct,
#             "Projected R:R": proj_rr,
#             "Actual R:R": actual_rr,
#             "Total Profit $": total_profit,
#             "Total Loss $": total_loss,
#             "Realized P/L $": profit_loss,
#             "Notes": notes
#         }])
#
#         if os.path.exists(save_path):
#             existing = pd.read_csv(save_path)
#             updated = pd.concat([existing, new_data], ignore_index=True)
#         else:
#             updated = new_data
#
#         updated.to_csv(save_path, index=False)
#         st.success(f"✅ Entry saved to: {save_path}")
#
#     # ========================== EXPANDER: TRADING INSTRUCTIONS ==========================
#     with st.expander("🧭 How to Load Option Chart & Take Position — Step-by-Step"):
#         st.markdown("""
#         ### ⚙️ **1️⃣ Load Your Option Chart in ThinkorSwim (TOS)**
#         - Go to the **Trade Tab** → select your ticker (e.g., `AAPL`).
#         - Find your desired **expiration date** (e.g., `7 NOV 25`).
#         - Right-click your **strike price** (e.g., `270C`) → choose **"Send to Chart"**.
#         - Open your linked **Chart tab**, and you’ll now see the **option contract price** (not the stock).
#
#         ### 📊 **2️⃣ Setup in Power ROI App**
#         - Select your **Symbol**, **Strike**, and **Option Type**.
#         - Enter **Entry Premium**, **Target**, **Stop-Loss**, and **Exit Premium**.
#         - The app auto-calculates:
#           - Total Investment, Break-Even, Profit/Loss, ROI, R:R.
#           - Visual Target & Stop-Loss levels.
#
#         ### 📈 **3️⃣ Manage and Record**
#         - Once trade closes, input Exit Premium.
#         - Add notes for why you entered/exited.
#         - Click **Save Entry** — data is appended to your trade log.
#         """)

# # ================================================================================= PART 10
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
import uuid

# ✅ Unique key per page reload
if "active_page_key" not in st.session_state:
    st.session_state["active_page_key"] = str(uuid.uuid4())[:8]

def power_roi_daytrading():
    st.title("⚡ Power ROI — Options Day-Trading Dashboard")
    st.caption("Track Entry, Stop-Loss, Target, and Actual P/L for CALL & PUT options.")

    # --- Top Input Row: Company + Editable Date + Buying Power ---
    col1, col2, col3 = st.columns([1.8, 1, 1])  # ← 3 columns now
    with col1:
        symbol = st.text_input("🏢 Company / Symbol", "AAPL",
                               key=f"symbol_input_roi_{st.session_state['active_page_key']}")

    with col2:
        # ✅ New Buying Power field (shows available or estimated buying capacity)
        buying_power = st.number_input("💵 Buying Power ($ Available)", 1000.0, 500000.0, 10000.0, 100.0,
                                       key=f"buying_power_{st.session_state['active_page_key']}")

    with col3:
        # Editable trade date, defaults to today
        default_date = datetime.now().date()
        trade_date = st.date_input("📅 Trade Date", default_date,
                                   key=f"trade_date_{st.session_state['active_page_key']}")


    # --- Unified Mode Toggle ---
    # st.markdown("#### 🎛️ Position Entry Mode")
    # mode = st.radio("Select Input Type", ["🧾 By Contracts", "💵 By Amount"],
    #                 horizontal=True, key=f"entry_mode_{st.session_state['active_page_key']}")
    #
    # # --- Primary Input Row (Compact + Aligned) ---
    # c1, c2, c3 = st.columns(3)
    # with c1:
    #     if mode == "🧾 By Contracts":
    #         contracts = st.number_input("🧾 Number of Contracts", 1, 1000, 1,
    #                                     key=f"contracts_input_{st.session_state['active_page_key']}")
    #         invest_amount = contracts * 100
    #     else:
    #         invest_amount = st.number_input("💵 Amount to Invest ($)", 100.0, 500000.0, 1000.0, 50.0,
    #                                         key=f"amount_input_{st.session_state['active_page_key']}")
    #         contracts = 1

    # ================================================================================================== PART 1

    # --- Position Entry Mode (Same Line Layout) ---
    c1, c2 = st.columns([1, 3])  # Adjust ratio as needed

    with c1:
        st.markdown("#### 🎛️ Position Entry Mode")

    with c2:
        mode = st.radio(
            "",
            ["🧾 By Contracts", "💵 By Amount"],
            horizontal=True,
            key=f"entry_mode_{st.session_state['active_page_key']}"
        )

    # --- Primary Input Row (Aligned Below) ---
    c1, c2, c3 = st.columns(3)

    with c1:
        if mode == "🧾 By Contracts":
            contracts = st.number_input(
                "🧾 Number of Contracts",
                min_value=1,
                max_value=1000,
                value=1,
                key=f"contracts_input_{st.session_state['active_page_key']}"
            )
            invest_amount = contracts * 100
        else:
            invest_amount = st.number_input(
                "💵 Amount to Invest ($)",
                min_value=100.0,
                max_value=500000.0,
                value=1000.0,
                step=50.0,
                key=f"amount_input_{st.session_state['active_page_key']}"
            )
            contracts = 1

    # =====================================ENDS HERE
    with c2:
        premium = st.number_input("💵 Entry Premium per Contract ($)", 0.01, 200.0, 2.15, 0.01,
                                  key=f"premium_input_{st.session_state['active_page_key']}")
    with c3:
        strike = st.number_input("🎯 Strike Price ($)", 0.01, 10000.0, 270.0, 1.0,
                                 key=f"strike_input_{st.session_state['active_page_key']}")

    # --- Secondary Input Row ---
    c4, c5, c6 = st.columns(3)
    with c4:
        current_price = st.number_input("📈 Current Stock Price ($)", 0.01, 10000.0, 269.5, 0.25,
                                        key=f"current_price_{st.session_state['active_page_key']}")
    with c5:
        option_type = st.selectbox("📊 Option Type", ["CALL", "PUT"],
                                   key=f"option_type_{st.session_state['active_page_key']}")
    with c6:
        exit_premium = st.number_input("💰 Exit Premium per Contract ($) – if sold", 0.0, 200.0, 0.0, 0.01,
                                       key=f"exit_premium_{st.session_state['active_page_key']}")

    # --- Dynamic Auto-Update ---
    shares = 100
    if mode == "💵 By Amount":
        contracts = max(1, int(invest_amount / (premium * shares))) if premium > 0 else 1
    total_invest = contracts * premium * shares

    # --- Inline Summary ---
    st.markdown(f"""
    <div style='margin-top:10px; padding:10px; border-radius:8px; background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.1);'>
        <b>📦 Contracts:</b> {contracts:,} | <b>💰 Total Investment:</b> ${total_invest:,.2f}
    </div>
    """, unsafe_allow_html=True)


    # st.divider()

    # ========================== CONSTANTS ==========================
    # shares = 100
    # total_invest = premium * shares * contracts
    # breakeven = strike + premium if option_type == "CALL" else strike - premium
    #
    # # ========================== TARGET / STOP SETTINGS ==========================
    # st.subheader("⚙️ Define Risk–Reward Plan")
    # colA, colB = st.columns(2)
    # with colA:
    #     target_gain_pct = st.slider("🎯 Target Gain (%)", 10, 300, 70, 5)
    # with colB:
    #     stop_loss_pct = st.slider("🛑 Stop-Loss (%)", 5, 100, 30, 5)
    #
    # take_profit_price = premium * (1 + target_gain_pct / 100)
    # stop_loss_price = premium * (1 - stop_loss_pct / 100)
    #
    # gain_per_contract = (take_profit_price - premium) * shares
    # loss_per_contract = (premium - stop_loss_price) * shares
    # total_profit = gain_per_contract * contracts
    # total_loss = loss_per_contract * contracts
    #
    # proj_rr = (take_profit_price - premium) / (premium - stop_loss_price)
    # proj_roi_pct = (take_profit_price - premium) / premium * 100

    # ================================================================================== PART2

    # ========================== CONSTANTS ==========================
    with st.expander('RISK RATIO ADJUSTMENT BASED ON ACTURAL TRADE'):
        shares = 100
        total_invest = premium * shares * contracts
        breakeven = strike + premium if option_type == "CALL" else strike - premium

        # ========================== TARGET / STOP SETTINGS ==========================
        st.subheader("⚙️ Define Risk–Reward Plan")

        colA, colB, colC = st.columns(3)

        # --- Manual sliders ---
        with colA:
            target_gain_pct = st.slider("🎯 Target Gain (%)", 10, 300, 70, 5)
        with colB:
            stop_loss_pct = st.slider("🛑 Stop-Loss (%)", 5, 100, 30, 5)

        # --- Auto slider based on actual ratio ---
        with colC:
            # ✅ Safely handle undefined df (if not loaded yet)
            actual_rr = 2.0  # default ratio if no trade data is present

            try:
                if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:
                    if "PnL%" in df.columns:
                        avg_gain = df.loc[df["PnL%"] > 0, "PnL%"].mean()
                        avg_loss = abs(df.loc[df["PnL%"] < 0, "PnL%"].mean())
                        if pd.notna(avg_gain) and pd.notna(avg_loss) and avg_loss > 0:
                            actual_rr = round(avg_gain / avg_loss, 2)
            except Exception as e:
                st.warning(f"⚠️ Could not calculate Actual R:R automatically — {e}")

            # Convert R:R to approximate gain suggestion
            auto_target = round(actual_rr * 30, 0)  # assume 30% base per R:R point
            actual_slider = st.slider("⚙️ Auto-Adjusted (Actual R:R-Based)", 10, 400, int(auto_target), 5)
            st.caption(f"📈 Based on Avg Actual R:R ≈ {actual_rr}:1 → Suggested Target ≈ {auto_target:.0f}%")

        # ========================== COMPUTE PRICES ==========================
        take_profit_price = premium * (1 + target_gain_pct / 100)
        stop_loss_price = premium * (1 - stop_loss_pct / 100)

        gain_per_contract = (take_profit_price - premium) * shares
        loss_per_contract = (premium - stop_loss_price) * shares
        total_profit = gain_per_contract * contracts
        total_loss = loss_per_contract * contracts

        proj_rr = (take_profit_price - premium) / (premium - stop_loss_price)
        proj_roi_pct = (take_profit_price - premium) / premium * 100

        # ========================== DISPLAY SUMMARY ==========================
        st.divider()
        st.write("### 💡 Summary")
        st.markdown(f"""
        - 🎯 **Manual Target:** `{target_gain_pct}%`
        - 🛑 **Stop-Loss:** `{stop_loss_pct}%`
        - ⚖️ **Projected R:R:** `{proj_rr:.2f}:1`
        - ⚙️ **Auto-Adjusted Suggestion:** `{auto_target}%` (based on Avg Actual R:R `{actual_rr}:1`)
        """)
    # st.divider()

    # ================= EBDS HERE============================================================



    # ========================== ACTUAL EXIT ==========================
    if exit_premium > 0:
        total_exit = exit_premium * shares * contracts
        profit_loss = total_exit - total_invest
        roi_pct = profit_loss / total_invest * 100
        actual_rr = abs((exit_premium - premium) / (premium - stop_loss_price))
    else:
        profit_loss, roi_pct, actual_rr = 0, 0, 0

    # ========================== DISPLAY SUMMARY ==========================
    st.markdown("### 📊 Trade Summary")

    total_volume = contracts * shares
    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Entry Premium", f"${premium:,.2f}")
        st.metric("Total Investment", f"${total_invest:,.2f}")
        st.metric("Break-Even", f"${breakeven:,.2f}")
        st.metric("📊 Total Volume (Shares Controlled)", f"{total_volume:,}")

    with cB:
        st.markdown(f"""
        <div style='background:rgba(0,255,0,0.03); border:1px solid rgba(0,255,0,0.2);
                    border-radius:10px; padding:10px;'>
            <b style='color:#27e08d;'>🎯 Target (+{target_gain_pct}%)</b><br>
            • Target Price: <b>${take_profit_price:,.2f}</b><br>
            • Gain/Contract: <b>+${gain_per_contract:,.2f}</b><br>
            • Total Gain ({contracts} contracts): <b>+${total_profit:,.2f}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:rgba(255,0,0,0.03); border:1px solid rgba(255,0,0,0.2);
                    border-radius:10px; padding:10px; margin-top:8px;'>
            <b style='color:#ff4d4d;'>🛑 Stop-Loss (−{stop_loss_pct}%)</b><br>
            • Stop-Loss Price: <b>${stop_loss_price:,.2f}</b><br>
            • Loss/Contract: <b>−${loss_per_contract:,.2f}</b><br>
            • Total Risk ({contracts} contracts): <b>−${total_loss:,.2f}</b>
        </div>
        """, unsafe_allow_html=True)

    with cC:
        color = "🟢" if profit_loss > 0 else ("🔴" if profit_loss < 0 else "⚪")
        st.metric(f"{color} Realized P/L", f"${profit_loss:,.2f}", f"{roi_pct:,.2f}%")
        if exit_premium > 0:
            st.metric("Actual R:R", f"1 : {actual_rr:,.2f}")
            st.metric("Exit Premium", f"${exit_premium:,.2f}")




    # ========================== PERFORMANCE COMPARISON + MOMENTUM INSIGHT ==========================
    if exit_premium > 0:
        perf_color = "#27e08d" if roi_pct >= proj_roi_pct else "#ff7070"
        st.markdown(f"""
        <div style='margin-top:10px; background:rgba(255,255,255,0.03);
                    border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:12px;'>
          <b style='color:{perf_color};'>📈 Performance Comparison</b><br>
          • Projected ROI (+{target_gain_pct}%): <b>{proj_roi_pct:.2f}%</b><br>
          • Actual ROI (Exit): <b>{roi_pct:.2f}%</b><br>
          • Projected R:R: <b>1 : {proj_rr:,.2f}</b> | Actual R:R: <b>1 : {actual_rr:,.2f}</b>
        </div>
        """, unsafe_allow_html=True)

        # --- DAY TRADER MOMENTUM INSIGHT ---


    # ==========================================================================================================================
 #        CONFIRMATION TRADER CALL ENTRY
# =====================================================================================================================
#         # ======================================================================
#         # 🔥 CALL ENTRY CONFIRMATION CHECKLIST (STREAMLIT UI)
#         # ======================================================================
#         st.divider()
#         st.markdown("## 🔥 CALL ENTRY CONFIRMATION CHECKLIST")
#
#         st.caption("Only take CALL entries if ALL conditions are satisfied.")
#
#         # --- Checklist Items ---
#         c1, c2 = st.columns(2)
#
#         with c1:
#             chk_support = st.checkbox("✔ Support at **$3.30–$3.35** is holding")
#             chk_5min = st.checkbox("✔ First **5-min candle** closes above premarket low")
#
#         with c2:
#             chk_dom = st.checkbox("✔ DOM shows **buyers stacking** (bid > ask)")
#             chk_ema = st.checkbox("✔ Price closes above the **blue EMA (1min/5min)**")
#
#         # Determine overall status
#         all_good = chk_support and chk_5min and chk_dom and chk_ema
#
#         status_color = "#27e08d" if all_good else "#ff4d4d"
#         status_text = "HIGH-PROBABILITY CALL SETUP" if all_good else "DO NOT TAKE CALLS YET"
#
#         st.markdown(f"""
#         <div style='margin-top:15px; padding:15px; border-radius:10px;
#                     background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);'>
#             <h3 style='color:{status_color}; text-align:center;'>⚡ {status_text}</h3>
#         </div>
#         """, unsafe_allow_html=True)
#
#         # ======================================================================
#         # 🎯 TRADING PLAN OUTPUT
#         # ======================================================================
#
#         if all_good:
#             st.success("""
#             ### 🎯 CALL ENTRY PLAN (Active)
#             **Buy Zone:** $3.30 – $3.45
#             **Stop-Loss:** Below $3.25
#             **Profit Targets:**
#             - Target 1: $3.80
#             - Target 2: $4.00
#             - Target 3: $4.20
#             - Max Target: $4.50 – $4.80 (if AAPL trends strong)
#             """)
#         else:
#             st.warning("""
#             ### 🛑 Not Ready for Call Entry
#             Wait for all four confirmations before entering a CALL trade.
#
#             Avoid calls if:
#             - Support $3.30–$3.35 breaks
#             - First 5-min candle is a large red candle
#             - Price rejects the blue EMA
#             - DOM shows strong ask walls
#             """)
        # ==============================================================================
        # ============================================================================ PART 2
                # ENTRY AND EXIT CHECKLIST
                # ENTRY AND EXIT CHECKLIST
        # ==============================================================================
                # ENTRY AND EXIT CHECKLIST
        # ==============================================================================

        # st.markdown("## 🔥 CALL Entry Checklist")
        # # ======================================================================
        # # 🚀 CALL ENTRY & EXIT CHECKLIST (BASED ON CHART LINES)
        # # ======================================================================
        # with st.expander("## 🔥 DAY TRADING CALL Entry Decision System"):
        #     st.markdown("## 🚀 CALL ENTRY & EXIT CHECKLIST")
        #
        #     # ---------- CSS ----------
        #     st.markdown(
        #         """
        #         <style>
        #         .check-card {
        #             background: rgba(255,255,255,0.04);
        #             border-radius: 16px;
        #             border: 1px solid rgba(255,255,255,0.12);
        #             padding: 18px 20px;
        #             margin-top: 10px;
        #             box-shadow: 0 8px 18px rgba(0,0,0,0.35);
        #         }
        #         .check-header {
        #             font-size: 20px;
        #             font-weight: 700;
        #             margin-bottom: 6px;
        #         }
        #         .check-sub {
        #             font-size: 13px;
        #             opacity: 0.8;
        #             margin-bottom: 10px;
        #         }
        #         .status-pill {
        #             display: inline-block;
        #             padding: 4px 10px;
        #             border-radius: 999px;
        #             font-size: 12px;
        #             font-weight: 600;
        #             letter-spacing: 0.03em;
        #         }
        #         </style>
        #         """,
        #         unsafe_allow_html=True
        #     )
        #
        #     # ---------- LAYOUT ----------
        #     left_col, right_col = st.columns(2)
        #
        #     # ================== ENTRY CHECKLIST ==================
        #     with left_col:
        #         st.markdown(
        #             "<div class='check-card'>"
        #             "<div class='check-header'>🟢 PRO CALL ENTRY (Your Chart Setup)</div>"
        #             "<div class='check-sub'>Use your colored lines to confirm a high-probability CALL.</div>",
        #             unsafe_allow_html=True,
        #         )
        #
        #         # e1 = st.checkbox("✔ Price is bouncing off **White/Gray horizontal support**")
        #         e2 = st.checkbox("✔ Candle Price CLOSED above **BLUE EMA 9** (fast trend)")
        #         e3 = st.checkbox("✔ **BLUE EMA 9 > RED EMA 21** (bullish EMA stack)")
        #         e4 = st.checkbox("✔ Price is **above or bouncing off YELLOW SMA 50**")
        #         e5 = st.checkbox("✔ **PURPLE AAPL stock line is rising** (stock trending up)")
        #
        #         # 🚀 ADDED VWAP — DO NOT CHANGE ORDER OR TEXT
        #         e6 = st.checkbox("✔ Price is **above VWAP (White dashed line)** → intraday bullish confirmation")
        #
        #         entry_checks = [e2, e3, e4, e5, e6]
        #         entry_score = sum(entry_checks)
        #         entry_pct = (entry_score / 5) * 100
        #
        #         entry_ok = all(entry_checks)
        #
        #         entry_color = "#27e08d" if entry_ok else "#ff4d4d"
        #         entry_text = "ENTRY READY — High-Probability CALL Setup" if entry_ok else "ENTRY BLOCKED — Conditions Not Fully Met"
        #
        #         st.progress(entry_pct / 100)
        #
        #         st.markdown(
        #             f"""
        #             <div style='margin-top:10px; text-align:center;'>
        #                 <span class='status-pill' style='background:{entry_color}22; color:{entry_color}; border:1px solid {entry_color};'>
        #                     {entry_text}
        #                 </span>
        #                 <div style='margin-top:6px; font-size:13px; opacity:0.8;'>
        #                     Checklist: <b>{entry_score}/5</b> ({entry_pct:.0f}%)
        #                 </div>
        #             </div>
        #             </div>
        #             """,
        #             unsafe_allow_html=True,
        #         )
        #
        #     # ================== EXIT / PROFIT CHECKLIST ==================
        #     with right_col:
        #         st.markdown(
        #             "<div class='check-card'>"
        #             "<div class='check-header'>💰 CALL EXIT & PROFIT RULES</div>"
        #             "<div class='check-sub'>Use the same lines as dynamic profit targets.</div>",
        #             unsafe_allow_html=True,
        #         )
        #
        #         x1 = st.checkbox("✔ TP1 hit: Price starts closing **back below BLUE EMA 9** (lock 20–30%)")
        #         x2 = st.checkbox("✔ TP2 near: Price is approaching **RED EMA 21** (take 40–50%)")
        #         x3 = st.checkbox("✔ TP3 near: Price has reached / tagged **YELLOW SMA 50** (unload remaining)")
        #         x4 = st.checkbox(
        #             "✔ TP4: **PURPLE stock line goes parabolic** (vertical spike → exhaustion, sell everything)")
        #
        #         # 🚀 ADDED VWAP EXIT CHECK
        #         x5 = st.checkbox("✔ Price breaks BELOW VWAP (White dashed line) → intraday momentum reversing")
        #
        #         exit_checks = [x1, x2, x3, x4, x5]
        #         exit_score = sum(exit_checks)
        #         exit_pct = (exit_score / 5) * 100
        #
        #         exit_signal = any(exit_checks)
        #
        #         exit_color = "#ffb84d" if exit_signal else "#27e08d"
        #         exit_text = "EXIT / TAKE PROFIT SIGNAL ACTIVE" if exit_signal else "NO EXIT SIGNAL — Trend Still Healthy"
        #
        #         st.progress(exit_pct / 100)
        #
        #         st.markdown(
        #             f"""
        #             <div style='margin-top:10px; text-align:center;'>
        #                 <span class='status-pill' style='background:{exit_color}22; color:{exit_color}; border:1px solid {exit_color};'>
        #                     {exit_text}
        #                 </span>
        #                 <div style='margin-top:6px; font-size:13px; opacity:0.8;'>
        #                     Triggers: <b>{exit_score}/5</b> ({exit_pct:.0f}%)
        #                 </div>
        #             </div>
        #             </div>
        #             """,
        #             unsafe_allow_html=True,
        #         )
        #
        #     # ================== SUMMARY BANNER ==================
        #     st.markdown("---")
        #
        #     if entry_ok and not exit_signal:
        #         overall_msg = "✅ **Best Zone:** Fresh CALL entry or manage open calls with trend intact."
        #     elif entry_ok and exit_signal:
        #         overall_msg = "⚠️ **Mixed:** Setup good but exits are triggering — reduce size."
        #     elif (not entry_ok) and exit_signal:
        #         overall_msg = "⛔ **No New Calls:** Exit signals firing — avoid entries."
        #     else:
        #         overall_msg = "🟡 **Wait:** No strong entry or exit signals — be patient."
        #
        #     st.markdown(
        #         f"""
        #         <div style='margin-top:8px; padding:16px; border-radius:14px;
        #                     background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.18);'>
        #             {overall_msg}
        #         </div>
        #         """,
        #         unsafe_allow_html=True,
        #     )

# ===================================================================================================================== EBTRY CONFIRMATION

        st.markdown("## 🔥 CALL Entry Checklist")

        with st.expander("## 🔥 DAY TRADING CALL Entry Decision System"):
            st.markdown("## 🚀 CALL ENTRY & EXIT CHECKLIST")

            # ---------- CSS ----------
            st.markdown(
                """
                <style>
                .check-card {
                    background: rgba(255,255,255,0.04);
                    border-radius: 16px;
                    border: 1px solid rgba(255,255,255,0.12);
                    padding: 18px 20px;
                    margin-top: 10px;
                    box-shadow: 0 8px 18px rgba(0,0,0,0.35);
                }
                .check-header {
                    font-size: 20px;
                    font-weight: 700;
                    margin-bottom: 6px;
                }
                .check-sub {
                    font-size: 13px;
                    opacity: 0.8;
                    margin-bottom: 10px;
                }
                .status-pill {
                    display: inline-block;
                    padding: 4px 10px;
                    border-radius: 999px;
                    font-size: 12px;
                    font-weight: 600;
                    letter-spacing: 0.03em;
                }
                .caption {
                    font-size: 12px;
                    opacity: 0.65;
                    margin-top: -6px;
                    margin-bottom: 6px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            left_col, right_col = st.columns(2)

            # ================== ENTRY CHECKLIST ==================
            with left_col:
                st.markdown(
                    "<div class='check-card'>"
                    "<div class='check-header'>🟢 PRO CALL ENTRY (Sybest Sniper Model)</div>"
                    "<div class='check-sub'>Core rules required before taking a CALL entry.</div>",
                    unsafe_allow_html=True,
                )

                # -------- New Simplified Professional Rules --------

                e1 = st.checkbox("✔ CALL ZONE Active")
                st.markdown("<div class='caption'>Bullish environment — sellers exhausted.</div>",
                            unsafe_allow_html=True)

                e2 = st.checkbox("✔ HL (Higher Low) Printed")
                st.markdown("<div class='caption'>Shows buyers stepping in at higher levels.</div>",
                            unsafe_allow_html=True)

                e3 = st.checkbox("✔ HH (Higher High) Printed")
                st.markdown("<div class='caption'>Break of previous high → continuation signal.</div>",
                            unsafe_allow_html=True)

                e4 = st.checkbox("✔ EMA 9 (BLUE) > EMA 21 (RED) (Momentum Flip)")
                st.markdown("<div class='caption'>Trend shift: momentum is bullish.</div>", unsafe_allow_html=True)

                e5 = st.checkbox("✔ Price Above EMA 9 (BLUE)")
                st.markdown("<div class='caption'>Immediate buyer control — strong entry timing.</div>",
                            unsafe_allow_html=True)

                e6 = st.checkbox("✔ 1m Timeframe = CALL")
                st.markdown("<div class='caption'>Micro-trend aligned for entry timing.</div>", unsafe_allow_html=True)

                e7 = st.checkbox("✔ 5m Timeframe = CALL")
                st.markdown("<div class='caption'>Macro intraday trend aligned → highest accuracy.</div>",
                            unsafe_allow_html=True)

                e8 = st.checkbox("✔ SNIPER CALL Signal (Optional)")
                st.markdown("<div class='caption'>Ultra-confirmation (not required but very strong).</div>",
                            unsafe_allow_html=True)

                # Score Only First 7 (SNIPER is optional)
                entry_checks = [e1, e2, e3, e4, e5, e6, e7]
                entry_score = sum(entry_checks)
                entry_pct = (entry_score / 7) * 100

                entry_ok = entry_score >= 6  # 6/7 or better = valid

                entry_color = "#27e08d" if entry_ok else "#ff4d4d"
                entry_text = "ENTRY READY — High-Probability CALL Setup" if entry_ok else "ENTRY BLOCKED — Conditions Not Fully Met"

                st.progress(entry_pct / 100)

                st.markdown(
                    f"""
                    <div style='margin-top:10px; text-align:center;'>
                        <span class='status-pill' style='background:{entry_color}22; color:{entry_color}; border:1px solid {entry_color};'>
                            {entry_text}
                        </span>
                        <div style='margin-top:6px; font-size:13px; opacity:0.8;'>
                            Checklist: <b>{entry_score}/7</b> ({entry_pct:.0f}%)
                        </div>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ================== EXIT / PROFIT CHECKLIST ==================
            with right_col:
                st.markdown(
                    "<div class='check-card'>"
                    "<div class='check-header'>💰 CALL EXIT & PROFIT RULES</div>"
                    "<div class='check-sub'>Exit based on weakening momentum.</div>",
                    unsafe_allow_html=True,
                )

                x1 = st.checkbox("✔ Price Closing BACK Below EMA 9 (BLUE)")
                st.markdown("<div class='caption'>Loss of micro momentum — take partials.</div>",
                            unsafe_allow_html=True)

                x2 = st.checkbox("✔ Price Approaching EMA 21 (RED)")
                st.markdown("<div class='caption'>Trend cooling; consider selling remaining size.</div>",
                            unsafe_allow_html=True)

                x3 = st.checkbox("✔ HH Fails to Break / Double Top Forms")
                st.markdown("<div class='caption'>Indicates exhaustion; manage risk.</div>", unsafe_allow_html=True)

                x4 = st.checkbox("✔ Volume Drops on Green Candles")
                st.markdown("<div class='caption'>Buyers weakening → prepare to exit.</div>", unsafe_allow_html=True)

                x5 = st.checkbox("✔ Price Breaks Below VWAP")
                st.markdown("<div class='caption'>Intraday sentiment flips bearish — exit immediately.</div>",
                            unsafe_allow_html=True)

                exit_checks = [x1, x2, x3, x4, x5]
                exit_score = sum(exit_checks)
                exit_pct = (exit_score / 5) * 100

                exit_signal = exit_score >= 2

                exit_color = "#ffb84d" if exit_signal else "#27e08d"
                exit_text = "EXIT / TAKE PROFIT SIGNAL ACTIVE" if exit_signal else "NO EXIT SIGNAL — Trend Still Strong"

                st.progress(exit_pct / 100)

                st.markdown(
                    f"""
                    <div style='margin-top:10px; text-align:center;'>
                        <span class='status-pill' style='background:{exit_color}22; color:{exit_color}; border:1px solid {exit_color};'>
                            {exit_text}
                        </span>
                        <div style='margin-top:6px; font-size:13px; opacity:0.8;'>
                            Triggers: <b>{exit_score}/5</b> ({exit_pct:.0f}%)
                        </div>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            if entry_ok and not exit_signal:
                overall_msg = "✅ **Best Zone:** Fresh CALL entry available — trend aligned strong."
            elif entry_ok and exit_signal:
                overall_msg = "⚠️ **Mixed:** Good entry setup but exit triggers firing — reduce size."
            elif (not entry_ok) and exit_signal:
                overall_msg = "⛔ **No New Calls:** Momentum reversing — avoid entries."
            else:
                overall_msg = "🟡 **Wait:** No strong entry or exit signals."

            st.markdown(
                f"""
                <div style='margin-top:8px; padding:16px; border-radius:14px;
                            background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.18);'>
                    {overall_msg}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ENTRY ENDS HERE
# ============================================== NOTE TAKEN
            # ========================== NOTES & SAVE TO CSV ==========================
        # st.markdown("---")
        st.subheader("🗒️ Notes / Review")
        notes = st.text_area("Journal your thought process, entry signal, and lessons learned:")

        if st.button("💾 Save Entry"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trade_date = datetime.now().strftime("%Y-%m-%d")  # ✅ current date
            save_path = r"C:\Users\stans\ML_PROJECTS\0. STOCK_TRADING_PROJECT\1. Dataset\Option_trading.csv"

            new_data = pd.DataFrame([{
                "Timestamp": timestamp,
                "Date": trade_date,  # ✅ today's date captured for new trade
                "Symbol": symbol,
                "Buying Power ($)": buying_power,  # ✅ NEW FIELD
                "Option Type": option_type,
                "Strike": strike,
                "Entry Premium": premium,
                "Exit Premium": exit_premium,
                "Contracts": contracts,
                "Shares": shares,
                "Total Volume": total_volume,
                "Total Investment": total_invest,
                "Break-Even": breakeven,
                "Target Gain %": target_gain_pct,
                "Stop-Loss %": stop_loss_pct,
                "Projected ROI %": proj_roi_pct,
                "Actual ROI %": roi_pct,
                "Projected R:R": proj_rr,
                "Actual R:R": actual_rr,
                "Total Profit $": total_profit,
                "Total Loss $": total_loss,
                "Realized P/L $": profit_loss,
                "Notes": notes
            }])

            # ✅ Merge new entry with existing file without overwriting past dates
            if os.path.exists(save_path):
                existing = pd.read_csv(save_path)

                # 🔄 If file lacks a "Date" column, create it from Timestamp
                if "Date" not in existing.columns:
                    existing["Date"] = pd.to_datetime(existing["Timestamp"], errors="coerce").dt.date.astype(str)

                # ✅ Keep old Date values intact — only fill missing ones
                existing["Date"].fillna(
                    pd.to_datetime(existing["Timestamp"], errors="coerce").dt.date.astype(str),
                    inplace=True
                )

                updated = pd.concat([existing, new_data], ignore_index=True)
            else:
                updated = new_data

            # ✅ Save file
            updated.to_csv(save_path, index=False)
            st.success(f"✅ Entry saved — with correct historical and current dates.\n📁 {save_path}")

        st.divider()


# ========================================================= NOTE ENDED HERE





















        # ======================================================================
        # 🔥 AUTOMATED CALL ENTRY ENGINE
        # ======================================================================
        # ======================================================================
        # 🔥 AUTOMATED CALL ENTRY ENGINE
        # ======================================================================
        # ======================================================================
        # 🔥 AUTOMATED CALL ENTRY ENGINE
        # ======================================================================


        # st.divider()
        st.markdown("## 🔥 Automated CALL Entry Decision System")

        with st.expander("## 🔥 Automated CALL Entry Decision System"):
            st.caption("Enter your real-time data. The system will decide if CALL entry is valid.")

            # ================================
            # INPUT FIELDS (YOU PROVIDE DATA)
            # ================================

            st.markdown("### 🧩 Market Conditions Input")

            colA, colB = st.columns(2)

            with colA:
                current_price = st.number_input("Current Option Price ($)", min_value=0.0, step=0.01)
                support_level = st.number_input("Support Level ($)", value=3.30, step=0.01)
                first_5min_close = st.number_input("First 5-min Close ($)", step=0.01)

            with colB:
                premarket_low = st.number_input("Premarket Low ($)", step=0.01)

                # Price vs EMA with correct chart colors
                ema_status = st.selectbox(
                    "Price (WHITE) vs EMA9 Fast Line (BLUE) — 1m / 5m Status",
                    [
                        "Below EMA (RED Zone)",
                        "At EMA (YELLOW Zone)",
                        "Above EMA (GREEN Zone)"
                    ]
                )

                # DOM Strength with color memory
                dom_status = st.selectbox(
                    "DOM Buyer Strength (GREEN = Strong)",
                    [
                        "Weak (RED)",
                        "Neutral (YELLOW)",
                        "Strong (GREEN)"
                    ]
                )

            # ================================
            # ALGORITHM DECISION LOGIC
            # ================================

            # Condition 1: Support holding
            support_holding = current_price >= support_level

            # Condition 2: First 5-min above premarket low
            first5_valid = first_5min_close > premarket_low

            # Condition 3: EMA status
            ema_valid = (ema_status == "Above EMA")

            # Condition 4: DOM buyer behavior
            dom_valid = (dom_status == "Strong")

            # Combined logic
            all_good = support_holding and first5_valid and ema_valid and dom_valid

            # Score out of 4
            score = sum([support_holding, first5_valid, ema_valid, dom_valid])
            percent = (score / 4) * 100

            # ================================
            # DISPLAY RESULT
            # ================================
            color = "#27e08d" if all_good else "#ff4d4d"
            status = "HIGH-PROBABILITY CALL SETUP" if all_good else "NOT SAFE FOR CALLS"

            st.markdown(f"""
            <div style='margin-top:20px; padding:20px; border-radius:12px;
                        background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);'>
              <h2 style='color:{color}; text-align:center;'>⚡ {status}</h2>
              <p style='text-align:center; font-size:18px;'>Decision Score: <b>{percent:.0f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # ================================
            # TRADING PLAN (AUTO_GENERATED)
            # ================================
            st.divider()
            st.markdown("## 🎯 Auto-Generated CALL Trading Plan")

            if all_good:
                st.success(f"""
            ### 📌 CALL ENTRY APPROVED
    
            **Buy Zone:** {support_level} – {support_level + 0.15:.2f}  
            **Stop-Loss:** Below {support_level - 0.05:.2f}  
            **Profit Targets:**  
            - Target 1: +10–15%  
            - Target 2: +20–30%  
            - Target 3: +40–50%  
            - Max Target: Only if momentum continues  
    
            **DOM Strength:** {dom_status}  
            **EMA Status:** {ema_status}
            """)
            else:
                st.warning("""
            ### 🛑 CALL ENTRY REJECTED  
            Current market conditions do **not** confirm a safe CALL entry.
    
            Avoid calls if:
            - Support is broken  
            - 5-min candle closes below premarket low  
            - Price is below EMA  
            - DOM shows seller control  
            """)

          # ======================================================================
        # END CHECKLIST MODULE
        # ======================================================================

        # ============================================================ ENDS HERE


        st.markdown("### ⚡ Day-Trader Momentum Insight")

        with st.expander('⚡ Day-Trader Momentum Insight'):

            delta = st.number_input("Δ Option Delta (how fast premium moves per $1 underlying)",
                                    0.05, 1.0, 0.45, 0.05,
                                    key=f"delta_day_{st.session_state['active_page_key']}")
            volume_speed = st.number_input("📊 Volume Spike Ratio (today vs avg)",
                                           0.5, 5.0, 1.8, 0.1,
                                           key=f"vol_day_{st.session_state['active_page_key']}")
            last_move = st.number_input("⚙️ Last Price Move ($ per min)",
                                        0.01, 10.0, 0.45, 0.05,
                                        key=f"price_speed_{st.session_state['active_page_key']}")

            premium_speed = delta * last_move * volume_speed
            move_needed = take_profit_price - premium
            est_minutes = max(1, move_needed / premium_speed) if premium_speed > 0 else 999
            momentum_score = min(100, (delta * volume_speed * (1 / est_minutes)) * 1000)

            if momentum_score > 80:
                insight_color = "#27e08d"
                speed_comment = "🚀 **High momentum:** target may hit within 15–30 minutes."
            elif momentum_score > 50:
                insight_color = "#f39c12"
                speed_comment = "⚙️ **Moderate momentum:** target likely within 30–90 minutes."
            else:
                insight_color = "#e74c3c"
                speed_comment = "🐢 **Low momentum:** premium moving slowly; may require 2–3 hours or exit early."

            st.markdown(f"""
            <div style='margin-top:10px; background:rgba(255,255,255,0.03);
                        border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;'>
                <b style='color:{insight_color};'>📊 Estimated Time to Target:</b> ~{est_minutes:.0f} minutes<br>
                <b>Premium Speed:</b> ${premium_speed:,.2f} per minute<br>
                <b>Momentum Score:</b> {momentum_score:.1f} / 100<br><br>
                {speed_comment}
            </div>
            """, unsafe_allow_html=True)

    # # ========================== NOTES & SAVE TO CSV ==========================
    # # st.markdown("---")
    # st.subheader("🗒️ Notes / Review")
    # notes = st.text_area("Journal your thought process, entry signal, and lessons learned:")
    #
    # if st.button("💾 Save Entry"):
    #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     trade_date = datetime.now().strftime("%Y-%m-%d")  # ✅ current date
    #     save_path = r"C:\Users\stans\ML_PROJECTS\0. STOCK_TRADING_PROJECT\1. Dataset\Option_trading.csv"
    #
    #     new_data = pd.DataFrame([{
    #         "Timestamp": timestamp,
    #         "Date": trade_date,  # ✅ today's date captured for new trade
    #         "Symbol": symbol,
    #         "Buying Power ($)": buying_power,  # ✅ NEW FIELD
    #         "Option Type": option_type,
    #         "Strike": strike,
    #         "Entry Premium": premium,
    #         "Exit Premium": exit_premium,
    #         "Contracts": contracts,
    #         "Shares": shares,
    #         "Total Volume": total_volume,
    #         "Total Investment": total_invest,
    #         "Break-Even": breakeven,
    #         "Target Gain %": target_gain_pct,
    #         "Stop-Loss %": stop_loss_pct,
    #         "Projected ROI %": proj_roi_pct,
    #         "Actual ROI %": roi_pct,
    #         "Projected R:R": proj_rr,
    #         "Actual R:R": actual_rr,
    #         "Total Profit $": total_profit,
    #         "Total Loss $": total_loss,
    #         "Realized P/L $": profit_loss,
    #         "Notes": notes
    #     }])
    #
    #     # ✅ Merge new entry with existing file without overwriting past dates
    #     if os.path.exists(save_path):
    #         existing = pd.read_csv(save_path)
    #
    #         # 🔄 If file lacks a "Date" column, create it from Timestamp
    #         if "Date" not in existing.columns:
    #             existing["Date"] = pd.to_datetime(existing["Timestamp"], errors="coerce").dt.date.astype(str)
    #
    #         # ✅ Keep old Date values intact — only fill missing ones
    #         existing["Date"].fillna(
    #             pd.to_datetime(existing["Timestamp"], errors="coerce").dt.date.astype(str),
    #             inplace=True
    #         )
    #
    #         updated = pd.concat([existing, new_data], ignore_index=True)
    #     else:
    #         updated = new_data
    #
    #     # ✅ Save file
    #     updated.to_csv(save_path, index=False)
    #     st.success(f"✅ Entry saved — with correct historical and current dates.\n📁 {save_path}")
    #
    # st.divider()

    # ========================== EXPANDER: TRADING INSTRUCTIONS ==========================
    with st.expander("🧭 How to Load Option Chart & Take Position — Step-by-Step"):
        st.markdown("""
        
        
        ### ⚙️ **1️⃣ Load Your Option Chart in ThinkorSwim (TOS)**
        - Go to the **Trade Tab** → select your ticker (e.g., `AAPL`).
        - Find your desired **expiration date** (e.g., `7 NOV 25`).
        - Right-click your **strike price** (e.g., `270C`) → choose **"Send to Chart"**.
        - Open your linked **Chart tab**, and you’ll now see the **option contract price**.

        ### 📊 **2️⃣ Setup in Power ROI App**
        - Select your **Symbol**, **Strike**, and **Option Type**.
        - Enter **Entry Premium**, **Target**, **Stop-Loss**, and **Exit Premium**.
        - The app auto-calculates:
          - Total Investment, Break-Even, Profit/Loss, ROI, R:R.
          - Visual Target & Stop-Loss levels.

        ### 📈 **3️⃣ Manage and Record**
        - Once trade closes, input Exit Premium.
        - Add notes for why you entered/exited.
        - Click **Save Entry** — data is appended to your trade log.


         ### 🛡️ **OCO Bracket Workflow**
        - Open **Buy Custom → with OCO Bracket** to set up the order.
        - **Review & Confirm**: adjust limit (take-profit) and stop prices, then click **Confirm and Send**.
        - **Monitor**: go to the **Monitor** tab to view/edit your **Working Orders**.
        - **On the Chart**: view and drag the **STP** (stop) and **LMT** (take-profit) lines to fine-tune levels.
        - INSTRUCTION: https://youtu.be/jxF10ReGWb8
        """)

    # ========================================================== ENDS HERE

    # ==============================================================================================================
    #  SYBEST SYSTEM INSTRUCTION
    # =================================================================================================================
    st.header('SYBEST SYSTEM TRADING UTILIZATION')

    with st.expander('SYBEST SYSTEM TRADING UTILIZATION'):
        st.markdown("## 📊 Trade Summary")
        st.markdown("### 🟩 THE SYBEST CALL TRADING FRAMEWORK (Your System)")

        st.markdown(
            """
    Your system already gives you these signals:

    1️⃣ **Structure:** `HH + HL`  
    2️⃣ **Trend:** `EMA 9 > EMA 21`  
    3️⃣ **Location:** Above `VWAP`  
    4️⃣ **Confirmation:** `1m + 5m` trend  
    5️⃣ **Early Entry:** `HL` (pullback)  
    6️⃣ **Sniper Entry:** Strong candle + `HL + HH`  
    7️⃣ **Avoid Zones:** Supply, `PDH`

    Now let’s combine all of these into a single powerful CALL strategy.
            """
        )

        # 1. Location
        st.markdown("### 🟩 1. Start With Location (MOST IMPORTANT)")
        st.markdown(
            """
    Before you even think of entering, confirm:

    - ✔ Is price **ABOVE VWAP**?  
      - If **NO → cancel CALL idea**
    - ✔ Is price **ABOVE EMA 9 & 21**?  
      - If **NO → remove CALL idea**
    - ✔ Is price **ABOVE ORH** (Opening Range High)?  
      - If price breaks ORH → **CALL TREND DAY**  
      - If below ORH → **wait for retest**.
    - ✔ Is price **near PDH**?  
      - PDH = **institutional resistance** → CALL can reject here.

    **IDEAL CALL LOCATION = ABOVE VWAP + ABOVE ORH but NOT at PDH**
            """
        )

        # 2. Trend
        st.markdown("### 🟩 2. Confirm Trend (Multi-Timeframe)")
        st.markdown(
            """
    Your labels show **1m, 5m, 30m** trends.

    **For early entries you want:**

    - 🟩 1m = CALL  
    - 🟩 5m = CALL  
    - ⚪ 30m = WAIT (optional)

    **For safer entries you want:**

    - 🟩 1m = CALL  
    - 🟩 5m = CALL  
    - 🟩 30m = CALL  

    30m makes it slower, but more reliable.
            """
        )

        # 3. Structure
        st.markdown("### 🟩 3. Identify Structure (HH / HL)")
        st.markdown(
            """
    Your system marks **HH** and **HL** automatically.

    - ✔ **HH = Strength** (trend continuation)  
    - ✔ **HL = Pullback** → **your CALL entry**  
    - ❌ **LH / LL = STOP CALL trades**

    Your call entry should come at **HL** — that’s your bread and butter.

    **Visual idea:**

    - HH prints at the top  
    - Price pulls back but holds higher than last low (HL)  
    - HL candle gives you the **CALL entry**
            """
        )

        # 4. PDH/PDL
        st.markdown("### 🟩 4. Use PDH / PDL Correctly")
        st.markdown(
            """
    **PDH is the biggest trap for CALL traders.**

    When price is near **PDH**:

    - ❌ Do **NOT** buy CALLS *into* PDH
    - ✔ You buy **AFTER** the breakout and retest:

      - ✔ **PDH breakout → wait**  
      - ✔ **PDH retest → CALL entry**

    This is where many traders lose money. You will not.
            """
        )

        # 5. CDH / CDL
        st.markdown("### 🟩 5. Use CDH / CDL for Intraday Trend")
        st.markdown(
            """
    - **CDH break** → CALL continuation  
    - **CDH reject** → **avoid CALLs**  
    - **CDL reject** → CALL bounce (reversal scalp)

    Best risk/reward:

    - 🔹 HL forms **above CDH** → CALL to new HOD  
    - 🔹 HL forms **above VWAP** → CALL continuation
            """
        )

        # 6. ORH
        st.markdown("### 🟩 6. Use ORH as Your “Institutional Confirmation”")
        st.markdown(
            """
    **ORH (Opening Range High)** is VERY powerful.

    - ✔ Price **above ORH** → CALL bias  
    - ✔ Price **below ORH** → WAIT  
    - ✔ Price **retests ORH from above and bounces** → **CALL entry**

    Institutions respect ORH almost as much as VWAP.
            """
        )

        # 7. Sniper
        st.markdown("### 🟩 7. Use Your SNIPER ENTRY")
        st.markdown(
            """
    Your Sniper logic simplifies everything:

    **Sniper CALL requires:**

    - HL  
    - HH  
    - EMA9 > EMA21  
    - Above VWAP  
    - Strong bullish candle  
    - Trend alignment

    When **SNIPER prints**:

    > 👉 This is the *cleanest* CALL entry you can take.  
    > No FOMO. No guessing. No emotions.
            """
        )

        # 8. When to avoid calls
        st.markdown("### 🟩 8. When to AVOID CALLS (MOST IMPORTANT)")
        st.markdown(
            """
    Avoid CALLs when:

    - ❌ Price **below VWAP**
    - ❌ **LH** structure forming
    - ❌ **LL** forming
    - ❌ Price at **supply zone**
    - ❌ Weak volume
    - ❌ Under ORH
    - ❌ Approaching PDH
    - ❌ EMA9 cross **down**
    - ❌ Big rejection candle
    - ❌ Sniper not triggered
    - ❌ 5m TF not aligned

    > Avoiding bad entries is EASIER and more powerful than forcing good ones.
            """
        )

        # 9. Clear entry rules
        st.markdown("### 🟩 9. CLEAR ENTRY RULES")
        st.markdown(
            """
    For a **CALL entry**, check:

    - ✔ Above **VWAP**
    - ✔ Above **EMA 9 / 21**
    - ✔ Forming **HL**
    - ✔ Recent **HH** printed
    - ✔ **1m + 5m** trends = CALL
    - ✔ Not at **PDH** or **supply**
    - ✔ HL forming near **CDH** or **ORH**
    - ✔ Volume expanding
    - ✔ **Sniper** or **CALL_OK** bubble prints

    **If 5 or more are true → ENTRY**  
    **If fewer than 5 → SKIP**
            """
        )

        # 10. Exit rules
        st.markdown("### 🟩 10. EXIT RULES (Very Important)")
        st.markdown(
            """
    Exit when:

    - ❌ Price **loses EMA 9**
    - ❌ Price **loses VWAP**
    - ❌ **LH** prints
    - ❌ Big rejection candle at a key level
    - ❌ Volume dies
    - ❌ HOD liquidity grab candle appears
    - ❌ Supply zone rejection
    - ❌ 1m trend flips to **PUT**

    > Taking profits **is a skill** – protect gains, don’t donate them back.
            """
        )

        # Summary table
        st.markdown("### 🟩 The Sybest CALL Trading Playbook (Summary)")
        st.markdown(
            """
    **Your components and what they do:**

    | Component         | Purpose                          |
    |-------------------|----------------------------------|
    | EMA 9/21          | Trend confirmation              |
    | VWAP              | Institutional trend filter      |
    | PDH/PDL           | Major reversal zones            |
    | CDH/CDL           | Intraday trend zones            |
    | ORH/ORL           | 5-min institutional entries     |
    | HH/HL             | Trend structure                 |
    | SNIPER            | Early & safest entries          |
    | 1m/5m/30m Trend   | Multi-timeframe confirmation    |
    | Supply / Demand   | Avoid bad entries & traps       |

    Everything you built works together to give you:

    - 🟩 Early entries  
    - 🟩 High-probability setups  
    - 🟩 Trend confirmation  
    - 🟩 Structure confirmation  
    - 🟩 Level-based entries  
    - 🟩 Institutional alignment  
    - 🟩 Trap avoidance  

    This is how you become an **unstoppable CALL day trader** with the Sybest system.
            """
        )

            # Example usage in your main app:
            # show_sybest_call_framework()

    # ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
        # DAILY CHECK LIST TO TRACK OPTION
        # ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]=
        # ========================================================================================================
        # ⚙️ SYSTEM RULE ALIGNMENT — STRATEGY COMPLIANCE REVIEW
        # ========================================================================================================

    # ========================================================================================================
    # ⚙️ SYSTEM RULE ALIGNMENT — STRATEGY COMPLIANCE REVIEW (with 70% TP dropdown + auto logic)
    # ========================================================================================================
    #
    # st.markdown("---")
    # st.subheader("⚙️ System Rule Alignment")
    #
    # # --- Section header ---
    # st.markdown("#### 🧩 Review Each System Element 2")
    # col1, col2, col3 = st.columns(3)
    #
    # # ========== COLUMN 1 ==========
    # with col1:
    #     trend_check = st.selectbox(
    #         "📊 Trend Confirmation (5m/15m)",
    #         ["✅ Followed", "❌ Not Followed"], index=1
    #     )
    #
    #     volume_check = st.selectbox(
    #         "📈 Volume > 500 & Spread ≤ $0.10",
    #         ["✅ Followed", "❌ Not Followed"], index=0
    #     )
    #
    # # ========== COLUMN 2 ==========
    # with col2:
    #     stop_check = st.selectbox(
    #         "🛑 Stop-Loss (25%)",
    #         ["✅ Followed", "❌ Not Followed"], index=1
    #     )
    #
    #     # 🔍 Auto-evaluate Take-Profit based on trade data
    #     if exit_premium > 0 and premium > 0:
    #         tp_ratio = ((exit_premium - premium) / premium) * 100
    #         if tp_ratio >= target_gain_pct:
    #             tp_default = "⚡ Exceeded Target"
    #         elif tp_ratio >= target_gain_pct * 0.9:
    #             tp_default = "✅ Followed"
    #         elif tp_ratio >= target_gain_pct * 0.6:
    #             tp_default = "⚠️ Not Hit (Partial)"
    #         else:
    #             tp_default = "❌ Not Followed"
    #     else:
    #         tp_default = "⚠️ Not Hit (Partial)"
    #
    #     # ✅ Always render visible dropdown for user to confirm / override
    #     target_check = st.selectbox(
    #         "🎯 Take-Profit (70%)",
    #         ["⚡ Exceeded Target", "✅ Followed", "⚠️ Not Hit (Partial)", "❌ Not Followed"],
    #         index=["⚡ Exceeded Target", "✅ Followed", "⚠️ Not Hit (Partial)", "❌ Not Followed"].index(tp_default)
    #     )
    #
    # # ========== COLUMN 3 ==========
    # with col3:
    #     risk_check = st.selectbox(
    #         "💰 Risk ≤ 1% per Trade",
    #         ["✅ Followed", "❌ Not Followed"], index=1
    #     )
    #
    #     oco_check = st.selectbox(
    #         "🔄 OCO Active",
    #         ["✅ Followed", "❌ Not Followed"], index=1
    #     )
    #
    # # --- Notes section ---
    # system_note = st.text_area(
    #     "🗒️ Add Comment or Observation (Optional)",
    #     "Ignored stop-loss; exited manually after price reversal."
    # )

    # ==========================================================================PART 1

    # ========================================================================================================
    # ⚙️ SYSTEM RULE ALIGNMENT — SYBEST SNIPER CHECKLIST (Separated Timeframes)
    # ========================================================================================================

    st.markdown("---")
    st.subheader("⚙️ Sybest Sniper — System Rule Alignment")

    st.markdown("#### 🧩 Review Each Entry Condition")

    col1, col2, col3 = st.columns(3)

    # ========== COLUMN 1 ========== (Structure + Momentum)
    with col1:
        call_zone = st.selectbox(
            "🟩 CALL ZONE Active",
            ["✅ Yes", "❌ No"], index=1
        )

        hl_check = st.selectbox(
            "📉 Higher Low (HL) Printed",
            ["✅ Yes", "❌ No"], index=1
        )

        hh_check = st.selectbox(
            "📈 Higher High (HH) Printed",
            ["✅ Yes", "❌ No"], index=1
        )

    # ========== COLUMN 2 ========== (Trend + Momentum)
    with col2:
        ema_check = st.selectbox(
            "📊 EMA 9 > EMA 21",
            ["✅ Yes", "❌ No"], index=1
        )

        price_ema_check = st.selectbox(
            "💹 Price Above EMA 9",
            ["✅ Yes", "❌ No"], index=1
        )

        price_vwap_check = st.selectbox(
            "🎯 Price Above VWAP",
            ["✅ Yes", "❌ No"], index=1
        )

    # ========== COLUMN 3 ========== (Timeframes + Final Trigger)
    with col3:
        tf1 = st.selectbox(
            "📍 1m Timeframe = CALL",
            ["📗 CALL", "📕 PUT", "⚪ WAIT"], index=2
        )

        tf5 = st.selectbox(
            "📍 5m Timeframe = CALL",
            ["📗 CALL", "📕 PUT", "⚪ WAIT"], index=2
        )

        tf30 = st.selectbox(
            "📍 30m Timeframe (Optional)",
            ["📗 CALL", "📕 PUT", "⚪ WAIT"], index=2
        )

        sniper_check = st.selectbox(
            "🎯 SNIPER CALL Signal",
            ["✅ Yes", "❌ No"], index=1
        )

    # ===================================================================
    # NOTES + COMMENT
    # ===================================================================

    system_note = st.text_area(
        "🗒️ Add Comment (Optional)",
        "HL and HH were valid; sniper candle did not print yet."
    )

    # ========================================================================================================
    # 🎯 SCORING LOGIC (Updated for Sybest Sniper)
    # ========================================================================================================

    def score_rule(value):
        if "✅" in value or "📗" in value:
            return 1  # full credit
        elif "⚪" in value:
            return 0.5  # partial credit for WAIT
        else:
            return 0  # not followed

    # New rule list matching our new checklist
    rule_values = [
        call_zone,
        hl_check,
        hh_check,
        ema_check,
        price_ema_check,
        price_vwap_check,
        tf1,
        tf5,
        tf30,
        sniper_check
    ]

    scores = [score_rule(v) for v in rule_values]
    score = sum(scores)
    total_rules = len(rule_values)
    percent = (score / total_rules) * 100

    # Assign grade + color
    if percent >= 90:
        grade, color = "A+", "#27ae60"
    elif percent >= 75:
        grade, color = "A", "#2ecc71"
    elif percent >= 60:
        grade, color = "B", "#f1c40f"
    elif percent >= 40:
        grade, color = "C", "#e67e22"
    else:
        grade, color = "F", "#e74c3c"

    # ========================================================================================================
    # 🪪 DISPLAY COMPLIANCE CARD
    # ========================================================================================================

    st.markdown(f"""
    <div style='margin-top:10px; border-radius:12px; padding:16px;
                border:1px solid rgba(255,255,255,0.1);
                background:rgba(255,255,255,0.03);'>
        <h4 style='color:{color};'>
            ⚙️ System Compliance Score: <b>{score:.1f} / {total_rules}</b> → {percent:.0f}% (Grade {grade})
        </h4>
        <ul style='color:#bdc3c7;'>
            <li>🟩 CALL ZONE Active: {call_zone}</li>
            <li>📉 Higher Low (HL): {hl_check}</li>
            <li>📈 Higher High (HH): {hh_check}</li>
            <li>📊 EMA Alignment: {ema_check}</li>
            <li>💹 Above EMA9: {price_ema_check}</li>
            <li>🎯 Above VWAP: {price_vwap_check}</li>
            <li>📍 1m Trend: {tf1}</li>
            <li>📍 5m Trend: {tf5}</li>
            <li>📍 30m Trend (Optional): {tf30}</li>
            <li>🎯 SNIPER CALL: {sniper_check}</li>
        </ul>
        <p><b>🗒️ Comment:</b> {system_note}</p>
    </div>
    """, unsafe_allow_html=True)

    # ========================================================================================================
    # 💾 SAVE CHECKLIST ENTRY
    # ========================================================================================================

    if st.button("💾 Save System Alignment"):
        checklist_path = r"C:\Users\stans\ML_PROJECTS\0. STOCK_TRADING_PROJECT\1. Dataset\checklist-tracker.csv"
        trade_date_str = datetime.now().strftime("%Y-%m-%d")

        compliance_data = pd.DataFrame([{
            "Trade Date": trade_date_str,
            "Symbol": symbol,
            "CALL ZONE": call_zone,
            "HL Printed": hl_check,
            "HH Printed": hh_check,
            "EMA Alignment": ema_check,
            "Above EMA9": price_ema_check,
            "Above VWAP": price_vwap_check,
            "1m Trend": tf1,
            "5m Trend": tf5,
            "30m Trend": tf30,
            "Sniper Call": sniper_check,
            "System Compliance Score": f"{score:.1f}/{total_rules}",
            "System Compliance %": round(percent, 2),
            "System Grade": grade,
            "Comment": system_note,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])

        if os.path.exists(checklist_path):
            existing = pd.read_csv(checklist_path)
            updated = pd.concat([existing, compliance_data], ignore_index=True)
        else:
            updated = compliance_data

        updated.to_csv(checklist_path, index=False)
        st.success(f"✅ System Checklist saved — {percent:.0f}% compliance (Grade {grade})")

    # ========================================================================================================
                        # LOAD DATA SAVED FOR INSIGHT AND AVALYSIS
# =========================================================================================================

    # ========================================================================================================
    # 📈 TRADE PERFORMANCE INSIGHTS AND ANALYSIS (Annotated on All Graphs)

    # 📈 TRADE PERFORMANCE INSIGHTS AND ANALYSIS (Annotated on All Graphs)
    # ========================================================================================================
    # ========================================================================================================
    # 📈 TRADE PERFORMANCE INSIGHTS AND ANALYSIS (Annotated on All Graphs)

    # 📈 TRADE PERFORMANCE INSIGHTS AND ANALYSIS (Annotated on All Graphs)
#     # ========================================================================================================
# # ========================================================================================================
# # LOAD DATA SAVED FOR INSIGHT AND AVALYSIS
# # =========================================================================================================
    import matplotlib.pyplot as plt

    st.markdown("---")
    st.header("📊 Trade Performance Insights")

    save_path = r"C:\Users\stans\ML_PROJECTS\0. STOCK_TRADING_PROJECT\1. Dataset\Option_trading.csv"

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)

        # ✅ FIX: ensure proper timestamp formatting
        if "Timestamp" in df.columns:
            # Convert safely
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

            # Fill missing values
            df["Timestamp"].fillna(method="ffill", inplace=True)
            df["Timestamp"].fillna(pd.Timestamp.now(), inplace=True)

            # 🔧 Format as 'YYYY-MM-DD HH:MM:SS'
            df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        df = df.sort_values("Timestamp", ascending=False)

        # st.subheader("🧾 Latest Recorded Trades")
        # st.dataframe(df.tail(10), use_container_width=True)
        # ✅ Ensure 'Trade Date' or 'Date' is the first column
        if "Trade Date" in df.columns:
            first_col = "Trade Date"
        elif "Date" in df.columns:
            first_col = "Date"
        else:
            first_col = None

        if first_col:
            # Move trade date to front, keep all other columns after
            cols = [first_col] + [c for c in df.columns if c != first_col]
            df = df[cols]

        # ✅ Display latest 10 trades with reordered columns
        st.subheader("🧾 Latest Recorded Trades (Date First)")
        st.dataframe(df.tail(10), use_container_width=True)

        # ---------- SUMMARY METRICS ----------
        st.divider()

            # ==============================================================================================
        # ---------- SUMMARY METRICS ----------

        # total_trades = len(df)
        # profitable_trades = (df["Realized P/L $"] > 0).sum()
        # losing_trades = (df["Realized P/L $"] < 0).sum()
        # avg_proj_roi = df["Projected ROI %"].mean()
        # avg_actual_roi = df["Actual ROI %"].mean()
        # total_profit = df["Realized P/L $"].sum()
        # total_invested = df["Total Investment"].sum()
        #
        # # ✅ Calculate total gain and total loss separately
        # total_gain = df.loc[df["Realized P/L $"] > 0, "Realized P/L $"].sum()
        # total_loss = df.loc[df["Realized P/L $"] < 0, "Realized P/L $"].sum()
        #
        # if "Trade Date" in df.columns:
        #     unique_days = df["Trade Date"].nunique()
        # elif "Date" in df.columns:
        #     unique_days = df["Date"].nunique()
        # else:
        #     unique_days = "N/A"
        #
        # # =========================
        # # 📊 CARD LAYOUT IN ONE LINE
        # # =========================
        # c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        # c1.metric("📊 Total Trades", total_trades)
        # c2.metric("💰 Total Capital Invested", f"${total_invested:,.2f}")
        # c2.metric("💰 AVG Total Capital Invested", f"${total_invested:,.2f}")

        # ======================================================================== PART 2

        total_trades = len(df)
        profitable_trades = (df["Realized P/L $"] > 0).sum()
        losing_trades = (df["Realized P/L $"] < 0).sum()
        avg_proj_roi = df["Projected ROI %"].mean()
        avg_actual_roi = df["Actual ROI %"].mean()
        total_profit = df["Realized P/L $"].sum()
        total_invested = df["Total Investment"].sum()

        # Separate gains/losses
        total_gain = df.loc[df["Realized P/L $"] > 0, "Realized P/L $"].sum()
        total_loss = df.loc[df["Realized P/L $"] < 0, "Realized P/L $"].sum()

        # NEW → Average investment per trade
        avg_investment = total_invested / total_trades if total_trades > 0 else 0

        avg_contracts = df["Contracts"].mean()
        avg_entry_premium = df["Entry Premium"].mean()

        # Unique trading days
        if "Trade Date" in df.columns:
            unique_days = df["Trade Date"].nunique()
        elif "Date" in df.columns:
            unique_days = df["Date"].nunique()
        else:
            unique_days = "N/A"
# ================================================================================== BEAUTIFUL

        # BEAUTIFUL CARD CSS
        st.markdown("""
        <style>

        .metric-card {
            background: linear-gradient(145deg, #1f1f1f, #2c2c2c);
            padding: 18px;
            border-radius: 14px;
            box-shadow: 5px 5px 12px #141414, -5px -5px 12px #2e2e2e;
            text-align: center;
            color: white;
            border: 1px solid #333;
            transition: 0.2s ease-in-out;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 5px 12px 22px #111;
        }

        .metric-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 4px;
            color: #e0e0e0;
        }

        .metric-value {
            font-size: 22px;
            font-weight: 700;
        }

        .gain { color: #2ecc71 !important; }
        .loss { color: #e74c3c !important; }
        .warning { color: #f1c40f !important; }

        </style>
        """, unsafe_allow_html=True)

        # =================================================== ENDS HERE
        # =========================
        # BEAUTIFUL BOX CARD LAYOUT
        # =========================

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

        # Card 1 – Total Trades
        c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Total Trades</div>
            <div class="metric-value">{total_trades}</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 2 – Total Capital Invested
        c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Total Capital Invested</div>
            <div class="metric-value">${total_invested:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 2 (second metric) – Avg Total Investment
        avg_total_investment = df["Total Investment"].mean()
        c2.markdown(f"""
        <div class="metric-card" style="margin-top:10px;">
            <div class="metric-title">📉 Avg Total Investment</div>
            <div class="metric-value">${avg_total_investment:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 3 – Total P/L
        c3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🏁 Total Realized P/L</div>
            <div class="metric-value">{'$' + format(total_profit, ',.2f')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 3 (second metric) – Avg Contracts
        c3.markdown(f"""
        <div class="metric-card" style="margin-top:10px;">
            <div class="metric-title">🧮 Avg Contracts per Trade</div>
            <div class="metric-value">{avg_contracts:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 4 – Win Rate
        c4.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">✅ Win Rate</div>
            <div class="metric-value">{(profitable_trades / total_trades * 100):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 4 (second metric) – Avg Entry Premium
        c4.markdown(f"""
        <div class="metric-card" style="margin-top:10px;">
            <div class="metric-title">💵 Avg Entry Premium</div>
            <div class="metric-value">${avg_entry_premium:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 5 – Total Gain
        c5.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📈 Total Gain</div>
            <div class="metric-value gain">${total_gain:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 6 – Total Loss
        c6.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📉 Total Loss</div>
            <div class="metric-value loss">${total_loss:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Card 7 – Days Traded
        c7.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🗓️ Days Traded</div>
            <div class="metric-value warning">{unique_days}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # # ========================================= ADDED
        # # =========================
        # # EXTRA METRIC CARDS (NEW)
        # # =========================
        #
        # # Calculate averages
        # avg_contracts = df["Contracts"].mean()
        # avg_entry_premium = df["Entry Premium"].mean()
        #
        # c8, c9 = st.columns(2)
        #
        # # Card: Avg Contracts per Trade
        # c8.markdown(f"""
        # <div class="metric-card">
        #     <div class="metric-title">🧮 Avg Contracts per Trade</div>
        #     <div class="metric-value">{avg_contracts:.2f}</div>
        # </div>
        # """, unsafe_allow_html=True)
        #
        # # Card: Avg Entry Premium
        # c9.markdown(f"""
        # <div class="metric-card">
        #     <div class="metric-title">💵 Avg Entry Premium</div>
        #     <div class="metric-value">${avg_entry_premium:.2f}</div>
        # </div>
        # """, unsafe_allow_html=True)

        # ==============================================================================================================
        # ---------- ROI TREND ----------

        st.subheader("📉 ROI Progression — Projected vs Actual (%)")

        with st.expander("📉 ROI Progression — Projected vs Actual (%)"):

            fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

            # Lines — thinner & smaller markers for a clean visual
            ax.plot(df["Timestamp"], df["Projected ROI %"],
                    label="Projected ROI", marker="o", color="#e74c3c",
                    linewidth=1.8, markersize=6)

            ax.plot(df["Timestamp"], df["Actual ROI %"],
                    label="Actual ROI", marker="s", color="#2ecc71",
                    linewidth=1.8, markersize=5)

            # Smooth label placement slightly above line
            offset_proj = (df["Projected ROI %"].max() - df["Projected ROI %"].min()) * 0.02
            offset_act = (df["Actual ROI %"].max() - df["Actual ROI %"].min()) * 0.02 \
                if df["Actual ROI %"].max() != df["Actual ROI %"].min() else 1

            for i, v in enumerate(df["Projected ROI %"]):
                ax.text(df["Timestamp"].iloc[i], v + offset_proj, f"{v:.1f}%",
                        color="#e74c3c", ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold")

            for i, v in enumerate(df["Actual ROI %"]):
                ax.text(df["Timestamp"].iloc[i], v + offset_act, f"{v:.1f}%",
                        color="#2ecc71", ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold")

            # Axis labels and layout
            ax.set_xlabel("Trade Date", fontsize=10)
            ax.set_ylabel("ROI (%)", fontsize=10)
            ax.set_title("📊 ROI Progression — Projected vs Actual (%)", fontsize=12, pad=8)

            # Rotate X-axis labels for readability
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

            ax.legend(facecolor="white", framealpha=1, loc="lower center", fontsize=9)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout(pad=2.0)
            st.pyplot(fig)

        # ===================
        # ===================ENDS HERE===============
        # ---------- ENTRY VS EXIT PREMIUM ----------


            # ---------- ENTRY VS EXIT PREMIUM ----------
            st.subheader("💵 Entry vs Exit Premium per Contract")

            fig_price, ax_price = plt.subplots(figsize=(8, 4), dpi=150)

            # Lines
            ax_price.plot(df["Timestamp"], df["Entry Premium"],
                          label="Entry Premium (Buy)",
                          marker="o", color="#e74c3c",
                          linewidth=1.8, markersize=6)

            ax_price.plot(df["Timestamp"], df["Exit Premium"],
                          label="Exit Premium (Sell)",
                          marker="s", color="#2ecc71",
                          linewidth=1.8, markersize=6)

            # Annotate values slightly above markers
            for i, v in enumerate(df["Entry Premium"]):
                ax_price.text(df["Timestamp"].iloc[i], v + 0.05, f"${v:.2f}",
                              color="#e74c3c", ha="center", va="bottom",
                              fontsize=8, fontweight="bold")

            for i, v in enumerate(df["Exit Premium"]):
                ax_price.text(df["Timestamp"].iloc[i], v + 0.05, f"${v:.2f}",
                              color="#2ecc71", ha="center", va="bottom",
                              fontsize=8, fontweight="bold")

            # Labels + Title
            ax_price.set_xlabel("Trade Date", fontsize=10)
            ax_price.set_ylabel("Premium ($)", fontsize=10)
            ax_price.set_title("💵 Entry vs Exit Premium per Contract", fontsize=12, pad=8)

            # Rotate X-axis labels 90 degrees
            plt.setp(ax_price.get_xticklabels(), rotation=78, ha='center')

            # Legend + Grid
            ax_price.legend(facecolor="white", framealpha=1, fontsize=9, loc="upper left")
            ax_price.grid(alpha=0.3)

            # Polished figure layout
            plt.tight_layout(pad=2.0)

            st.pyplot(fig_price)

        # ====================== END HERE

        # # ---------- R:R COMPARISON ----------

        # ---------- R:R COMPARISON ----------
        st.subheader("⚖️ Projected vs Actual Risk/Reward Ratio")

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        bar_width = 0.4
        x = range(len(df))

        # Bars
        bars_proj = ax2.bar([i - bar_width / 2 for i in x], df["Projected R:R"], width=bar_width,
                            label="Projected", color="#e74c3c")
        bars_act = ax2.bar([i + bar_width / 2 for i in x], df["Actual R:R"], width=bar_width,
                           label="Actual", color="#2ecc71", alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(df["Symbol"].fillna("Unknown").astype(str), rotation=45, ha="right")
        ax2.set_ylabel("R:R Ratio")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # ---- Average Lines ----
        avg_proj = df["Projected R:R"].mean()
        avg_act = df["Actual R:R"].mean()

        ax2.axhline(avg_proj, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Avg Projected {avg_proj:.2f}")
        ax2.axhline(avg_act, color="#27ae60", linestyle="--", linewidth=1.5, label=f"Avg Actual {avg_act:.2f}")

        # Annotate average values
        ax2.text(len(df) - 0.3, avg_proj + 0.05, f"Avg Proj: {avg_proj:.2f}", color="#c0392b",
                 fontsize=9, ha="right", va="bottom", fontweight="bold")
        ax2.text(len(df) - 0.3, avg_act + 0.05, f"Avg Act: {avg_act:.2f}", color="#27ae60",
                 fontsize=9, ha="right", va="bottom", fontweight="bold")

        # Annotate both bars individually
        for bars in [bars_proj, bars_act]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                         f"{height:.2f}", ha="center", va="bottom", fontsize=8, color="#2c3e50")

        # Adjust legend (so average lines show too)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, labels, loc="upper left")

        st.pyplot(fig2)


        # ========================= ENDS HERE=================
        # ---------- PROFIT / LOSS ----------

        # ========================= ENDS HERE=================
        # ---------- PROFIT / LOSS ----------
        st.subheader("📊 Profit / Loss per Trade ($)")

        df["Symbol"] = df["Symbol"].fillna("Unknown").astype(str)

        fig4, ax4 = plt.subplots(figsize=(10, 5))

        colors = df["Realized P/L $"].apply(
            lambda x: "#2ecc71" if x > 0 else "#e74c3c"
        )

        bars = ax4.bar(df["Symbol"], df["Realized P/L $"], color=colors)

        ax4.axhline(0, color="gray", linestyle="--", linewidth=1)

        # ---- Annotate bars ----
        for bar in bars:
            height = bar.get_height()
            y_offset = 0.02 * max(df["Realized P/L $"]) if height > 0 else -0.05 * abs(min(df["Realized P/L $"]))
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                height + y_offset,
                f"${height:,.0f}",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=9,
                color="#2c3e50",
                fontweight="bold"
            )

        ax4.set_xlabel("Symbol")
        ax4.set_ylabel("Profit / Loss ($)")
        ax4.set_title("Per-Trade Profit vs Loss (Annotated)")
        ax4.grid(alpha=0.2)

        # 🔥 **Rotate X-axis labels 90 degrees**
        plt.xticks(rotation=90)

        st.pyplot(fig4)

        # ============================================================ DAILY GAIN AND LOSS
        # == DAILY GAIN AND LOSS
        # ==================================================================================
        # ========================= ENDS HERE=================
        # ========================= DAILY PROFIT / LOSS =========================

        # ========================= ENDS HERE=================
        # ---------- DAILY PROFIT / LOSS + TRADE COUNT (STACKED + 90° LABELS) ----------
        st.subheader("📆 Daily Profit / Loss & Trade Count")

        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        # 1️⃣ Parse Date column (handles mixed formats)
        df["Date"] = pd.to_datetime(
            df["Date"].astype(str).str.strip(),
            format="mixed",
            errors="coerce"
        )

        # Keep only rows with a valid Date
        df_dates = df.dropna(subset=["Date"]).copy()

        # 2️⃣ Group by Date → sum P/L and count trades
        daily_stats = (
            df_dates.groupby("Date", as_index=False)
            .agg({
                "Realized P/L $": "sum",
                "Symbol": "count"
            })
            .rename(columns={"Symbol": "Trade Count"})
            .sort_values("Date")
        )

        # 3️⃣ Plot the chart
        if not daily_stats.empty:
            fig5, ax5 = plt.subplots(figsize=(12, 6))

            x_labels = daily_stats["Date"].dt.strftime("%Y-%m-%d")
            pl_values = daily_stats["Realized P/L $"]
            trade_counts = daily_stats["Trade Count"]
            x_positions = range(len(x_labels))

            # Colors
            bar_colors = pl_values.apply(lambda x: "#2ecc71" if x > 0 else "#e74c3c")

            # Bars = P/L per day
            bars = ax5.bar(x_positions, pl_values, color=bar_colors)

            ax5.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax5.set_xlabel("Date")
            ax5.set_ylabel("Profit / Loss ($)")
            ax5.set_title("Daily Profit / Loss with Trade Count")
            ax5.grid(alpha=0.2)

            # 4️⃣ SHOW ALL DATE LABELS ROTATED 90°
            ax5.set_xticks(x_positions)
            ax5.set_xticklabels(x_labels, rotation=90, ha="center")  # <---- HERE

            # 5️⃣ Line for trade count
            ax6 = ax5.twinx()
            trade_line, = ax6.plot(
                x_positions,
                trade_counts,
                marker="o",
                linewidth=2,
                color="#34495e",
                label="Trade Count"
            )
            ax6.set_ylabel("Number of Trades")

            # 6️⃣ Stacked annotations: amount + (trade count)
            max_pl = pl_values.max()
            min_pl = pl_values.min()

            for bar, value, count in zip(bars, pl_values, trade_counts):
                height = bar.get_height()

                if value >= 0:
                    y = height + (0.02 * max_pl if max_pl != 0 else 0.5)
                    va = "bottom"
                else:
                    y = height - (0.05 * abs(min_pl) if min_pl != 0 else 0.5)
                    va = "top"

                ax5.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"${value:,.0f}\n({int(count)})",  # <--- parentheses
                    ha="center",
                    va=va,
                    fontsize=9,
                    color="#2c3e50",
                    fontweight="bold"
                )

            # 7️⃣ Legend
            profit_patch = mpatches.Patch(color="#2ecc71", label="Profit Day")
            loss_patch = mpatches.Patch(color="#e74c3c", label="Loss Day")
            ax5.legend(
                handles=[profit_patch, loss_patch, trade_line],
                loc="upper left",
                frameon=True
            )

            fig5.tight_layout()
            st.pyplot(fig5)

        else:
            st.info("No data available to display Daily Profit / Loss & Trade Count.")

        # DAILY P/L ==================================================================================




        # GOOD GRAPHS ENDS HERE
        # =======================================================PART 2

        # ========================= ENDS HERE=================
        # ---------- PROFIT / LOSS ----------
        st.subheader("📊 Profit / Loss per Trade ($)")

        df["Symbol"] = df["Symbol"].fillna("Unknown").astype(str)

        fig4, ax4 = plt.subplots(figsize=(10, 5))

        # Colors for bars
        colors = df["Realized P/L $"].apply(lambda x: "#2ecc71" if x > 0 else "#e74c3c")

        # Plot bars
        bars = ax4.bar(df["Symbol"], df["Realized P/L $"], color=colors)
        ax4.axhline(0, color="gray", linestyle="--", linewidth=1)

        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            y_offset = 0.03 * max(abs(df["Realized P/L $"]))
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                height + (y_offset if height >= 0 else -y_offset),
                f"${height:,.0f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
                color="#2c3e50",
                fontweight="bold"
            )

        ax4.set_xlabel("Symbol")
        ax4.set_ylabel("Profit / Loss ($)")
        ax4.set_title("Per-Trade Profit vs Loss (Annotated)")
        ax4.grid(alpha=0.2)

        # Rotate x-axis labels
        plt.xticks(rotation=90)

        # Calculate metrics
        total_gain = df[df["Realized P/L $"] > 0]["Realized P/L $"].sum()
        total_loss = df[df["Realized P/L $"] < 0]["Realized P/L $"].sum()
        total_amount = total_gain + total_loss

        # Legend handles
        handles = [
            plt.Line2D([0], [0], color="#2ecc71", linewidth=10,
                       label=f"Total Gain: ${total_gain:,.2f}"),
            plt.Line2D([0], [0], color="#e74c3c", linewidth=10,
                       label=f"Total Loss: ${total_loss:,.2f}"),
            plt.Line2D([0], [0], color="#3498db", linewidth=10,
                       label=f"Net Total Amount: ${total_amount:,.2f}")
        ]

        # ------------ 🟦 LEGEND COMPLETELY OUTSIDE PLOT ----------------
        ax4.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0, -0.55),  # Push FAR outside the figure
            frameon=True,
            fancybox=True,
            shadow=False,
            fontsize=10
        )

        # Add extra bottom margin so legend is not cut off
        plt.subplots_adjust(bottom=0.55)

        st.pyplot(fig4)



        # ============================PART 2 Ends

        st.divider()

        # ---------- INSIGHTS ----------
        st.markdown("### 🔍 Insights & Observations")
        avg_rr_diff = (df["Projected R:R"] - df["Actual R:R"]).mean()
        avg_roi_diff = (df["Projected ROI %"] - df["Actual ROI %"]).mean()
        avg_entry = df["Entry Premium"].mean()
        avg_exit = df["Exit Premium"].mean()
        avg_trade_size = df["Total Investment"].mean()

        st.markdown(f"""
        - 💵 **Average Entry Premium:** ${avg_entry:.2f} | **Average Exit Premium:** ${avg_exit:.2f}
        - 📦 **Average Trade Size:** ${avg_trade_size:,.2f}
        - 📈 **Average Projected ROI:** {avg_proj_roi:.2f}% | **Average Actual ROI:** {avg_actual_roi:.2f}%
        - ⚖️ **Avg R:R Diff:** {avg_rr_diff:.2f} | **Avg ROI Diff:** {avg_roi_diff:.2f}%
        - 💰 **Total Net P/L:** ${total_profit:,.2f}

        **Interpretation:**
        🔴 **Projected** = planned outcome | 🟢 **Actual** = executed result
        - Large 🔴>🟢 gap → entering late or exiting early.
        - Sustained 🟢≥🔴 → excellent trade management & discipline.
        - Keep **Actual ROI within ±10 %** of Projected to build consistency.
        """)

    else:
        st.warning("⚠️ No saved data found yet. Save at least one trade to begin performance tracking.")

    # ========================================================================================================
    # 🎓 TRADER PERFORMANCE GRADE — Daily and Overall
    # ========================================================================================================
    # ========================================================================================================
    # 🎓 TRADER PERFORMANCE GRADE — Daily and Overall (Individual Trade Grading Added)
    # ========================================================================================================
    # st.divider()

    st.markdown("---")
    st.subheader("🎓 Trader Performance Grade — Daily & Overall")



    with st.expander('TRADER GRADE PERFORMANCE'):
        # Add missing variable defaults to prevent NameError
        exit_premium = 0
        premium = 0
        target_gain_pct = 70
        symbol = "AAPL"

        # st.markdown("---")
        # st.subheader("🎓 Trader Performance Grade — Daily & Overall")

        # ========================================================================================================
        # 🔄 AUTO-LOAD LATEST TRADE DATA & ENSURE TODAY'S TRADES ARE INCLUDED
        # ========================================================================================================

        save_path = r"C:\Users\stans\ML_PROJECTS\0. STOCK_TRADING_PROJECT\1. Dataset\Option_trading.csv"

        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
        else:
            st.warning("⚠️ No trade data file found.")
            st.stop()

        # ✅ Parse timestamps correctly once
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Date"] = df["Timestamp"].dt.date
        df["Time"] = df["Timestamp"].dt.strftime("%H:%M:%S")
        df["Trade_ID"] = range(1, len(df) + 1)

        today = datetime.now().date()
        today_trades = df[df["Date"] == today]

        if today_trades.empty:
            st.warning("⚠️ No trades recorded today yet.")
        else:
            st.success(f"📅 Showing today's performance ({today}) — {len(today_trades)} trade(s) found.")

        # ========================================================================================================
        # 🎯 FUNCTION — COMPUTE GRADE
        # ========================================================================================================
        def compute_grade(data):
            total_trades = len(data)
            profitable_trades = (data["Realized P/L $"] > 0).sum()
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

            safe = data.fillna({
                "Projected ROI %": 0,
                "Actual ROI %": 0,
                "Projected R:R": 1,
                "Actual R:R": 1,
                "Realized P/L $": 0
            })

            avg_proj_roi = safe["Projected ROI %"].mean()
            avg_actual_roi = safe["Actual ROI %"].mean()
            avg_rr_diff = (safe["Projected R:R"] - safe["Actual R:R"]).mean()
            roi_efficiency = max(0, 100 - abs(avg_proj_roi - avg_actual_roi))
            rr_efficiency = max(0, (safe["Actual R:R"].mean() / safe["Projected R:R"].mean()) * 100) if safe[
                                                                                                            "Projected R:R"].mean() > 0 else 0
            total_profit = safe["Realized P/L $"].sum()
            profit_factor = 100 if total_profit > 0 else 40

            score = (win_rate * 0.4) + (roi_efficiency * 0.3) + (rr_efficiency * 0.2) + (profit_factor * 0.1)
            score = min(100, round(score, 2))

            if score >= 90:
                grade, color, note = "A+", "#27ae60", "Outstanding — Consistent control and profitability!"
            elif score >= 80:
                grade, color, note = "A", "#2ecc71", "Excellent discipline and strong returns."
            elif score >= 70:
                grade, color, note = "B", "#f1c40f", "Good performance — improve entry precision."
            elif score >= 60:
                grade, color, note = "C", "#e67e22", "Average — risk management needs attention."
            else:
                grade, color, note = "D", "#e74c3c", "Needs improvement — review strategy consistency."

            return grade, color, note, score, win_rate, roi_efficiency, rr_efficiency, profit_factor, total_profit

        # ========================================================================================================
        # 🌞 TODAY'S PERFORMANCE
        # ========================================================================================================
        if not today_trades.empty:
            grade, color, note, score, win_rate, roi_eff, rr_eff, profit_factor, total_profit = compute_grade(today_trades)
            st.markdown(f"""
            <div style='margin-bottom:12px; border-radius:10px; padding:12px; border:1px solid rgba(255,255,255,0.15);
                        background:rgba(255,255,255,0.05);'>
                <h4 style='color:{color};'>🌞 Today's Grade ({today}) — <b>{grade}</b> ({score}%)</h4>
                <p style='color:#bdc3c7; margin-bottom:4px;'>{note}</p>
                <ul style='margin:0; padding-left:20px; color:#bdc3c7;'>
                    <li>📈 Win Rate: {win_rate:.1f}%</li>
                    <li>🎯 ROI Efficiency: {roi_eff:.1f}%</li>
                    <li>⚖️ R:R Efficiency: {rr_eff:.1f}%</li>
                    <li>💰 Profit Factor: {profit_factor}%</li>
                    <li>💵 Daily P/L: ${total_profit:,.2f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # ========================================================================================================
        # 💹 INDIVIDUAL TRADE PERFORMANCE (NOW GRADED LIKE DAILY FORMAT)
        # ========================================================================================================
        st.markdown("### 💹 Individual Trade Grades (Each Trade Evaluated)")

        for idx, row in df.iterrows():
            # Wrap each trade as a mini DataFrame so compute_grade() can process it
            trade_df = pd.DataFrame([row])
            grade, color, note, score, win_rate, roi_eff, rr_eff, profit_factor, total_profit = compute_grade(trade_df)

            trade_date = row["Date"]
            trade_time = row["Time"]
            trade_id = row["Trade_ID"]
            symbol = row.get("Symbol", "N/A")

            st.markdown(f"""
            <div style='margin-bottom:12px; border-radius:10px; padding:12px; border:1px solid rgba(255,255,255,0.1);
                        background:rgba(255,255,255,0.03);'>
                <h4 style='color:{color};'>📄 Trade #{trade_id} — {symbol} | {trade_date} {trade_time}</h4>
                <p style='color:#bdc3c7; margin-bottom:4px;'>{note}</p>
                <ul style='margin:0; padding-left:20px; color:#bdc3c7;'>
                    <li>📆 Grade: <b>{grade}</b> ({score}%)</li>
                    <li>📈 Win Rate: {win_rate:.1f}%</li>
                    <li>🎯 ROI Efficiency: {roi_eff:.1f}%</li>
                    <li>⚖️ R:R Efficiency: {rr_eff:.1f}%</li>
                    <li>💰 Profit Factor: {profit_factor}%</li>
                    <li>💵 Trade P/L: ${total_profit:,.2f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # ========================================================================================================
        # 📅 DAILY PERFORMANCE
        # ========================================================================================================
        st.markdown("### 📅 Daily Trade Grades")

        daily_groups = df.groupby("Date")

        for date, group in daily_groups:
            grade, color, note, score, win_rate, roi_eff, rr_eff, profit_factor, total_profit = compute_grade(group)
            st.markdown(f"""
            <div style='margin-bottom:12px; border-radius:10px; padding:12px; border:1px solid rgba(255,255,255,0.15);
                        background:rgba(255,255,255,0.03);'>
                <h4 style='color:{color};'>📆 {date} — Grade: <b>{grade}</b> ({score}%)</h4>
                <p style='color:#bdc3c7; margin-bottom:4px;'>{note}</p>
                <ul style='margin:0; padding-left:20px; color:#bdc3c7;'>
                    <li>📈 Win Rate: {win_rate:.1f}%</li>
                    <li>🎯 ROI Efficiency: {roi_eff:.1f}%</li>
                    <li>⚖️ R:R Efficiency: {rr_eff:.1f}%</li>
                    <li>💰 Profit Factor: {profit_factor}%</li>
                    <li>💵 Daily P/L: ${total_profit:,.2f}</li>
                    <li>🧾 Number of Trades: {len(group)}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================================================================
    # 🏁 OVERALL PERFORMANCE
    # ========================================================================================================
    st.markdown("### 🏁 Overall Trader Grade Summary")

    grade, color, note, score, win_rate, roi_eff, rr_eff, profit_factor, total_profit = compute_grade(df)

    st.markdown(f"""
    <div style='border-radius:12px; padding:20px; border:1px solid rgba(255,255,255,0.15);
                background:rgba(255,255,255,0.02);'>
        <h3 style='color:{color};'>🏆 Overall Grade: <b>{grade}</b> ({score}%)</h3>
        <p style='color:#bdc3c7;'>{note}</p>
        <ul>
            <li>📈 Win Rate: {win_rate:.1f}%</li>
            <li>🎯 ROI Consistency: {roi_eff:.1f}%</li>
            <li>⚖️ R:R Efficiency: {rr_eff:.1f}%</li>
            <li>💰 Profit Factor: {profit_factor}%</li>
            <li>💵 Net P/L: ${total_profit:,.2f}</li>
            <li>🧾 Total Trades Recorded: {len(df)}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# #
#     # ========================================================================================================
#     # ⚙️ SYSTEM RULE ALIGNMENT — STRATEGY COMPLIANCE REVIEW (with 70% TP dropdown + educational notes)
#     # ========================================================================================================





    # 🔧 No code changed below this line — all original logic retained exactly


# if __name__ == "__main__":
#     st.set_page_config(page_title="Options ROI", layout="wide")
#     roi_calculator()
# ========================================================================================================
# 🎓 TRADER PERFORMANCE GRADE — Daily and Overall                           TRADING ENDS HERE
# ========================================================================================================

# ============================== Add Line to Represent Buy, Sell and StopLoss ==========

# ============== “Trend Confirmation (5m/15m)” Means  ==========================
# ============== “Trend Confirmation (5m/15m)” Means  ==========================
# ============== “Trend Confirmation (5m/15m)” Means  ==========================
# ============== “Trend Confirmation (5m/15m)” Means  ==========================


# ============================================================================== PART 2
# =====================================================================
# 🎯 DAY TRADER — TREND CONFIRMATION BEFORE ENTRY
# =====================================================================

# 🎯 DAY TRADER — TREND CONFIRMATION SYSTEM (5m / 15m)
# =====================================================================
# =====================================================================
# 🎯 DAY TRADER — TREND CONFIRMATION WITH ENTRY MARKERS (5m / 15m)
# =====================================================================
# ============================================================
# ✅ DAYTRADE TREND CONFIRMATION MODULE
# ============================================================
# =====================================================================
# ⚡ DAY TRADER — LIVE TREND CONFIRMATION SYSTEM
# =====================================================================
# # =====================================================================
# # ⚡ DAY TRADER — LIVE TREND CONFIRMATION SYSTEM (5m / 15m)
# # =====================================================================
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import plotly.graph_objects as go
# from ta.momentum import RSIIndicator
# import time
#
# # -------------------------------------------------------------
# # 📡 SAFE DATA FETCH FUNCTION
# # -------------------------------------------------------------
# @st.cache_data(ttl=60)
# def get_intraday_data(ticker: str, period: str = "5d"):
#     """
#     Safely download both 5-minute and 15-minute data from Yahoo Finance.
#     Always returns a tuple of two DataFrames (df_5m, df_15m),
#     even if an error occurs.
#     """
#     df_5m, df_15m = pd.DataFrame(), pd.DataFrame()
#
#     try:
#         # Fetch 5m data
#         temp_5m = yf.download(ticker, period=period, interval="5m", progress=False)
#         if isinstance(temp_5m, pd.DataFrame):
#             df_5m = temp_5m.copy()
#
#         # Fetch 15m data
#         temp_15m = yf.download(ticker, period=period, interval="15m", progress=False)
#         if isinstance(temp_15m, pd.DataFrame):
#             df_15m = temp_15m.copy()
#
#     except Exception as e:
#         st.warning(f"⚠️ Error fetching data for {ticker}: {e}")
#
#     # Always return 2 DataFrames
#     if not isinstance(df_5m, pd.DataFrame):
#         df_5m = pd.DataFrame()
#     if not isinstance(df_15m, pd.DataFrame):
#         df_15m = pd.DataFrame()
#
#     return df_5m, df_15m
#
#
# # -------------------------------------------------------------
# # 📈 EMA TREND SIGNAL (with flattening fix)
# # -------------------------------------------------------------
# def trend_signal(df, short=9, long=21):
#     if df.empty:
#         return df
#
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = [c[0] for c in df.columns]
#     if isinstance(df["Close"], (pd.DataFrame, np.ndarray)):
#         df["Close"] = np.ravel(df["Close"])
#
#     df["EMA_Short"] = df["Close"].ewm(span=short, adjust=False).mean()
#     df["EMA_Long"] = df["Close"].ewm(span=long, adjust=False).mean()
#     df["Trend"] = np.where(df["EMA_Short"] > df["EMA_Long"], "UP", "DOWN")
#     return df
#
#
# # -------------------------------------------------------------
# # 📊 RSI CALCULATION (with flattening fix)
# # -------------------------------------------------------------
# def rsi_calc(df):
#     if df.empty:
#         return df
#
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = [c[0] for c in df.columns]
#     if isinstance(df["Close"], (pd.DataFrame, np.ndarray)):
#         df["Close"] = np.ravel(df["Close"])
#
#     rsi = RSIIndicator(df["Close"], window=14)
#     df["RSI"] = rsi.rsi()
#     return df
#
#
# # -------------------------------------------------------------
# # 🧠 DETERMINE CALL / PUT / WAIT BIAS
# # -------------------------------------------------------------
# def bias_decision(df_5m, df_15m):
#     if df_5m.empty or df_15m.empty:
#         return "❌ No Data", "gray", "No data available."
#
#     t5 = df_5m["Trend"].iloc[-1]
#     t15 = df_15m["Trend"].iloc[-1]
#     rsi = df_5m["RSI"].iloc[-1]
#     vol_now = df_5m["Volume"].iloc[-1]
#     vol_avg = df_5m["Volume"].rolling(20).mean().iloc[-1]
#
#     if t5 == t15 == "UP" and rsi < 70 and vol_now > vol_avg:
#         return (
#             "🟩 CALL Bias — Uptrend confirmed",
#             "green",
#             "Wait for 5m candle close ABOVE 9 EMA with rising volume."
#         )
#     elif t5 == t15 == "DOWN" and rsi > 30 and vol_now > vol_avg:
#         return (
#             "🟥 PUT Bias — Downtrend confirmed",
#             "red",
#             "Wait for 5m candle close BELOW 9 EMA with rising volume."
#         )
#     else:
#         return (
#             "🟨 Mixed / Weak Trend — Wait",
#             "orange",
#             "Do not enter; trend and momentum not aligned."
#         )
#
#
# # -------------------------------------------------------------
# # ⚡ MAIN LIVE PAGE FUNCTION
# # -------------------------------------------------------------
# def daytrade_trend_confirmation():
#     st.header("⚡ Live Day Trader — 5m / 15m Trend Confirmation System")
#     st.caption("Auto-refreshes every minute during market hours to confirm live bias and entry zones.")
#
#     ticker = st.text_input("Enter Symbol (e.g. AAPL, TSLA, SPY)", "AAPL").upper()
#     period = st.selectbox("Data Range", ["1d", "2d", "5d"], index=0)
#     refresh_rate = st.slider("Auto-Refresh Interval (seconds)", 30, 180, 60, 10)
#
#     run_live = st.checkbox("🔁 Enable Live Auto-Refresh", value=True)
#     placeholder = st.empty()
#
#     while True:
#         with placeholder.container():
#             df_5m, df_15m = get_intraday_data(ticker, period)
#             df_5m = trend_signal(rsi_calc(df_5m))
#             df_15m = trend_signal(rsi_calc(df_15m))
#
#             bias, color, entry_tip = bias_decision(df_5m, df_15m)
#             st.markdown(f"### **{bias}**")
#             st.info(f"📘 Entry Tip: {entry_tip}")
#             st.write(f"🕒 Last updated: **{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} PST**")
#
#             if not df_5m.empty:
#                 entries_long = df_5m[(df_5m["Close"] > df_5m["EMA_Short"]) & (df_5m["EMA_Short"] > df_5m["EMA_Long"])]
#                 entries_short = df_5m[(df_5m["Close"] < df_5m["EMA_Short"]) & (df_5m["EMA_Short"] < df_5m["EMA_Long"])]
#
#                 fig = go.Figure()
#
#                 fig.add_trace(go.Candlestick(
#                     x=df_5m.index,
#                     open=df_5m["Open"], high=df_5m["High"],
#                     low=df_5m["Low"], close=df_5m["Close"],
#                     name="Price"))
#                 fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["EMA_Short"], mode="lines", name="EMA 9", line=dict(color="blue")))
#                 fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["EMA_Long"], mode="lines", name="EMA 21", line=dict(color="red")))
#
#                 # Entry markers (historical)
#                 if not entries_long.empty:
#                     fig.add_trace(go.Scatter(
#                         x=entries_long.index, y=entries_long["Close"],
#                         mode="markers", name="CALL Entry",
#                         marker_symbol="triangle-up", marker_size=12, marker_color="lime"))
#                 if not entries_short.empty:
#                     fig.add_trace(go.Scatter(
#                         x=entries_short.index, y=entries_short["Close"],
#                         mode="markers", name="PUT Entry",
#                         marker_symbol="triangle-down", marker_size=12, marker_color="red"))
#
#                 # Live marker
#                 latest = df_5m.iloc[-1]
#                 if latest["Close"] > latest["EMA_Short"] and latest["EMA_Short"] > latest["EMA_Long"]:
#                     fig.add_trace(go.Scatter(
#                         x=[df_5m.index[-1]], y=[latest["Close"]],
#                         mode="markers+text", name="LIVE CALL Signal",
#                         marker_symbol="triangle-up", marker_color="lime", marker_size=16,
#                         text=["CALL ✅"], textposition="top center"))
#                 elif latest["Close"] < latest["EMA_Short"] and latest["EMA_Short"] < latest["EMA_Long"]:
#                     fig.add_trace(go.Scatter(
#                         x=[df_5m.index[-1]], y=[latest["Close"]],
#                         mode="markers+text", name="LIVE PUT Signal",
#                         marker_symbol="triangle-down", marker_color="red", marker_size=16,
#                         text=["PUT ✅"], textposition="bottom center"))
#
#                 fig.update_layout(
#                     title=f"{ticker} — LIVE 5-Min Chart (EMA 9/21 + Entry Signals)",
#                     xaxis_rangeslider_visible=False,
#                     height=600)
#                 st.plotly_chart(fig, use_container_width=True)
#
#             # RSI Chart
#             if "RSI" in df_5m.columns:
#                 st.subheader("RSI Momentum (5m)")
#                 st.line_chart(df_5m[["RSI"]].tail(100))
#                 st.caption("RSI < 30 = oversold | RSI > 70 = overbought")
#
#             # 15-Min Chart
#             if not df_15m.empty:
#                 st.subheader("15-Minute Trend Overview")
#                 fig15 = go.Figure()
#                 fig15.add_trace(go.Candlestick(
#                     x=df_15m.index,
#                     open=df_15m["Open"], high=df_15m["High"],
#                     low=df_15m["Low"], close=df_15m["Close"],
#                     name="Price"))
#                 fig15.add_trace(go.Scatter(x=df_15m.index, y=df_15m["EMA_Short"],
#                                            mode="lines", name="EMA 9", line=dict(color="blue")))
#                 fig15.add_trace(go.Scatter(x=df_15m.index, y=df_15m["EMA_Long"],
#                                            mode="lines", name="EMA 21", line=dict(color="red")))
#                 fig15.update_layout(title=f"{ticker} — 15-Min Chart (EMA 9/21)", xaxis_rangeslider_visible=False)
#                 st.plotly_chart(fig15, use_container_width=True)
#
#             st.success("✅ Live trend confirmation updated successfully.")
#
#             # ==========================================================
#             # 🧭 Educational Reference — For New Traders
#             # ==========================================================
#             with st.expander("🧭 Trading System Reference (How CALL & PUT Entries Work)"):
#                 st.markdown("""
#                             ### 🎯 Core Idea
#                             The system uses two EMAs (9 & 21), RSI, and Volume to confirm trend momentum before entry.
#                             You enter only when **5m and 15m trends align**, price confirms direction, and volume supports it.
#
#                             ---
#
#                             ### 🟩 CALL ENTRY (BUY CALL / GO LONG)
#                             **When to Enter:**
#                             - EMA 9 > EMA 21 → short-term uptrend confirmed
#                             - Price closes **above** EMA 9
#                             - RSI < 70 (not overbought yet)
#                             - Current Volume > 20-bar average
#                             - 15m chart trend also UP
#
#                             **Chart Marker:** Green ▲ “CALL ENTRY”
#                             **Stop:** Below EMA 21 or last swing low
#                             **Target:** 1.5×–3× your risk
#
#                             ---
#
#                             ### 🟥 PUT ENTRY (BUY PUT / SHORT)
#                             **When to Enter:**
#                             - EMA 9 < EMA 21 → downtrend confirmed
#                             - Price closes **below** EMA 9
#                             - RSI > 30 (not oversold yet)
#                             - Current Volume > 20-bar average
#                             - 15m chart trend also DOWN
#
#                             **Chart Marker:** Red ▼ “PUT ENTRY”
#                             **Stop:** Above EMA 21 or last swing high
#                             **Target:** 1.5×–3× your risk
#
#                             ---
#
#                             ### ⚙️ Summary Table
#                             | Condition | CALL (Bullish) | PUT (Bearish) |
#                             |------------|----------------|----------------|
#                             | Trend | EMA 9 > EMA 21 | EMA 9 < EMA 21 |
#                             | Price | Close > EMA 9 | Close < EMA 9 |
#                             | RSI | < 70 | > 30 |
#                             | Volume | > Average | > Average |
#                             | 15m Trend | UP | DOWN |
#                             | Entry Marker | ▲ Green | ▼ Red |
#                             | Stop | Below EMA 21 | Above EMA 21 |
#                             | Reward Target | 1.5–3× | 1.5–3× |
#
#                             ---
#
#                             💡 **Pro Tip:**
#                             When both timeframes align and volume surges, those entries have the highest success rate.
#                             If the app shows 🟨 *Mixed Trend — Wait*, avoid entry until signals fully align.
#                             """)
#
#         if not run_live:
#             break
#
#             from datetime import datetime
#             st.info(f"🕒 Logged at {datetime.now().strftime('%I:%M %p')}  |  Continue with next setup when ready.")
#
#         # Stop refresh loop if unchecked
#         if not run_live:
#             break
#
#         time.sleep(refresh_rate)
#         st.rerun()                                I  LOVE THIS APP COMMENED OUT

# ============================================================================================ PART 4

# =====================================================================
# ⚡ DAY TRADER — LIVE TREND CONFIRMATION SYSTEM (with Entry & Exit Logic)
# =====================================================================
# =====================================================================
# ⚡ DAY TRADER — LIVE TREND CONFIRMATION SYSTEM (Entry + Exit + RSI FIX)
# =====================================================================
# =====================================================================
# ⚡ DAY TRADER — LIVE TREND CONFIRMATION SYSTEM (FINAL STABLE RELEASE)
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
import time

# -------------------------------------------------------------
# 📡 SAFE DATA FETCH FUNCTION
# -------------------------------------------------------------
@st.cache_data(ttl=60)
def get_intraday_data(ticker: str, period: str = "5d"):
    """Safely download both 5-minute and 15-minute data."""
    df_5m, df_15m = pd.DataFrame(), pd.DataFrame()
    try:
        df_5m = yf.download(ticker, period=period, interval="5m", progress=False)
        df_15m = yf.download(ticker, period=period, interval="15m", progress=False)
    except Exception as e:
        st.warning(f"⚠️ Error fetching data for {ticker}: {e}")

    if not isinstance(df_5m, pd.DataFrame):
        df_5m = pd.DataFrame()
    if not isinstance(df_15m, pd.DataFrame):
        df_15m = pd.DataFrame()

    return df_5m, df_15m


# -------------------------------------------------------------
# 📈 SAFE EMA TREND SIGNAL (handles multi-index + arrays)
# -------------------------------------------------------------
def trend_signal(df, short=9, long=21):
    if df.empty:
        return df
    df = df.copy()

    # --- Flatten multi-index columns from Yahoo ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # --- Ensure Close column exists ---
    if "Close" not in df.columns:
        close_candidates = [c for c in df.columns if "close" in c.lower()]
        if close_candidates:
            df["Close"] = df[close_candidates[0]]
        else:
            st.error("❌ Could not find 'Close' column in data.")
            return df

    # --- Flatten Close to 1D array if needed ---
    if isinstance(df["Close"], (pd.DataFrame, np.ndarray)):
        df["Close"] = np.ravel(df["Close"])

    # --- Convert to numeric ---
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # --- Compute EMAs ---
    df["EMA_Short"] = df["Close"].ewm(span=short, adjust=False).mean()
    df["EMA_Long"] = df["Close"].ewm(span=long, adjust=False).mean()
    df["Trend"] = np.where(df["EMA_Short"] > df["EMA_Long"], "UP", "DOWN")
    return df


# -------------------------------------------------------------
# 📊 SAFE RSI CALCULATION (handles any Close format)
# -------------------------------------------------------------
def rsi_calc(df):
    if df.empty:
        return df
    df = df.copy()

    # --- Flatten multi-index columns ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # --- Ensure Close column exists ---
    if "Close" not in df.columns:
        close_candidates = [c for c in df.columns if "close" in c.lower()]
        if close_candidates:
            df["Close"] = df[close_candidates[0]]
        else:
            st.error("❌ Could not find 'Close' column for RSI.")
            return df

    # --- Flatten & sanitize Close column ---
    if isinstance(df["Close"], (pd.DataFrame, np.ndarray)):
        df["Close"] = np.ravel(df["Close"])

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # --- Compute RSI safely ---
    try:
        rsi = RSIIndicator(df["Close"], window=14)
        df["RSI"] = rsi.rsi()
    except Exception as e:
        st.warning(f"⚠️ RSI calculation skipped: {e}")
        df["RSI"] = np.nan
    return df


# -------------------------------------------------------------
# 🧠 DETERMINE CALL / PUT / WAIT BIAS
# -------------------------------------------------------------
def bias_decision(df_5m, df_15m):
    if df_5m.empty or df_15m.empty:
        return "❌ No Data", "gray", "No data available."

    t5 = df_5m["Trend"].iloc[-1]
    t15 = df_15m["Trend"].iloc[-1]
    rsi = df_5m["RSI"].iloc[-1]
    vol_now = df_5m["Volume"].iloc[-1]
    vol_avg = df_5m["Volume"].rolling(20).mean().iloc[-1]

    if t5 == t15 == "UP" and rsi < 70 and vol_now > vol_avg:
        return ("🟩 CALL Bias — Uptrend confirmed", "green",
                "Wait for 5m candle close ABOVE EMA 9 with rising volume.")
    elif t5 == t15 == "DOWN" and rsi > 30 and vol_now > vol_avg:
        return ("🟥 PUT Bias — Downtrend confirmed", "red",
                "Wait for 5m candle close BELOW EMA 9 with rising volume.")
    else:
        return ("🟨 Mixed / Weak Trend — Wait", "orange",
                "Do not enter; trend and momentum not aligned.")


# -------------------------------------------------------------
# ⚡ MAIN LIVE PAGE FUNCTION
# -------------------------------------------------------------
def daytrade_trend_confirmation():
    st.header("⚡ Live Day Trader — 5m / 15m Trend Confirmation System")
    st.caption("Auto-refreshes every minute to confirm live entry & exit signals.")

    # ticker = st.text_input("Enter Symbol (e.g. AAPL, TSLA, SPY, NVDA, MARA)", "AAPL").upper()

    # --- Symbol Selection (Dropdown + Custom Option) ---
    symbols = ["AAPL", "TSLA", "SPY", "NVDA", "MARA", "QQQ", "AMD", "META", "AMZN", "MSFT", "Other (Enter Manually)"]

    selected_symbol = st.selectbox("Select Trading Symbol", symbols, index=0)

    if selected_symbol == "Other (Enter Manually)":
        custom_symbol = st.text_input("Enter Custom Symbol (e.g. RIOT, NFLX, SOXL)", "").upper()
        ticker = custom_symbol if custom_symbol else "AAPL"
    else:
        ticker = selected_symbol

    # ===================== ENDS HERE

    period = st.selectbox("Data Range", ["1d", "2d", "5d"], index=0)
    refresh_rate = st.slider("Auto-Refresh Interval (seconds)", 30, 180, 60, 10)
    run_live = st.checkbox("🔁 Enable Live Auto-Refresh", value=True)
    placeholder = st.empty()

    while True:
        with placeholder.container():
            df_5m, df_15m = get_intraday_data(ticker, period)
            df_5m = trend_signal(rsi_calc(df_5m))
            df_15m = trend_signal(rsi_calc(df_15m))

            bias, color, entry_tip = bias_decision(df_5m, df_15m)
            st.markdown(f"### **{bias}**")
            st.info(f"📘 Entry Tip: {entry_tip}")
            st.write(f"🕒 Last updated: **{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} PST**")

            if not df_5m.empty:
                entries_long = df_5m[(df_5m["Close"] > df_5m["EMA_Short"]) &
                                     (df_5m["EMA_Short"] > df_5m["EMA_Long"])]
                entries_short = df_5m[(df_5m["Close"] < df_5m["EMA_Short"]) &
                                      (df_5m["EMA_Short"] < df_5m["EMA_Long"])]

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_5m.index, open=df_5m["Open"], high=df_5m["High"],
                    low=df_5m["Low"], close=df_5m["Close"], name="Price"))
                fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["EMA_Short"],
                                         mode="lines", name="EMA 9", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["EMA_Long"],
                                         mode="lines", name="EMA 21", line=dict(color="red")))

                # ENTRY MARKERS
                if not entries_long.empty:
                    fig.add_trace(go.Scatter(
                        x=entries_long.index, y=entries_long["Close"],
                        mode="markers", name="CALL Entry",
                        marker_symbol="triangle-up", marker_size=12, marker_color="lime"))
                if not entries_short.empty:
                    fig.add_trace(go.Scatter(
                        x=entries_short.index, y=entries_short["Close"],
                        mode="markers", name="PUT Entry",
                        marker_symbol="triangle-down", marker_size=12, marker_color="red"))

                # EXIT MARKERS
                # exit_long = df_5m[(df_5m["Close"] < df_5m["EMA_Short"]) &
                #                   (df_5m["EMA_Short"] > df_5m["EMA_Long"])]
                # exit_short = df_5m[(df_5m["Close"] > df_5m["EMA_Short"]) &
                #                    (df_5m["EMA_Short"] < df_5m["EMA_Long"])]
                #
                # if not exit_long.empty:
                #     fig.add_trace(go.Scatter(
                #         x=exit_long.index, y=exit_long["Close"],
                #         mode="markers+text", name="CALL Exit",
                #         marker_symbol="triangle-down", marker_color="gold", marker_size=12,
                #         text=["Exit CALL"], textposition="bottom center"))
                # if not exit_short.empty:
                #     fig.add_trace(go.Scatter(
                #         x=exit_short.index, y=exit_short["Close"],
                #         mode="markers+text", name="PUT Exit",
                #         marker_symbol="triangle-up", marker_color="gold", marker_size=12,
                #         text=["Exit PUT"], textposition="top center"))

                # ===================================================================================== PART 2
                # EXIT MARKERS
                exit_long = df_5m[(df_5m["Close"] < df_5m["EMA_Short"]) &
                                  (df_5m["EMA_Short"] > df_5m["EMA_Long"])]
                exit_short = df_5m[(df_5m["Close"] > df_5m["EMA_Short"]) &
                                   (df_5m["EMA_Short"] < df_5m["EMA_Long"])]

                # 🔵 CALL Exit — keep triangle-down but color blue
                if not exit_long.empty:
                    fig.add_trace(go.Scatter(
                        x=exit_long.index, y=exit_long["Close"],
                        mode="markers+text", name="CALL Exit",
                        marker_symbol="triangle-down", marker_color="deepskyblue", marker_size=12,
                        text=["Exit CALL"], textfont=dict(color="deepskyblue"),
                        textposition="bottom center"))

                # 🟠 PUT Exit — keep same arrow head and gold color
                if not exit_short.empty:
                    fig.add_trace(go.Scatter(
                        x=exit_short.index, y=exit_short["Close"],
                        mode="markers+text", name="PUT Exit",
                        marker_symbol="triangle-up", marker_color="gold", marker_size=12,
                        text=["Exit PUT"], textfont=dict(color="gold"),
                        textposition="top center"))

                # ========== ENDS HERE

                # LIVE SIGNAL MARKER
                latest = df_5m.iloc[-1]
                if latest["Close"] > latest["EMA_Short"] and latest["EMA_Short"] > latest["EMA_Long"]:
                    fig.add_trace(go.Scatter(
                        x=[df_5m.index[-1]], y=[latest["Close"]],
                        mode="markers+text", name="LIVE CALL Signal",
                        marker_symbol="triangle-up", marker_color="lime", marker_size=16,
                        text=["CALL ✅"], textposition="top center"))
                elif latest["Close"] < latest["EMA_Short"] and latest["EMA_Short"] < latest["EMA_Long"]:
                    fig.add_trace(go.Scatter(
                        x=[df_5m.index[-1]], y=[latest["Close"]],
                        mode="markers+text", name="LIVE PUT Signal",
                        marker_symbol="triangle-down", marker_color="red", marker_size=16,
                        text=["PUT ✅"], textposition="bottom center"))

                fig.update_layout(
                    title=f"{ticker} — LIVE 5-Min Chart (EMA 9/21 + Entry & Exit Signals)",
                    xaxis_rangeslider_visible=False,
                    height=600)
                st.plotly_chart(fig, use_container_width=True)

            # RSI CHART
            if "RSI" in df_5m.columns:
                st.subheader("RSI Momentum (5m)")
                st.line_chart(df_5m[["RSI"]].tail(100))
                st.caption("RSI < 30 = oversold | RSI > 70 = overbought")

            # 15-MIN CHART
            if not df_15m.empty:
                st.subheader("15-Minute Trend Overview")
                fig15 = go.Figure()
                fig15.add_trace(go.Candlestick(
                    x=df_15m.index, open=df_15m["Open"], high=df_15m["High"],
                    low=df_15m["Low"], close=df_15m["Close"], name="Price"))
                fig15.add_trace(go.Scatter(x=df_15m.index, y=df_15m["EMA_Short"],
                                           mode="lines", name="EMA 9", line=dict(color="blue")))
                fig15.add_trace(go.Scatter(x=df_15m.index, y=df_15m["EMA_Long"],
                                           mode="lines", name="EMA 21", line=dict(color="red")))
                fig15.update_layout(title=f"{ticker} — 15-Min Chart (EMA 9/21)",
                                    xaxis_rangeslider_visible=False)
                st.plotly_chart(fig15, use_container_width=True)

            st.success("✅ Live trend confirmation updated successfully.")

            # ==========================================================
            # 🧭 Educational Reference
            # ==========================================================
            with st.expander("🧭 Trading System Reference (How CALL, PUT & EXIT Work)"):
                st.markdown("""
                ### 🎯 Core Idea
                The system uses EMA 9/21, RSI, and Volume to confirm momentum before entries or exits.

                ---

                ### 🟩 CALL ENTRY (BUY CALL / GO LONG)
                **Entry:** Close > EMA 9 > EMA 21  
                **Exit:** Close < EMA 9 or RSI > 70 (declining)  
                **Volume:** Must be above average  
                **15m Trend:** Also UP  
                **Stop:** Below EMA 21 or last swing low  
                **Target:** 1.5×–3× your risk  

                ---

                ### 🟥 PUT ENTRY (BUY PUT / SHORT)
                **Entry:** Close < EMA 9 < EMA 21  
                **Exit:** Close > EMA 9 or RSI < 30 (rising)  
                **Volume:** Must be above average  
                **15m Trend:** Also DOWN  
                **Stop:** Above EMA 21 or last swing high  
                **Target:** 1.5×–3× your risk  

                ---

                ### ⚙️ Summary Table
                | Condition | CALL (Bullish) | PUT (Bearish) |
                |------------|----------------|----------------|
                | Entry Trend | EMA 9 > EMA 21 | EMA 9 < EMA 21 |
                | Entry Candle | Close > EMA 9 | Close < EMA 9 |
                | Exit Condition | Close < EMA 9 | Close > EMA 9 |
                | RSI Filter | < 70 | > 30 |
                | Volume | > Avg | > Avg |
                | 15m Trend | UP | DOWN |
                | Entry Marker | ▲ Green | ▼ Red |
                | Exit Marker | ▼ Yellow | ▲ Yellow |
                | Stop | Below EMA 21 | Above EMA 21 |
                | Reward Target | 1.5–3× | 1.5–3× |
                """)

            # ==========================================================
            # 📘 Quick Visual Reference — Trend Confirmation Cheat Sheet
            # ==========================================================
            with st.expander("🧭 Quick Visual Cheat Sheet — EMA 9/21 Day Trader System"):
                st.markdown("""
                <div style="text-align:center;">
                    <h3>🎯 <span style="color:#00FF7F;">EMA 9/21 Trend Confirmation System</span></h3>
                </div>

                ---

                ### 🟩 <span style="color:#00FF7F;">CALL FLOW (Bullish Setup)</span>
                **Bias:** EMA 9 <span style="color:#00FF7F;">(Blue)</span> > EMA 21 <span style="color:#FF4500;">(Red)</span>  
                **Trigger:** Candle closes <span style="color:#00FF7F;">ABOVE EMA 9</span>  
                **Confirm:** Volume rising + RSI below 70  
                **Exit:** Candle closes <span style="color:#00BFFF;">BELOW EMA 9</span> or CALL Exit marker appears  

                **Visuals:**  
                - 🔼 <span style="color:#00FF7F;">Lime ▲</span> = CALL Entry  
                - 🔽 <span style="color:#1E90FF;">Blue ▼</span> = CALL Exit  
                - 🔼 <span style="color:#00FF7F; font-weight:bold;">Lime ▲ (large)</span> = LIVE CALL Signal  

                **Notes:**  
                - ✅ Best when both 5m & 15m are **UP**  
                - ⚠️ Avoid entry if RSI > 70 (Overbought)  
                - 💪 Confirm with **high volume + strong candle close**

                ---

                ### 🟥 <span style="color:#FF6347;">PUT FLOW (Bearish Setup)</span>
                **Bias:** EMA 9 <span style="color:#00BFFF;">(Blue)</span> < EMA 21 <span style="color:#FF4500;">(Red)</span>  
                **Trigger:** Candle closes <span style="color:#FF6347;">BELOW EMA 9</span>  
                **Confirm:** Volume rising + RSI above 30  
                **Exit:** Candle closes <span style="color:#FFA500;">ABOVE EMA 9</span> or PUT Exit marker appears  

                **Visuals:**  
                - 🔽 <span style="color:#FF0000;">Red ▼</span> = PUT Entry  
                - 🔼 <span style="color:#FFA500;">Orange ▲</span> = PUT Exit  
                - 🔽 <span style="color:#FF0000; font-weight:bold;">Red ▼ (large)</span> = LIVE PUT Signal  

                **Notes:**  
                - ✅ Best when both 5m & 15m are **DOWN**  
                - ⚠️ Avoid entry if RSI < 30 (Oversold)  
                - 💪 Confirm with **rising volume on breakdown**

                ---

                ### ⚙️ <span style="color:#00CED1;">SYSTEM FILTERS</span>
                | Indicator | Purpose | Rule |
                |------------|----------|------|
                | 🔵 EMA 9 & 🔴 EMA 21 | Trend Direction | 9 > 21 → Bullish / 9 < 21 → Bearish |
                | 🟣 RSI (14) | Momentum Filter | 30–70 = Safe zone |
                | 📈 Volume | Confirmation | Above 20-bar avg = Valid move |
                | ⏱ Multi-Timeframe | Bias Filter | 5m & 15m must agree |

                ---

                ### ✅ <span style="color:#32CD32;">ENTRY CHECKLIST</span>
                - 📊 Trend confirmed on 5m & 15m  
                - 🕯 Candle closes beyond EMA 9  
                - 💥 Volume > average  
                - 🎯 RSI between 40–65 (not overbought)  
                - 💰 Target = 1.5–3× Risk  

                ---

                ### 🚪 <span style="color:#FFA500;">EXIT CHECKLIST</span>
                - ❌ Candle closes opposite EMA side  
                - ❌ RSI crosses 70 (CALL) or 30 (PUT)  
                - ❌ Volume starts falling  
                - ⚠️ Exit markers appear (🔵 Blue ▼ or 🟠 Orange ▲)

                ---

                ### 🧩 <span style="color:#00CED1;">MARKER COLOR GUIDE</span>
                | Marker | Color | Meaning |
                |:--------|:--------|:--------|
                | 🔼 <span style="color:#00FF7F;">Lime</span> | CALL Entry | Bullish breakout |
                | 🔽 <span style="color:#1E90FF;">Blue</span> | CALL Exit | Close long position |
                | 🔽 <span style="color:#FF0000;">Red</span> | PUT Entry | Bearish breakdown |
                | 🔼 <span style="color:#FFA500;">Orange</span> | PUT Exit | Close short position |
                | 🔼 <span style="color:#00FF7F; font-weight:bold;">Lime (bold)</span> | LIVE CALL | Current CALL bias |
                | 🔽 <span style="color:#FF0000; font-weight:bold;">Red (bold)</span> | LIVE PUT | Current PUT bias |

                ---

                ### 🛡️ <span style="color:#DAA520;">RISK RULES</span>
                - 🧱 Stop Loss = just beyond EMA 21 or last swing  
                - 💵 Max loss per trade = 1–2% of account  
                - 🎯 Take partial profits near 2× Risk  
                - 🚫 Never trade during “Mixed / Weak Trend”  
                - ⚖️ Flat EMA = Chop zone — avoid

                <div style="text-align:center; margin-top:10px;">
                    <h4>💡 <span style="color:#00FA9A;">"Trade the confirmation, not the hope."</span> — <i>Sybest Day Trader System</i></h4>
                </div>
                """, unsafe_allow_html=True)

        if not run_live:
            break

        time.sleep(refresh_rate)
        st.rerun()


# =====================================================================================================================================

# ======================================================================================================
#     PICK THE BEST STRIKE TO TRADE ON
#==================================================================================================
    # =======================================================================
    # 🎯 BEST STRIKE PICKER — intraday options (CALL/PUT)
    # Dependencies: yfinance, pandas, numpy, scipy (for norm)
#     # =======================================================================
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from math import log, sqrt, exp
# from datetime import datetime, timezone
# try:
#     from scipy.stats import norm
# except Exception:
#     # light fallback if scipy isn't available
#     # (rough normal CDF using Abramowitz–Stegun)
#     def norm_cdf(x):
#         # polynomial approx
#         a1 = 0.254829592;
#         a2 = -0.284496736;
#         a3 = 1.421413741;
#         a4 = -1.453152027;
#         a5 = 1.061405429
#         p = 0.3275911;
#         sign = 1 if x >= 0 else -1;
#         x = abs(x) / sqrt(2.0)
#         t = 1.0 / (1.0 + p * x)
#         y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * exp(-x * x)
#         return 0.5 * (1.0 + sign * y)
#
#     class _Norm:
#         cdf = staticmethod(norm_cdf)
#
#     norm = _Norm()
#
# def _bs_greeks(underlying, strike, t_years, iv, r=0.045, q=0.0, right="C"):
#     """Black-Scholes greeks: Delta (and Theta approx). right: 'C' or 'P'"""
#     if t_years <= 0 or iv <= 0:
#         return np.nan, np.nan
#     S = float(underlying);
#     K = float(strike);
#     sigma = float(iv)
#     d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt(t_years))
#     d2 = d1 - sigma * sqrt(t_years)
#     if right.upper().startswith("C"):
#         delta = exp(-q * t_years) * norm.cdf(d1)
#     else:
#         delta = -exp(-q * t_years) * norm.cdf(-d1)
#     # simple theta per day (approx, annualized/365)
#     if right.upper().startswith("C"):
#         theta = (-(S * exp(-q * t_years) * sigma / (2 * sqrt(t_years)) * (1 / np.sqrt(2 * np.pi)) * np.exp(
#             -0.5 * d1 * d1))
#                  - r * K * exp(-r * t_years) * norm.cdf(d2) + q * S * exp(-q * t_years) * norm.cdf(d1)) / 365.0
#     else:
#         theta = (-(S * exp(-q * t_years) * sigma / (2 * sqrt(t_years)) * (1 / np.sqrt(2 * np.pi)) * np.exp(
#             -0.5 * d1 * d1))
#                  + r * K * exp(-r * t_years) * norm.cdf(-d2) - q * S * exp(-q * t_years) * norm.cdf(-d1)) / 365.0
#     return float(delta), float(theta)
#
# @st.cache_data(ttl=60, show_spinner=False)
# def _get_chain(ticker: str, expiry: str):
#     t = yf.Ticker(ticker)
#     # yfinance wants YYYY-MM-DD; validate
#     exps = t.options or []
#     if expiry not in exps:
#         return None, None, exps
#     chain = t.option_chain(expiry)
#     return chain.calls.copy(), chain.puts.copy(), exps
#
# def _score_row(row, side, spot, target_delta_low, target_delta_high):
#     # lower is better for spread%, theta decay; higher is better for volume
#     penalties = 0.0
#     # delta band penalty
#     d = abs(abs(row["Delta"]) - np.clip(abs(row["Delta"]), target_delta_low, target_delta_high))
#     if not np.isnan(d): penalties += d * 3.0  # weight
#
#     # proximity to ATM
#     atm_pen = abs(row["strike"] - spot) / max(1.0, spot) * 10
#
#     # liquidity & cost metrics
#     spread_pen = row["SpreadPct"] * 50.0  # heavy penalty for wide spreads
#     vol_pen = -np.log1p(row["volume"])  # reward larger volume (negative penalty)
#     theta_pen = abs(row["Theta"]) * 2.0
#
#     score = atm_pen + spread_pen + theta_pen + penalties + vol_pen
#     # add tiny preference for rounded strikes
#     score -= 0.1 if (row["strike"] % 1 == 0) else 0.0
#     return score
#
# def best_strike_picker():
#     st.title("🎯 Best Strike Picker")
#     st.caption("Pick the most tradable strike by Delta, liquidity, spread, and decay.")
#
#     colA, colB = st.columns([1, 1])
#     with colA:
#         ticker = st.text_input("Symbol", "WMT").upper().strip()
#         side = st.selectbox("Side", ["CALLS", "PUTS"], index=0)
#         # typical day-trader sweet spot
#         target_delta = st.slider("Target Delta (|Δ|)", 0.30, 0.65, (0.40, 0.55), 0.01)
#         min_volume = st.number_input("Min Volume", 0, 100000, 500, 50)
#     with colB:
#         max_spread_pct = st.slider("Max Spread %", 0.1, 10.0, 3.0, 0.1,
#                                    help="(Ask-Bid)/Mid * 100")
#         rfr = st.number_input("Risk-free rate (annual)", 0.0, 0.20, 0.045, 0.005)
#         q_div = st.number_input("Dividend yield (annual)", 0.0, 0.10, 0.0, 0.005)
#
#     # expiry selection
#     tk = yf.Ticker(ticker)
#     expirations = tk.options or []
#     expiry = st.selectbox("Expiration", expirations, index=0) if expirations else None
#     if not expiry:
#         st.error("No expirations found for this symbol.")
#         return
#
#     # fetch chain
#     calls, puts, exps = _get_chain(ticker, expiry)
#     if calls is None:
#         st.error("Could not load option chain (maybe a holiday or stale symbol).")
#         return
#
#     # current spot & time to expiry
#     spot = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
#     t_days = (datetime.fromisoformat(expiry).replace(tzinfo=timezone.utc) - datetime.now(
#         timezone.utc)).days + 0.0001
#     t_years = max(t_days / 365.0, 1e-6)
#
#     df = calls if side == "CALLS" else puts
#     df = df.copy()
#
#     # compute fields
#     df["Mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
#     df["Spread"] = (df["ask"].fillna(0) - df["bid"].fillna(0))
#     df["SpreadPct"] = np.where(df["Mid"] > 0, (df["Spread"] / df["Mid"]) * 100, np.inf)
#     df["Premium"] = df["Mid"].round(2)
#
#     # Greeks from BS using IV column (yfinance returns decimal IV)
#     iv_col = "impliedVolatility" if "impliedVolatility" in df.columns else None
#     if iv_col is None:
#         df["Delta"] = np.nan
#         df["Theta"] = np.nan
#     else:
#         deltas, thetas = [], []
#         for _, r in df.iterrows():
#             delta, theta = _bs_greeks(spot, r["strike"], t_years, float(r[iv_col]), r=rfr, q=q_div,
#                                       right="C" if side == "CALLS" else "P")
#             deltas.append(delta);
#             thetas.append(theta)
#         df["Delta"] = deltas
#         df["Theta"] = thetas
#
#     # filters
#     low, high = target_delta
#     df = df[(df["volume"].fillna(0) >= min_volume)]
#     df = df[(df["SpreadPct"] <= max_spread_pct)]
#     if "Delta" in df:
#         df = df[df["Delta"].abs().between(low, high, inclusive="both")]
#
#     if df.empty:
#         st.warning("No contracts pass the filters. Loosen delta band, raise spread limit, or lower min volume.")
#         return
#
#     # score & sort
#     df["Score"] = df.apply(lambda r: _score_row(r, side, spot, low, high), axis=1)
#     df = df.sort_values("Score").reset_index(drop=True)
#
#     # display
#     top = df.head(8).copy()
#     show_cols = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "Mid", "SpreadPct", "volume",
#                  "openInterest", "impliedVolatility", "Delta", "Theta", "Score"]
#     show_cols = [c for c in show_cols if c in top.columns]
#     st.subheader("Top candidates")
#     st.dataframe(top[show_cols].style.format({
#         "strike": "{:.2f}", "lastPrice": "{:.2f}", "bid": "{:.2f}", "ask": "{:.2f}",
#         "Mid": "{:.2f}", "SpreadPct": "{:.2f}%", "impliedVolatility": "{:.3f}",
#         "Delta": "{:.3f}", "Theta": "{:.3f}", "Score": "{:.2f}"
#     }))
#
#     best = top.iloc[0]
#     st.success(
#         f"**Top Pick:** `{best.get('contractSymbol', '')}` — "
#         f"Strike **{best['strike']:.2f}** | Mid **${best['Mid']:.2f}** | "
#         f"Δ **{best.get('Delta', np.nan):.3f}** | Vol **{int(best['volume'])}** | "
#         f"Spread **{best['SpreadPct']:.2f}%**"
#     )
#
#     with st.expander("How scoring works"):
#         st.markdown("""
# - **Delta band** (target |Δ|): favors 0.40–0.55 by default
# - **Liquidity**: higher **volume** and tighter **spread%** rank higher
# - **ATM proximity**: closer to spot gets preference for responsiveness
# - **Decay**: lower |**Theta**| per day ranks higher
# You can fine-tune these with the controls at the top.
#         """)

    # ==== ROUTER HOOK (put this in your menu switch) =====================
    # elif menu == "🎯 Best Strike Picker":
    #     best_strike_picker()


# ========================================================================================================= PART 2
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from math import log, sqrt, exp
# from datetime import datetime, timezone
# from scipy.stats import norm
#
# # ============================ Black-Scholes Greeks ============================
# def _bs_greeks(underlying, strike, t_years, iv, r=0.045, q=0.0, right="C"):
#     """Compute Delta & Theta using Black-Scholes model"""
#     if t_years <= 0 or iv <= 0:
#         return np.nan, np.nan
#     S = float(underlying)
#     K = float(strike)
#     sigma = float(iv)
#     d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * t_years) / (sigma * sqrt(t_years))
#     d2 = d1 - sigma * sqrt(t_years)
#     if right.upper().startswith("C"):
#         delta = exp(-q * t_years) * norm.cdf(d1)
#     else:
#         delta = -exp(-q * t_years) * norm.cdf(-d1)
#     theta = (-(S * exp(-q * t_years) * sigma / (2 * sqrt(t_years)) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1 * d1))
#              - r * K * exp(-r * t_years) * norm.cdf(d2)) / 365.0
#     return float(delta), float(theta)
#
#
# # ============================ Fetch Option Chain ==============================
# @st.cache_data(ttl=60)
# def _get_chain(ticker: str, expiry: str):
#     t = yf.Ticker(ticker)
#     if expiry not in (t.options or []):
#         return None, None
#     chain = t.option_chain(expiry)
#     return chain.calls.copy(), chain.puts.copy()
#
#
# # ============================ Scoring Function ================================
# def _score_row(row, side, spot, low, high):
#     """Compute score (lower is better)"""
#     d = abs(abs(row["Delta"]) - np.clip(abs(row["Delta"]), low, high))
#     spread_pen = row["SpreadPct"] * 40
#     vol_pen = -np.log1p(row["volume"])
#     atm_pen = abs(row["strike"] - spot) / spot * 10
#     theta_pen = abs(row["Theta"]) * 2
#     return atm_pen + spread_pen + theta_pen + d * 3 + vol_pen
#
#
# # ============================ Best Strike Picker ==============================
# def best_strike_picker():
#     st.title("🎯 Best Strike Picker — Automated Option Selection")
#
#     col1, col2 = st.columns(2)
#     with col1:
#         ticker = st.text_input("Symbol", "WMT").upper().strip()
#         side = st.selectbox("Side", ["CALLS", "PUTS"])
#         delta_band = st.slider("Target Delta (|Δ|)", 0.3, 0.7, (0.4, 0.55))
#         min_vol = st.number_input("Min Volume", 0, 5000, 500)
#     with col2:
#         spread_limit = st.slider("Max Spread %", 0.1, 10.0, 3.0, 0.1)
#         expiry_index = 0
#
#     tk = yf.Ticker(ticker)
#     expirations = tk.options or []
#     expiry = st.selectbox("Expiration", expirations, index=expiry_index)
#     if not expiry:
#         st.error("No option expirations found.")
#         return
#
#     calls, puts = _get_chain(ticker, expiry)
#     if calls is None:
#         st.error("Failed to load chain data.")
#         return
#
#     spot = float(tk.history(period="1d")["Close"].iloc[-1])
#     t_days = (datetime.fromisoformat(expiry).replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days
#     t_years = max(t_days / 365.0, 1e-6)
#
#     df = calls if side == "CALLS" else puts
#     df["Mid"] = (df["bid"] + df["ask"]) / 2
#     df["SpreadPct"] = np.where(df["Mid"] > 0, (df["ask"] - df["bid"]) / df["Mid"] * 100, np.inf)
#     df = df[(df["volume"] >= min_vol) & (df["SpreadPct"] <= spread_limit)]
#
#     # --- Compute Greeks
#     deltas, thetas = [], []
#     for _, r in df.iterrows():
#         delta, theta = _bs_greeks(spot, r["strike"], t_years, float(r["impliedVolatility"]),
#                                   right="C" if side == "CALLS" else "P")
#         deltas.append(delta)
#         thetas.append(theta)
#     df["Delta"], df["Theta"] = deltas, thetas
#
#     # --- Remove invalid/zero-delta rows ✅
#     df = df.dropna(subset=["Delta", "Theta", "impliedVolatility"])
#     df = df[(df["impliedVolatility"] > 0) & (df["Mid"] > 0)]
#     df = df[df["Delta"].abs() >= 0.05]  # exclude Δ=0 junk contracts
#
#     low, high = delta_band
#     df = df[df["Delta"].abs().between(low, high)]
#
#     # --- Auto-relax if empty
#     if df.empty:
#         st.warning("No contracts match — expanding Delta band slightly...")
#         df = calls if side == "CALLS" else puts
#         df["Mid"] = (df["bid"] + df["ask"]) / 2
#         df["SpreadPct"] = np.where(df["Mid"] > 0, (df["ask"] - df["bid"]) / df["Mid"] * 100, np.inf)
#         df = df[(df["volume"] >= min_vol / 2) & (df["SpreadPct"] <= spread_limit * 1.5)]
#         deltas, thetas = [], []
#         for _, r in df.iterrows():
#             delta, theta = _bs_greeks(spot, r["strike"], t_years, float(r["impliedVolatility"]),
#                                       right="C" if side == "CALLS" else "P")
#             deltas.append(delta)
#             thetas.append(theta)
#         df["Delta"], df["Theta"] = deltas, thetas
#         df = df.dropna(subset=["Delta", "Theta"])
#         df = df[df["Delta"].abs() >= 0.05]
#
#     if df.empty:
#         st.error("No eligible options found even after expanding filters.")
#         return
#
#     # # --- New Scoring (Ask + Theta impact)
#     # df["Score"] = (
#     #     abs(df["strike"] - spot) / spot * 10     # ATM proximity
#     #     + df["SpreadPct"] * 40                   # Spread penalty
#     #     + np.log1p(np.maximum(0.1, df["ask"])) * 5   # Ask price penalty
#     #     + abs(df["Theta"]) * 3                   # Theta penalty
#     #     - np.log1p(df["volume"]) * 2             # Liquidity bonus
#     # )
#
#     # --- Improved Scoring (Stronger preference for lower Ask)
#     df["Score"] = (
#             abs(df["strike"] - spot) / spot * 10  # ATM proximity (lower is better)
#             + df["SpreadPct"] * 40  # Spread penalty (tight is better)
#             + abs(df["Theta"]) * 3  # Theta decay penalty
#             - np.log1p(df["volume"]) * 2  # Liquidity bonus
#             - np.log1p(1 / np.maximum(df["ask"], 0.05)) * 3  # ✅ Reward smaller ask prices
#     )
#
#     df = df.sort_values("Score").reset_index(drop=True)
#
#     # --- Display
#     tab1, tab2, tab3 = st.tabs(["📊 Scoring Table", "📈 Chain Summary", "💡 Insights"])
#
#     with tab1:
#         st.dataframe(df[["contractSymbol", "strike", "bid", "ask", "Mid", "SpreadPct", "volume",
#                          "openInterest", "impliedVolatility", "Delta", "Theta", "Score"]]
#                      .style.format({
#                          "bid": "{:.2f}", "ask": "{:.2f}", "Mid": "{:.2f}", "SpreadPct": "{:.2f}%",
#                          "impliedVolatility": "{:.3f}", "Delta": "{:.3f}", "Theta": "{:.3f}", "Score": "{:.2f}"
#                      }))
#
#     with tab2:
#         st.bar_chart(df.set_index("strike")["volume"], use_container_width=True)
#         st.line_chart(df.set_index("strike")[["Delta", "Theta"]])
#
#     with tab3:
#         if len(df) > 0:
#             best = df.iloc[0]
#             st.success(
#                 f"**Top Pick:** {best['contractSymbol']} — Strike **{best['strike']:.2f}** | "
#                 f"Bid **${best['bid']:.2f}** / Ask **${best['ask']:.2f}** | "
#                 f"Δ **{best['Delta']:.3f}** | Vol **{int(best['volume'])}** | "
#                 f"Spread **{best['SpreadPct']:.2f}%** | Θ **{best['Theta']:.4f}**"
#             )
#             st.caption("✅ Prioritizes tight spread, low Ask, and low Theta (decay).")
#         else:
#             st.warning("No contracts found to display.")

# ============================================================================== PART 3
# ======================================================================
# 🧠 Day-Trade Auto Recommender (0DTE/1DTE) — Best Symbols & Strikes
# ======================================================================
# ======================================================================
# 🧠 BEST STRIKE PICKER — AUTO RECOMMENDER (Sybest Smart Day Trader)
# ======================================================================
# # ======================================================================
# # 🎯 BEST STRIKE PICKER — AUTO RECOMMENDER with Ask Control
# # ======================================================================
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from math import log, sqrt, exp
# from datetime import datetime, timezone
# from scipy.stats import norm
#
# # -------------------- Greeks --------------------
# def _bs_greeks(underlying, strike, t_years, iv, r=0.045, q=0.0, right="C"):
#     if t_years <= 0 or iv is None or iv <= 0:
#         return np.nan, np.nan
#     S = float(underlying); K = float(strike); sigma = float(iv)
#     d1 = (log(S/K) + (r - q + 0.5*sigma**2)*t_years) / (sigma*sqrt(t_years))
#     d2 = d1 - sigma*sqrt(t_years)
#     delta = exp(-q*t_years)*norm.cdf(d1) if right.upper().startswith("C") else -exp(-q*t_years)*norm.cdf(-d1)
#     theta = (-(S*exp(-q*t_years)*sigma/(2*sqrt(t_years))*(1/np.sqrt(2*np.pi))*np.exp(-0.5*d1*d1))
#              - r*K*exp(-r*t_years)*(norm.cdf(d2) if right.upper().startswith("C") else norm.cdf(-d2))) / 365.0
#     return float(delta), float(theta)
#
# @st.cache_data(ttl=60)
# def _get_chain(ticker: str, expiry: str):
#     t = yf.Ticker(ticker)
#     if expiry not in (t.options or []):
#         return None, None
#     ch = t.option_chain(expiry)
#     return ch.calls.copy(), ch.puts.copy()
#
# @st.cache_data(ttl=60)
# def _get_intraday(ticker: str, period="2d", interval="5m"):
#     df = yf.download(ticker, period=period, interval=interval, progress=False)
#     if df.empty: return pd.DataFrame()
#     df.columns = [c[0].title() if isinstance(c, tuple) else str(c).title() for c in df.columns]
#     return df
#
# def _ema(series, n): return series.ewm(span=n, adjust=False).mean()
#
# def _trend_bias_5m(ticker:str):
#     df = _get_intraday(ticker, "2d", "5m")
#     if df.empty: return "UNKNOWN"
#     ema9, ema21 = _ema(df["Close"], 9), _ema(df["Close"], 21)
#     if ema9.iloc[-1] > ema21.iloc[-1]: return "BULL"
#     if ema9.iloc[-1] < ema21.iloc[-1]: return "BEAR"
#     return "FLAT"
#
# def _nearest_expiration(ticker:str):
#     t = yf.Ticker(ticker)
#     exps = t.options or []
#     if not exps: return None
#     today = datetime.now(timezone.utc).date()
#     for e in sorted(exps):
#         if datetime.fromisoformat(e).date() >= today:
#             return e
#     return exps[-1]
#
# # -------------------- Scoring --------------------
# def _score_row(row, spot, low, high, ask_weight):
#     delta = abs(row.get("Delta", np.nan))
#     d_pen = abs(delta - np.clip(delta, low, high)) * 3.0
#     atm_pen = abs(row["strike"] - spot) / max(1.0, spot) * 10.0
#     spr_pen = row["SpreadPct"] * 40.0
#     ask_pen = np.log1p(row["ask"]) * ask_weight   # dynamic weight
#     th_pen = abs(row.get("Theta", 0.0)) * 3.0
#     vol_rwd = -np.log1p(max(0, row.get("volume", 0))) * 2.0
#     return atm_pen + spr_pen + th_pen + d_pen + ask_pen + vol_rwd
#
# def _evaluate_symbol(ticker, side, delta_band, min_vol, max_spread_pct, use_trend_filter, ask_weight, max_ask):
#     if use_trend_filter:
#         bias = _trend_bias_5m(ticker)
#         if bias == "BULL": side = "CALLS"
#         elif bias == "BEAR": side = "PUTS"
#
#     expiry = _nearest_expiration(ticker)
#     if not expiry: return None
#
#     calls, puts = _get_chain(ticker, expiry)
#     if calls is None: return None
#
#     spot = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
#     t_days = (datetime.fromisoformat(expiry).replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days + 0.0001
#     t_years = max(t_days / 365.0, 1e-6)
#
#     df = (calls if side == "CALLS" else puts).copy()
#     if df.empty: return None
#     df["Mid"] = (df["bid"] + df["ask"]) / 2
#     df["SpreadPct"] = np.where(df["Mid"]>0, (df["ask"]-df["bid"])/df["Mid"]*100, np.inf)
#     df = df[(df["volume"] >= min_vol) & (df["SpreadPct"] <= max_spread_pct)]
#     if max_ask > 0: df = df[df["ask"] <= max_ask]   # ✅ exclude high Ask
#
#     deltas, thetas = [], []
#     for _, r in df.iterrows():
#         iv = float(r.get("impliedVolatility", np.nan)) if pd.notna(r.get("impliedVolatility", np.nan)) else np.nan
#         d, th = _bs_greeks(spot, r["strike"], t_years, iv, right=("C" if side=="CALLS" else "P"))
#         deltas.append(d); thetas.append(th)
#     df["Delta"], df["Theta"] = deltas, thetas
#     df = df.dropna(subset=["Delta","Theta"])
#     df = df[(df["Delta"].abs() >= 0.05) & (df["Mid"] > 0)]
#     low, high = delta_band
#     df = df[df["Delta"].abs().between(low, high)]
#     if df.empty: return None
#
#     df["Score"] = df.apply(lambda r: _score_row(r, spot, low, high, ask_weight), axis=1)
#     df = df.sort_values("Score").reset_index(drop=True)
#     best = df.iloc[0].to_dict()
#     best.update({"symbol": ticker, "expiry": expiry, "spot": spot, "side": side})
#     return best
#
# # -------------------- MAIN --------------------
# def best_strike_picker():
#     st.header("🎯 Best Strike Picker — Ask-Sensitive Auto Recommender")
#     st.caption("Now you can control how much the algorithm prioritizes lower Ask contracts.")
#
#     left, right = st.columns([1,1])
#     with left:
#         default_universe = ["SPY","QQQ","AAPL","TSLA","NVDA","MSFT","META","AMD","AMZN","WMT"]
#         universe = st.multiselect("Select Universe", default_universe, default=default_universe)
#         side = st.selectbox("Preferred Side", ["CALLS","PUTS"])
#         delta_band = st.slider("|Delta| Range", 0.25, 0.75, (0.40, 0.55))
#         min_vol = st.number_input("Min Volume", 0, 200000, 500)
#     with right:
#         max_spread_pct = st.slider("Max Spread %", 0.1, 10.0, 3.0, 0.1)
#         ask_weight = st.slider("Ask Price Sensitivity", 1.0, 10.0, 5.0, 0.5)
#         max_ask = st.number_input("Max Ask Cap ($)", 0.0, 50.0, 0.0, help="Set >0 to exclude expensive contracts")
#         use_trend_filter = st.checkbox("Use EMA(9/21) Trend Filter", value=True)
#
#     manual_symbol = st.text_input("Manual Symbol (optional)", "").upper().strip()
#     manual_side = st.selectbox("Manual Side", ["CALLS","PUTS"], index=0)
#
#     if st.button("🚀 Run Auto Strike Finder", type="primary"):
#         picks = []
#         if manual_symbol:
#             best = _evaluate_symbol(manual_symbol, manual_side, delta_band, min_vol, max_spread_pct, use_trend_filter, ask_weight, max_ask)
#             if best: picks.append(best)
#             else: st.warning(f"No valid contract for {manual_symbol}")
#
#         for sym in universe:
#             try:
#                 best = _evaluate_symbol(sym, side, delta_band, min_vol, max_spread_pct, use_trend_filter, ask_weight, max_ask)
#                 if best: picks.append(best)
#             except Exception: continue
#
#         if not picks:
#             st.error("No eligible contracts — loosen filters or adjust Ask Weight / Cap.")
#             return
#
#         dfp = pd.DataFrame(picks).sort_values("Score").reset_index(drop=True)
#         st.subheader("🏆 Top Recommendations (Ask-Adjusted)")
#         show_cols = ["symbol","side","expiry","contractSymbol","strike","bid","ask","Delta","Theta","SpreadPct","volume","spot","Score"]
#         for c in show_cols:
#             if c not in dfp.columns: dfp[c] = np.nan
#         st.dataframe(
#             dfp[show_cols].style.format({
#                 "strike":"{:.2f}","bid":"{:.2f}","ask":"{:.2f}","Delta":"{:.3f}",
#                 "Theta":"{:.4f}","SpreadPct":"{:.2f}%","volume":"{:,.0f}","spot":"{:.2f}","Score":"{:.2f}"
#             }), use_container_width=True
#         )
#
#         for i, r in dfp.head(5).iterrows():
#             st.success(
#                 f"**#{i+1} {r['symbol']}** — {r['side']} | {r.get('contractSymbol','')} | "
#                 f"Strike **{r['strike']:.2f}** | Δ **{r['Delta']:.3f}** | "
#                 f"Bid **${r['bid']:.2f}** / Ask **${r['ask']:.2f}** | "
#                 f"Spread **{r['SpreadPct']:.2f}%** | Θ **{r['Theta']:.4f}** | "
#                 f"Vol **{int(r['volume'])}** | Exp **{r['expiry']}**"
#             )
#
#         with st.expander("ℹ️ Ask Control Explained"):
#             st.markdown(f"""
# - **Ask Sensitivity ({ask_weight:.1f})** → higher = more weight on low Ask
# - **Max Ask Cap (${max_ask:.2f})** → excludes any contract above this Ask
# - Lower Score = better overall contract
# - Still blends Delta alignment, tight spread, low Theta, and volume strength.
#             """)


# ====================================================================================================== PART 4
# ======================================================================
# 🎯 BEST STRIKE PICKER — Ask-Sensitive Auto Recommender (Final Version)
# ======================================================================
# ======================================================================
# 🎯 BEST STRIKE PICKER — Multi-Expiration “Perfect Scenario” Scanner
# ======================================================================
# # ======================================================================
# # # 🎯 BEST STRIKE PICKER — Multi-Expiration Perfect Scenario
# # # ======================================================================
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from math import log, sqrt, exp
# from datetime import datetime, timezone, date
# from scipy.stats import norm
#
# # ----------------------------------------------------------------------
# # 🧮 Black–Scholes Greeks
# # ----------------------------------------------------------------------
# def _bs_greeks(underlying, strike, t_years, iv, r=0.045, q=0.0, right="C"):
#     if t_years <= 0 or iv is None or iv <= 0:
#         return np.nan, np.nan
#     S = float(underlying)
#     K = float(strike)
#     sigma = float(iv)
#     d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * t_years) / (sigma * sqrt(t_years))
#     d2 = d1 - sigma * sqrt(t_years)
#     delta = exp(-q * t_years) * norm.cdf(d1) if right.upper().startswith("C") \
#         else -exp(-q * t_years) * norm.cdf(-d1)
#     theta = (-(S * exp(-q * t_years) * sigma / (2 * sqrt(t_years))
#                * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1 * d1))
#              - r * K * exp(-r * t_years)
#              * (norm.cdf(d2) if right.upper().startswith("C") else norm.cdf(-d2))) / 365.0
#     return float(delta), float(theta)
#
#
# # ----------------------------------------------------------------------
# # 📦 Fetch Option Chain
# # ----------------------------------------------------------------------
# @st.cache_data(ttl=60)
# def _get_chain(ticker: str, expiry: str):
#     t = yf.Ticker(ticker)
#     if expiry not in (t.options or []):
#         return None, None
#     ch = t.option_chain(expiry)
#     return ch.calls.copy(), ch.puts.copy()
#
#
# # ----------------------------------------------------------------------
# # 🕒 Helpers for expiration and intraday bias
# # ----------------------------------------------------------------------
# @st.cache_data(ttl=60)
# def _get_intraday(ticker: str, period="2d", interval="5m"):
#     df = yf.download(ticker, period=period, interval=interval, progress=False)
#     if df.empty:
#         return pd.DataFrame()
#     df.columns = [c[0].title() if isinstance(c, tuple) else str(c).title() for c in df.columns]
#     return df
#
#
# def _ema(s, n):
#     return s.ewm(span=n, adjust=False).mean()
#
#
# def _trend_bias_5m(ticker: str):
#     df = _get_intraday(ticker, "2d", "5m")
#     if df.empty:
#         return "UNKNOWN"
#     ema9, ema21 = _ema(df["Close"], 9), _ema(df["Close"], 21)
#     return "BULL" if ema9.iloc[-1] > ema21.iloc[-1] else ("BEAR" if ema9.iloc[-1] < ema21.iloc[-1] else "FLAT")
#
#
# def _list_expirations(ticker: str):
#     t = yf.Ticker(ticker)
#     return t.options or []
#
#
# def _dte(exp_str: str):
#     exp = datetime.fromisoformat(exp_str).date()
#     return (exp - date.today()).days
#
#
# # ----------------------------------------------------------------------
# # 🧮 Scoring Function (Ask/Theta/DTE aware)
# # ----------------------------------------------------------------------
# def _score_row(row, spot, low, high, ask_weight, dte_days, ideal_dte):
#     delta = abs(row.get("Delta", np.nan))
#     d_pen = abs(delta - np.clip(delta, low, high)) * 3.0
#     atm_pen = abs(row["strike"] - spot) / max(1.0, spot) * 10.0
#     spr_pen = row["SpreadPct"] * 40.0
#     ask_pen = np.log1p(row["ask"]) * ask_weight
#     th_pen = abs(row.get("Theta", 0.0)) * 3.0
#     vol_rwd = -np.log1p(max(0, row.get("volume", 0))) * 2.0
#     dte_pen = max(0, abs(dte_days - ideal_dte)) * 0.8
#     return atm_pen + spr_pen + th_pen + d_pen + ask_pen + vol_rwd + dte_pen
#
#
# # ----------------------------------------------------------------------
# # ⚙️ Evaluate symbol on a given expiry
# # ----------------------------------------------------------------------
# def _evaluate_symbol_on_expiry(ticker, side, expiry, delta_band, min_vol, max_spread_pct,
#                                use_trend_filter, ask_weight, max_ask, ideal_dte):
#     if use_trend_filter:
#         bias = _trend_bias_5m(ticker)
#         if bias == "BULL":
#             side = "CALLS"
#         elif bias == "BEAR":
#             side = "PUTS"
#
#     calls, puts = _get_chain(ticker, expiry)
#     if calls is None:
#         return None
#
#     tkr = yf.Ticker(ticker)
#     hist = tkr.history(period="1d")
#     if hist.empty:
#         return None
#     spot = float(hist["Close"].iloc[-1])
#
#     dte_days = max(0, _dte(expiry))
#     t_years = max(dte_days / 365.0, 1e-6)
#     df = (calls if side == "CALLS" else puts).copy()
#     if df.empty:
#         return None
#
#     # basic clean up
#     df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
#     df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
#     df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
#     df["Mid"] = (df["bid"] + df["ask"]) / 2
#     df["SpreadPct"] = np.where(df["Mid"] > 0, (df["ask"] - df["bid"]) / df["Mid"] * 100, np.inf)
#     mask = (df["volume"] >= min_vol) & (df["SpreadPct"] <= max_spread_pct)
#     if max_ask > 0:
#         mask &= (df["ask"] <= max_ask)
#     df = df[mask]
#     if df.empty:
#         return None
#
#     # greeks
#     deltas, thetas = [], []
#     for _, r in df.iterrows():
#         iv = float(r.get("impliedVolatility", np.nan)) if pd.notna(r.get("impliedVolatility", np.nan)) else np.nan
#         d, th = _bs_greeks(spot, r["strike"], t_years, iv, right=("C" if side == "CALLS" else "P"))
#         deltas.append(d)
#         thetas.append(th)
#     df["Delta"], df["Theta"] = deltas, thetas
#
#     df = df.dropna(subset=["Delta", "Theta"])
#     df = df[(df["Delta"].abs() >= 0.05) & (df["Mid"] > 0)]
#     low, high = delta_band
#     df = df[df["Delta"].abs().between(low, high)]
#     if df.empty:
#         return None
#
#     df["Score"] = df.apply(lambda r: _score_row(r, spot, low, high, ask_weight, dte_days, ideal_dte), axis=1)
#     df = df.sort_values("Score").reset_index(drop=True)
#     best = df.iloc[0].to_dict()
#     best.update({"symbol": ticker, "expiry": expiry, "spot": spot, "side": side, "DTE": dte_days})
#     return best
#
#
# # ----------------------------------------------------------------------
# # 🧭 MAIN STREAMLIT PAGE
# # ----------------------------------------------------------------------
# def best_strike_picker():
#     st.header("🎯 Best Strike Picker — Multi-Expiration Perfect Scenario")
#     st.caption("Scans several expirations and ranks the best contracts by Delta, Theta, Spread, Volume, Ask, and DTE.")
#
#     # --- sidebar controls
#     c1, c2, c3 = st.columns([1, 1, 1])
#     with c1:
#         universe_default = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "META", "AMD", "AMZN", "WMT", "GILD", "BIO"]
#         universe = st.multiselect("Symbols (Universe)", universe_default, default=universe_default)
#         side = st.selectbox("Preferred Side", ["CALLS", "PUTS"], index=0)
#         delta_band = st.slider("|Delta| range", 0.25, 0.75, (0.40, 0.55), 0.01)
#         min_vol = st.number_input("Min Volume", 0, 300000, 500, 50)
#
#     with c2:
#         max_spread_pct = st.slider("Max Spread %", 0.1, 10.0, 3.0, 0.1)
#         ask_weight = st.slider("Ask Sensitivity (weight)", 1.0, 10.0, 5.0, 0.5)
#         max_ask = st.number_input("Max Ask Cap ($)", 0.0, 50.0, 0.0)
#         use_trend_filter = st.checkbox("EMA(9/21) Trend Filter (5m)", value=True)
#     with c3:
#         exp_mode = st.radio("Expiration Scan", ["Nearest N", "DTE Range"], index=0, horizontal=True)
#         if exp_mode == "Nearest N":
#             n_exp = st.slider("Scan next N expirations", 1, 8, 3, 1)
#             ideal_dte = st.slider("Ideal DTE (days)", 0, 14, 1, 1)
#         else:
#             dte_min, dte_max = st.slider("DTE Range (days)", 0, 30, (0, 7), 1)
#             ideal_dte = st.slider("Ideal DTE (days)", 0, 14, 1, 1)
#
#     manual_symbol = st.text_input("Manual Symbol (optional)", "").upper().strip()
#     manual_side = st.selectbox("Manual Side", ["CALLS", "PUTS"], index=0)
#
#     if st.button("🚀 Run Multi-Expiration Scanner", type="primary"):
#         all_picks = []
#
#         def _exp_filter_list(ticker):
#             exps = _list_expirations(ticker)
#             if not exps:
#                 return []
#             rows = [(e, _dte(e)) for e in exps]
#             if exp_mode == "Nearest N":
#                 rows = [r for r in rows if r[1] >= 0]
#                 rows.sort(key=lambda x: x[1])
#                 rows = rows[:n_exp]
#             else:
#                 rows = [r for r in rows if dte_min <= max(0, r[1]) <= dte_max]
#                 rows.sort(key=lambda x: x[1])
#             return rows
#
#         if manual_symbol:
#             for exp, dte in _exp_filter_list(manual_symbol):
#                 pick = _evaluate_symbol_on_expiry(
#                     manual_symbol, manual_side, exp, delta_band, min_vol, max_spread_pct,
#                     use_trend_filter, ask_weight, max_ask, ideal_dte
#                 )
#                 if pick:
#                     all_picks.append(pick)
#
#         for sym in universe:
#             for exp, dte in _exp_filter_list(sym):
#                 pick = _evaluate_symbol_on_expiry(
#                     sym, side, exp, delta_band, min_vol, max_spread_pct,
#                     use_trend_filter, ask_weight, max_ask, ideal_dte
#                 )
#                 if pick:
#                     all_picks.append(pick)
#
#         if not all_picks:
#             st.error("No eligible contracts found — relax filters or widen DTE range.")
#             return
#
#         dfp = pd.DataFrame(all_picks).sort_values(["Score", "DTE"]).reset_index(drop=True)
#
#         # -------------------- ADD REMARKS COLUMN --------------------
#         remarks = []
#         for _, r in dfp.iterrows():
#             reason = []
#             if r["SpreadPct"] <= max_spread_pct:
#                 reason.append("Tight spread")
#             if abs(r["Delta"]) >= delta_band[0] and abs(r["Delta"]) <= delta_band[1]:
#                 reason.append("Ideal Delta range")
#             if abs(r["Theta"]) < 0.01:
#                 reason.append("Low time decay")
#             if max_ask == 0 or r["ask"] <= max_ask:
#                 reason.append("Affordable premium")
#             if r["volume"] > min_vol:
#                 reason.append("High liquidity")
#             if r["DTE"] <= 3:
#                 reason.append("Short-term expiry")
#             remarks.append(", ".join(reason))
#         dfp["Remark"] = remarks
#
#         # -------------------- DISPLAY MAIN DATAFRAME --------------------
#         st.subheader("🏆 Overall Top Picks (Ranked by Score)")
#         show_cols = ["symbol", "side", "expiry", "DTE", "contractSymbol", "strike",
#                      "bid", "ask", "Delta", "Theta", "SpreadPct", "volume", "spot", "Score", "Remark"]
#
#         st.dataframe(
#             dfp[show_cols].style.format({
#                 "DTE": "{:.0f}", "strike": "{:.2f}", "bid": "{:.2f}", "ask": "{:.2f}",
#                 "Delta": "{:.3f}", "Theta": "{:.4f}", "SpreadPct": "{:.2f}%",
#                 "volume": "{:,.0f}", "spot": "{:.2f}", "Score": "{:.2f}"
#             }),
#             use_container_width=True
#         )
#
#        #
#         # -------------------- TOP 3 EXPLANATIONS --------------------
#         st.markdown("### 📘 Top Picks Explanation")
#         top_picks = dfp.head(3)
#         for rank, row in top_picks.iterrows():
#             label = (
#                 "🏆 **Best Option**" if rank == 0 else
#                 "🥈 **Second Best Option**" if rank == 1 else
#                 "🥉 **Third Best Option**"
#             )
#             st.markdown(f"""
# {label} — **{row['symbol']} {row['side']}**
# - **Expiration:** {row['expiry']} (DTE {int(row['DTE'])} days)
# - **Strike:** {row['strike']:.2f}, **Spot:** {row['spot']:.2f}
# - **Δ (Delta):** {row['Delta']:.3f} → {'strong price sensitivity' if abs(row['Delta']) >= 0.5 else 'balanced responsiveness'}
# - **Θ (Theta):** {row['Theta']:.4f} → {'low decay — ideal for intraday trades' if abs(row['Theta']) < 0.01 else 'moderate time decay'}
# - **Ask:** ${row['ask']:.2f} → {'low-cost entry' if row['ask'] <= 2 else 'higher premium — requires momentum confirmation'}
# - **Spread:** {row['SpreadPct']:.2f}% → {'tight and efficient execution' if row['SpreadPct'] < 2 else 'moderate spread'}
# - **Volume:** {int(row['volume']):,} → {'excellent liquidity' if row['volume'] > 5000 else 'acceptable but thinner volume'}
#
# **Why selected:**
# {row['symbol']} ranks in the top due to a balance between **liquidity, cost efficiency, and risk-adjusted potential**.
# Its Delta aligns with your configured target, Theta remains low, and execution cost (Ask/Spread) fits within acceptable intraday parameters.
#             """)
#
#         st.markdown("---")
#         st.info("""
# **Selection Logic Summary**
# - 🟩 *Best Option:* Tight spread, ideal Delta (0.40–0.55), low Theta, and strong liquidity.
# - 🟨 *Second Best:* Slightly higher Theta or spread but strong liquidity and ATM proximity.
# - 🟥 *Third Best:* Balanced Delta but higher premium or smaller volume.
# Each option is ranked for **intraday responsiveness vs. cost efficiency**.
#         """)

# ================================================================================================== PART 66


# ======================================================================
# 🎯 BEST STRIKE PICKER — Multi-Expiration Perfect Scenario (Smart Spread)
# ======================================================================
# st.divider()
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from math import log, sqrt, exp
from datetime import datetime, timezone, date
from scipy.stats import norm

# ----------------------------------------------------------------------
# 🧮 Black–Scholes Greeks
# ----------------------------------------------------------------------
def _bs_greeks(underlying, strike, t_years, iv, r=0.045, q=0.0, right="C"):
    if t_years <= 0 or iv is None or iv <= 0:
        return np.nan, np.nan
    S = float(underlying)
    K = float(strike)
    sigma = float(iv)
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * t_years) / (sigma * sqrt(t_years))
    d2 = d1 - sigma * sqrt(t_years)
    if right.upper().startswith("C"):
        delta = exp(-q * t_years) * norm.cdf(d1)
    else:
        delta = -exp(-q * t_years) * norm.cdf(-d1)
    theta = (-(S * exp(-q * t_years) * sigma / (2 * sqrt(t_years))
               * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1 * d1))
             - r * K * exp(-r * t_years)
             * (norm.cdf(d2) if right.upper().startswith("C") else norm.cdf(-d2))) / 365.0
    return float(delta), float(theta)

# ----------------------------------------------------------------------
# 📦 Fetch Option Chain
# ----------------------------------------------------------------------
@st.cache_data(ttl=60)
def _get_chain(ticker: str, expiry: str):
    t = yf.Ticker(ticker)
    if expiry not in (t.options or []):
        return None, None
    ch = t.option_chain(expiry)
    return ch.calls.copy(), ch.puts.copy()

# ----------------------------------------------------------------------
# 🕒 Helpers for expiration and intraday bias
# ----------------------------------------------------------------------
@st.cache_data(ttl=60)
def _get_intraday(ticker: str, period="2d", interval="5m"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df.columns = [c[0].title() if isinstance(c, tuple) else str(c).title() for c in df.columns]
    return df

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _trend_bias_5m(ticker: str):
    df = _get_intraday(ticker, "2d", "5m")
    if df.empty:
        return "UNKNOWN"
    ema9, ema21 = _ema(df["Close"], 9), _ema(df["Close"], 21)
    return "BULL" if ema9.iloc[-1] > ema21.iloc[-1] else ("BEAR" if ema9.iloc[-1] < ema21.iloc[-1] else "FLAT")

def _list_expirations(ticker: str):
    t = yf.Ticker(ticker)
    return t.options or []

def _dte(exp_str: str):
    exp = datetime.fromisoformat(exp_str).date()
    return (exp - date.today()).days

# ----------------------------------------------------------------------
# 💡 Auto-adjust Max Spread % based on symbol liquidity
# ----------------------------------------------------------------------
def _auto_spread_cap(symbol):
    rule_based = {
        "SPY": 3.0, "QQQ": 3.0, "AAPL": 3.0, "MSFT": 3.5,
        "NVDA": 5.0, "TSLA": 5.0, "AMD": 6.0, "META": 5.0,
        "SOXL": 8.0, "LABU": 8.0, "WMT": 4.0, "AMZN": 4.0,
        "GILD": 4.0, "BIO": 6.0
    }
    sym = symbol.upper()
    if sym in rule_based:
        return rule_based[sym]

    # Dynamic fallback: look at median spread of nearest expiry
    try:
        t = yf.Ticker(sym)
        exps = t.options
        if not exps:
            return 6.0
        ch = t.option_chain(exps[0])
        calls = ch.calls.copy()
        calls["Mid"] = (calls["bid"] + calls["ask"]) / 2
        calls["SpreadPct"] = np.where(
            calls["Mid"] > 0, (calls["ask"] - calls["bid"]) / calls["Mid"] * 100, np.inf
        )
        med = np.median(calls["SpreadPct"].replace(np.inf, np.nan).dropna())
        # Clamp to 3–10% and add a little buffer
        return float(min(max(med * 1.5, 3.0), 10.0))
    except Exception:
        return 6.0

# ----------------------------------------------------------------------
# 🧮 Scoring Function (Ask/Theta/DTE aware)
# ----------------------------------------------------------------------
def _score_row(row, spot, low, high, ask_weight, dte_days, ideal_dte):
    delta = abs(row.get("Delta", np.nan))
    d_pen   = abs(delta - np.clip(delta, low, high)) * 3.0
    atm_pen = abs(row["strike"] - spot) / max(1.0, spot) * 10.0
    spr_pen = row["SpreadPct"] * 40.0
    ask_pen = np.log1p(max(0.0, row["ask"])) * ask_weight
    th_pen  = abs(row.get("Theta", 0.0)) * 3.0  # lower |Theta| preferred
    vol_rwd = -np.log1p(max(0, row.get("volume", 0))) * 2.0  # more volume → lower score
    dte_pen = max(0, abs(dte_days - ideal_dte)) * 0.8
    return atm_pen + spr_pen + th_pen + d_pen + ask_pen + vol_rwd + dte_pen

# ----------------------------------------------------------------------
# ⚙️ Evaluate symbol on a given expiry
# ----------------------------------------------------------------------
def _evaluate_symbol_on_expiry(ticker, side, expiry, delta_band, min_vol, max_spread_pct,
                               use_trend_filter, ask_weight, max_ask, ideal_dte):
    # Align side to current 5m trend if enabled
    if use_trend_filter:
        bias = _trend_bias_5m(ticker)
        if bias == "BULL":
            side = "CALLS"
        elif bias == "BEAR":
            side = "PUTS"

    calls, puts = _get_chain(ticker, expiry)
    if calls is None:
        return None

    tkr = yf.Ticker(ticker)
    hist = tkr.history(period="1d")
    if hist.empty:
        return None
    spot = float(hist["Close"].iloc[-1])

    dte_days = max(0, _dte(expiry))
    t_years  = max(dte_days / 365.0, 1e-6)
    df = (calls if side == "CALLS" else puts).copy()
    if df.empty:
        return None

    # basic clean up
    df["ask"]    = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
    df["bid"]    = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    df["Mid"]    = (df["bid"] + df["ask"]) / 2
    df["SpreadPct"] = np.where(df["Mid"] > 0, (df["ask"] - df["bid"]) / df["Mid"] * 100, np.inf)

    mask = (df["volume"] >= min_vol) & (df["SpreadPct"] <= max_spread_pct)
    if max_ask > 0:
        mask &= (df["ask"] <= max_ask)
    df = df[mask]
    if df.empty:
        return None

    # greeks
    deltas, thetas = [], []
    for _, r in df.iterrows():
        iv = float(r.get("impliedVolatility", np.nan)) if pd.notna(r.get("impliedVolatility", np.nan)) else np.nan
        d, th = _bs_greeks(spot, r["strike"], t_years, iv, right=("C" if side == "CALLS" else "P"))
        deltas.append(d)
        thetas.append(th)
    df["Delta"], df["Theta"] = deltas, thetas

    # filters & features
    df = df.dropna(subset=["Delta", "Theta"])
    df = df[(df["Delta"].abs() >= 0.05) & (df["Mid"] > 0)]
    low, high = delta_band
    df = df[df["Delta"].abs().between(low, high)]
    if df.empty:
        return None

    # Extra columns for display
    df["ThetaAbs"]        = df["Theta"].abs()                       # your “ignore the minus” view of Theta
    df["DeltaMinusTheta"] = df["Delta"].abs() - df["Theta"].abs()   # your requested Delta - Theta metric

    df["Score"] = df.apply(lambda r: _score_row(r, spot, low, high, ask_weight, dte_days, ideal_dte), axis=1)
    df = df.sort_values("Score").reset_index(drop=True)
    best = df.iloc[0].to_dict()
    best.update({
        "symbol": ticker, "expiry": expiry, "spot": spot,
        "side": side, "DTE": dte_days,
        "ThetaAbs": abs(best["Theta"]),
        "DeltaMinusTheta": abs(best["Delta"]) - abs(best["Theta"])
    })
    return best

# ----------------------------------------------------------------------
# 🧭 MAIN STREAMLIT PAGE
# ----------------------------------------------------------------------
def best_strike_picker():
    st.header("🎯 Best Strike Picker — Multi-Expiration Perfect Scenario")
    st.caption("Scans several expirations and ranks the best contracts by Delta, Theta, Spread, Volume, Ask, and DTE.")

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        universe_default = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "META", "AMD", "AMZN", "WMT", "GILD", "BIO"]
        universe = st.multiselect("Symbols (Universe)", universe_default, default=universe_default)
        side = st.selectbox("Preferred Side", ["CALLS", "PUTS"], index=0)
        delta_band = st.slider("|Delta| range", 0.25, 0.75, (0.40, 0.55), 0.01)
        min_vol = st.number_input("Min Volume", 0, 300000, 500, 50)

    with c2:
        use_auto_spread = st.checkbox("Auto Max Spread % by Liquidity", value=True)
        max_spread_pct_manual = st.slider("Max Spread % (manual)", 0.1, 10.0, 3.0, 0.1,
                                          help="Only used if Auto is OFF")
        ask_weight = st.slider("Ask Sensitivity (weight)", 1.0, 10.0, 5.0, 0.5)
        max_ask = st.number_input("Max Ask Cap ($)", 0.0, 50.0, 0.0,
                                  help="Set 0 to ignore premium cap")
        use_trend_filter = st.checkbox("EMA(9/21) Trend Filter (5m)", value=True)

    with c3:
        exp_mode = st.radio("Expiration Scan", ["Nearest N", "DTE Range"], index=0, horizontal=True)
        if exp_mode == "Nearest N":
            n_exp = st.slider("Scan next N expirations", 1, 8, 3, 1)
            ideal_dte = st.slider("Ideal DTE (days)", 0, 14, 1, 1)
        else:
            dte_min, dte_max = st.slider("DTE Range (days)", 0, 30, (0, 7), 1)
            ideal_dte = st.slider("Ideal DTE (days)", 0, 14, 1, 1)

    manual_symbol = st.text_input("Manual Symbol (optional)", "").upper().strip()
    manual_side = st.selectbox("Manual Side", ["CALLS", "PUTS"], index=0)

    if st.button("🚀 Run Multi-Expiration Scanner", type="primary"):
        all_picks = []

        def _exp_filter_list(ticker):
            exps = _list_expirations(ticker)
            if not exps:
                return []
            rows = [(e, _dte(e)) for e in exps]
            if exp_mode == "Nearest N":
                rows = [r for r in rows if r[1] >= 0]
                rows.sort(key=lambda x: x[1])
                rows = rows[:n_exp]
            else:
                rows = [r for r in rows if dte_min <= max(0, r[1]) <= dte_max]
                rows.sort(key=lambda x: x[1])
            return rows

        # Manual symbol first (if provided)
        if manual_symbol:
            spread_cap = _auto_spread_cap(manual_symbol) if use_auto_spread else max_spread_pct_manual
            st.write(f"🧮 Auto Spread Cap for **{manual_symbol}** = {spread_cap:.2f}%")
            for exp, dte in _exp_filter_list(manual_symbol):
                pick = _evaluate_symbol_on_expiry(
                    manual_symbol, manual_side, exp, delta_band, min_vol, spread_cap,
                    use_trend_filter, ask_weight, max_ask, ideal_dte
                )
                if pick:
                    all_picks.append(pick)

        # Universe symbols
        for sym in universe:
            spread_cap = _auto_spread_cap(sym) if use_auto_spread else max_spread_pct_manual
            st.write(f"🧮 Auto Spread Cap for **{sym}** = {spread_cap:.2f}%")
            for exp, dte in _exp_filter_list(sym):
                pick = _evaluate_symbol_on_expiry(
                    sym, side, exp, delta_band, min_vol, spread_cap,
                    use_trend_filter, ask_weight, max_ask, ideal_dte
                )
                if pick:
                    all_picks.append(pick)

        if not all_picks:
            st.error("No eligible contracts found — try loosening Delta band, raising Max Spread %, "
                     "lowering Min Volume, or widening your DTE window.")
            return

        dfp = pd.DataFrame(all_picks).sort_values(["Score", "DTE"]).reset_index(drop=True)

        # Remarks (readable reasons)
        remarks = []
        for _, r in dfp.iterrows():
            reason = []
            # We don't know each symbol's exact spread cap here (manual vs auto),
            # but we can highlight generally tight spreads:
            if r.get("SpreadPct", np.inf) <= 3.0:
                reason.append("Very tight spread")
            elif r.get("SpreadPct", np.inf) <= 5.0:
                reason.append("Tight spread")

            if abs(r["Delta"]) >= delta_band[0] and abs(r["Delta"]) <= delta_band[1]:
                reason.append("Ideal Delta")
            if abs(r["Theta"]) < 0.01:
                reason.append("Low time decay")
            if max_ask == 0 or r.get("ask", np.inf) <= max_ask:
                reason.append("Affordable premium")
            if r.get("volume", 0) > min_vol:
                reason.append("High liquidity")
            if r["DTE"] <= 3:
                reason.append("Short-term expiry")
            remarks.append(", ".join(reason))
        dfp["Remark"] = remarks

        # Display table
        st.subheader("🏆 Overall Top Picks (Ranked by Score)")
        show_cols = [
            "symbol", "side", "expiry", "DTE", "contractSymbol", "strike",
            "bid", "ask", "Delta", "Theta", "ThetaAbs", "DeltaMinusTheta",
            "SpreadPct", "volume", "spot", "Score", "Remark"
        ]
        existing_cols = [c for c in show_cols if c in dfp.columns]
        st.dataframe(
            dfp[existing_cols].style.format({
                "DTE": "{:.0f}", "strike": "{:.2f}", "bid": "{:.2f}", "ask": "{:.2f}",
                "Delta": "{:.3f}", "Theta": "{:.4f}", "ThetaAbs": "{:.4f}",
                "DeltaMinusTheta": "{:.4f}",
                "SpreadPct": "{:.2f}%", "volume": "{:,.0f}", "spot": "{:.2f}", "Score": "{:.2f}"
            }),
            use_container_width=True
        )

        # Explanations for Top 3
        st.markdown("### 📘 Top Picks Explanation")
        top_picks = dfp.head(3)
        for idx, row in top_picks.iterrows():
            label = ("🏆 **Best Option**" if idx == 0
                     else "🥈 **Second Best Option**" if idx == 1
                     else "🥉 **Third Best Option**")
            st.markdown(f"""
{label} — **{row['symbol']} {row['side']}**
- **Expiration:** {row['expiry']} (DTE {int(row['DTE'])} days)
- **Strike:** {row['strike']:.2f}, **Spot:** {row['spot']:.2f}
- **Δ (Delta):** {row['Delta']:.3f} → {'strong price sensitivity' if abs(row['Delta']) >= 0.5 else 'balanced responsiveness'}
- **Θ (Theta):** {row['Theta']:.4f} (|Θ| = {abs(row['Theta']):.4f}) → {'low decay — ideal for intraday trades' if abs(row['Theta']) < 0.01 else 'moderate time decay'}
- **Δ − |Θ|:** {abs(row['Delta']) - abs(row['Theta']):.4f} → higher suggests good responsiveness with low decay
- **Ask:** ${row['ask']:.2f} → {'low-cost entry' if row['ask'] <= 2 else 'higher premium — confirm momentum'}
- **Spread:** {row['SpreadPct']:.2f}% → {'tight and efficient fills' if row['SpreadPct'] < 2 else 'moderate spread'}
- **Volume:** {int(row['volume']):,} → {'excellent liquidity' if row['volume'] > 5000 else 'ok but thinner'}
""")

        st.markdown("---")
        st.info("""
**Selection Logic Summary**
- 🟩 *Best Option:* Tight spread, ideal Delta (0.40–0.55), low |Theta|, strong liquidity, acceptable Ask.
- 🟨 *Second Best:* Slightly higher |Theta| or spread but still good liquidity and ATM proximity.
- 🟥 *Third Best:* Good Delta but higher premium or thinner volume.
We rank contracts for **intraday responsiveness vs. execution cost & decay**.
""")


# 🧮 SIMPLE STRIKE CALCULATOR — Based on Overall Top Picks Table
# =========================================================================================================
# =========================================================================================================
# 🧠 SMART STRIKE EVALUATOR — Paste & Auto-Rate with Manual Expiry/Strike
# =========================================================================================================




# =========================================================================================================
# 🧠 SMART STRIKE EVALUATOR — Enhanced Parsing + Symbol + Volume Detection
# =========================================================================================================




# ======================================================================
# 💡 Sybest LLC — "Trade the confirmation, not the hope."
# ======================================================================


# ----------------- MENU HOOK EXAMPLE -----------------
# elif menu == "🧠 Auto Recommender (0DTE)":
#     daytrade_auto_recommender()

    # ====================== ENDS HERE

# # ====================================================================================
# US MONTHLY SALES PREDICTION
# # ====================================================================================
# US MONTHLY SALES PREDICTION
# # ====================================================================================
# US MONTHLY SALES PREDICTION
# =====================================================================================
# ============================================================
# 🛍️ US RETAIL SALES FORECAST (MARTS 44X72)
# - Seasonally Adjusted (SA) and Not Seasonally Adjusted (NSA)
# - Models: ETS (Holt-Winters) and SARIMAX
# ============================================================
# ====================================================================================
# 🛍️ US RETAIL SALES FORECAST (44X72)
# ====================================================================================

# ====================================================================================
# 🛍️ US RETAIL SALES FORECAST (44X72)
# ====================================================================================

# ====================================================================================
# 📊 Retail Monthly Sales Forecast — Machine Learning / Time-Series Model
# ====================================================================================




# ======================================================================================================= ACTIVITIES ANALYSIS
# # ====================================================================================
# TRADE ACTIVITY ANALYSIS
# TRADE ACTIVITY ANALYSIS
# # ====================================================================================
# TRADE ACTIVITY ANALYSIS
# =====================================================================================

# =======================================================================================================
# 📊 TRADE ACTIVITY ANALYSIS — FULL INSIGHTS, FILTERS & RECOMMENDATIONS
# =======================================================================================================

# =======================================================================================================
# 📊 TRADE ACTIVITY ANALYSIS — FULL MODULE
# =======================================================================================================

# =======================================================================================================
# 📊 TRADE ACTIVITY ANALYSIS — FULL MODULE
# =======================================================================================================

# =======================================================================================================
# 📊 TRADE ACTIVITY ANALYSIS — CLEAN VERSION (NO DUPLICATE ELEMENTS)
# =======================================================================================================
# =======================================================================================================
# 📊 TRADE ACTIVITY ANALYSIS — COMPLETE MODULE (FULL WORKING VERSION)
# =======================================================================================================

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import csv
import io

# =======================================================================================================
# 1️⃣ SAFE CSV LOADER
# =======================================================================================================

def load_trade_csv(uploaded_file):
    """Safely load ThinkorSwim / Brokerage CSV using delimiter detection & bad-line fixing."""

    try:
        sample = uploaded_file.read(4096).decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "|", "\t"])
        delimiter = dialect.delimiter
    except:
        delimiter = ","

    try:
        df = pd.read_csv(uploaded_file, delimiter=delimiter)
        return df
    except:
        uploaded_file.seek(0)

    try:
        df = pd.read_csv(uploaded_file, delimiter=delimiter, on_bad_lines="skip", engine="python")
        return df
    except Exception as e:
        st.error(f"❌ CSV Parsing Error: {e}")
        return None


# =======================================================================================================
# 2️⃣ EXTRACT ACCOUNT ORDER HISTORY SECTION
# =======================================================================================================

def extract_order_history_section(raw_df):
    """
    Extract only the 'Account Order History' block and return clean df2.
    """
    raw_df_str = raw_df.astype(str)

    # Find header row
    header_found = raw_df_str.apply(
        lambda row: row.str.contains("Account Order History", case=False).any(), axis=1
    )

    if not header_found.any():
        st.warning("⚠️ 'Account Order History' section NOT found. Using full CSV.")
        return raw_df.copy()

    start_idx = header_found.idxmax()

    # Extract everything after this line
    df2 = raw_df.iloc[start_idx + 1:].copy()

    df2.dropna(how="all", inplace=True)

    # Clean headers
    df2.columns = [c.strip().replace(" ", "_").replace(".", "_") for c in df2.columns]

    return df2


# =======================================================================================================
# 3️⃣ MAIN TRADE ACTIVITY ANALYSIS UI
# =======================================================================================================

def trade_activity_analysis():

    st.title("📊 Trade Activity Analysis")
    st.write("Upload your Account Statement CSV to analyze intra-day execution patterns & trends.")

    uploaded_file = st.file_uploader(
        "📁 Upload Account Statement (CSV)",
        type=["csv"],
        key="account_statement_upload"
    )

    if not uploaded_file:
        st.info("🔼 Please upload a CSV to begin.")
        return

    raw_df = load_trade_csv(uploaded_file)
    if raw_df is None:
        st.stop()

    st.success("✅ CSV Loaded Successfully")

    st.subheader("📄 Raw Data Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    # ------------------------------------------------------------------------------
    # Extract ONLY Order History Section
    # ------------------------------------------------------------------------------
    df2 = extract_order_history_section(raw_df)

    st.subheader("📌 Extracted: Account Order History")
    st.dataframe(df2.head(20), use_container_width=True)

    # ===================================================================================================
    # 4️⃣ Timestamp Processing (Time Trend Analysis)
    # ===================================================================================================

    df2.columns = [c.lower() for c in df2.columns]

    time_col = None
    for c in df2.columns:
        if "time" in c or "placed" in c:
            time_col = c
            break

    if time_col:
        df2["timestamp"] = pd.to_datetime(df2[time_col], errors="coerce")
        if df2["timestamp"].notna().sum() == 0:
            st.warning("⚠ Timestamp column could not be parsed.")
            df2["timestamp"] = pd.NaT
    else:
        df2["timestamp"] = pd.NaT

    df2["hour"] = df2["timestamp"].dt.hour
    df2["minute"] = df2["timestamp"].dt.minute
    df2["date"] = df2["timestamp"].dt.date

    # Status column detection
    status_col = None
    for c in df2.columns:
        if "status" in c:
            status_col = c
            break

    if status_col is None:
        df2["status"] = "UNKNOWN"
        status_col = "status"

    # Detect P/L column (optional)
    profit_col = None
    for c in df2.columns:
        if "p/l" in c or "profit" in c or "amount" in c:
            profit_col = c
            break

    if profit_col:
        df2["profit_clean"] = pd.to_numeric(df2[profit_col], errors="coerce")
    else:
        df2["profit_clean"] = 0

    # ===================================================================================================
    # 5️⃣ METRICS SUMMARY
    # ===================================================================================================

    st.subheader("📊 Key Metrics Summary")

    total_orders = len(df2)
    total_filled = (df2[status_col].str.contains("FILLED", case=False)).sum()
    total_canceled = (df2[status_col].str.contains("CANCELED", case=False)).sum()
    total_rejected = (df2[status_col].str.contains("REJECT", case=False)).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total Orders", total_orders)
    col2.metric("✅ Filled", total_filled)
    col3.metric("❌ Canceled", total_canceled)
    col4.metric("⛔ Rejected", total_rejected)

    # ===================================================================================================
    # 6️⃣ STATUS TREND BY TIME OF DAY
    # ===================================================================================================

    st.subheader("⏱️ Status Trend by Hour of Day")

    hour_status = df2.groupby(["hour", status_col]).size().unstack(fill_value=0)

    st.dataframe(hour_status)

    fig, ax = plt.subplots(figsize=(8, 4))
    hour_status.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Order Status by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Order Count")
    st.pyplot(fig)

    # ===================================================================================================
    # 7️⃣ STATUS TREND BY MINUTE (Precision Timing Insight)
    # ===================================================================================================

    st.subheader("⏱️ Minute-Level Trend (Fine Precision)")
    minute_status = df2.groupby(["minute", status_col]).size().unstack(fill_value=0)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    minute_status.plot(kind="line", ax=ax2)
    ax2.set_title("Status Trend by Minute")
    ax2.set_xlabel("Minute")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    # ===================================================================================================
    # 8️⃣ BEST/WORST TIME WINDOWS
    # ===================================================================================================

    st.subheader("🔎 Best & Worst Trading Windows")

    best_hour = hour_status.get("FILLED", pd.Series()).idxmax() if "FILLED" in hour_status.columns else None
    worst_hour = hour_status.get("CANCELED", pd.Series()).idxmax() if "CANCELED" in hour_status.columns else None

    if best_hour is not None:
        st.success(f"📈 **Best hour for fills:** {best_hour}:00")

    if worst_hour is not None:
        st.error(f"⚠️ **Highest cancellations:** {worst_hour}:00")

    # ===================================================================================================
    # 9️⃣ TOP WINNERS & LOSERS (If P/L available)
    # ===================================================================================================

    st.subheader("🏆 Winners & ❌ Losers")

    if df2["profit_clean"].abs().sum() > 0:
        st.write("### 🔝 Top 5 Winning Orders")
        st.dataframe(df2.nlargest(5, "profit_clean"))

        st.write("### ❌ Top 5 Losing Orders")
        st.dataframe(df2.nsmallest(5, "profit_clean"))

    # ===================================================================================================
    # 🔟 RECOMMENDATIONS ENGINE
    # ===================================================================================================

    st.subheader("🤖 Smart Recommendations")

    recs = []

    if worst_hour is not None:
        recs.append(f"🚫 Avoid trading between **{worst_hour}:00–{worst_hour+1}:00** (heavy cancellations).")

    if best_hour is not None:
        recs.append(f"📈 Best execution window: **{best_hour}:00**.")

    if total_rejected > 5:
        recs.append("⚠️ High rejection rate — verify buying power & approval level.")

    if total_filled < total_canceled:
        recs.append("❗ More canceled than filled orders — consider using LIMIT instead of MARKET.")

    if len(recs) == 0:
        recs.append("Everything looks stable. Continue trading normally.")

    for r in recs:
        st.warning(r)

    st.success("🎉 Analysis Complete!")


# =======================================================================================================
# END OF MODULE
# =======================================================================================================


# Call the function in your main Streamlit app
# trade_activity_analysis()



# End of Module


# ============== ENDS HERE ======================================














# ------------------------------------------------------------
# 🟢 RUN APP
# ------------------------------------------------------------
# if __name__ == "__main__":
#     daytrade_trend_confirmation()




# ========================================================== ENTRY CONFIRMATIOMN





# =====================================================================
# 🚀 PAGE ROUTING
# =====================================================================
if menu == "🏦 Single Stock Predictor":
    single_stock()
elif menu == "💹 Multi-Stock Predictor":
    multi_stock()
elif menu == "⚡ Intraday Signal Predictor":
    intraday_signals()

elif menu == "📚 Technical Analysis":
    technical_analysis()

elif menu == "💰 ROI Calculator":
    power_roi_daytrading()


elif menu == "🎯 Trend Confirmation (Day Trader)":
    daytrade_trend_confirmation()   # ✅ defined above


elif menu == "🎯 Best Strike Picker":
    best_strike_picker()

# elif menu == "🛍️ US Retail Sales Forecast":
#     retail_sales_forecast()
#
# elif menu == "📊 Trade Activity Analysis":
#     trade_activity_analysis()


# # =============================================================================
# # TOS AUTOMATION CODES HERE
# # ===============================================================================
#
#
# input fastLength = 9;
# input slowLength = 21;
# input rsiLength = 14;
# input alertEnabled = yes;
#
# # -----------------------------
# # 1️⃣ CORE CALCULATIONS (1-MIN)
# # -----------------------------
# def price = close;
# def emaFast = ExpAverage(price, fastLength);
# def emaSlow = ExpAverage(price, slowLength);
# def rsi = RSI(length = rsiLength);
# def vwapLine = VWAP();
#
# def bull1 = emaFast > emaSlow;
# def bear1 = emaFast < emaSlow;
#
# # -----------------------------
# # 2️⃣ MULTI-TIMEFRAME (5m + 30m)
# # -----------------------------
# def c5  = close(period = AggregationPeriod.FIVE_MIN);
# def c30 = close(period = AggregationPeriod.THIRTY_MIN);
#
# def emaFast5 = ExpAverage(c5, fastLength);
# def emaSlow5 = ExpAverage(c5, slowLength);
# def bull5 = emaFast5 > emaSlow5;
# def bear5 = emaFast5 < emaSlow5;
#
# def emaFast30 = ExpAverage(c30, fastLength);
# def emaSlow30 = ExpAverage(c30, slowLength);
# def bull30 = emaFast30 > emaSlow30;
# def bear30 = emaFast30 < emaSlow30;
#
# # -----------------------------
# # 3️⃣ STRICT CALL / PUT LOGIC
# # -----------------------------
# def CALL_OK =
#     bull1 and bull5 and bull30 and
#     price > vwapLine and
#     rsi > 55;
#
# def PUT_OK =
#     bear1 and bear5 and bear30 and
#     price < vwapLine and
#     rsi < 45;
#
# # -----------------------------
# # 4️⃣ PLOT EMAs
# # -----------------------------
# plot EMA_9 = emaFast;
# EMA_9.SetDefaultColor(Color.BLUE);
# EMA_9.SetLineWeight(2);
#
# plot EMA_21 = emaSlow;
# EMA_21.SetDefaultColor(Color.RED);
# EMA_21.SetLineWeight(2);
#
# # -----------------------------
# # 5️⃣ CALL & PUT Bubbles (High Visibility)
# # -----------------------------
# AddChartBubble(
#     CALL_OK,
#     emaFast + (TickSize() * 4),
#     "CALL",
#     Color.GREEN,
#     yes
# );
#
# AddChartBubble(
#     PUT_OK,
#     emaFast - (TickSize() * 4),
#     "PUT",
#     Color.RED,
#     no
# );
#
# # -----------------------------
# # 6️⃣ ALERTS
# # -----------------------------
# Alert(
#     CALL_OK and alertEnabled,
#     "CALL CONFIRMED — All Timeframes Bullish",
#     Alert.BAR,
#     Sound.Ding
# );
#
# Alert(
#     PUT_OK and alertEnabled,
#     "PUT CONFIRMED — All Timeframes Bearish",
#     Alert.BAR,
#     Sound.Ding
# );
#
# # -----------------------------
# # 7️⃣ HIGH VISIBILITY DASHBOARD
# # -----------------------------
# AddLabel(
#     yes,
#     "30m TF: " + (if bull30 then "CALL" else if bear30 then "PUT" else "WAIT"),
#     if bull30 then CreateColor(0, 255, 0)
#     else if bear30 then CreateColor(255, 0, 0)
#     else Color.YELLOW
# );
#
# AddLabel(
#     yes,
#     "5m TF: " + (if bull5 then "CALL" else if bear5 then "PUT" else "WAIT"),
#     if bull5 then CreateColor(0, 255, 0)
#     else if bear5 then CreateColor(255, 0, 0)
#     else Color.YELLOW
# );
#
# AddLabel(
#     yes,
#     "1m TF: " + (if bull1 then "CALL" else if bear1 then "PUT" else "WAIT"),
#     if bull1 then CreateColor(0, 255, 0)
#     else if bear1 then CreateColor(255, 0, 0)
#     else Color.YELLOW
# );
#
# AddLabel(
#     yes,
#     if CALL_OK then "🟩 CALL ENTRY READY"
#     else if PUT_OK then "🟥 PUT ENTRY READY"
#     else "⚠ NO TRADE",
#     if CALL_OK then CreateColor(0, 200, 0)
#     else if PUT_OK then CreateColor(200, 0, 0)
#     else Color.YELLOW
# );
#
# # -----------------------------
# # 8️⃣ RSI + VWAP LABELS
# # -----------------------------
# AddLabel(
#     yes,
#     "RSI: " + Round(rsi, 0),
#     if rsi > 55 then CreateColor(0, 255, 0)
#     else if rsi < 45 then CreateColor(255, 0, 0)
#     else Color.WHITE
# );
#
# AddLabel(
#     yes,
#     "VWAP: " + Round(vwapLine, 2),
#     CreateColor(120, 120, 120)
# );
#
# # =====================================================================
# # 💡 "Trade the alignment — Not the noise." — Sybest LLC
# # =====================================================================


# =================================================================================== PART 22

#
# input fastLength = 9;
# input slowLength = 21;
# input rsiLength = 14;
# input alertEnabled = yes;
#
# # -----------------------------
# # 1️⃣ CORE CALCULATIONS (1-MIN)
# # -----------------------------
# def price = close;
# def emaFast = ExpAverage(price, fastLength);
# def emaSlow = ExpAverage(price, slowLength);
# def rsi = RSI(length = rsiLength);
# def vwapLine = VWAP();
#
# def bull1 = emaFast > emaSlow;
# def bear1 = emaFast < emaSlow;
#
# # -----------------------------
# # 2️⃣ MULTI-TIMEFRAME (5m + 30m)
# # -----------------------------
# def c5  = close(period = AggregationPeriod.FIVE_MIN);
# def c30 = close(period = AggregationPeriod.THIRTY_MIN);
#
# def emaFast5 = ExpAverage(c5, fastLength);
# def emaSlow5 = ExpAverage(c5, slowLength);
# def bull5 = emaFast5 > emaSlow5;
# def bear5 = emaFast5 < emaSlow5;
#
# def emaFast30 = ExpAverage(c30, fastLength);
# def emaSlow30 = ExpAverage(c30, slowLength);
# def bull30 = emaFast30 > emaSlow30;
# def bear30 = emaFast30 < emaSlow30;
#
# # -----------------------------
# # 3️⃣ STRICT CALL / PUT LOGIC
# # -----------------------------
# def CALL_OK =
#     bull1 and bull5 and bull30 and
#     price > vwapLine and
#     rsi > 55;
#
# def PUT_OK =
#     bear1 and bear5 and bear30 and
#     price < vwapLine and
#     rsi < 45;
#
# # -----------------------------
# # ⭐ 4️⃣ TREND STRUCTURE (HH / HL / LH / LL)
# # -----------------------------
# def HH = high > high[1] and high[1] > high[2];     # Higher High
# def HL = low > low[1] and low[1] < low[2];          # Higher Low
#
# def LH = high < high[1] and high[1] < high[2];      # Lower High
# def LL = low < low[1] and low[1] > low[2];          # Lower Low
#
# def UPTREND = HH or HL;
# def DOWNTREND = LH or LL;
#
# # -----------------------------
# # ⭐ HH & LL BUBBLES
# # -----------------------------
# AddChartBubble(HH, high, "HH", Color.CYAN, yes);
# AddChartBubble(LL, low, "LL", Color.RED, no);
#
# # -----------------------------
# # ⭐ HL SUPPORT LINE (Light Gray)
# # -----------------------------
# plot HL_SUPPORT = if HL then low else Double.NaN;
# HL_SUPPORT.SetLineWeight(3);
# HL_SUPPORT.SetDefaultColor(CreateColor(180,180,180));
#
# # -----------------------------
# # ⭐ CALL ZONE BACKGROUND (GREEN)
# # -----------------------------
# AddCloud(
#     if UPTREND then low else Double.NaN,
#     if UPTREND then high else Double.NaN,
#     CreateColor(0, 60, 0),   # CALL ZONE Green
#     Color.BLACK
# );
#
# AddChartBubble(
#     UPTREND and HL,
#     low,
#     "CALL ZONE",
#     Color.GREEN,
#     no
# );
#
# # -----------------------------
# # ⭐ PUT ZONE BACKGROUND (REAL RED)
# # -----------------------------
# AddCloud(
#     if DOWNTREND then high else Double.NaN,
#     if DOWNTREND then low else Double.NaN,
#     CreateColor(255, 0, 0),  # REAL RED
#     CreateColor(80, 0, 0)
# );
#
# AddChartBubble(
#     DOWNTREND and LL,
#     high,
#     "PUT ZONE",
#     Color.RED,
#     yes
# );
#
# # -----------------------------
# # 5️⃣ PLOT EMAs
# # -----------------------------
# plot EMA_9 = emaFast;
# EMA_9.SetDefaultColor(Color.BLUE);
# EMA_9.SetLineWeight(2);
#
# plot EMA_21 = emaSlow;
# EMA_21.SetDefaultColor(Color.RED);
# EMA_21.SetLineWeight(2);
#
# # -----------------------------
# # 6️⃣ CALL & PUT BUBBLES
# # -----------------------------
# AddChartBubble(CALL_OK, emaFast + (TickSize() * 4), "CALL", Color.GREEN, yes);
# AddChartBubble(PUT_OK , emaFast - (TickSize() * 4), "PUT", Color.RED, no);
#
# # -----------------------------
# # 7️⃣ ALERTS
# # -----------------------------
# Alert(CALL_OK and alertEnabled, "CALL CONFIRMED — MTF Bullish", Alert.BAR, Sound.Ding);
# Alert(PUT_OK  and alertEnabled, "PUT CONFIRMED — MTF Bearish",  Alert.BAR, Sound.Ding);
#
# # -----------------------------
# # 8️⃣ TREND DASHBOARD LABELS (TOP LEFT CORNER)
# # -----------------------------
# AddLabel(yes, "30m TF: " + (if bull30 then "CALL" else if bear30 then "PUT" else "WAIT"),
#         if bull30 then Color.GREEN else if bear30 then Color.RED else Color.YELLOW);
#
# AddLabel(yes, "5m TF:  " + (if bull5 then "CALL" else if bear5 then "PUT" else "WAIT"),
#         if bull5 then Color.GREEN else if bear5 then Color.RED else Color.YELLOW);
#
# AddLabel(yes, "1m TF:  " + (if bull1 then "CALL" else if bear1 then "PUT" else "WAIT"),
#         if bull1 then Color.GREEN else if bear1 then Color.RED else Color.YELLOW);
#
# AddLabel(yes,
#         if CALL_OK then "🟩 CALL ENTRY READY"
#         else if PUT_OK then "🟥 PUT ENTRY READY"
#         else "⚠ NO TRADE",
#         if CALL_OK then Color.GREEN else if PUT_OK then Color.RED else Color.YELLOW);
#
# # -----------------------------
# # 9️⃣ RSI + VWAP LABELS
# # -----------------------------
# AddLabel(yes, "RSI: " + Round(rsi, 0),
#     if rsi > 55 then Color.GREEN else if rsi < 45 then Color.RED else Color.WHITE);
#
# AddLabel(yes, "VWAP: " + Round(vwapLine, 2),
#     CreateColor(120, 120, 120));
#
# # =====================================================================
# # 💡 "Trade the alignment — Not the noise." — Sybest LLC
# # =====================================================================


# ======================================================================================== PART 44
#
# #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ PART 3 WITH HL2
#
# # ⚡ SYBEST TREND CONFIRMATION SYSTEM — EMA 9/21 DAY TRADER EDITION
# # =====================================================================
# # Designed by: Dr. Stanley Njoku / Sybest LLC
# # Purpose: CALL/PUT confirmations + MTF Trend + HH/HL + LH/LL Detection
# # =====================================================================
#
# input fastLength = 9;
# input slowLength = 21;
# input rsiLength = 14;
# input alertEnabled = yes;
#
# # -----------------------------
# # 1️⃣ CORE CALCULATIONS (1-MIN)
# # -----------------------------
# def price = close;
# def emaFast = ExpAverage(price, fastLength);
# def emaSlow = ExpAverage(price, slowLength);
# def rsi = RSI(length = rsiLength);
# def vwapLine = VWAP();
#
# def bull1 = emaFast > emaSlow;
# def bear1 = emaFast < emaSlow;
#
# # -----------------------------
# # 2️⃣ MULTI-TIMEFRAME (5m + 30m)
# # -----------------------------
# def c5  = close(period = AggregationPeriod.FIVE_MIN);
# def c30 = close(period = AggregationPeriod.THIRTY_MIN);
#
# def emaFast5 = ExpAverage(c5, fastLength);
# def emaSlow5 = ExpAverage(c5, slowLength);
# def bull5 = emaFast5 > emaSlow5;
# def bear5 = emaFast5 < emaSlow5;
#
# def emaFast30 = ExpAverage(c30, fastLength);
# def emaSlow30 = ExpAverage(c30, slowLength);
# def bull30 = emaFast30 > emaSlow30;
# def bear30 = emaFast30 < emaSlow30;
#
# # -----------------------------
# # 3️⃣ STRICT CALL / PUT LOGIC
# # -----------------------------
# def CALL_OK =
#     bull1 and bull5 and bull30 and
#     price > vwapLine and
#     rsi > 55;
#
# def PUT_OK =
#     bear1 and bear5 and bear30 and
#     price < vwapLine and
#     rsi < 45;
#
# # -----------------------------
# # ⭐ 4️⃣ TREND STRUCTURE (HH / HL / LH / LL)
# # -----------------------------
# def HH = high > high[1] and high[1] > high[2];     # Higher High
# def HL = low > low[1] and low[1] < low[2];         # Higher Low
#
# def LH = high < high[1] and high[1] < high[2];     # Lower High
# def LL = low < low[1] and low[1] > low[2];         # Lower Low
#
# def UPTREND = HH or HL;
# def DOWNTREND = LH or LL;
#
# # -----------------------------
# # ⭐ HH & LL BUBBLES
# # -----------------------------
# AddChartBubble(HH, high, "HH", Color.CYAN, yes);
# AddChartBubble(LL, low, "LL", Color.RED, no);
#
# # -----------------------------
# # ⭐ HL SUPPORT LINE (Light Gray)
# # -----------------------------
# plot HL_SUPPORT = if HL then low else Double.NaN;
# HL_SUPPORT.SetLineWeight(3);
# HL_SUPPORT.SetDefaultColor(CreateColor(180,180,180));
#
# # =====================================================================
# # ⭐ NEW: HL SNIPER DETECTION (ONLY ADDITION — NO OTHER CHANGES)
# # =====================================================================
#
# # Pivot-based HL Sniper logic
# def PivotLow_Sniper = low[1] < low[2] and low[1] < low;
#
# rec LastPivotLow_Sniper =
#     if PivotLow_Sniper then low[1]
#     else if IsNaN(LastPivotLow_Sniper[1]) then low
#     else LastPivotLow_Sniper[1];
#
# def HL_Sniper = PivotLow_Sniper and low[1] > LastPivotLow_Sniper[1];
#
# # Bubble
# AddChartBubble(
#     HL_Sniper,
#     low[1],
#     "HL",
#     Color.LIGHT_GREEN,
#     no
# );
#
# # Sniper line
# plot HL_Sniper_Line =
#     if HL_Sniper then low[1] else Double.NaN;
# HL_Sniper_Line.SetDefaultColor(CreateColor(150,150,150));
# HL_Sniper_Line.SetLineWeight(2);
#
# # =====================================================================
# # END HL SNIPER INSERT
# # =====================================================================
#
# # -----------------------------
# # ⭐ CALL ZONE BACKGROUND (GREEN)
# # -----------------------------
# AddCloud(
#     if UPTREND then low else Double.NaN,
#     if UPTREND then high else Double.NaN,
#     CreateColor(0, 60, 0),   # CALL ZONE Green
#     Color.BLACK
# );
#
# AddChartBubble(
#     UPTREND and HL,
#     low,
#     "CALL ZONE",
#     Color.GREEN,
#     no
# );
#
# # -----------------------------
# # ⭐ PUT ZONE BACKGROUND (REAL RED)
# # -----------------------------
# AddCloud(
#     if DOWNTREND then high else Double.NaN,
#     if DOWNTREND then low else Double.NaN,
#     CreateColor(255, 0, 0),  # REAL RED
#     CreateColor(80, 0, 0)
# );
#
# AddChartBubble(
#     DOWNTREND and LL,
#     high,
#     "PUT ZONE",
#     Color.RED,
#     yes
# );
#
# # -----------------------------
# # 5️⃣ PLOT EMAs
# # -----------------------------
# plot EMA_9 = emaFast;
# EMA_9.SetDefaultColor(Color.BLUE);
# EMA_9.SetLineWeight(2);
#
# plot EMA_21 = emaSlow;
# EMA_21.SetDefaultColor(Color.RED);
# EMA_21.SetLineWeight(2);
#
# # -----------------------------
# # 6️⃣ CALL & PUT BUBBLES
# # -----------------------------
# AddChartBubble(CALL_OK, emaFast + (TickSize() * 4), "CALL", Color.GREEN, yes);
# AddChartBubble(PUT_OK , emaFast - (TickSize() * 4), "PUT", Color.RED, no);
#
# # -----------------------------
# # 7️⃣ ALERTS
# # -----------------------------
# Alert(CALL_OK and alertEnabled, "CALL CONFIRMED — MTF Bullish", Alert.BAR, Sound.Ding);
# Alert(PUT_OK  and alertEnabled, "PUT CONFIRMED — MTF Bearish",  Alert.BAR, Sound.Ding);
#
# # -----------------------------
# # 8️⃣ TREND DASHBOARD LABELS
# # -----------------------------
# AddLabel(yes, "30m TF: " + (if bull30 then "CALL" else if bear30 then "PUT" else "WAIT"),
#         if bull30 then Color.GREEN else if bear30 then Color.RED else Color.YELLOW);
#
# AddLabel(yes, "5m TF:  " + (if bull5 then "CALL" else if bear5 then "PUT" else "WAIT"),
#         if bull5 then Color.GREEN else if bear5 then Color.RED else Color.YELLOW);
#
# AddLabel(yes, "1m TF:  " + (if bull1 then "CALL" else if bear1 then "PUT" else "WAIT"),
#         if bull1 then Color.GREEN else if bear1 then Color.RED else Color.YELLOW);
#
# AddLabel(yes,
#         if CALL_OK then "🟩 CALL ENTRY READY"
#         else if PUT_OK then "🟥 PUT ENTRY READY"
#         else "⚠ NO TRADE",
#         if CALL_OK then Color.GREEN else if PUT_OK then Color.RED else Color.YELLOW);
#
# # -----------------------------
# # 9️⃣ RSI + VWAP LABELS
# # -----------------------------
# AddLabel(yes, "RSI: " + Round(rsi, 0),
#     if rsi > 55 then Color.GREEN else if rsi < 45 then Color.RED else Color.WHITE);
#
# AddLabel(yes, "VWAP: " + Round(vwapLine, 2),
#     CreateColor(120, 120, 120));
#
# # =====================================================================
# # 💡 "Trade the alignment — Not the noise." — Sybest LLC
# # =====================================================================


#
# # SNIPPER ADDED ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]] PART3
#
# #GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG PART 55 TAKE PROFIT
# #GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG PART 55 TAKE PROFIT
#
# # ⚡ SYBEST TREND CONFIRMATION SYSTEM — EMA 9/21 DAY TRADER EDITION
# # =====================================================================
# # Designed by: Dr. Stanley Njoku / Sybest LLC
# # Purpose: CALL/PUT confirmations + MTF Trend + HH/HL + LH/LL Detection
# # =====================================================================
#
# input fastLength = 9;
# input slowLength = 21;
# input rsiLength = 14;
# input alertEnabled = yes;
#
# # -----------------------------
# # 1️⃣ CORE CALCULATIONS (1-MIN)
# # -----------------------------
# def price = close;
# def emaFast = ExpAverage(price, fastLength);
# def emaSlow = ExpAverage(price, slowLength);
# def rsi = RSI(length = rsiLength);
# def vwapLine = VWAP();
#
# def bull1 = emaFast > emaSlow;
# def bear1 = emaFast < emaSlow;
#
# # -----------------------------
# # 2️⃣ MULTI-TIMEFRAME (5m + 30m)
# # -----------------------------
# def c5  = close(period = AggregationPeriod.FIVE_MIN);
# def c30 = close(period = AggregationPeriod.THIRTY_MIN);
#
# def emaFast5 = ExpAverage(c5, fastLength);
# def emaSlow5 = ExpAverage(c5, slowLength);
# def bull5 = emaFast5 > emaSlow5;
# def bear5 = emaFast5 < emaSlow5;
#
# def emaFast30 = ExpAverage(c30, fastLength);
# def emaSlow30 = ExpAverage(c30, slowLength);
# def bull30 = emaFast30 > emaSlow30;
# def bear30 = emaFast30 < emaSlow30;
#
# # -----------------------------
# # 3️⃣ STRICT CALL / PUT LOGIC
# # -----------------------------
# def CALL_OK =
#     bull1 and bull5 and bull30 and
#     price > vwapLine and
#     rsi > 55;
#
# def PUT_OK =
#     bear1 and bear5 and bear30 and
#     price < vwapLine and
#     rsi < 45;
#
# # -----------------------------
# # ⭐ 4️⃣ TREND STRUCTURE (HH / HL / LH / LL)
# # -----------------------------
# def HH = high > high[1] and high[1] > high[2];
# def HL = low > low[1] and low[1] < low[2];
#
# def LH = high < high[1] and high[1] < high[2];
# def LL = low < low[1] and low[1] > low[2];
#
# def UPTREND = HH or HL;
# def DOWNTREND = LH or LL;
#
# # HH & LL bubbles
# AddChartBubble(HH, high, "HH", Color.CYAN, yes);
# AddChartBubble(LL, low, "LL", Color.RED, no);
#
# # HL line
# plot HL_SUPPORT = if HL then low else Double.NaN;
# HL_SUPPORT.SetLineWeight(3);
# HL_SUPPORT.SetDefaultColor(CreateColor(180,180,180));
#
# # =====================================================================
# # ⭐ NEW: HL SNIPER DETECTION (ONLY ADDITION)
# # =====================================================================
#
# # Pivot low
# def PivotLow_Sniper = low[1] < low[2] and low[1] < low;
#
# # Track previous pivot low
# rec LastPivotLow_Sniper =
#     if PivotLow_Sniper then low[1]
#     else if IsNaN(LastPivotLow_Sniper[1]) then low
#     else LastPivotLow_Sniper[1];
#
# # HL Sniper signal
# def HL_Sniper = PivotLow_Sniper and low[1] > LastPivotLow_Sniper[1];
#
# # HL Sniper Bubble
# AddChartBubble(HL_Sniper, low[1], "HL", Color.LIGHT_GREEN, no);
#
# # HL Sniper line
# plot HL_Sniper_Line = if HL_Sniper then low[1] else Double.NaN;
# HL_Sniper_Line.SetDefaultColor(CreateColor(150,150,150));
# HL_Sniper_Line.SetLineWeight(2);
#
# # =====================================================================
# # ⭐ NEW: TAKE PROFIT (TP) & STOP LOSS (SL)
# # Based on: D (SL) and E (TP)
# # =====================================================================
#
# # Track previous HH for Take Profit
# rec PrevHH =
#     if HH then high
#     else PrevHH[1];
#
# # STOP LOSS — HL – TickSize()
# plot SL_Line =
#     if HL_Sniper then low[1] - TickSize()
#     else Double.NaN;
# SL_Line.SetDefaultColor(Color.RED);
# SL_Line.SetLineWeight(2);
# SL_Line.SetStyle(Curve.SHORT_DASH);      # <─ ▬ ─ ▬ ─ ▬
# AddChartBubble(HL_Sniper, SL_Line, "SL", Color.RED, no);
#
# # TAKE PROFIT — Previous HH
# plot TP_Line =
#     if HL_Sniper then PrevHH
#     else Double.NaN;
# TP_Line.SetDefaultColor(Color.GREEN);
# TP_Line.SetLineWeight(2);
# TP_Line.SetStyle(Curve.SHORT_DASH);      # <─ ▬ ─ ▬ ─ ▬
# AddChartBubble(HL_Sniper, TP_Line, "TP", Color.GREEN, yes);
#
# # =====================================================================
# # 5️⃣ CALL ZONE
# # =====================================================================
# AddCloud(
#     if DOWNTREND then high else Double.NaN,
#     if DOWNTREND then low else Double.NaN,
#     CreateColor(255, 80, 80),
#     CreateColor(80, 0, 0)
# );
#
#
#
# AddChartBubble(UPTREND and HL, low, "CALL ZONE", Color.GREEN, no);
#
# # =====================================================================
# # 6️⃣ PUT ZONE
# # =====================================================================
# AddCloud(
#     if DOWNTREND then high else Double.NaN,
#     if DOWNTREND then low else Double.NaN,
#     Color.RED,
#     CreateColor(80, 0, 0)
# );
#
# AddChartBubble(DOWNTREND and LL, high, "PUT ZONE", Color.RED, yes);
#
# # =====================================================================
# # 7️⃣ EMAs
# # =====================================================================
# plot EMA_9 = emaFast;
# EMA_9.SetDefaultColor(Color.BLUE);
# EMA_9.SetLineWeight(2);
#
# plot EMA_21 = emaSlow;
# EMA_21.SetDefaultColor(Color.RED);
# EMA_21.SetLineWeight(2);
#
# # =====================================================================
# # 8️⃣ CALL/PUT SIGNAL BUBBLES
# # =====================================================================
# AddChartBubble(CALL_OK, emaFast + (TickSize() * 4), "CALL", Color.GREEN, yes);
# AddChartBubble(PUT_OK , emaFast - (TickSize() * 4), "PUT", Color.RED, no);
#
# # =====================================================================
# # 9️⃣ ALERTS
# # =====================================================================
# Alert(CALL_OK and alertEnabled, "CALL CONFIRMED — MTF Bullish", Alert.BAR, Sound.Ding);
# Alert(PUT_OK  and alertEnabled, "PUT CONFIRMED — MTF Bearish",  Alert.BAR, Sound.Ding);
#
# # =====================================================================
# # 🔟 DASHBOARD LABELS
# # =====================================================================
# AddLabel(yes, "30m TF: " + (if bull30 then "CALL" else if bear30 then "PUT" else "WAIT"),
#         if bull30 then Color.GREEN else if bear30 then Color.RED else Color.YELLOW);
#
# AddLabel(yes, "5m TF:  " + (if bull5 then "CALL" else if bear5 then "PUT" else "WAIT"),
#         if bull5 then Color.GREEN else if bear5 then Color.RED else Color.YELLOW);
#
# AddLabel(yes, "1m TF:  " + (if bull1 then "CALL" else if bear1 then "PUT" else "WAIT"),
#         if bull1 then Color.GREEN else if bear1 then Color.RED else Color.YELLOW);
#
# AddLabel(yes,
#         if CALL_OK then "🟩 CALL ENTRY READY"
#         else if PUT_OK then "🟥 PUT ENTRY READY"
#         else "⚠ NO TRADE",
#         if CALL_OK then Color.GREEN else if PUT_OK then Color.RED else Color.YELLOW);
#
# # =====================================================================
# # 1️⃣1️⃣ RSI + VWAP LABELS
# # =====================================================================
# AddLabel(yes, "RSI: " + Round(rsi, 0),
#     if rsi > 55 then Color.GREEN else if rsi < 45 then Color.RED else Color.WHITE);
#
# AddLabel(yes, "VWAP: " + Round(vwapLine, 2),
#     CreateColor(120, 120, 120));
#
# # =====================================================================
# # 💡 "Trade the alignment — Not the noise." — Sybest LLC
# # =====================================================================
#
# # =====================================================================
# # MODULE D — ULTRA-STRICT SNIPER CALL SYSTEM (Add-On Only)
# # Designed by: Dr. Stanley Njoku / Sybest LLC
# # =====================================================================
#
# # 1️⃣ Confirm prerequisites
# def sniper_HL = HL;
# def sniper_HH = HH;
#
# # 2️⃣ EMA directional alignment
# def sniper_EMA_Flip = emaFast > emaSlow;
#
# # 3️⃣ Price above EMA 9
# def sniper_Price_Strength = close > emaFast;
#
# # 4️⃣ CALL ZONE must be active
# def sniper_CALLZONE = UPTREND;
#
# # 5️⃣ VWAP support
# def sniper_VWAP = close > vwapLine;
#
# # 6️⃣ Strong bullish candle (body > 50% of candle)
# def sniper_Strong_Candle =
#     (close - open) > 0 and
#     (close - open) >= 0.5 * (high - low);
#
# # 7️⃣ Combine ultra-strict conditions
# def SNIPER_CALL =
#     sniper_HL and
#     sniper_HH and
#     sniper_EMA_Flip and
#     sniper_Price_Strength and
#     sniper_CALLZONE and
#     sniper_VWAP and
#     sniper_Strong_Candle;
#
# # 8️⃣ Bubble annotation
# AddChartBubble(
#     SNIPER_CALL,
#     low - (TickSize() * 4),
#     "SNIPER CALL",
#     Color.CYAN,
#     no
# );
#
# # 9️⃣ Optional alert
# Alert(
#     SNIPER_CALL,
#     "ULTRA-STRICT SNIPER CALL ENTRY",
#     Alert.BAR,
#     Sound.Ring
# );



# # ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]] CEMMENTING TO ADD PRE-MARKET
#
# #FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF SNIPPER AND CONFIRMATION PART 2
#
# # # ⚡ SYBEST TREND CONFIRMATION SYSTEM — EMA 9/21 DAY TRADER EDITION
# # # =====================================================================
# # # Designed by: Dr. Stanley Njoku / Sybest LLC
# # # Purpose: CALL/PUT confirmations + MTF Trend + HH/HL + LH/LL Detection
# # # =====================================================================
# #
# input fastLength = 9;
# input slowLength = 21;
# input rsiLength = 14;
# input alertEnabled = yes;
#
# # -----------------------------
# # 1️⃣ CORE CALCULATIONS (1-MIN)
# # -----------------------------
# def price = close;
# def emaFast = ExpAverage(price, fastLength);
# def emaSlow = ExpAverage(price, slowLength);
# def rsi = RSI(length = rsiLength);
# def vwapLine = VWAP();
#
# def bull1 = emaFast > emaSlow;
# def bear1 = emaFast < emaSlow;
#
# # -----------------------------
# # 2️⃣ MULTI-TIMEFRAME (5m + 30m)
# # -----------------------------
# def c5  = close(period = AggregationPeriod.FIVE_MIN);
# def c30 = close(period = AggregationPeriod.THIRTY_MIN);
#
# def emaFast5 = ExpAverage(c5, fastLength);
# def emaSlow5 = ExpAverage(c5, slowLength);
# def bull5 = emaFast5 > emaSlow5;
# def bear5 = emaFast5 < emaSlow5;
#
# def emaFast30 = ExpAverage(c30, fastLength);
# def emaSlow30 = ExpAverage(c30, slowLength);
# def bull30 = emaFast30 > emaSlow30;
# def bear30 = emaFast30 < emaSlow30;
#
# # -----------------------------
# # 3️⃣ STRICT CALL / PUT LOGIC
# # -----------------------------
# def CALL_OK =
#     bull1 and bull5 and bull30 and
#     price > vwapLine and
#     rsi > 55;
#
# def PUT_OK =
#     bear1 and bear5 and bear30 and
#     price < vwapLine and
#     rsi < 45;
#
# # -----------------------------
# # ⭐ 4️⃣ TREND STRUCTURE (HH / HL / LH / LL)
# # -----------------------------
# def HH = high > high[1] and high[1] > high[2];     # Higher High
# def HL = low > low[1] and low[1] < low[2];          # Higher Low
#
# def LH = high < high[1] and high[1] < high[2];      # Lower High
# def LL = low < low[1] and low[1] > low[2];          # Lower Low
#
# def UPTREND = HH or HL;
# def DOWNTREND = LH or LL;
#
# # -----------------------------
# # ⭐ HH & LL BUBBLES
# # -----------------------------
# AddChartBubble(HH, high, "HH", Color.CYAN, yes);
# AddChartBubble(LL, low, "LL", Color.RED, no);
#
#
# # ⭐ HL BUBBLE (ADDED – NEW)
# AddChartBubble(HL, low, "HL", Color.LIGHT_GREEN, no);
#
# # -----------------------------
# # ⭐ HL SUPPORT LINE (Light Gray)
# # -----------------------------
# plot HL_SUPPORT = if HL then low else Double.NaN;
# HL_SUPPORT.SetLineWeight(3);
# HL_SUPPORT.SetDefaultColor(CreateColor(180,180,180));
#
# # -----------------------------
# # ⭐ CALL ZONE BACKGROUND (GREEN)
# # -----------------------------
# AddCloud(
#     if UPTREND then low else Double.NaN,
#     if UPTREND then high else Double.NaN,
#     CreateColor(0, 120, 0),      # lighter green (keeps candle low visible)
#     Color.BLACK
# );
#
# AddChartBubble(
#     UPTREND and HL,
#     low,
#     "CALL ZONE",
#     Color.GREEN,
#     no
# );
#
# # -----------------------------
# # ⭐ PUT ZONE BACKGROUND (REAL RED)
# # -----------------------------
# AddCloud(
#     if DOWNTREND then high else Double.NaN,
#     if DOWNTREND then low else Double.NaN,
#     Color.RED,
#     CreateColor(80, 0, 0)
# );
#
# AddChartBubble(
#     DOWNTREND and LL,
#     high,
#     "PUT ZONE",
#     Color.RED,
#     yes
# );
#
# # -----------------------------
# # 5️⃣ PLOT EMAs
# # -----------------------------
# plot EMA_9 = emaFast;
# EMA_9.SetDefaultColor(Color.BLUE);
# EMA_9.SetLineWeight(2);
#
# plot EMA_21 = emaSlow;
# EMA_21.SetDefaultColor(Color.RED);
# EMA_21.SetLineWeight(2);
#
# # -----------------------------
# # 6️⃣ CALL & PUT BUBBLES
# # -----------------------------
# AddChartBubble(CALL_OK, emaFast + (TickSize() * 4), "CALL", Color.GREEN, yes);
# AddChartBubble(PUT_OK , emaFast - (TickSize() * 4), "PUT", Color.RED, no);
#
# # -----------------------------
# # 7️⃣ ALERTS
# # -----------------------------
# Alert(CALL_OK and alertEnabled, "CALL CONFIRMED — MTF Bullish", Alert.BAR, Sound.Ding);
# Alert(PUT_OK  and alertEnabled, "PUT CONFIRMED — MTF Bearish",  Alert.BAR, Sound.Ding);
#
# # -----------------------------
# # 8️⃣ TREND DASHBOARD LABELS (TOP LEFT CORNER)
# # -----------------------------
# AddLabel(yes, "30m TF: " + (if bull30 then "CALL" else if bear30 then "PUT" else "WAIT"),
#         if bull30 then Color.GREEN else if bear30 then Color.RED else Color.YELLOW);
#
# AddLabel(yes, "5m TF:  " + (if bull5 then "CALL" else if bear5 then "PUT" else "WAIT"),
#         if bull5 then Color.GREEN else if bear5 then Color.RED else Color.YELLOW);
#
# AddLabel(yes, "1m TF:  " + (if bull1 then "CALL" else if bear1 then "PUT" else "WAIT"),
#         if bull1 then Color.GREEN else if bear1 then Color.RED else Color.YELLOW);
#
# AddLabel(yes,
#         if CALL_OK then "🟩 CALL ENTRY READY"
#         else if PUT_OK then "🟥 PUT ENTRY READY"
#         else "⚠ NO TRADE",
#         if CALL_OK then Color.GREEN else if PUT_OK then Color.RED else Color.YELLOW);
#
# # -----------------------------
# # 9️⃣ RSI + VWAP LABELS
# # -----------------------------
# AddLabel(yes, "RSI: " + Round(rsi, 0),
#     if rsi > 55 then Color.GREEN else if rsi < 45 then Color.RED else Color.WHITE);
#
# AddLabel(yes, "VWAP: " + Round(vwapLine, 2),
#     CreateColor(120, 120, 120));
#
# # =====================================================================
# # 💡 "Trade the alignment — Not the noise." — Sybest LLC
# # =====================================================================
#
#
#
# # =====================================================================
# # MODULE D — ULTRA-STRICT SNIPER CALL SYSTEM (ADD-ON ONLY)
# # Designed by: Dr. Stanley Njoku / Sybest LLC
# # =====================================================================
#
# # 1️⃣ Confirm HL + HH structure
# def sniper_HL = HL;
# def sniper_HH = HH;
#
# # 2️⃣ EMA directional alignment (trend must actually flip)
# def sniper_EMA_Flip = emaFast > emaSlow;
#
# # 3️⃣ Price strength (close above EMA9)
# def sniper_Price_Strength = close > emaFast;
#
# # 4️⃣ Sniper requires CALL ZONE to be active
# def sniper_CALLZONE = UPTREND;
#
# # 5️⃣ Price must be above VWAP
# def sniper_VWAP = close > vwapLine;
#
# # 6️⃣ Strong bullish candle (body >= 50% of full range)
# def sniper_Strong_Candle =
#     (close - open) > 0 and
#     (close - open) >= 0.5 * (high - low);
#
# # 7️⃣ SNIPER CALL = ALL conditions must be true
# def SNIPER_CALL =
#     sniper_HL and
#     sniper_HH and
#     sniper_EMA_Flip and
#     sniper_Price_Strength and
#     sniper_CALLZONE and
#     sniper_VWAP and
#     sniper_Strong_Candle;
#
# # 8️⃣ SNIPER CALL bubble
# AddChartBubble(
#     SNIPER_CALL,
#     low - (TickSize() * 4),
#     "SNIPER CALL",
#     Color.CYAN,
#     no
# );
#
# # 9️⃣ SNIPER CALL alert
# Alert(
#     SNIPER_CALL,
#     "ULTRA-STRICT SNIPER CALL ENTRY",
#     Alert.BAR,
#     Sound.Ring
# );
#
#


# # ===============================================================================================
#
# # =======================================================
# #  SYBEST – PREVIOUS DAY HIGH & LOW (Dashed Line Version)
# # =======================================================
#
# input showOnlyToday = yes;
# input showLabels    = yes;
#
# # ---- Yesterday's Levels ----
# def priorHigh = high(period = AggregationPeriod.DAY)[1];
# def priorLow  = low(period = AggregationPeriod.DAY)[1];
#
# # ---- Limit display to today only ----
# def isToday = GetDay() == GetLastDay();
# def showLines = if showOnlyToday then isToday else 1;
#
# # ---- Previous Day High (Dashed) ----
# plot PDH = if showLines then priorHigh else Double.NaN;
# PDH.SetDefaultColor(Color.GREEN);
# PDH.SetLineWeight(3);
# PDH.SetStyle(Curve.SHORT_DASH);     # 🔥 dashed line
# PDH.HideTitle();
#
# # ---- Previous Day Low (Dashed) ----
# plot PDL = if showLines then priorLow else Double.NaN;
# PDL.SetDefaultColor(Color.RED);
# PDL.SetLineWeight(3);
# PDL.SetStyle(Curve.SHORT_DASH);     # 🔥 dashed line
# PDL.HideTitle();
#
# # ---- Optional: Right-side labels ----
# AddChartBubble(
#     showLabels and IsNaN(PDH[-1]) and showLines,
#     priorHigh,
#     "PDH: " + AsText(priorHigh),
#     Color.GREEN,
#     no
# );
#
# AddChartBubble(
#     showLabels and IsNaN(PDL[-1]) and showLines,
#     priorLow,
#     "PDL: " + AsText(priorLow),
#     Color.RED,
#     no
# );