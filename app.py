import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(
    page_title="Live Market Signal Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Live Market Signal Analyzer")
st.caption("Research / paper-trading tool. Signals are not guaranteed predictions.")


# ---------- INDICATORS ----------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    prev = df.close.shift(1)

    tr = pd.concat(
        [
            df.high - df.low,
            (df.high - prev).abs(),
            (df.low - prev).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(alpha=1/n, adjust=False).mean()


def adx(df, n=14):
    up = df.high.diff()
    down = -df.low.diff()

    plus = up.where(
        (up > down) & (up > 0),
        0.0
    )

    minus = down.where(
        (down > up) & (down > 0),
        0.0
    )

    a = atr(df, n)

    pdi = (
        100
        * plus.ewm(alpha=1/n, adjust=False).mean()
        / a.replace(0, np.nan)
    )

    mdi = (
        100
        * minus.ewm(alpha=1/n, adjust=False).mean()
        / a.replace(0, np.nan)
    )

    dx = (
        100
        * (pdi - mdi).abs()
        / (pdi + mdi).replace(0, np.nan)
    )

    return dx.ewm(alpha=1/n, adjust=False).mean()


def indicators(df):

    x = df.copy()

    x["ema20"] = ema(x.close, 20)
    x["ema50"] = ema(x.close, 50)

    x["rsi"] = rsi(x.close)

    macd_line = ema(x.close, 12) - ema(x.close, 26)

    x["macd"] = macd_line
    x["macd_sig"] = ema(macd_line, 9)

    x["adx"] = adx(x)

    x["atr"] = atr(x)

    return x


# ---------- SIGNAL ENGINE ----------

def make_signal(x):

    last = x.iloc[-1]

    bull = 0
    bear = 0

    reasons = []

    # EMA
    if last.ema20 > last.ema50:

        bull += 2

        reasons.append(
            "EMA20 > EMA50: bullish trend"
        )

    elif last.ema20 < last.ema50:

        bear += 2

        reasons.append(
            "EMA20 < EMA50: bearish trend"
        )

    # RSI
    if last.rsi > 55:

        bull += 1

        reasons.append(
            f"RSI {last.rsi:.1f}: bullish momentum"
        )

    elif last.rsi < 45:

        bear += 1

        reasons.append(
            f"RSI {last.rsi:.1f}: bearish momentum"
        )

    else:

        reasons.append(
            f"RSI {last.rsi:.1f}: neutral"
        )

    # MACD
    if last.macd > last.macd_sig:

        bull += 1

        reasons.append(
            "MACD bullish"
        )

    elif last.macd < last.macd_sig:

        bear += 1

        reasons.append(
            "MACD bearish"
        )

    # Price vs EMA20
    if last.close > last.ema20:

        bull += 1

    elif last.close < last.ema20:

        bear += 1

    # ADX filter
    if last.adx < 20:

        reasons.append(
            f"ADX {last.adx:.1f}: weak trend → NO TRADE"
        )

        return "NO TRADE", 0, reasons

    total = bull + bear

    if total == 0:

        return "NO TRADE", 0, reasons

    confidence = max(bull, bear) / total * 100

    # CALL
    if bull >= 4 and bull > bear:

        return "CALL", confidence, reasons

    # PUT
    if bear >= 4 and bear > bull:

        return "PUT", confidence, reasons

    return "NO TRADE", confidence, reasons


# ---------- SIDEBAR ----------

st.sidebar.header("⚙️ Market Settings")

symbol = st.sidebar.text_input(
    "Symbol",
    "EUR/USD"
)

interval = st.sidebar.selectbox(
    "Timeframe",
    [
        "1min",
        "5min",
        "15min",
        "30min",
        "1h"
    ],
    index=1
)

bars = st.sidebar.slider(
    "Number of candles",
    60,
    300,
    120,
    10
)

api_key = st.sidebar.text_input(
    "Twelve Data API Key",
    type="password"
)

refresh = st.sidebar.button(
    "🔄 Get Latest Market Data",
    use_container_width=True
)


# ---------- INFORMATION ----------

st.info(
    "Uses Twelve Data market data. "
    "It does NOT connect to Quotex or place trades."
)


if not api_key:

    st.warning(
        "Enter your Twelve Data API key in the sidebar."
    )

    st.stop()


# ---------- GET MARKET DATA ----------

if refresh or "market_df" not in st.session_state:

    try:

        response = requests.get(
            "https://api.twelvedata.com/time_series",

            params={
                "symbol": symbol.strip().upper(),
                "interval": interval,
                "outputsize": bars,
                "apikey": api_key.strip(),
                "format": "JSON"
            },

            timeout=15
        )

        data = response.json()

        if (
            response.status_code != 200
            or data.get("status") == "error"
        ):

            st.error(
                data.get(
                    "message",
                    "Market-data request failed"
                )
            )

            st.stop()

        values = data.get("values", [])

        if len(values) < 55:

            st.error(
                "Not enough candles returned."
            )

            st.stop()

        df = pd.DataFrame(values)

        for column in [
            "open",
            "high",
            "low",
            "close"
        ]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close"
            ]
        )

        df = df.sort_values(
            "datetime"
        ).reset_index(drop=True)

        st.session_state.market_df = df

    except Exception as error:

        st.error(
            f"Market data error: {error}"
        )

        st.stop()


# ---------- ANALYSIS ----------

df = st.session_state.market_df

x = indicators(df)

signal, confidence, reasons = make_signal(x)


# ---------- TOP METRICS ----------

a, b, c, d = st.columns(4)

a.metric(
    "Signal",
    signal
)

b.metric(
    "Score",
    f"{confidence:.0f}/100"
)

c.metric(
    "Last Price",
    f"{x.close.iloc[-1]:.6g}"
)

d.metric(
    "ADX",
    f"{x.adx.iloc[-1]:.1f}"
)


st.caption(
    f"{symbol.upper()} • {interval} • "
    f"Last candle: {x.datetime.iloc[-1]}"
)


# ---------- REASONS ----------

st.subheader(
    "🔎 Why this signal?"
)

for reason in reasons:

    st.write(
        "• " + reason
    )


# ---------- INDICATORS ----------

st.subheader(
    "📊 Indicator Snapshot"
)

st.dataframe(

    x[
        [
            "datetime",
            "close",
            "ema20",
            "ema50",
            "rsi",
            "macd",
            "macd_sig",
            "adx",
            "atr"
        ]
    ].tail(15),

    use_container_width=True,

    hide_index=True
)


# ---------- WARNING ----------

st.warning(
    "Technical signal only. "
    "There is no guaranteed win rate. "
    "Validate with historical backtesting and "
    "demo/paper trading before risking money."
)
