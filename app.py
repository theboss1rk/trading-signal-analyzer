import streamlit as st
import pandas as pd
import numpy as np
import requests


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SCALPING MASTER",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #050805;
        color: #e8ffe8;
    }

    [data-testid="stSidebar"] {
        background-color: #071007;
    }

    .main-title {
        font-size: 42px;
        font-weight: 900;
        color: #39ff88;
        text-shadow: 0 0 14px rgba(57,255,136,0.55);
        margin-bottom: 0;
    }

    .subtitle {
        color: #8ca88f;
        margin-bottom: 25px;
    }

    .signal-box {
        padding: 28px;
        border: 1px solid #1eff72;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            rgba(20,45,25,0.95),
            rgba(5,15,7,0.98)
        );
        text-align: center;
        box-shadow: 0 0 20px rgba(30,255,114,0.10);
    }

    .signal-call {
        color: #39ff88;
        font-size: 48px;
        font-weight: 900;
    }

    .signal-put {
        color: #ff5c5c;
        font-size: 48px;
        font-weight: 900;
    }

    .signal-none {
        color: #ffd166;
        font-size: 42px;
        font-weight: 900;
    }

    .small-note {
        color: #8ca88f;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚡ SCALPING MASTER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Market Analysis • Backtesting • Paper Trading Research</div>',
    unsafe_allow_html=True
)


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    result = result.where(
        ~((avg_loss == 0) & (avg_gain > 0)),
        100
    )

    result = result.where(
        ~((avg_gain == 0) & (avg_loss > 0)),
        0
    )

    return result


def atr(df, period=14):

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def adx(df, period=14):

    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0.0
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0.0
    )

    atr_value = atr(df, period)

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()
        / atr_value.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()
        / atr_value.replace(0, np.nan)
    )

    denominator = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_indicators(df):

    x = df.copy()

    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)

    x["rsi"] = rsi(x["close"], 14)

    x["macd"] = (
        ema(x["close"], 12)
        - ema(x["close"], 26)
    )

    x["macd_signal"] = ema(
        x["macd"],
        9
    )

    x["adx"] = adx(x, 14)
    x["atr"] = atr(x, 14)

    return x


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(
    x,
    adx_threshold=20
):

    if len(x) < 55:
        return "NO TRADE", 0, 0, []

    last = x.iloc[-1]

    required = [
        "close",
        "ema20",
        "ema50",
        "rsi",
        "macd",
        "macd_signal",
        "adx",
        "atr"
    ]

    if any(
        pd.isna(last[column])
        for column in required
    ):
        return (
            "NO TRADE",
            0,
            0,
            ["Indicators are not ready."]
        )

    bull = 0
    bear = 0
    reasons = []

    # EMA
    if last["ema20"] > last["ema50"]:
        bull += 2
        reasons.append(
            "EMA20 > EMA50 → bullish trend"
        )

    elif last["ema20"] < last["ema50"]:
        bear += 2
        reasons.append(
            "EMA20 < EMA50 → bearish trend"
        )

    else:
        reasons.append(
            "EMA20 = EMA50 → neutral"
        )

    # RSI
    if 55 <= last["rsi"] <= 70:
        bull += 1
        reasons.append(
            f"RSI {last['rsi']:.1f} → bullish momentum"
        )

    elif 30 <= last["rsi"] <= 45:
        bear += 1
        reasons.append(
            f"RSI {last['rsi']:.1f} → bearish momentum"
        )

    elif last["rsi"] > 70:
        reasons.append(
            f"RSI {last['rsi']:.1f} → overbought caution"
        )

    elif last["rsi"] < 30:
        reasons.append(
            f"RSI {last['rsi']:.1f} → oversold caution"
        )

    else:
        reasons.append(
            f"RSI {last['rsi']:.1f} → neutral"
        )

    # MACD
    if last["macd"] > last["macd_signal"]:
        bull += 1
        reasons.append(
            "MACD > Signal → bullish"
        )

    elif last["macd"] < last["macd_signal"]:
        bear += 1
        reasons.append(
            "MACD < Signal → bearish"
        )

    # Price vs EMA20
    if last["close"] > last["ema20"]:
        bull += 1
        reasons.append(
            "Price above EMA20"
        )

    elif last["close"] < last["ema20"]:
        bear += 1
        reasons.append(
            "Price below EMA20"
        )

    # ADX filter
    if last["adx"] < adx_threshold:
        reasons.append(
            f"ADX {last['adx']:.1f} < {adx_threshold} "
            "→ weak trend → NO TRADE"
        )

        return (
            "NO TRADE",
            bull,
            bear,
            reasons
        )

    if bull >= 4 and bull > bear:
        reasons.append(
            f"Bullish confirmation: {bull}/5 points"
        )

        return (
            "CALL",
            bull,
            bear,
            reasons
        )

    if bear >= 4 and bear > bull:
        reasons.append(
            f"Bearish confirmation: {bear}/5 points"
        )

        return (
            "PUT",
            bull,
            bear,
            reasons
        )

    reasons.append(
        "Signals are not sufficiently aligned → NO TRADE"
    )

    return (
        "NO TRADE",
        bull,
        bear,
        reasons
    )


# ============================================================
# DATA FETCH
# ============================================================

def fetch_market_data(
    symbol,
    interval,
    bars,
    api_key
):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": int(bars),
        "apikey": api_key,
        "format": "JSON"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "Twelve Data request timed out. "
            "Please try again."
        )

    except requests.exceptions.RequestException:

        raise RuntimeError(
            "Could not connect to Twelve Data."
        )

    if response.status_code == 429:

        raise RuntimeError(
            "Twelve Data rate limit reached. "
            "Wait a little and try again."
        )

    if response.status_code != 200:

        raise RuntimeError(
            f"Twelve Data HTTP error: "
            f"{response.status_code}"
        )

    try:

        data = response.json()

    except ValueError:

        raise RuntimeError(
            "Twelve Data returned invalid data."
        )

    message = str(
        data.get("message", "")
    ).lower()

    if (
        data.get("status") == "error"
        or "rate limit" in message
        or "api credits" in message
    ):

        raise RuntimeError(
            data.get(
                "message",
                "Twelve Data API error."
            )
        )

    values = data.get("values")

    if not values:

        raise RuntimeError(
            "No market candles returned."
        )

    df = pd.DataFrame(values)

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing columns: "
            + ", ".join(missing)
        )

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

    df = df.drop_duplicates(
        subset=["datetime"]
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    if len(df) < 55:

        raise RuntimeError(
            f"Only {len(df)} valid candles available. "
            "At least 55 are required."
        )

    return df


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    df,
    expiry_candles,
    payout_percent,
    adx_threshold
):

    if len(df) < 60:

        return pd.DataFrame(), {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "calls": 0,
            "puts": 0,
            "max_consecutive_losses": 0,
            "profit_loss": 0,
            "max_drawdown": 0
        }

    x = calculate_indicators(df)

    trades = []

    start_index = 54

    last_signal_index = (
        len(x) - expiry_candles - 1
    )

    for i in range(
        start_index,
        last_signal_index + 1
    ):

        history = x.iloc[:i + 1]

        signal, bull, bear, reasons = generate_signal(
            history,
            adx_threshold
        )

        if signal not in [
            "CALL",
            "PUT"
        ]:
            continue

        entry_price = float(
            x.iloc[i]["close"]
        )

        expiry_price = float(
            x.iloc[
                i + expiry_candles
            ]["close"]
        )

        if signal == "CALL":

            result = (
                "WIN"
                if expiry_price > entry_price
                else "LOSS"
            )

        else:

            result = (
                "WIN"
                if expiry_price < entry_price
                else "LOSS"
            )

        trades.append(
            {
                "Signal Time": x.iloc[i]["datetime"],
                "Expiry Time": x.iloc[
                    i + expiry_candles
                ]["datetime"],
                "Signal": signal,
                "Entry": entry_price,
                "Expiry Price": expiry_price,
                "Result": result
            }
        )

    results = pd.DataFrame(trades)

    if results.empty:

        return results, {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "calls": 0,
            "puts": 0,
            "max_consecutive_losses": 0,
            "profit_loss": 0,
            "max_drawdown": 0
        }

    results["P/L"] = np.where(
        results["Result"] == "WIN",
        payout_percent / 100,
        -1.0
    )

    total = len(results)

    wins = int(
        (results["Result"] == "WIN").sum()
    )

    losses = int(
        (results["Result"] == "LOSS").sum()
    )

    win_rate = (
        wins / total * 100
        if total > 0
        else 0
    )

    calls = int(
        (results["Signal"] == "CALL").sum()
    )

    puts = int(
        (results["Signal"] == "PUT").sum()
    )

    max_consecutive_losses = 0
    current_losses = 0

    for result in results["Result"]:

        if result == "LOSS":

            current_losses += 1

            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses
            )

        else:

            current_losses = 0

    equity = results["P/L"].cumsum()

    peak = equity.cummax()

    drawdown = equity - peak

    max_drawdown = float(
        drawdown.min()
    )

    stats = {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "calls": calls,
        "puts": puts,
        "max_consecutive_losses":
            max_consecutive_losses,
        "profit_loss":
            float(results["P/L"].sum()),
        "max_drawdown":
            max_drawdown
    }

    return results, stats


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ⚙️ MARKET SETTINGS"
)

broker = st.sidebar.selectbox(
    "Broker / Reference",
    [
        "Research / Demo",
        "Quotex (Reference Only)"
    ]
)

symbol = st.sidebar.selectbox(
    "Currency Pair",
    [
        "EUR/USD",
        "GBP/USD",
        "USD/JPY",
        "AUD/USD",
        "USD/CAD",
        "USD/CHF",
        "NZD/USD"
    ]
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

expiry_candles = st.sidebar.selectbox(
    "Expiry",
    [1, 2, 3, 5],
    index=0
)

bars = st.sidebar.slider(
    "Historical Candles",
    60,
    500,
    200,
    10
)

adx_threshold = st.sidebar.slider(
    "ADX Minimum",
    10,
    40,
    20
)

api_key = st.sidebar.text_input(
    "Twelve Data API Key",
    type="password"
)

analyze_button = st.sidebar.button(
    "⚡ ANALYZE MARKET",
    use_container_width=True
)


# ============================================================
# INFO
# ============================================================

st.info(
    "Research / paper-trading system only. "
    "No broker account connection and no automatic trade execution."
)

if broker == "Quotex (Reference Only)":

    st.warning(
        "Quotex is only a reference label. "
        "This application does not connect to Quotex "
        "or place orders."
    )


# ============================================================
# LIVE ANALYSIS
# ============================================================

if analyze_button:

    if not api_key.strip():

        st.error(
            "Please enter your Twelve Data API key."
        )

    else:

        with st.spinner(
            "Fetching latest market data..."
        ):

            try:

                live_df = fetch_market_data(
                    symbol,
                    interval,
                    bars,
                    api_key
                )

                live_x = calculate_indicators(
                    live_df
                )

                signal, bull, bear, reasons = generate_signal(
                    live_x,
                    adx_threshold
                )

                st.session_state.live_x = live_x
                st.session_state.live_signal = signal
                st.session_state.live_bull = bull
                st.session_state.live_bear = bear
                st.session_state.live_reasons = reasons

            except Exception as error:

                st.error(
                    f"Market-data error: {error}"
                )


# ============================================================
# LIVE RESULT
# ============================================================

if "live_x" in st.session_state:

    live_x = st.session_state.live_x

    signal = st.session_state.live_signal
    bull = st.session_state.live_bull
    bear = st.session_state.live_bear
    reasons = st.session_state.live_reasons

    last = live_x.iloc[-1]

    if signal == "CALL":

        signal_class = "signal-call"

    elif signal == "PUT":

        signal_class = "signal-put"

    else:

        signal_class = "signal-none"

    total_score = bull + bear

    technical_score = (
        max(bull, bear)
        / total_score
        * 100
        if total_score > 0
        else 0
    )

    st.markdown(
        f"""
        <div class="signal-box">

            <div class="{signal_class}">
                {signal}
            </div>

            <div>
                Technical Score:
                <strong>{technical_score:.0f}/100</strong>
            </div>

            <div class="small-note">
                Technical score is NOT a probability
                of winning.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Price",
        f"{last['close']:.6g}"
    )

    c2.metric(
        "EMA20",
        f"{last['ema20']:.6g}"
    )

    c3.metric(
        "EMA50",
        f"{last['ema50']:.6g}"
    )

    c4.metric(
        "RSI",
        f"{last['rsi']:.1f}"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "MACD",
        f"{last['macd']:.6g}"
    )

    c6.metric(
        "ADX",
        f"{last['adx']:.1f}"
    )

    c7.metric(
        "ATR",
        f"{last['atr']:.6g}"
    )

    c8.metric(
        "Direction",
        f"Bull {bull} / Bear {bear}"
    )

    st.caption(
        f"{symbol} • {interval} • "
        f"Latest available candle: "
        f"{last['datetime']}"
    )

    st.subheader(
        "🔎 Signal Analysis"
    )

    for reason in reasons:

        st.write(
            "• " + reason
        )

    with st.expander(
        "📊 Indicator Snapshot"
    ):

        st.dataframe(
            live_x[
                [
                    "datetime",
                    "close",
                    "ema20",
                    "ema50",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "adx",
                    "atr"
                ]
            ].tail(15),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# BACKTEST
# ============================================================

st.divider()

st.header(
    "🧪 Historical Backtesting"
)

st.write(
    "Each historical signal uses only information available "
    "at that c
