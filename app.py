
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI Signal Analyzer", page_icon="📈", layout="wide")
st.title("📈 Multi-Timeframe Trading Signal Analyzer")
st.caption("Research / paper-trading tool — not a guaranteed-profit system.")

uploaded = st.file_uploader(
    "Upload OHLC CSV (columns: time, open, high, low, close)",
    type=["csv"]
)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(df, n=14):
    prev = df.close.shift(1)
    tr = pd.concat([(df.high-df.low), (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def macd(s):
    m = ema(s,12)-ema(s,26)
    sig = ema(m,9)
    return m, sig

def adx(df, n=14):
    up = df.high.diff()
    down = -df.low.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    a = atr(df,n)
    pdi = 100 * plus.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    mdi = 100 * minus.ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean()

def indicators(df):
    x=df.copy()
    x["ema20"]=ema(x.close,20); x["ema50"]=ema(x.close,50)
    x["rsi"]=rsi(x.close)
    x["macd"],x["macd_sig"]=macd(x.close)
    x["adx"]=adx(x)
    mid=x.close.rolling(20).mean()
    sd=x.close.rolling(20).std()
    x["bb_up"]=mid+2*sd; x["bb_dn"]=mid-2*sd
    x["atr"]=atr(x)
    return x

def score(x):
    last=x.iloc[-1]
    bull=bear=0
    reasons=[]
    if last.ema20>last.ema50: bull+=2; reasons.append("EMA trend bullish")
    else: bear+=2; reasons.append("EMA trend bearish")
    if last.rsi>55: bull+=1; reasons.append("RSI bullish")
    elif last.rsi<45: bear+=1; reasons.append("RSI bearish")
    if last.macd>last.macd_sig: bull+=1; reasons.append("MACD bullish")
    elif last.macd<last.macd_sig: bear+=1; reasons.append("MACD bearish")
    if last.adx>=20:
        reasons.append(f"ADX trend strength {last.adx:.1f}")
    else:
        reasons.append("ADX weak — avoid low-quality setup")
    if last.close>last.ema20: bull+=1
    else: bear+=1
    total=bull+bear
    if last.adx<20 or total<4:
        return "NO TRADE", max(bull,bear)/max(total,1)*100, reasons
    if bull>bear and bull>=4:
        return "CALL", bull/total*100, reasons
    if bear>bull and bear>=4:
        return "PUT", bear/total*100, reasons
    return "NO TRADE", max(bull,bear)/max(total,1)*100, reasons

if uploaded:
    df=pd.read_csv(uploaded)
    df.columns=[c.lower().strip() for c in df.columns]
    required={"open","high","low","close"}
    if not required.issubset(df.columns):
        st.error("CSV must contain open, high, low, close columns.")
    else:
        for c in ["open","high","low","close"]:
            df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=list(required)).reset_index(drop=True)
        x=indicators(df)
        sig,conf,reasons=score(x)
        a,b,c=st.columns(3)
        a.metric("Signal",sig)
        b.metric("Confidence",f"{conf:.0f}/100")
        c.metric("Latest Close",f"{x.close.iloc[-1]:.5g}")
        st.subheader("Indicator snapshot")
        st.dataframe(x[["close","ema20","ema50","rsi","macd","macd_sig","adx","atr"]].tail(10),use_container_width=True)
        st.subheader("Why")
        for r in reasons: st.write("• "+r)
        st.warning("This is a research signal, not a prediction guarantee. Test in paper/demo mode first.")
else:
    st.info("Upload historical OHLC CSV to generate a signal.")
