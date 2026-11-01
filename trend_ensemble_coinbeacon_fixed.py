# Trend Ensemble — Binance USD-M Futures PAPER trading dashboard
# PAPER ONLY: this app never sends orders and never requests API credentials.
# Run locally: pip install streamlit pandas numpy requests
#             streamlit run trend_ensemble_paper_dashboard.py

import time
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

st.set_page_config(page_title="Trend Ensemble Paper Terminal", layout="wide")
API_BASE = "https://api.coinbeacon.io"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
FEE_RATE = 0.0006  # Pine commission_value=0.06%, charged on traded notional
INTERVAL_MS = 60 * 60 * 1000

st.title("Trend Ensemble — Futures Paper Trading")
st.caption("Binance Futures market data via CoinBeacon • Virtual fills only • No exchange orders")

with st.sidebar:
    st.header("Paper account")
    starting_balance = st.number_input("Starting virtual USDT", min_value=100.0, value=10000.0, step=500.0)
    target_vol = st.number_input("Target annual volatility", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
    risk_mult = st.number_input("Risk multiplier", min_value=0.25, max_value=3.5, value=2.0, step=0.25)
    n_coins = st.number_input("Coins sharing account", min_value=1, max_value=10, value=4, step=1)
    band_base = st.number_input("No-trade band", min_value=0.0, max_value=1.0, value=0.075, step=0.025)
    max_lev = st.number_input("Max leverage per coin sleeve", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    fee_rate = st.number_input("Commission per side (%)", min_value=0.0, max_value=1.0, value=0.06, step=0.01) / 100
    refresh_secs = st.selectbox("Refresh interval", [15, 30, 60, 120], index=1)
    st.info("This is a simulation. Funding, slippage, liquidation engine and exchange quantity filters are not modelled.")

@st.cache_data(ttl=600, show_spinner=False)
def get_klines(symbol, api_key):
    url = f"{API_BASE}/public/ohlcv/binance-futures/{symbol}/1h"
    response = requests.get(url, headers={"X-API-Key": api_key},
                            params={"limit": 1500}, timeout=25)
    response.raise_for_status()
    payload = response.json()
    bars = payload.get("bars", [])
    if not bars:
        raise RuntimeError(f"CoinBeacon returned no hourly bars for {symbol}: {payload}")
    df = pd.DataFrame(bars)
    required = ["openTime", "open", "high", "low", "close", "volume", "closeTime"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Unexpected CoinBeacon candle response; missing {missing}")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["openTime"]), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(pd.to_numeric(df["closeTime"]), unit="ms", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] < now].copy()
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]


@st.cache_data(ttl=600, show_spinner=False)
def get_live_quotes(api_key):
    markets = ",".join(f"binance-futures:{s}" for s in SYMBOLS)
    response = requests.get(
        f"{API_BASE}/public/market/quotes/batch",
        headers={"X-API-Key": api_key}, params={"m": markets}, timeout=25
    )
    response.raise_for_status()
    payload = response.json()
    result = {}
    for row in payload.get("quotes", []):
        symbol = row.get("m", "").split(":", 1)[-1]
        price = row.get("price")
        if symbol in SYMBOLS and price is not None:
            result[symbol] = float(price)
    return result


def tanh_clip(x):
    if pd.isna(x) or not np.isfinite(x):
        return np.nan
    x = float(np.clip(x, -15.0, 15.0))
    return math.tanh(x)

def calc_components(df):
    c, h, l = df["close"], df["high"], df["low"]
    ret = c.pct_change()
    out = pd.DataFrame(index=df.index)
    for n in [96, 192, 384, 768]:
        sd = c.rolling(n, min_periods=n).std(ddof=0)
        # Pine ta.stdev uses population standard deviation by default.
        out[f"z{n}"] = ((c - c.rolling(n, min_periods=n).mean()) / sd / 2).map(tanh_clip)
    sd720 = ret.rolling(720, min_periods=720).std(ddof=0)
    for n in [168, 336, 720]:
        roc = c.pct_change(n)
        denom = sd720 * math.sqrt(n)
        out[f"ts{n}"] = ((roc / denom) / 1.5).map(tanh_clip)
    for n in [120, 240, 480, 960, 168, 336, 720]:
        hh = h.rolling(n, min_periods=n).max()
        ll = l.rolling(n, min_periods=n).min()
        out[f"dc{n}"] = np.where(hh > ll, (c - ll) / (hh - ll) * 2 - 1, 0.0)
    out["annvol120"] = ret.rolling(120, min_periods=120).std(ddof=0) * math.sqrt(8760)
    out["annvol240"] = ret.rolling(240, min_periods=240).std(ddof=0) * math.sqrt(8760)
    out["annvol480"] = ret.rolling(480, min_periods=480).std(ddof=0) * math.sqrt(8760)
    out["annvol720"] = ret.rolling(720, min_periods=720).std(ddof=0) * math.sqrt(8760)
    out["zt"] = out[[f"z{n}" for n in [96,192,384,768]]].mean(axis=1)
    out["tsm"] = out[[f"ts{n}" for n in [168,336,720]]].mean(axis=1)
    out["dc1"] = out[[f"dc{n}" for n in [120,240,480,960]]].mean(axis=1)
    out["dc2"] = out[[f"dc{n}" for n in [168,336,720]]].mean(axis=1)
    out["score"] = out[["zt","tsm","dc1","dc2"]].mean(axis=1)
    invs = 1.0 / out[["annvol120","annvol240","annvol480","annvol720"]]
    out["ivol"] = invs.mean(axis=1)
    return out

def compute_signal(df, target_vol, risk_mult, n_coins, max_lev):
    comp = calc_components(df)
    if len(comp) < 1:
        return None
    row = comp.iloc[-1]
    if not np.isfinite(row["score"]) or not np.isfinite(row["ivol"]):
        return None
    raw = row["score"] * target_vol * row["ivol"]
    target = float(np.clip(raw * math.sqrt(n_coins) * risk_mult, -max_lev, max_lev))
    return {"score": float(row["score"]), "ivol": float(row["ivol"]), "target": target,
            "bar_time": df.iloc[-1]["close_time"], "price": float(df.iloc[-1]["close"])}

def init_state(balance):
    st.session_state.paper = {
        "cash": float(balance), "positions": {s: {"qty": 0.0, "avg": 0.0, "weight": 0.0} for s in SYMBOLS},
        "last_bar": {}, "trades": [], "equity_history": [], "started": datetime.now(timezone.utc).isoformat()
    }

if "paper" not in st.session_state:
    init_state(starting_balance)
if "running" not in st.session_state:
    st.session_state.running = False
if "last_config_balance" not in st.session_state:
    st.session_state.last_config_balance = starting_balance

c1, c2 = st.columns([1, 4])
with c1:
    if st.button("▶ START / RESUME", use_container_width=True):
        st.session_state.running = True
    if st.button("Ⅱ PAUSE", use_container_width=True):
        st.session_state.running = False
    if st.button("↻ RESET PAPER ACCOUNT", use_container_width=True):
        init_state(starting_balance)
        st.session_state.running = False
with c2:
    st.warning("PAPER MODE ONLY — simulated positions and P&L. No exchange order endpoint is called.")

if starting_balance != st.session_state.last_config_balance and not st.session_state.paper["trades"]:
    st.session_state.paper["cash"] = float(starting_balance)
    st.session_state.last_config_balance = starting_balance

def mark_equity(prices):
    p = st.session_state.paper
    unreal = 0.0
    gross = 0.0
    for s, pos in p["positions"].items():
        px = prices.get(s)
        if px and pos["qty"] != 0:
            unreal += pos["qty"] * (px - pos["avg"])
            gross += abs(pos["qty"] * px)
    return p["cash"] + unreal, unreal, gross

def rebalance(symbol, sig, price):
    p = st.session_state.paper
    pos = p["positions"][symbol]
    equity, _, _ = mark_equity({symbol: price, **{k: v.get("last_price", price) for k,v in p["positions"].items()}})
    sleeve_equity = max(equity, 0.0) / float(n_coins)
    band = band_base * n_coins * risk_mult
    target_weight = sig["target"]
    if abs(target_weight - pos["weight"]) <= band:
        return
    desired_notional = target_weight * sleeve_equity
    desired_qty = desired_notional / price
    delta_qty = desired_qty - pos["qty"]
    if abs(delta_qty * price) < 5:
        return
    fee = abs(delta_qty * price) * fee_rate
    old_qty = pos["qty"]
    new_qty = old_qty + delta_qty
    # Realize P&L on the closed portion; handle flips correctly.
    realized = 0.0
    if old_qty != 0 and np.sign(delta_qty) != np.sign(old_qty):
        close_qty = min(abs(old_qty), abs(delta_qty)) * np.sign(old_qty)
        realized = close_qty * (price - pos["avg"])
    p["cash"] += realized - fee
    if new_qty == 0 or (old_qty != 0 and np.sign(new_qty) != np.sign(old_qty)):
        new_avg = price if new_qty != 0 else 0.0
    elif old_qty == 0:
        new_avg = price
    elif np.sign(delta_qty) == np.sign(old_qty):
        new_avg = (abs(old_qty)*pos["avg"] + abs(delta_qty)*price) / (abs(old_qty)+abs(delta_qty))
    else:
        new_avg = pos["avg"]
    pos["qty"], pos["avg"], pos["weight"] = new_qty, new_avg, target_weight
    p["trades"].append({"time": datetime.now(timezone.utc), "symbol": symbol,
                        "side": "BUY" if delta_qty > 0 else "SELL", "qty": abs(delta_qty),
                        "price": price, "notional": abs(delta_qty*price), "fee": fee,
                        "realized_pnl": realized, "new_target_weight": target_weight})

# Read the API key from Streamlit Secrets; never put it in this source file.
try:
    COINBEACON_API_KEY = str(st.secrets["COINBEACON_API_KEY"]).strip()
except Exception:
    st.error("Missing COINBEACON_API_KEY. Add it in Streamlit → Manage app → Settings → Secrets.")
    st.stop()

# Fetch all four symbols. Signals are acted on once per newly closed 1h candle.
errors, signals, prices = {}, {}, {}
with st.spinner("Loading Binance Futures candles through CoinBeacon..."):
    try:
        quote_prices = get_live_quotes(COINBEACON_API_KEY)
    except Exception as quote_error:
        quote_prices = {}
        errors["Quotes"] = f"Live quote request failed: {quote_error}"

    for sym in SYMBOLS:
        try:
            df = get_klines(sym, COINBEACON_API_KEY)
            if len(df) < 960:
                errors[sym] = f"Only {len(df)} confirmed hourly candles returned; strategy needs at least 960 confirmed candles (960-bar Donchian lookback; volatility lookback is 720)."
                continue
            sig = compute_signal(df, target_vol, risk_mult, n_coins, max_lev)
            if sig is None:
                errors[sym] = "Indicators are still warming up; more hourly history is needed."
                continue
            signals[sym] = sig
            # Use a current public ticker quote for simulated execution/mark-to-market,
            # while keeping the signal based only on the last confirmed 1H candle.
            # Current quote is used for paper marking; signal uses confirmed 1H candles.
            live_price = quote_prices.get(sym, sig["price"])
            prices[sym] = live_price
            st.session_state.paper["positions"][sym]["last_price"] = live_price
            last = st.session_state.paper["last_bar"].get(sym)
            bar_id = sig["bar_time"].isoformat()
            if st.session_state.running and last != bar_id:
                rebalance(sym, sig, live_price)
                st.session_state.paper["last_bar"][sym] = bar_id
        except Exception as e:
            errors[sym] = str(e)

if prices:
    equity, unreal, gross = mark_equity(prices)
    p = st.session_state.paper
    start = float(starting_balance)
    realized_total = sum(t["realized_pnl"] for t in p["trades"])
    fees_total = sum(t["fee"] for t in p["trades"])
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Paper equity", f"${equity:,.2f}", f"{equity-start:+,.2f}")
    m2.metric("Unrealized P&L", f"${unreal:,.2f}")
    m3.metric("Realized P&L", f"${realized_total:,.2f}")
    m4.metric("Fees paid", f"${fees_total:,.2f}")
    st.subheader("Live strategy signals & positions")
    rows=[]
    for s in SYMBOLS:
        if s not in signals: continue
        sig=signals[s]; pos=p["positions"][s]; px=prices[s]
        upnl=pos["qty"]*(px-pos["avg"]) if pos["qty"] else 0.0
        rows.append({"Symbol":s,"Last confirmed 1H close":sig["bar_time"].strftime("%Y-%m-%d %H:%M UTC"),
                     "Price":px,"Score":sig["score"],"Target exposure (x sleeve)":sig["target"],
                     "Position qty":pos["qty"],"Avg entry":pos["avg"],"Unrealized P&L":upnl})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Gross position notional: ${gross:,.2f} • Gross leverage vs equity: {gross/max(equity,1e-9):.2f}x • Bot: {'RUNNING' if st.session_state.running else 'PAUSED'}")
else:
    st.error("No usable market data was loaded. Check CoinBeacon API key, API response and candle-history depth.")

if errors:
    with st.expander("Market data/API diagnostics"):
        for sym, err in errors.items():
            st.write(f"**{sym}:** {err}")

st.subheader("Paper trade log")
trades = st.session_state.paper["trades"]
if trades:
    log=pd.DataFrame(trades).sort_values("time", ascending=False)
    st.dataframe(log, use_container_width=True, hide_index=True)
else:
    st.info("No simulated orders yet. Press START / RESUME. The engine will rebalance only when a new confirmed hourly candle is available.")

st.caption("Model assumptions: one quarter of total equity per coin sleeve; max leverage is applied to each sleeve; 0.06% fee on traded notional; funding, slippage, maintenance margin, liquidation and exchange filters are not included. This is not a broker-connected system.")
time.sleep(refresh_secs)
st.rerun()
