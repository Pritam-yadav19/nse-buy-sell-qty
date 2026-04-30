import streamlit as st
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import numpy as np
import time
from urllib.parse import urlencode

# ─── Fetch & cache (2 min refresh) ──────────────────────────────────
@st.cache_data(ttl=120)
def get_option_chain(symbol: str, is_index: bool = True, expiry: str | None = None):
    base = "https://www.nseindia.com"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
        "X-Requested-With": "XMLHttpRequest",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
    }

    session = requests.Session()
    session.headers.update(headers)

    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # Warmup the NSE site so the cookie and headers are prepared
    session.get(f"{base}/option-chain", timeout=10)
    time.sleep(0.5)

    if expiry is None:
        url = f"{base}/api/option-chain-contract-info?{urlencode({'symbol': symbol})}"
        res = session.get(url, timeout=10)

        if "application/json" not in res.headers.get("Content-Type", ""):
            raise Exception("Blocked by NSE (non-JSON response)")

        res.raise_for_status()
        json_data = res.json()
        return {"records": {"expiryDates": json_data.get("expiryDates", [])}}

    option_type = "Indices" if is_index else "Equity"
    url = f"{base}/api/option-chain-v3?{urlencode({'type': option_type, 'symbol': symbol, 'expiry': expiry})}"
    res = session.get(url, timeout=10)

    if "application/json" not in res.headers.get("Content-Type", ""):
        raise Exception("Blocked by NSE (non-JSON response)")

    res.raise_for_status()
    return res.json()

# ─── Parse & calculate OI % change ─────────────────────────────────
def parse_oi_change(raw_json):
    data = (
        raw_json.get("filtered", {}).get("data")
        or raw_json.get("records", {}).get("data")
        or raw_json.get("data")
    )

    call_rows, put_rows = [], []

    for item in data:
        strike = item.get("strikePrice")

        ce = item.get("CE", {}) or {}
        pe = item.get("PE", {}) or {}

        if ce:
            oi = ce.get("openInterest", 0)
            chg_oi = ce.get("changeinOpenInterest", 0)
            prev_oi = oi - chg_oi
            pct = (chg_oi / prev_oi * 100) if prev_oi > 0 else 0

            call_rows.append({
                "Strike": strike,
                "OI": oi,
                "OI Change %": round(pct, 2),
                "IV": ce.get("impliedVolatility", np.nan)
            })

        if pe:
            oi = pe.get("openInterest", 0)
            chg_oi = pe.get("changeinOpenInterest", 0)
            prev_oi = oi - chg_oi
            pct = (chg_oi / prev_oi * 100) if prev_oi > 0 else 0

            put_rows.append({
                "Strike": strike,
                "OI": oi,
                "OI Change %": round(pct, 2),
                "IV": pe.get("impliedVolatility", np.nan)
            })

    df_calls = pd.DataFrame(call_rows)
    df_puts = pd.DataFrame(put_rows)

    # Filter for meaningful open interest
    df_calls = df_calls[df_calls["OI"] > 30000]
    df_puts = df_puts[df_puts["OI"] > 30000]

    # Remove extreme noise (garbage spikes)
    df_calls = df_calls[df_calls["OI Change %"] < 500]
    df_puts = df_puts[df_puts["OI Change %"] < 500]

    # Smart scoring (real activity)
    if not df_calls.empty:
        df_calls["Score"] = df_calls["OI"] * df_calls["OI Change %"]
    if not df_puts.empty:
        df_puts["Score"] = df_puts["OI"] * df_puts["OI Change %"]

    return df_calls, df_puts

# ─── UI ────────────────────────────────────────────────────────────
def main():
    st.markdown('<meta http-equiv="refresh" content="120">', unsafe_allow_html=True)

    st.title("⚡ Smart OI Change Tracker")

    option_type = st.sidebar.radio("Type", ["Index", "Equity"])
    is_index = option_type == "Index"

    if is_index:
        symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
    else:
        symbol = st.sidebar.text_input("Symbol", "RELIANCE").upper().strip()

    if not symbol:
        st.error("Enter symbol")
        return

    try:
        raw_initial = get_option_chain(symbol, is_index)
    except Exception as e:
        st.error(f"NSE blocked request: {e}")
        return

    expiry_dates = raw_initial.get("records", {}).get("expiryDates", [])
    selected_expiry = None

    if expiry_dates:
        selected_expiry = st.sidebar.selectbox("Expiry", expiry_dates)

    try:
        raw = get_option_chain(symbol, is_index, selected_expiry)
    except Exception as e:
        st.error(f"NSE blocked request: {e}")
        return

    df_calls, df_puts = parse_oi_change(raw)

    if df_calls.empty or df_puts.empty:
        st.error("No usable data (filtered out or blocked)")
        return

    # Top 5 based on decreasing OI Change % (max first)
    top_calls = df_calls.sort_values(by="OI Change %", ascending=False).head(5)
    top_puts = df_puts.sort_values(by="OI Change %", ascending=False).head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Top Calls (Real Activity)")
        st.dataframe(
            top_calls[["Strike", "OI", "OI Change %", "IV"]].reset_index(drop=True),
            use_container_width=True,
        )

    with col2:
        st.subheader("🔥 Top Puts (Real Activity)")
        st.dataframe(
            top_puts[["Strike", "OI", "OI Change %", "IV"]].reset_index(drop=True),
            use_container_width=True,
        )

if __name__ == "__main__":
    main()