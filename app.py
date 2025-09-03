import streamlit as st
import requests
import pandas as pd
import numpy as np

# ─── Fetch & cache ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_option_chain(symbol: str, is_index: bool = True):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en,hi;q=0.9",
    }
    session = requests.Session()
    url_base = "https://www.nseindia.com"

    # First warm-up request (longer timeout)
    session.get(f"{url_base}/option-chain", headers=headers, timeout=20)

    if is_index:
        api_url = f"{url_base}/api/option-chain-indices?symbol={symbol}"
    else:
        api_url = f"{url_base}/api/option-chain-equities?symbol={symbol}"

    # Main API request (longer timeout)
    resp = session.get(api_url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()

# ─── Parse into separate Call / Put DataFrames ────────────────────────────────
def parse_into_dfs(raw_json: dict):
    call_rows, put_rows = [], []
    data = raw_json.get("filtered", {}).get("data") or raw_json.get("data")
    if data is None:
        return pd.DataFrame(), pd.DataFrame()
    for itm in data:
        strike = itm.get("strikePrice")
        ce = itm.get("CE", {}) or {}
        pe = itm.get("PE", {}) or {}

        ce_oi = ce.get("openInterest", ce.get("openInterestQty", 0)) if ce else 0
        pe_oi = pe.get("openInterest", pe.get("openInterestQty", 0)) if pe else 0

        call_rows.append({
            "Strike": strike,
            "LTP": ce.get("lastPrice", np.nan),
            "Volume": ce.get("totalTradedVolume", 0),
            "OI": ce_oi,
            "Total Buy Qty": ce.get("totalBuyQuantity", 0),
            "Total Sell Qty": ce.get("totalSellQuantity", 0),
            "Buy/Sell Ratio": round(
                (ce.get("totalBuyQuantity", 0) / ce.get("totalSellQuantity", 1)), 2
            ) if ce.get("totalSellQuantity", 0) > 0 else None,
        })
        put_rows.append({
            "Strike": strike,
            "LTP": pe.get("lastPrice", np.nan),
            "Volume": pe.get("totalTradedVolume", 0),
            "OI": pe_oi,
            "Total Buy Qty": pe.get("totalBuyQuantity", 0),
            "Total Sell Qty": pe.get("totalSellQuantity", 0),
            "Buy/Sell Ratio": round(
                (pe.get("totalBuyQuantity", 0) / pe.get("totalSellQuantity", 1)), 2
            ) if pe.get("totalSellQuantity", 0) > 0 else None,
        })
    df_calls = pd.DataFrame(call_rows)
    df_puts = pd.DataFrame(put_rows)

    # Ensure numeric types for arithmetic
    for df in (df_calls, df_puts):
        if not df.empty:
            df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
            df["LTP"] = pd.to_numeric(df["LTP"], errors="coerce")
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)
            df["OI"] = pd.to_numeric(df["OI"], errors="coerce").fillna(0).astype(int)
            df["Total Buy Qty"] = pd.to_numeric(df["Total Buy Qty"], errors="coerce").fillna(0).astype(int)
            df["Total Sell Qty"] = pd.to_numeric(df["Total Sell Qty"], errors="coerce").fillna(0).astype(int)
    return df_calls, df_puts

# ─── Calculate Max Pain ──────────────────────────────────────────────────────
def calculate_max_pain(df_calls: pd.DataFrame, df_puts: pd.DataFrame):
    strikes = sorted(set(df_calls['Strike']).union(df_puts['Strike']))
    pains = []
    for p in strikes:
        call_pain = ((np.maximum(df_calls['Strike'] - p, 0)) * df_calls['Volume']).sum()
        put_pain  = ((np.maximum(p - df_puts['Strike'], 0)) * df_puts['Volume']).sum()
        pains.append({'Strike': p, 'Pain': call_pain + put_pain})
    pain_df = pd.DataFrame(pains)
    if pain_df.empty:
        return None, None
    max_row = pain_df.loc[pain_df['Pain'].idxmin()]
    return int(max_row['Strike']), max_row['Pain']

# ─── Streamlit UI ────────────────────────────────────────────────────────────
def main():
    # Auto-refresh every 60 seconds
    st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

    st.title("📈 NSE Option Chain – Top 5 by Volume + Buy/Sell Ratio + LTP + Max Pain + PCR")
    st.markdown(
        "Choose an Index or Equity to see the top 5 Call & Put strikes by volume, "
        "their LTP, total buy/sell quantities, buy/sell ratio, Max Pain, and PCR."
    )

    option_type = st.sidebar.radio("Option Type", ["Index", "Equity"])
    is_index = (option_type == "Index")
    if is_index:
        symbol = st.sidebar.selectbox("Index Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
    else:
        symbol = st.sidebar.text_input("Equity Symbol", "RELIANCE").upper().strip()

    if not symbol:
        st.error("Please enter a symbol to proceed.")
        return

    with st.spinner(f"Fetching {symbol} option chain..."):
        try:
            raw = get_option_chain(symbol, is_index)
        except Exception as e:
            st.error(f"Failed to fetch data for {symbol}: {e}")
            return

    # Underlying
    underlying = raw.get("records", {}).get("underlyingValue")
    if underlying is not None:
        st.subheader(f"Underlying Price: {underlying}")

    # Parse data
    df_calls, df_puts = parse_into_dfs(raw)
    if df_calls.empty or df_puts.empty:
        st.error("No data available.")
        return

    # ─── NEW: restrict to union of top-20 by Volume from each side ───────────
    try:
        top_calls_strikes = set(df_calls.nlargest(20, "Volume")["Strike"].tolist())
        top_puts_strikes = set(df_puts.nlargest(20, "Volume")["Strike"].tolist())
        top_strikes_union = sorted(top_calls_strikes.union(top_puts_strikes))
    except Exception:
        top_strikes_union = sorted(set(df_calls["Strike"]).union(df_puts["Strike"]))

    df_calls = df_calls[df_calls["Strike"].isin(top_strikes_union)].reset_index(drop=True)
    df_puts = df_puts[df_puts["Strike"].isin(top_strikes_union)].reset_index(drop=True)

    if df_calls.empty or df_puts.empty:
        st.error("After filtering to top strikes there is no data to display.")
        return

    # ─── PCR calculation ─────────────────────────────────────────────────────
    calls_oi_sum = int(df_calls["OI"].sum()) if "OI" in df_calls.columns else 0
    puts_oi_sum = int(df_puts["OI"].sum()) if "OI" in df_puts.columns else 0

    if calls_oi_sum > 0 or puts_oi_sum > 0:
        denom = calls_oi_sum if calls_oi_sum > 0 else 1
        pcr_value = puts_oi_sum / denom if denom else None
        pcr_source = "OI"
    else:
        calls_vol_sum = int(df_calls["Volume"].sum())
        puts_vol_sum = int(df_puts["Volume"].sum())
        denom = calls_vol_sum if calls_vol_sum > 0 else 1
        pcr_value = puts_vol_sum / denom if denom else None
        pcr_source = "Volume"

    if pcr_value is None:
        st.metric("PCR", "N/A", delta=None, help="PCR could not be calculated (division by zero).")
    else:
        st.metric(label=f"PCR ({pcr_source})", value=f"{pcr_value:.2f}", delta=None,
                  help=f"Put/Call ratio using {pcr_source}.")
        if pcr_value < 1:
            interp = "PCR < 1 ⇒ More Calls than Puts (generally bearish)."
        elif pcr_value > 1:
            interp = "PCR > 1 ⇒ More Puts than Calls (generally bullish)."
        else:
            interp = "PCR ≈ 1 ⇒ Balanced positioning."
        st.caption(interp)

    # Max Pain
    max_pain_strike, max_pain_value = calculate_max_pain(df_calls, df_puts)
    if max_pain_strike is None:
        st.info("Max Pain could not be calculated for the selected strikes.")
    else:
        st.metric("Max Pain Strike", max_pain_strike, delta=None, help=f"Total pain: {max_pain_value:.0f}")

    # Display top 5
    st.subheader(f"Top 5 Call Strikes by Volume for {symbol} (from top-20-set)")
    st.dataframe(df_calls.nlargest(5, "Volume").reset_index(drop=True))

    st.subheader(f"Top 5 Put Strikes by Volume for {symbol} (from top-20-set)")
    st.dataframe(df_puts.nlargest(5, "Volume").reset_index(drop=True))

    st.sidebar.markdown("Data is cached for 60 seconds. Filtered to top-20 strikes per side for calculations.")

if __name__ == "__main__":
    main()
