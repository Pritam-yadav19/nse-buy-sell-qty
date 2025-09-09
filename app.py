import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# File for persistent Top-5 PCR storage
PCR_FILE = "pcr_history.csv"

# ─── Fetch & cache ──────────────────────────────────────────────────
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
    # warm-up request to obtain cookies
    session.get(f"{url_base}/option-chain", headers=headers, timeout=5)
    if is_index:
        api_url = f"{url_base}/api/option-chain-indices?symbol={symbol}"
    else:
        api_url = f"{url_base}/api/option-chain-equities?symbol={symbol}"
    resp = session.get(api_url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()

# ─── Parse into separate Call / Put DataFrames ──────────────────────
def parse_into_dfs(raw_json: dict):
    call_rows, put_rows = [], []
    data = raw_json.get("filtered", {}).get("data") or raw_json.get("data")
    if data is None:
        return pd.DataFrame(), pd.DataFrame()
    for itm in data:
        strike = itm.get("strikePrice")
        ce = itm.get("CE", {}) or {}
        pe = itm.get("PE", {}) or {}
        # try to read Open Interest (OI) if present, fallback to 0
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

# ─── Calculate Max Pain ─────────────────────────────────────────────
def calculate_max_pain(df_calls: pd.DataFrame, df_puts: pd.DataFrame):
    # expects numeric 'Strike' and 'Volume' columns
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

# ─── Streamlit UI ───────────────────────────────────────────────────
def main():
    # Auto-refresh every 60 seconds
    st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

    st.title("📈 NSE Option Chain – Top-5 / Top-10 / Top-20 PCR (OI-based)")
    st.markdown("""Calculate three PCRs using OI only:
- PCR Top-20 (union of top20 by Volume)
- PCR Top-10 (union of top10 by Volume)
- PCR Top-5  (union of top5 by Volume) — stored & plotted""")

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

    # Parse
    df_calls, df_puts = parse_into_dfs(raw)
    if df_calls.empty or df_puts.empty:
        st.error("No data available.")
        return

    # ─── Top-20 union (existing behavior) ─────────────────────────────
    try:
        top_calls_20 = set(df_calls.nlargest(20, "Volume")["Strike"].tolist())
        top_puts_20 = set(df_puts.nlargest(20, "Volume")["Strike"].tolist())
        top20_union = sorted(top_calls_20.union(top_puts_20))
    except Exception:
        top20_union = sorted(set(df_calls["Strike"]).union(df_puts["Strike"]))

    df_calls_20 = df_calls[df_calls["Strike"].isin(top20_union)].reset_index(drop=True)
    df_puts_20 = df_puts[df_puts["Strike"].isin(top20_union)].reset_index(drop=True)

    if df_calls_20.empty or df_puts_20.empty:
        st.error("After filtering to top-20 strikes there is no data to display.")
        return

    # ─── Top-10 union (existing) ────────────────────────────────────
    try:
        top_calls_10 = set(df_calls.nlargest(10, "Volume")["Strike"].tolist())
        top_puts_10 = set(df_puts.nlargest(10, "Volume")["Strike"].tolist())
        top10_union = sorted(top_calls_10.union(top_puts_10))
    except Exception:
        top10_union = top20_union[:10] if len(top20_union) >= 10 else top20_union

    df_calls_10 = df_calls[df_calls["Strike"].isin(top10_union)].reset_index(drop=True)
    df_puts_10 = df_puts[df_puts["Strike"].isin(top10_union)].reset_index(drop=True)

    if df_calls_10.empty or df_puts_10.empty:
        st.warning("Top-10 strike set is empty — Top-10 PCR cannot be computed.")

    # ─── Top-5 union (NEW: stored & plotted) ──────────────────────────
    try:
        top_calls_5 = set(df_calls.nlargest(5, "Volume")["Strike"].tolist())
        top_puts_5 = set(df_puts.nlargest(5, "Volume")["Strike"].tolist())
        top5_union = sorted(top_calls_5.union(top_puts_5))
    except Exception:
        top5_union = top10_union[:5] if len(top10_union) >= 5 else top10_union

    df_calls_5 = df_calls[df_calls["Strike"].isin(top5_union)].reset_index(drop=True)
    df_puts_5 = df_puts[df_puts["Strike"].isin(top5_union)].reset_index(drop=True)

    if df_calls_5.empty or df_puts_5.empty:
        st.warning("Top-5 strike set is empty — Top-5 PCR cannot be computed.")

    # ─── PCR calculations (OI only) ──────────────────────────────────
    def compute_pcr_oi(calls_df, puts_df):
        calls_oi = int(calls_df["OI"].sum()) if not calls_df.empty else 0
        puts_oi = int(puts_df["OI"].sum()) if not puts_df.empty else 0
        if calls_oi == 0 and puts_oi == 0:
            return None
        if calls_oi == 0:
            return None  # avoid divide-by-zero; show N/A
        return puts_oi / calls_oi

    pcr_top20 = compute_pcr_oi(df_calls_20, df_puts_20)
    pcr_top10 = compute_pcr_oi(df_calls_10, df_puts_10)
    pcr_top5  = compute_pcr_oi(df_calls_5, df_puts_5)  # stored & plotted

    # ─── Persist only Top-5 PCR to CSV (timestamp + value) ─────────
    if pcr_top5 is not None:
        entry = {"timestamp": datetime.utcnow().isoformat(), "pcr_top5": pcr_top5}
        if os.path.exists(PCR_FILE):
            df_hist = pd.read_csv(PCR_FILE)
            df_hist = pd.concat([df_hist, pd.DataFrame([entry])], ignore_index=True)
        else:
            df_hist = pd.DataFrame([entry])
        # ensure column name pcr_top5 exists and save
        df_hist.to_csv(PCR_FILE, index=False)
    else:
        df_hist = pd.read_csv(PCR_FILE) if os.path.exists(PCR_FILE) else pd.DataFrame()

    # ─── Display metrics ─────────────────────────────────────────────
    # Show Top-20 PCR metric
    if pcr_top20 is None:
        st.metric("PCR (Top-20)", "N/A", delta=None, help="Computed using OI across top-20 union")
    else:
        st.metric("PCR (Top-20)", f"{pcr_top20:.2f}", delta=None, help="Computed using OI across top-20 union")

    # Show Top-10 PCR metric
    if pcr_top10 is None:
        st.metric("PCR (Top-10)", "N/A", delta=None, help="Computed using OI across top-10 union")
    else:
        st.metric("PCR (Top-10)", f"{pcr_top10:.2f}", delta=None, help="Computed using OI across top-10 union")

    # Show Top-5 PCR metric (new)
    if pcr_top5 is None:
        st.metric("PCR (Top-5)", "N/A", delta=None, help="Computed using OI across top-5 union")
    else:
        st.metric("PCR (Top-5)", f"{pcr_top5:.2f}", delta=None, help="Computed using OI across top-5 union")

    # Interpretation caption for Top-10 and Top-5
    if pcr_top10 is not None:
        if pcr_top10 < 1:
            st.caption("Top-10 PCR < 1 ⇒ More Calls than Puts (bearish bias).")
        elif pcr_top10 > 1:
            st.caption("Top-10 PCR > 1 ⇒ More Puts than Calls (bullish bias).")
        else:
            st.caption("Top-10 PCR ≈ 1 ⇒ Balanced positioning.")

    if pcr_top5 is not None:
        if pcr_top5 < 1:
            st.caption("Top-5 PCR < 1 ⇒ More Calls than Puts (bearish bias).")
        elif pcr_top5 > 1:
            st.caption("Top-5 PCR > 1 ⇒ More Puts than Calls (bullish bias).")
        else:
            st.caption("Top-5 PCR ≈ 1 ⇒ Balanced positioning.")

    # Max Pain (use df_calls_20/df_puts_20 for calculations)
    max_pain_strike, max_pain_value = calculate_max_pain(df_calls_20, df_puts_20)
    if max_pain_strike is not None:
        st.metric("Max Pain Strike", max_pain_strike, delta=None, help=f"Total pain: {max_pain_value:.0f}")

    # ─── Display top-5 tables with Buy/Sell Ratio ───────────────────
    st.subheader(f"Top 5 Call Strikes by Volume for {symbol} (from top-20-set)")
    display_calls = df_calls_20.nlargest(5, "Volume").reset_index(drop=True)
    st.dataframe(display_calls)

    st.subheader(f"Top 5 Put Strikes by Volume for {symbol} (from top-20-set)")
    display_puts = df_puts_20.nlargest(5, "Volume").reset_index(drop=True)
    st.dataframe(display_puts)

    st.sidebar.markdown("Data is cached for 60 seconds. Top-20 used for main calculations; Top-10 shown as a metric; Top-5 PCR stored & plotted.")

    # ─── Continuous Top-5 PCR Plot (x = entry count, y = PCR value) ──
    if not df_hist.empty and "pcr_top5" in df_hist.columns:
        # drop rows where pcr_top5 is NaN (older files without this column may exist)
        plot_series = df_hist["pcr_top5"].dropna().astype(float)
        if not plot_series.empty:
            fig, ax = plt.subplots()
            ax.plot(plot_series.index + 1, plot_series.values, marker="o")
            ax.set_title("Top-5 PCR Trend (Persistent)")
            ax.set_xlabel("Entry Count")
            ax.set_ylabel("PCR (Top-5, OI)")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
