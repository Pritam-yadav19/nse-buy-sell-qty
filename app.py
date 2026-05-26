import streamlit as st
import pandas as pd
import numpy as np
from nsepython import option_chain

# ─────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_option_chain_data(symbol):
    return option_chain(symbol)

# ─────────────────────────────────────────────────────────────
# PARSE DATA
# ─────────────────────────────────────────────────────────────
def parse_oi_change(raw_json):

    data = raw_json["records"]["data"]

    call_rows = []
    put_rows = []

    for item in data:

        strike = item.get("strikePrice")

        ce = item.get("CE", {})
        pe = item.get("PE", {})

        # CALLS
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

        # PUTS
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

    # ─────────────────────────────────────────────────────────
    # FILTERS
    # ─────────────────────────────────────────────────────────

    # meaningful OI only
    df_calls = df_calls[df_calls["OI"] > 30000]
    df_puts = df_puts[df_puts["OI"] > 30000]

    # remove garbage spikes
    df_calls = df_calls[
        (df_calls["OI Change %"] < 500)
        & (df_calls["OI Change %"] > -500)
    ]

    df_puts = df_puts[
        (df_puts["OI Change %"] < 500)
        & (df_puts["OI Change %"] > -500)
    ]

    return df_calls, df_puts

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
def main():

    st.set_page_config(
        page_title="Smart OI Tracker",
        layout="wide"
    )

    st.title("⚡ Smart OI Change Tracker")

    # Manual refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

    option_type = st.sidebar.radio(
        "Type",
        ["Index", "Equity"]
    )

    is_index = option_type == "Index"

    if is_index:

        symbol = st.sidebar.selectbox(
            "Symbol",
            ["NIFTY", "BANKNIFTY", "FINNIFTY"]
        )

    else:

        symbol = st.sidebar.text_input(
            "Symbol",
            "RELIANCE"
        ).upper().strip()

    if not symbol:
        st.error("Enter symbol")
        return

    # ─────────────────────────────────────────────────────────
    # FETCH DATA
    # ─────────────────────────────────────────────────────────

    try:

        raw = get_option_chain_data(symbol)

    except Exception as e:

        st.error(f"Failed to fetch NSE data: {e}")
        return

    # ─────────────────────────────────────────────────────────
    # PARSE
    # ─────────────────────────────────────────────────────────

    try:

        df_calls, df_puts = parse_oi_change(raw)

    except Exception as e:

        st.error(f"Data parsing failed: {e}")
        return

    if df_calls.empty or df_puts.empty:

        st.error("No usable data found")
        return

    # ─────────────────────────────────────────────────────────
    # TOP / LOWEST
    # ─────────────────────────────────────────────────────────

    top_calls = df_calls.sort_values(
        by="OI Change %",
        ascending=False
    ).head(5)

    top_puts = df_puts.sort_values(
        by="OI Change %",
        ascending=False
    ).head(5)

    low_calls = df_calls.sort_values(
        by="OI Change %",
        ascending=True
    ).head(5)

    low_puts = df_puts.sort_values(
        by="OI Change %",
        ascending=True
    ).head(5)

    # ─────────────────────────────────────────────────────────
    # DISPLAY
    # ─────────────────────────────────────────────────────────

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🔥 Highest OI Change Calls")

        st.dataframe(
            top_calls[
                ["Strike", "OI", "OI Change %", "IV"]
            ].reset_index(drop=True),
            use_container_width=True
        )

    with col2:

        st.subheader("🔥 Highest OI Change Puts")

        st.dataframe(
            top_puts[
                ["Strike", "OI", "OI Change %", "IV"]
            ].reset_index(drop=True),
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("📉 Lowest OI Change Calls")

        st.dataframe(
            low_calls[
                ["Strike", "OI", "OI Change %", "IV"]
            ].reset_index(drop=True),
            use_container_width=True
        )

    with col4:

        st.subheader("📉 Lowest OI Change Puts")

        st.dataframe(
            low_puts[
                ["Strike", "OI", "OI Change %", "IV"]
            ].reset_index(drop=True),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()