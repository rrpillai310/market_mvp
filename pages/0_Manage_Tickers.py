import time

import streamlit as st

from market_mvp import ui_data as d

st.set_page_config(page_title="Manage Tickers · Market MVP", page_icon="🔧", layout="wide")
st.title("🔧 Manage Tickers")

# ── Tracked tickers ───────────────────────────────────────────────────────────

tracked = d.get_tracked_symbols()
st.subheader(f"Tracked tickers ({len(tracked)})")

if tracked:
    cols = st.columns(min(len(tracked), 6))
    for i, sym in enumerate(tracked):
        ticker = sym["ticker"]
        col = cols[i % 6]
        has_data = d.symbol_has_data(ticker)
        model_ready = d.model_exists(ticker, 5)
        badge = "✅ model ready" if model_ready else ("📥 data only" if has_data else "⬜ no data")
        col.metric(ticker, sym["type"], badge)
        if col.button("Remove", key=f"rm_{ticker}"):
            d.remove_symbol(ticker)
            st.rerun()
else:
    st.info("No tickers tracked yet.")

st.divider()

# ── Add ticker ────────────────────────────────────────────────────────────────

st.subheader("Add ticker")

col_in, col_btn = st.columns([3, 1])
new_ticker = col_in.text_input("Symbol", placeholder="e.g. NVDA, XLE, AAPL").upper().strip()
pull_now = st.checkbox("Pull data & train model immediately", value=True)

if col_btn.button("Add", type="primary", disabled=not new_ticker):
    if new_ticker in [s["ticker"] for s in d.get_tracked_symbols()]:
        st.warning(f"{new_ticker} is already tracked.")
    else:
        with st.spinner(f"Validating {new_ticker}…"):
            try:
                import yfinance as yf
                info = yf.Ticker(new_ticker).fast_info
                # fast_info raises or returns empty if ticker invalid
                _ = info.market_cap
                quote_type = yf.Ticker(new_ticker).info.get("quoteType", "EQUITY")
                sym_type = "etf" if quote_type in ("ETF", "MUTUALFUND") else "stock"
            except Exception:
                st.error(f"Could not validate {new_ticker} on Yahoo Finance. Check the symbol.")
                st.stop()

        d.add_symbol(new_ticker, sym_type)
        st.success(f"Added **{new_ticker}** ({sym_type}).")

        if pull_now:
            if d.is_pipeline_running():
                st.warning("A pipeline is already running — data will be pulled in the next daily run.")
            else:
                pid = d.trigger_pipeline(new_ticker)
                st.info(f"⏳ Pipeline started for **{new_ticker}** (PID {pid}). "
                        f"Check back in a few minutes — the Predictions page will show results once training completes.")
        st.rerun()

st.divider()

# ── Pipeline status ───────────────────────────────────────────────────────────

st.subheader("Pipeline status")

if d.is_pipeline_running():
    st.info("⏳ Pipeline is running. Auto-refreshing…")
    time.sleep(8)
    st.rerun()
else:
    st.success("✅ No pipeline running.")
    last_log = sorted(
        __import__("pathlib").Path("/Users/rakeshpillai/market_mvp/logs").glob("pipeline_*.log"),
        default=None,
    )
    if last_log:
        with st.expander("Last pipeline log"):
            st.code(last_log[-1].read_text()[-3000:], language="")
