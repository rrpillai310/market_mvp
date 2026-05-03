"""Market MVP — Streamlit dashboard entry point.

Run from project root:
    streamlit run market_mvp/app.py
"""
import streamlit as st
from market_mvp import ui_data as d

st.set_page_config(
    page_title="Market MVP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Market MVP")
st.caption("SPY / QQQ forward-return predictor — powered by LightGBM + DuckDB")

st.divider()

# ── Database status ───────────────────────────────────────────────────────────

if not d.db_exists():
    st.warning(
        "No database found. Run the pipeline first:\n\n"
        "```bash\ncd /Users/rakeshpillai\n"
        "python3 -m market_mvp.pipeline --symbols SPY QQQ --skip-social\n```"
    )
    st.stop()

st.subheader("Pipeline status")

col1, col2, col3, col4 = st.columns(4)

for sym, col in zip(["SPY", "QQQ"], [col1, col2]):
    last = d.get_last_ingest_date(sym)
    col.metric(f"{sym} last ingest", last or "—")

counts = d.get_row_counts()
col3.metric("Price rows", f"{counts.get('prices_daily', 0):,}")
col4.metric("Feature rows", f"{counts.get('features_daily', 0):,}")

st.divider()

# ── Model status ──────────────────────────────────────────────────────────────

st.subheader("Saved models")

any_model = False
cols = st.columns(len(d.SYMBOLS) * len(d.HORIZONS))
for i, sym in enumerate(d.SYMBOLS):
    for j, h in enumerate(d.HORIZONS):
        col = cols[i * len(d.HORIZONS) + j]
        metrics = d.load_metrics(sym, h)
        if metrics:
            any_model = True
            dir_acc = metrics.get("mean_dir_acc")
            col.metric(
                f"{sym} h={h}",
                f"{dir_acc:.1%} dir acc" if dir_acc else "trained",
                f"RMSE {metrics['mean_rmse']:.4f}" if metrics.get("mean_rmse") else None,
            )
        else:
            col.metric(f"{sym} h={h}", "not trained", delta_color="off")

if not any_model:
    st.info(
        "No trained models yet. After ingesting data, run:\n\n"
        "```bash\npython3 -m market_mvp.train --symbol SPY --horizon 5\n```"
    )

st.divider()

# ── Data source status ────────────────────────────────────────────────────────

st.subheader("Data sources")

rows = [
    ("Alpha Vantage (prices/options/news)", counts.get("prices_daily", 0) > 0),
    ("EDGAR earnings", counts.get("edgar_earnings", 0) > 0),
    ("Fed minutes", counts.get("fed_minutes", 0) > 0),
    ("Social sentiment", counts.get("social_sentiment_daily", 0) > 0),
    ("News sentiment", counts.get("news_sentiment", 0) > 0),
]

for label, ok in rows:
    icon = "✅" if ok else "⬜"
    n = {
        "Alpha Vantage (prices/options/news)": counts.get("prices_daily", 0),
        "EDGAR earnings": counts.get("edgar_earnings", 0),
        "Fed minutes": counts.get("fed_minutes", 0),
        "Social sentiment": counts.get("social_sentiment_daily", 0),
        "News sentiment": counts.get("news_sentiment", 0),
    }.get(label, 0)
    st.write(f"{icon} **{label}** — {n:,} rows")

st.divider()
st.caption("Use the sidebar to navigate to Predictions, Signals, or Performance.")
