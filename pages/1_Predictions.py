import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_mvp import ui_data as d

st.set_page_config(page_title="Predictions · Market MVP", page_icon="🎯", layout="wide")

st.title("🎯 Predictions")

# ── Controls ──────────────────────────────────────────────────────────────────

col_sym, col_h, _ = st.columns([1, 1, 3])
symbol = col_sym.selectbox("Symbol", d.get_symbols())
horizon = col_h.selectbox("Horizon (days)", d.HORIZONS)

st.divider()

# ── LightGBM prediction ───────────────────────────────────────────────────────

if not d.model_exists(symbol, horizon):
    st.warning(
        f"No model for {symbol} h={horizon}. Run:\n\n"
        f"```bash\npython3 -m market_mvp.train --symbol {symbol} --horizon {horizon}\n```"
    )
    st.stop()

result = d.predict(symbol, horizon)
if result is None:
    st.error("Model found but no features in DB. Run the full pipeline first.")
    st.stop()

# ── Model comparison cards ────────────────────────────────────────────────────

tft_result = d.predict_tft(symbol, horizon)

if tft_result:
    lgbm_col, tft_col = st.columns(2)
else:
    lgbm_col = st.container()


def _pred_card(container, res, label):
    pred_pct = res["predicted_return"] * 100
    direction = res["direction"]
    color = "#00c853" if direction == "UP" else "#d50000"
    arrow = "▲" if direction == "UP" else "▼"
    container.markdown(
        f"""
        <div style="
            background:{color}18;
            border:2px solid {color};
            border-radius:12px;
            padding:20px 28px;
            margin-bottom:8px;
        ">
            <div style="font-size:12px;color:#888;margin-bottom:4px;">{label}</div>
            <div style="display:flex;align-items:center;gap:16px;">
                <span style="font-size:40px;">{arrow}</span>
                <div>
                    <div style="font-size:30px;font-weight:700;color:{color};">{direction}</div>
                    <div style="font-size:17px;color:{color};">{pred_pct:+.2f}% over {horizon}d</div>
                    <div style="font-size:12px;color:#888;margin-top:2px;">As of {res['as_of_date']}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_pred_card(lgbm_col, result, "LightGBM")
if tft_result:
    _pred_card(tft_col, tft_result, f"TFT  (val_loss={tft_result.get('val_loss', 0):.4f})")
    if "p10" in tft_result:
        tft_col.caption(
            f"90% CI: {tft_result['p10']*100:+.2f}% — {tft_result['p90']*100:+.2f}%"
        )
else:
    st.caption(
        "TFT model not available. Train on DGX: "
        f"`python /workspace/market_mvp/train_dgx.py --symbol {symbol} --horizon {horizon}`"
    )

st.divider()

# ── Feature importance ────────────────────────────────────────────────────────

metrics = d.load_metrics(symbol, horizon)
if metrics and metrics.get("feature_importances"):
    st.subheader("What's driving this prediction")

    imps = metrics["feature_importances"]
    imp_df = pd.DataFrame(
        sorted(imps.items(), key=lambda x: x[1], reverse=True)[:15],
        columns=["Feature", "Importance"],
    )

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation="h",
        marker_color="#1976d2",
        marker_line_width=0,
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis_title="Importance score",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    st.plotly_chart(fig, width="stretch")

st.divider()

# ── Recent feature values ─────────────────────────────────────────────────────

st.subheader("Latest feature snapshot")

latest = d.get_latest_features(symbol, horizon)
if not latest.empty:
    display_cols = [
        "date", "rsi_14", "ret_5d", "ret_20d", "pcr", "voi",
        "news_sent", "fed_net_score", "stocktwits_bull_ratio",
        "vol_ratio_20d", "price_ma20_ratio",
    ]
    show = [c for c in display_cols if c in latest.columns]
    row = latest[show].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    def _fmt(val, pct=False):
        if pd.isna(val):
            return "—"
        return f"{val:.1%}" if pct else f"{val:.3f}"

    c1.metric("RSI (14d)", _fmt(row.get("rsi_14")))
    c2.metric("5d return", _fmt(row.get("ret_5d"), pct=True))
    c3.metric("20d return", _fmt(row.get("ret_20d"), pct=True))
    c4.metric("Put/Call ratio", _fmt(row.get("pcr")))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("News sentiment", _fmt(row.get("news_sent")))
    c2.metric("Fed net score", _fmt(row.get("fed_net_score")))
    c3.metric("StockTwits bull%", _fmt(row.get("stocktwits_bull_ratio"), pct=True))
    c4.metric("Vol ratio 20d", _fmt(row.get("vol_ratio_20d")))
