from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_mvp import ui_data as d

st.set_page_config(page_title="Signals · Market MVP", page_icon="📊", layout="wide")

st.title("📊 Signals Explorer")

# ── Controls ──────────────────────────────────────────────────────────────────

col_sym, col_days, _ = st.columns([1, 1, 3])
symbol = col_sym.selectbox("Symbol", d.SYMBOLS)
days = col_days.selectbox("Lookback", [30, 60, 90, 180], index=1, format_func=lambda x: f"{x} days")

st.divider()

# ── Price + Volume ────────────────────────────────────────────────────────────

prices = d.get_prices(symbol, days=days)

if prices.empty:
    st.warning("No price data. Run `python3 -m market_mvp.ingest` first.")
else:
    st.subheader(f"{symbol} — Price & Volume")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Candlestick(
            x=prices["date"],
            open=prices["open"], high=prices["high"],
            low=prices["low"], close=prices["adjusted_close"],
            name="Price",
            increasing_line_color="#00c853",
            decreasing_line_color="#d50000",
        ),
        row=1, col=1,
    )
    colors = ["#00c853" if c >= o else "#d50000"
              for c, o in zip(prices["adjusted_close"], prices["open"])]
    fig.add_trace(
        go.Bar(x=prices["date"], y=prices["volume"], name="Volume",
               marker_color=colors, showlegend=False),
        row=2, col=1,
    )
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=8, b=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.05),
        font=dict(size=12),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, width="stretch")

st.divider()

# ── RSI ───────────────────────────────────────────────────────────────────────

feats = d.get_features_history(symbol, horizon=5, days=days)

if not feats.empty and "rsi_14" in feats.columns:
    st.subheader("RSI (14d)")
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="#d50000", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="#00c853", opacity=0.08, line_width=0)
    fig.add_hline(y=70, line_dash="dot", line_color="#d50000", line_width=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00c853", line_width=1)
    fig.add_trace(go.Scatter(
        x=feats["date"], y=feats["rsi_14"],
        mode="lines", name="RSI", line=dict(color="#1976d2", width=2),
    ))
    fig.update_layout(
        height=240, margin=dict(l=0, r=0, t=8, b=0),
        yaxis=dict(range=[0, 100], title="RSI"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12), showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")
    st.divider()

# ── Momentum ──────────────────────────────────────────────────────────────────

if not feats.empty:
    mom_cols = [c for c in ["ret_1d", "ret_5d", "ret_20d", "ret_60d"] if c in feats.columns]
    if mom_cols:
        st.subheader("Price Momentum")
        colors_mom = ["#42a5f5", "#1976d2", "#0d47a1", "#82b1ff"]
        fig = go.Figure()
        for col, clr in zip(mom_cols, colors_mom):
            fig.add_trace(go.Scatter(
                x=feats["date"], y=feats[col] * 100,
                mode="lines", name=col.replace("ret_", "").replace("d", "d return"),
                line=dict(color=clr, width=1.5),
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1)
        fig.update_layout(
            height=260, margin=dict(l=0, r=0, t=8, b=0),
            yaxis_title="Return (%)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1), font=dict(size=12),
        )
        st.plotly_chart(fig, width="stretch")
        st.divider()

# ── Options ───────────────────────────────────────────────────────────────────

options = d.get_options(symbol, days=days)

if not options.empty:
    st.subheader("Options Signals")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Put/Call Ratio", "Volume / OI Ratio"])

    if "put_call_ratio" in options.columns:
        fig.add_trace(go.Scatter(
            x=options["date"], y=options["put_call_ratio"],
            mode="lines", name="PCR", line=dict(color="#ab47bc", width=2),
        ), row=1, col=1)
        fig.add_hline(y=1.0, line_dash="dot", line_color="#888", line_width=1, row=1, col=1)

    if "volume_oi_ratio" in options.columns:
        fig.add_trace(go.Scatter(
            x=options["date"], y=options["volume_oi_ratio"],
            mode="lines", name="VOI", line=dict(color="#ff7043", width=2),
        ), row=1, col=2)

    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=32, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(size=12),
    )
    st.plotly_chart(fig, width="stretch")
    st.divider()

# ── News Sentiment ────────────────────────────────────────────────────────────

news = d.get_news_sentiment(symbol, days=days)

if not news.empty:
    st.subheader("News Sentiment")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=news["date"], y=news["article_count"],
        name="Articles", marker_color="#b0bec5", opacity=0.5,
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=news["date"], y=news["avg_sentiment"],
        mode="lines+markers", name="Sentiment",
        line=dict(color="#26a69a", width=2),
        marker=dict(size=5),
    ), secondary_y=False)
    fig.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1)
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1), font=dict(size=12),
    )
    fig.update_yaxes(title_text="Sentiment score", secondary_y=False)
    fig.update_yaxes(title_text="Article count", secondary_y=True)
    st.plotly_chart(fig, width="stretch")
    st.divider()

# ── Fed Minutes ───────────────────────────────────────────────────────────────

fed = d.get_fed_history()

if not fed.empty:
    st.subheader("Fed FOMC — Hawkish / Dovish Score")
    statements = fed[fed["document_type"] == "statement"]
    minutes = fed[fed["document_type"] == "minutes"]

    fig = go.Figure()
    for df_src, name, clr in [(statements, "Statement", "#ef5350"), (minutes, "Minutes", "#ef9a9a")]:
        if not df_src.empty:
            fig.add_trace(go.Bar(
                x=df_src["meeting_date"], y=df_src["net_score"],
                name=name,
                marker_color=[clr if v > 0 else "#42a5f5" for v in df_src["net_score"]],
            ))
    fig.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1)
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=8, b=0),
        yaxis_title="Net hawkish score (per 1k words)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1), font=dict(size=12),
        barmode="overlay",
    )
    st.plotly_chart(fig, width="stretch")
    st.divider()

# ── Social Sentiment ──────────────────────────────────────────────────────────

social = d.get_social(symbol, days=days)

if not social.empty:
    st.subheader("Social Sentiment")
    st_df = social[social["source"] == "stocktwits"]
    rd_df = social[social["source"] == "reddit"]

    fig = go.Figure()
    if not st_df.empty and st_df["bull_ratio"].notna().any():
        fig.add_trace(go.Scatter(
            x=st_df["date"], y=st_df["bull_ratio"] * 100,
            mode="lines+markers", name="StockTwits bull%",
            line=dict(color="#66bb6a", width=2), marker=dict(size=5),
        ))
        fig.add_hline(y=50, line_dash="dot", line_color="#888", line_width=1)
    if not rd_df.empty:
        fig.add_trace(go.Scatter(
            x=rd_df["date"], y=rd_df["sentiment_score"],
            mode="lines+markers", name="Reddit sentiment",
            line=dict(color="#ffa726", width=2), marker=dict(size=5),
        ))
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1), font=dict(size=12),
    )
    st.plotly_chart(fig, width="stretch")

if feats.empty and options.empty and news.empty and fed.empty and social.empty:
    st.info("No signal data yet. Run the full pipeline to populate the database.")
