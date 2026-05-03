import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_mvp import ui_data as d

st.set_page_config(page_title="Performance · Market MVP", page_icon="⚡", layout="wide")

st.title("⚡ Model Performance")

# ── Controls ──────────────────────────────────────────────────────────────────

col_sym, col_h, _ = st.columns([1, 1, 3])
symbol = col_sym.selectbox("Symbol", d.SYMBOLS)
horizon = col_h.selectbox("Horizon (days)", d.HORIZONS)

st.divider()

# ── Load metrics ──────────────────────────────────────────────────────────────

metrics = d.load_metrics(symbol, horizon)

if metrics is None:
    st.warning(
        f"No metrics for {symbol} h={horizon}. Run:\n\n"
        f"```bash\npython3 -m market_mvp.train --symbol {symbol} --horizon {horizon}\n```"
    )
    st.stop()

# ── Top-line metrics ──────────────────────────────────────────────────────────

st.subheader("Overall")

c1, c2, c3, c4 = st.columns(4)
mean_rmse = metrics.get("mean_rmse")
mean_dir_acc = metrics.get("mean_dir_acc")
train_rows = metrics.get("total_rows") or metrics.get("train_rows")
model_name = metrics.get("model", "—")
trained_at = metrics.get("trained_at", "—")[:10] if metrics.get("trained_at") else "—"

c1.metric("Directional accuracy", f"{mean_dir_acc:.1%}" if mean_dir_acc else "—",
          help="% of test periods where direction (up/down) was correct")
c2.metric("Mean RMSE", f"{mean_rmse:.5f}" if mean_rmse else "—",
          help="Root mean squared error of return predictions across walk-forward folds")
c3.metric("Training rows", f"{train_rows:,}" if train_rows else "—")
c4.metric("Trained", trained_at)

st.caption(f"Model: `{model_name}`")

st.divider()

# ── Walk-forward folds ────────────────────────────────────────────────────────

folds = metrics.get("walk_forward_folds")

if folds:
    st.subheader("Walk-forward folds")

    fold_df = pd.DataFrame(folds)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Directional Accuracy per Fold", "RMSE per Fold"],
    )
    colors = ["#00c853" if v >= 0.5 else "#d50000" for v in fold_df["dir_acc"]]
    fig.add_trace(go.Bar(
        x=fold_df["fold"].astype(str),
        y=fold_df["dir_acc"] * 100,
        marker_color=colors,
        name="Dir Acc %",
        text=[f"{v:.1%}" for v in fold_df["dir_acc"]],
        textposition="outside",
    ), row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#888", line_width=1, row=1, col=1)

    fig.add_trace(go.Bar(
        x=fold_df["fold"].astype(str),
        y=fold_df["rmse"],
        marker_color="#1976d2",
        name="RMSE",
        text=[f"{v:.4f}" for v in fold_df["rmse"]],
        textposition="outside",
    ), row=1, col=2)

    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=32, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(size=12),
    )
    fig.update_yaxes(title_text="Dir Acc (%)", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

    # Fold detail table
    with st.expander("Fold detail table"):
        show_df = fold_df[["fold", "train_rows", "test_rows", "train_end_date", "rmse", "dir_acc"]].copy()
        show_df["dir_acc"] = show_df["dir_acc"].map("{:.1%}".format)
        show_df["rmse"] = show_df["rmse"].map("{:.5f}".format)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.divider()

# ── Feature importance ────────────────────────────────────────────────────────

imps = metrics.get("feature_importances")
if imps:
    st.subheader("Feature importance")

    imp_df = pd.DataFrame(
        sorted(imps.items(), key=lambda x: x[1], reverse=True),
        columns=["Feature", "Importance"],
    )
    total = imp_df["Importance"].sum()
    imp_df["Share"] = imp_df["Importance"] / total if total > 0 else 0

    fig = go.Figure(go.Bar(
        x=imp_df["Share"][:20] * 100,
        y=imp_df["Feature"][:20],
        orientation="h",
        marker=dict(
            color=imp_df["Share"][:20] * 100,
            colorscale="Blues",
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in imp_df["Share"][:20] * 100],
        textposition="outside",
    ))
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=8, b=0),
        xaxis_title="Share of total importance (%)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

# ── Actual vs predicted scatter ───────────────────────────────────────────────

st.subheader("Actual vs predicted returns")

feats = d.get_features_history(symbol, horizon, days=365)
bundle = d.load_model(symbol, horizon)

if not feats.empty and bundle is not None:
    feature_cols = bundle["feature_cols"]
    for c in feature_cols:
        if c not in feats.columns:
            feats[c] = 0.0

    X = feats[feature_cols].ffill().fillna(0.0).to_numpy()
    preds = bundle["model"].predict(X)
    actuals = feats["y_fwd_return"].to_numpy()

    # Colour by correctness of direction
    correct = [(a > 0) == (p > 0) for a, p in zip(actuals, preds)]
    point_colors = ["#00c853" if c else "#d50000" for c in correct]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actuals * 100, y=preds * 100,
        mode="markers",
        marker=dict(color=point_colors, size=5, opacity=0.65),
        text=[str(d)[:10] for d in feats["date"]],
        hovertemplate="Date: %{text}<br>Actual: %{x:.2f}%<br>Predicted: %{y:.2f}%<extra></extra>",
        name="",
    ))
    # Perfect prediction line
    rng = max(abs(actuals * 100).max(), abs(preds * 100).max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[-rng, rng], y=[-rng, rng],
        mode="lines", line=dict(color="#888", dash="dot", width=1),
        showlegend=False,
    ))
    fig.update_layout(
        height=400, margin=dict(l=0, r=0, t=8, b=0),
        xaxis_title="Actual return (%)",
        yaxis_title="Predicted return (%)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12), showlegend=False,
    )
    pct_correct = sum(correct) / len(correct) * 100
    st.caption(f"Green = correct direction · Red = wrong direction · {pct_correct:.1f}% correct on this sample")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Run the pipeline to generate features and retrain to see this chart.")
