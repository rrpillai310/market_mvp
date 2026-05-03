"""Shared cached data-access layer for the Streamlit UI.

All functions open DuckDB read-only and cache results for 5 minutes.
Returns empty DataFrames / None on missing data — UI pages must handle those.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

_DB_PATH = Path(__file__).parent.parent / "data" / "market_mvp.duckdb"
_MODELS_DIR = Path(__file__).parent.parent / "models"

SYMBOLS = ["SPY", "QQQ"]
HORIZONS = [5, 20]


@st.cache_resource
def _get_con() -> duckdb.DuckDBPyConnection:
    if not _DB_PATH.exists():
        return None
    return duckdb.connect(str(_DB_PATH), read_only=True)


def _con():
    c = _get_con()
    if c is None:
        st.error(f"Database not found at `{_DB_PATH}`. Run the pipeline first.")
        st.stop()
    return c


# ── Status helpers ────────────────────────────────────────────────────────────

def db_exists() -> bool:
    return _DB_PATH.exists()


def model_exists(symbol: str, horizon: int) -> bool:
    return (_MODELS_DIR / f"{symbol}_h{horizon}.pkl").exists()


@st.cache_data(ttl=300)
def get_last_ingest_date(symbol: str) -> str | None:
    try:
        row = _con().execute(
            "SELECT MAX(date) FROM prices_daily WHERE symbol=?", (symbol,)
        ).fetchone()
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_row_counts() -> dict:
    con = _con()
    tables = ["prices_daily", "features_daily", "edgar_earnings",
              "fed_minutes", "social_sentiment_daily", "news_sentiment"]
    counts = {}
    for t in tables:
        try:
            counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            counts[t] = 0
    return counts


# ── Predictions ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_latest_features(symbol: str, horizon: int) -> pd.DataFrame:
    return _con().execute(
        "SELECT * FROM features_daily WHERE symbol=? AND horizon=? ORDER BY date DESC LIMIT 1",
        (symbol, horizon),
    ).df()


@st.cache_data(ttl=300)
def get_features_history(symbol: str, horizon: int, days: int = 120) -> pd.DataFrame:
    df = _con().execute(
        """
        SELECT * FROM features_daily
        WHERE symbol=? AND horizon=?
        ORDER BY date DESC LIMIT ?
        """,
        (symbol, horizon, days),
    ).df()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


def load_model(symbol: str, horizon: int):
    path = _MODELS_DIR / f"{symbol}_h{horizon}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metrics(symbol: str, horizon: int) -> dict | None:
    path = _MODELS_DIR / f"{symbol}_h{horizon}_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def predict(symbol: str, horizon: int) -> dict | None:
    bundle = load_model(symbol, horizon)
    if bundle is None:
        return None
    latest = get_latest_features(symbol, horizon)
    if latest.empty:
        return None

    feature_cols = bundle["feature_cols"]
    for c in feature_cols:
        if c not in latest.columns:
            latest[c] = 0.0

    X = latest[feature_cols].ffill().fillna(0.0).to_numpy()
    pred = float(bundle["model"].predict(X)[0])
    return {
        "symbol": symbol,
        "horizon": horizon,
        "as_of_date": str(latest.iloc[0]["date"]),
        "predicted_return": pred,
        "direction": "UP" if pred > 0 else "DOWN",
    }


# ── Price & OHLCV ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_prices(symbol: str, days: int = 120) -> pd.DataFrame:
    df = _con().execute(
        """
        SELECT date, open, high, low, close, adjusted_close, volume
        FROM prices_daily WHERE symbol=? ORDER BY date DESC LIMIT ?
        """,
        (symbol, days),
    ).df()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


# ── Options ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_options(symbol: str, days: int = 120) -> pd.DataFrame:
    pcr = _con().execute(
        "SELECT date, put_call_ratio FROM options_pcr_daily WHERE symbol=? ORDER BY date DESC LIMIT ?",
        (symbol, days),
    ).df()
    voi = _con().execute(
        "SELECT date, volume_oi_ratio FROM options_voi_daily WHERE symbol=? ORDER BY date DESC LIMIT ?",
        (symbol, days),
    ).df()
    if pcr.empty and voi.empty:
        return pd.DataFrame()
    dfs = []
    for df in [pcr, voi]:
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            dfs.append(df.sort_values("date").set_index("date"))
    return pd.concat(dfs, axis=1).reset_index() if dfs else pd.DataFrame()


# ── News ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_news_sentiment(symbol: str, days: int = 60) -> pd.DataFrame:
    df = _con().execute(
        """
        SELECT CAST(time_published AS DATE) AS date,
               AVG(overall_sentiment_score) AS avg_sentiment,
               COUNT(*) AS article_count
        FROM news_sentiment WHERE symbol=?
        GROUP BY 1 ORDER BY 1 DESC LIMIT ?
        """,
        (symbol, days),
    ).df()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


# ── Fed ───────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_fed_history() -> pd.DataFrame:
    df = _con().execute(
        "SELECT meeting_date, document_type, net_score, hawkish_score, dovish_score FROM fed_minutes ORDER BY meeting_date"
    ).df()
    if not df.empty:
        df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    return df


# ── Social ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_social(symbol: str, days: int = 60) -> pd.DataFrame:
    df = _con().execute(
        """
        SELECT date, source, bull_ratio, sentiment_score, message_count
        FROM social_sentiment_daily WHERE symbol=?
        ORDER BY date DESC LIMIT ?
        """,
        (symbol, days),
    ).df()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


# ── Earnings ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_earnings(symbols: list[str]) -> pd.DataFrame:
    placeholders = ",".join("?" * len(symbols))
    df = _con().execute(
        f"""
        SELECT symbol, period_end, eps_actual, rev_actual, eps_surprise_pct, rev_surprise_pct
        FROM edgar_earnings WHERE symbol IN ({placeholders})
        ORDER BY period_end DESC LIMIT 100
        """,
        symbols,
    ).df()
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"])
    return df
