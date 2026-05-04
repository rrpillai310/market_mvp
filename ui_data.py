"""Shared cached data-access layer for the Streamlit UI.

All functions open DuckDB read-only and cache results for 5 minutes.
Returns empty DataFrames / None on missing data — UI pages must handle those.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from market_mvp.storage import get_data_dir, get_models_dir

_DB_PATH = get_data_dir() / "market_mvp.duckdb"
_MODELS_DIR = get_models_dir()
_CONFIG_PATH = Path(__file__).parent / "config" / "symbols.json"
_PIPELINE_STATE = Path(__file__).parent / "config" / "pipeline_state.json"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {"symbols": [{"ticker": t, "type": "etf"} for t in ["SPY", "QQQ", "VXUS", "XSD", "XLK"]]}


def get_symbols() -> list[str]:
    return [s["ticker"] for s in _load_config().get("symbols", [])]


def get_tracked_symbols() -> list[dict]:
    return _load_config().get("symbols", [])


def add_symbol(ticker: str, symbol_type: str = "stock") -> None:
    cfg = _load_config()
    if ticker.upper() not in [s["ticker"] for s in cfg["symbols"]]:
        cfg["symbols"].append({"ticker": ticker.upper(), "type": symbol_type})
        with open(_CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)


def remove_symbol(ticker: str) -> None:
    cfg = _load_config()
    cfg["symbols"] = [s for s in cfg["symbols"] if s["ticker"] != ticker.upper()]
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def symbol_has_data(symbol: str) -> bool:
    try:
        c = _get_con()
        if c is None:
            return False
        row = c.execute("SELECT COUNT(*) FROM prices_daily WHERE symbol=?", (symbol,)).fetchone()
        return bool(row and row[0] > 0)
    except Exception:
        return False


def is_pipeline_running() -> bool:
    if not _PIPELINE_STATE.exists():
        return False
    try:
        with open(_PIPELINE_STATE) as f:
            state = json.load(f)
        pid = state.get("pid")
        if pid:
            os.kill(int(pid), 0)
            return True
    except (OSError, ProcessLookupError, ValueError):
        _PIPELINE_STATE.unlink(missing_ok=True)
    return False


def trigger_pipeline(ticker: str) -> int:
    cmd = [
        sys.executable, "-m", "market_mvp.pipeline",
        "--symbols", ticker.upper(),
        "--skip-social",
    ]
    env = {**os.environ, "PYTHONPATH": str(_DB_PATH.parent.parent)}
    proc = subprocess.Popen(cmd, cwd=str(_DB_PATH.parent.parent), env=env)
    state = {"pid": proc.pid, "symbol": ticker.upper()}
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_PIPELINE_STATE, "w") as f:
        json.dump(state, f)
    return proc.pid


SYMBOLS = get_symbols()
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


def load_tft_metrics(symbol: str, horizon: int) -> dict | None:
    path = _MODELS_DIR / f"{symbol}_h{horizon}_tft_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def predict_tft(symbol: str, horizon: int) -> dict | None:
    """Load the latest TFT prediction JSON written by train_dgx.py."""
    path = _MODELS_DIR / f"{symbol}_h{horizon}_tft_pred.json"
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
