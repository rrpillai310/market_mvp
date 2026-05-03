from __future__ import annotations

import argparse
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from market_mvp.db import DB, init_db


FEATURE_COLS = [
    "pcr", "pcr_chg_5",
    "voi", "voi_chg_5",
    "news_sent", "news_sent_chg_5", "news_count",
    "div_ex_days", "div_amount",
]


def _time_split(df: pd.DataFrame, *, train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] <= train_end].copy()
    test = df[df["date"] > train_end].copy()
    return train, test


def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _dir_acc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true > 0) == (y_pred > 0)))


def train_and_eval(con: duckdb.DuckDBPyConnection, symbol: str, horizon: int) -> dict:
    df = con.execute(
        """
        SELECT *
        FROM features_daily
        WHERE symbol=? AND horizon=?
        ORDER BY date
        """,
        (symbol, int(horizon)),
    ).df()
    if df.empty:
        return {"error": "no features; run ingest + features first"}

    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    # Basic missing handling
    df = df.dropna(subset=["y_fwd_return"])
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(method="ffill").fillna(0.0)

    # Simple last-20% as test by time
    split_idx = int(len(df) * 0.8)
    train_end = df.iloc[split_idx]["date"]
    train, test = _time_split(df, train_end=train_end)
    if len(test) < 50:
        # If not enough, just do last 30 rows as test
        train = df.iloc[:-30].copy()
        test = df.iloc[-30:].copy()

    X_train = train[FEATURE_COLS].to_numpy()
    y_train = train["y_fwd_return"].to_numpy()
    X_test = test[FEATURE_COLS].to_numpy()
    y_test = test["y_fwd_return"].to_numpy()

    # LightGBM if available, else fallback to sklearn HistGradientBoostingRegressor
    model_name = "lightgbm"
    try:
        import lightgbm as lgb

        m = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
    except Exception:
        model_name = "sklearn_hgbr"
        from sklearn.ensemble import HistGradientBoostingRegressor

        m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

    return {
        "symbol": symbol,
        "horizon": int(horizon),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": train_end,
        "rmse": _rmse(y_test, y_pred),
        "dir_acc": _dir_acc(y_test, y_pred),
        "test_mean_return": float(np.mean(y_test)),
        "pred_mean_return": float(np.mean(y_pred)),
        "model": model_name,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/market_mvp.duckdb")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--horizon", type=int, default=5)
    ns = ap.parse_args()
    db = DB(path=__import__("pathlib").Path(ns.db))
    con = db.connect()
    init_db(con)
    res = train_and_eval(con, ns.symbol, ns.horizon)
    print(res)
    con.close()

