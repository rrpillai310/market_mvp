from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

from market_mvp.db import DB, init_db


def _roll_chg(s: pd.Series, n: int) -> pd.Series:
    return s - s.shift(n)


def _rsi(returns: pd.Series, period: int = 14) -> pd.Series:
    gain = returns.clip(lower=0)
    loss = (-returns).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # All-gain periods: avg_loss=0 → RSI=100 (not NaN)
    rsi = pd.Series(
        np.where(
            avg_gain.isna(),
            np.nan,
            np.where(avg_loss == 0, 100.0, 100 - 100 / (1 + avg_gain / avg_loss)),
        ),
        index=returns.index,
    )
    return rsi


def build_features(con: duckdb.DuckDBPyConnection, symbol: str, *, horizons: list[int]) -> None:
    prices = con.execute(
        "SELECT date, open, high, low, close, adjusted_close, volume, dividend_amount FROM prices_daily WHERE symbol=? ORDER BY date",
        (symbol,),
    ).df()
    if prices.empty:
        return
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.set_index("date").sort_index()

    pcr = con.execute(
        "SELECT date, put_call_ratio FROM options_pcr_daily WHERE symbol=? ORDER BY date",
        (symbol,),
    ).df()
    pcr["date"] = pd.to_datetime(pcr["date"]).dt.date
    pcr = pcr.set_index("date").sort_index()

    voi = con.execute(
        "SELECT date, volume_oi_ratio FROM options_voi_daily WHERE symbol=? ORDER BY date",
        (symbol,),
    ).df()
    voi["date"] = pd.to_datetime(voi["date"]).dt.date
    voi = voi.set_index("date").sort_index()

    news = con.execute(
        """
        SELECT CAST(time_published AS DATE) AS date,
               AVG(overall_sentiment_score) AS news_sent,
               COUNT(*) AS news_count
        FROM news_sentiment
        WHERE symbol=?
        GROUP BY 1
        ORDER BY 1
        """,
        (symbol,),
    ).df()
    if not news.empty:
        news["date"] = pd.to_datetime(news["date"]).dt.date
        news = news.set_index("date").sort_index()
    else:
        news = pd.DataFrame(columns=["news_sent", "news_count"])

    df = prices.join(pcr, how="left").join(voi, how="left").join(news, how="left")

    # --- Dividend features ---
    div = df["dividend_amount"].fillna(0.0)
    div_dates = pd.Series(pd.to_datetime(df.index), index=df.index)
    div_dates = div_dates.where(div > 0)
    last_div_date = div_dates.ffill()
    current_dates = pd.to_datetime(pd.Series(df.index, index=df.index))
    df["div_ex_days"] = (current_dates - last_div_date).dt.days.fillna(-1).astype(int)
    df["div_amount"] = div

    # --- Options/news change features ---
    df["pcr"] = df["put_call_ratio"]
    df["voi"] = df["volume_oi_ratio"]
    df["pcr_chg_5"] = _roll_chg(df["pcr"], 5)
    df["voi_chg_5"] = _roll_chg(df["voi"], 5)
    df["news_sent_chg_5"] = _roll_chg(df["news_sent"], 5)

    # --- Price momentum ---
    ac = df["adjusted_close"]
    returns = ac.pct_change()
    df["ret_1d"] = returns
    df["ret_5d"] = ac / ac.shift(5) - 1
    df["ret_20d"] = ac / ac.shift(20) - 1
    df["ret_60d"] = ac / ac.shift(60) - 1

    # --- Volatility ---
    df["vol_10d"] = returns.rolling(10).std()
    df["vol_20d"] = returns.rolling(20).std()

    # --- MA ratios (price position relative to trend) ---
    df["price_ma5_ratio"] = ac / ac.rolling(5).mean() - 1
    df["price_ma20_ratio"] = ac / ac.rolling(20).mean() - 1
    df["price_ma60_ratio"] = ac / ac.rolling(60).mean() - 1

    # --- RSI ---
    df["rsi_14"] = _rsi(returns)

    # --- Volume ratio ---
    vol = df["volume"].astype(float)
    df["vol_ratio_20d"] = vol / vol.rolling(20).mean()

    # --- Intraday range ---
    df["intraday_range"] = (df["high"] - df["low"]) / ac.replace(0, np.nan)

    # --- Labels: forward returns on adjusted close ---
    for h in horizons:
        df[f"y_fwd_return_{h}"] = ac.shift(-h) / ac - 1.0

    feature_cols = [
        "pcr", "pcr_chg_5", "voi", "voi_chg_5",
        "news_sent", "news_sent_chg_5", "news_count",
        "div_ex_days", "div_amount",
        "ret_1d", "ret_5d", "ret_20d", "ret_60d",
        "vol_10d", "vol_20d",
        "price_ma5_ratio", "price_ma20_ratio", "price_ma60_ratio",
        "rsi_14", "vol_ratio_20d", "intraday_range",
    ]

    out_rows = []
    for h in horizons:
        tmp = df[feature_cols + [f"y_fwd_return_{h}"]].copy()
        tmp = tmp.rename(columns={f"y_fwd_return_{h}": "y_fwd_return"})
        tmp["symbol"] = symbol
        tmp["date"] = tmp.index.astype(str)
        tmp["horizon"] = int(h)
        tmp = tmp.dropna(subset=["y_fwd_return"])
        out_rows.append(tmp.reset_index(drop=True))

    if not out_rows:
        return
    out = pd.concat(out_rows, ignore_index=True)
    con.register("tmp_feat", out)
    con.execute(
        """
        INSERT OR REPLACE INTO features_daily
        (symbol, date, horizon, y_fwd_return,
         pcr, pcr_chg_5, voi, voi_chg_5,
         news_sent, news_sent_chg_5, news_count,
         div_ex_days, div_amount,
         ret_1d, ret_5d, ret_20d, ret_60d,
         vol_10d, vol_20d,
         price_ma5_ratio, price_ma20_ratio, price_ma60_ratio,
         rsi_14, vol_ratio_20d, intraday_range)
        SELECT symbol,
               CAST(date AS DATE) AS date,
               horizon,
               y_fwd_return,
               pcr, pcr_chg_5,
               voi, voi_chg_5,
               news_sent, news_sent_chg_5,
               CAST(news_count AS INTEGER),
               CAST(div_ex_days AS INTEGER),
               div_amount,
               ret_1d, ret_5d, ret_20d, ret_60d,
               vol_10d, vol_20d,
               price_ma5_ratio, price_ma20_ratio, price_ma60_ratio,
               rsi_14, vol_ratio_20d, intraday_range
        FROM tmp_feat
        """
    )
    con.unregister("tmp_feat")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/market_mvp.duckdb")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[5, 20])
    ns = ap.parse_args()
    db = DB(path=__import__("pathlib").Path(ns.db))
    con = db.connect()
    init_db(con)
    for sym in ns.symbols:
        print(f"[features] {sym}...")
        build_features(con, sym, horizons=list(ns.horizons))
    con.execute("CHECKPOINT")
    con.close()
    print("[features] done.")
