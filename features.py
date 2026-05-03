from __future__ import annotations

import argparse
from datetime import timedelta

import duckdb
import pandas as pd

from market_mvp.db import DB, init_db


def _roll_chg(s: pd.Series, n: int) -> pd.Series:
    return s - s.shift(n)


def build_features(con: duckdb.DuckDBPyConnection, symbol: str, *, horizons: list[int]) -> None:
    prices = con.execute(
        "SELECT date, adjusted_close, dividend_amount FROM prices_daily WHERE symbol=? ORDER BY date",
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

    # Simple daily dividend event features (ETF friendly):
    # "days until next ex-div" is hard without a calendar; approximate with "days since last dividend".
    div = df["dividend_amount"].fillna(0.0)
    last_div_date = div.where(div > 0).index.to_series().ffill()
    df["div_ex_days"] = (pd.Series(df.index).astype("datetime64[ns]") - pd.Series(last_div_date).astype("datetime64[ns]")).dt.days.values
    df["div_amount"] = div

    # Options/news changes
    df["pcr"] = df["put_call_ratio"]
    df["voi"] = df["volume_oi_ratio"]
    df["pcr_chg_5"] = _roll_chg(df["pcr"], 5)
    df["voi_chg_5"] = _roll_chg(df["voi"], 5)
    df["news_sent_chg_5"] = _roll_chg(df["news_sent"], 5)

    # Labels: forward returns on adjusted close (div/split adjusted).
    for h in horizons:
        df[f"y_fwd_return_{h}"] = df["adjusted_close"].shift(-h) / df["adjusted_close"] - 1.0

    out_rows = []
    for h in horizons:
        tmp = df[[
            "pcr", "pcr_chg_5", "voi", "voi_chg_5",
            "news_sent", "news_sent_chg_5", "news_count",
            "div_ex_days", "div_amount",
            f"y_fwd_return_{h}",
        ]].copy()
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
        SELECT symbol,
               CAST(date AS DATE) AS date,
               horizon,
               y_fwd_return,
               pcr, pcr_chg_5,
               voi, voi_chg_5,
               news_sent, news_sent_chg_5,
               CAST(news_count AS INTEGER),
               CAST(div_ex_days AS INTEGER),
               div_amount
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
        build_features(con, sym, horizons=list(ns.horizons))
    con.execute("CHECKPOINT")
    con.close()

