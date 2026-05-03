from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from market_mvp.alpha_vantage import AlphaVantageClient
from market_mvp.db import DB, init_db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_get(con, *, provider: str, endpoint: str, symbol: str | None, params: dict[str, Any]) -> dict | None:
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
    row = con.execute(
        "SELECT payload_json FROM api_cache WHERE provider=? AND endpoint=? AND symbol IS NOT DISTINCT FROM ? AND params_json=?",
        (provider, endpoint, symbol, params_json),
    ).fetchone()
    if not row:
        return None
    return json.loads(row[0])


def _cache_put(con, *, provider: str, endpoint: str, symbol: str | None, params: dict[str, Any], payload: dict) -> None:
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
    payload_json = json.dumps(payload, separators=(",", ":"))
    con.execute(
        "INSERT OR REPLACE INTO api_cache(provider, endpoint, symbol, params_json, fetched_at, payload_json) VALUES (?,?,?,?,?,?)",
        (provider, endpoint, symbol, params_json, _now_iso(), payload_json),
    )


def ingest_daily_adjusted(con, av: AlphaVantageClient, symbol: str) -> None:
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    raw = ticker.history(period="max", auto_adjust=False, actions=True)
    if raw.empty:
        print(f"[ingest] {symbol}: yfinance returned no data")
        return

    raw = raw.reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    df = pd.DataFrame({
        "symbol": symbol,
        "date": pd.to_datetime(raw["date"]).dt.date.astype(str),
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
        "adjusted_close": raw["adj_close"].astype(float),
        "volume": raw["volume"].astype(float),
        "dividend_amount": raw.get("dividends", pd.Series(0.0, index=raw.index)).fillna(0.0).astype(float),
        "split_coefficient": raw.get("stock_splits", pd.Series(1.0, index=raw.index)).replace(0, 1.0).astype(float),
    })

    con.register("tmp_prices", df)
    con.execute(
        """
        INSERT OR REPLACE INTO prices_daily
        SELECT symbol, CAST(date AS DATE), open, high, low, close, adjusted_close, volume, dividend_amount, split_coefficient
        FROM tmp_prices
        """
    )
    con.unregister("tmp_prices")


def ingest_options_pcr_yf(con, symbol: str) -> None:
    """Compute today's put/call ratio from yfinance options chains and store it."""
    import yfinance as yf
    from datetime import date as date_type
    ticker = yf.Ticker(symbol)
    expirations = ticker.options
    if not expirations:
        return
    put_vol = call_vol = 0
    # Sum volume across nearest 6 expirations to get a representative PCR
    for exp in expirations[:6]:
        try:
            chain = ticker.option_chain(exp)
            put_vol += chain.puts["volume"].fillna(0).sum()
            call_vol += chain.calls["volume"].fillna(0).sum()
        except Exception:
            continue
    if call_vol == 0:
        return
    pcr = float(put_vol) / float(call_vol)
    today = str(date_type.today())
    con.execute(
        "INSERT OR REPLACE INTO options_pcr_daily(symbol, date, put_call_ratio) VALUES (?, ?, ?)",
        (symbol, today, pcr),
    )


def ingest_options_pcr(con, av: AlphaVantageClient, symbol: str) -> None:
    # Historical put/call ratio endpoint (Options Data APIs).
    endpoint = "HISTORICAL_PUT_CALL_RATIO"
    params = {"symbol": symbol}
    cached = _cache_get(con, provider="alphavantage", endpoint=endpoint, symbol=symbol, params=params)
    if cached is None:
        data = av.get(function=endpoint, params=params)
        _cache_put(con, provider="alphavantage", endpoint=endpoint, symbol=symbol, params=params, payload=data)
    else:
        data = cached

    series = data.get("data") or data.get("historical_put_call_ratio") or data.get("put_call_ratio")
    if not isinstance(series, list):
        return
    rows = []
    for r in series:
        if not isinstance(r, dict):
            continue
        date = r.get("date") or r.get("timestamp") or r.get("time")
        val = r.get("put_call_ratio") or r.get("value")
        if not date or val is None:
            continue
        try:
            rows.append({"symbol": symbol, "date": str(date)[:10], "put_call_ratio": float(val)})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return
    con.register("tmp_pcr", df)
    con.execute(
        "INSERT OR REPLACE INTO options_pcr_daily SELECT symbol, CAST(date AS DATE), put_call_ratio FROM tmp_pcr"
    )
    con.unregister("tmp_pcr")


def ingest_options_voi(con, av: AlphaVantageClient, symbol: str) -> None:
    # Historical volume-to-open-interest ratio endpoint (Options Data APIs).
    endpoint = "HISTORICAL_VOLUME_OI_RATIO"
    params = {"symbol": symbol}
    cached = _cache_get(con, provider="alphavantage", endpoint=endpoint, symbol=symbol, params=params)
    if cached is None:
        data = av.get(function=endpoint, params=params)
        _cache_put(con, provider="alphavantage", endpoint=endpoint, symbol=symbol, params=params, payload=data)
    else:
        data = cached

    series = data.get("data") or data.get("historical_volume_oi_ratio") or data.get("volume_oi_ratio")
    if not isinstance(series, list):
        return
    rows = []
    for r in series:
        if not isinstance(r, dict):
            continue
        date = r.get("date") or r.get("timestamp") or r.get("time")
        val = r.get("volume_oi_ratio") or r.get("value")
        if not date or val is None:
            continue
        try:
            rows.append({"symbol": symbol, "date": str(date)[:10], "volume_oi_ratio": float(val)})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return
    con.register("tmp_voi", df)
    con.execute(
        "INSERT OR REPLACE INTO options_voi_daily SELECT symbol, CAST(date AS DATE), volume_oi_ratio FROM tmp_voi"
    )
    con.unregister("tmp_voi")


def ingest_news_sentiment(con, av: AlphaVantageClient, symbol: str, *, limit: int = 200) -> None:
    # Alpha Intelligence: NEWS_SENTIMENT
    endpoint = "NEWS_SENTIMENT"
    # AlphaVantage uses tickers=... for equities/ETFs. We'll keep it simple.
    params = {"tickers": symbol, "sort": "LATEST", "limit": int(limit)}
    cached = _cache_get(con, provider="alphavantage", endpoint=endpoint, symbol=symbol, params=params)
    if cached is None:
        data = av.get(function=endpoint, params=params)
        _cache_put(con, provider="alphavantage", endpoint=endpoint, symbol=symbol, params=params, payload=data)
    else:
        data = cached

    feed = data.get("feed")
    if not isinstance(feed, list):
        return
    rows = []
    for item in feed:
        if not isinstance(item, dict):
            continue
        tp = item.get("time_published")
        url = item.get("url") or ""
        if not tp or not url:
            continue
        # Parse AV time: YYYYMMDDThhmmss
        try:
            ts = datetime.strptime(tp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        ticker_sent = None
        rel = None
        for t in (item.get("ticker_sentiment") or []):
            if (t or {}).get("ticker") == symbol:
                ticker_sent = t.get("ticker_sentiment_score")
                rel = t.get("relevance_score")
                break
        try:
            rows.append({
                "symbol": symbol,
                "time_published": ts.isoformat(),
                "overall_sentiment_score": float(item.get("overall_sentiment_score")) if item.get("overall_sentiment_score") is not None else None,
                "overall_sentiment_label": item.get("overall_sentiment_label"),
                "relevance_score": float(rel) if rel is not None else None,
                "source": item.get("source"),
                "title": item.get("title"),
                "url": url,
                "raw_json": json.dumps(item, separators=(",", ":")),
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return
    con.register("tmp_news", df)
    con.execute(
        """
        INSERT OR REPLACE INTO news_sentiment
        SELECT symbol,
               CAST(time_published AS TIMESTAMP),
               overall_sentiment_score,
               overall_sentiment_label,
               relevance_score,
               source,
               title,
               url,
               raw_json
        FROM tmp_news
        """
    )
    con.unregister("tmp_news")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ingest market data into DuckDB via Alpha Vantage.")
    ap.add_argument("--db", default="data/market_mvp.duckdb")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--skip-news", action="store_true")
    ap.add_argument("--skip-options", action="store_true")
    ap.add_argument("--news-limit", type=int, default=200)
    ap.add_argument("--throttle-secs", type=float, default=12.5)
    return ap


def main():
    ns = _build_parser().parse_args()
    import pathlib
    db = DB(path=pathlib.Path(ns.db))
    con = db.connect()
    init_db(con)
    av = AlphaVantageClient(throttle_secs=float(ns.throttle_secs))

    for sym in ns.symbols:
        print(f"[ingest] {sym}: prices...")
        ingest_daily_adjusted(con, av, sym)
        if not ns.skip_options:
            print(f"[ingest] {sym}: options PCR...")
            ingest_options_pcr(con, av, sym)
            print(f"[ingest] {sym}: options VOI...")
            ingest_options_voi(con, av, sym)
        if not ns.skip_news:
            print(f"[ingest] {sym}: news sentiment...")
            ingest_news_sentiment(con, av, sym, limit=ns.news_limit)
        con.execute("CHECKPOINT")
        print(f"[ingest] {sym}: done.")

    con.close()


if __name__ == "__main__":
    main()
