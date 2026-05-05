from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from market_mvp.ingest import (
    _cache_get,
    _cache_put,
    ingest_daily_adjusted,
    ingest_options_pcr,
    ingest_options_voi,
    ingest_news_sentiment,
)


def test_cache_put_and_get_roundtrip(con):
    payload = {"Time Series (Daily)": {"2024-01-01": {"1. open": "400"}}}
    params = {"symbol": "SPY", "outputsize": "full"}
    _cache_put(con, provider="alphavantage", endpoint="TIME_SERIES_DAILY_ADJUSTED",
               symbol="SPY", params=params, payload=payload)
    result = _cache_get(con, provider="alphavantage", endpoint="TIME_SERIES_DAILY_ADJUSTED",
                        symbol="SPY", params=params)
    assert result == payload


def test_cache_miss_returns_none(con):
    result = _cache_get(con, provider="missing", endpoint="NONE", symbol="X", params={})
    assert result is None


def test_cache_key_is_param_order_independent(con):
    payload = {"data": 1}
    _cache_put(con, provider="p", endpoint="e", symbol="S",
               params={"b": 2, "a": 1}, payload=payload)
    result = _cache_get(con, provider="p", endpoint="e", symbol="S",
                        params={"a": 1, "b": 2})
    assert result == payload


def test_cache_put_overwrites_existing(con):
    params = {"symbol": "SPY"}
    _cache_put(con, provider="p", endpoint="e", symbol="SPY", params=params, payload={"v": 1})
    _cache_put(con, provider="p", endpoint="e", symbol="SPY", params=params, payload={"v": 2})
    result = _cache_get(con, provider="p", endpoint="e", symbol="SPY", params=params)
    assert result == {"v": 2}


def _make_av_client(payload: dict):
    """Return a mock AlphaVantageClient that returns payload without HTTP."""
    class FakeAV:
        def get(self, *, function, params):
            return payload
    return FakeAV()


def _make_yf_history(dates, closes):
    """Build a fake yfinance history DataFrame."""
    n = len(dates)
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Open": closes,
        "High": [c * 1.005 for c in closes],
        "Low": [c * 0.995 for c in closes],
        "Close": closes,
        "Adj Close": closes,
        "Volume": [75_000_000.0] * n,
        "Dividends": [0.0] * n,
        "Stock Splits": [0.0] * n,
    })


def _patch_yfinance(history_df):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = history_df
    return patch("yfinance.Ticker", return_value=mock_ticker)


def test_ingest_daily_adjusted_writes_prices(con):
    df = _make_yf_history(["2024-01-02", "2024-01-03"], [403.0, 406.0])
    with _patch_yfinance(df):
        ingest_daily_adjusted(con, "SPY")
    count = con.execute("SELECT COUNT(*) FROM prices_daily WHERE symbol='SPY'").fetchone()[0]
    assert count == 2


def test_ingest_daily_adjusted_is_idempotent(con):
    df = _make_yf_history(["2024-01-02"], [403.0])
    with _patch_yfinance(df):
        ingest_daily_adjusted(con, "SPY")
        ingest_daily_adjusted(con, "SPY")
    count = con.execute("SELECT COUNT(*) FROM prices_daily WHERE symbol='SPY'").fetchone()[0]
    assert count == 1


def test_ingest_options_pcr_writes_rows(con):
    payload = {
        "data": [
            {"date": "2024-01-02", "put_call_ratio": "0.85"},
            {"date": "2024-01-03", "put_call_ratio": "0.90"},
        ]
    }
    ingest_options_pcr(con, _make_av_client(payload), "SPY")
    count = con.execute("SELECT COUNT(*) FROM options_pcr_daily WHERE symbol='SPY'").fetchone()[0]
    assert count == 2


def test_ingest_options_voi_writes_rows(con):
    payload = {
        "data": [
            {"date": "2024-01-02", "volume_oi_ratio": "0.45"},
        ]
    }
    ingest_options_voi(con, _make_av_client(payload), "SPY")
    count = con.execute("SELECT COUNT(*) FROM options_voi_daily WHERE symbol='SPY'").fetchone()[0]
    assert count == 1


def test_ingest_news_sentiment_writes_rows(con):
    payload = {
        "feed": [
            {
                "time_published": "20240102T120000",
                "url": "https://example.com/news1",
                "title": "SPY rallies",
                "source": "Reuters",
                "overall_sentiment_score": 0.3,
                "overall_sentiment_label": "Somewhat-Bullish",
                "ticker_sentiment": [
                    {"ticker": "SPY", "ticker_sentiment_score": "0.25", "relevance_score": "0.8"}
                ],
            }
        ]
    }
    ingest_news_sentiment(con, _make_av_client(payload), "SPY")
    count = con.execute("SELECT COUNT(*) FROM news_sentiment WHERE symbol='SPY'").fetchone()[0]
    assert count == 1
