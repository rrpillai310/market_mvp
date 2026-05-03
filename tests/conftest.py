"""Shared fixtures for all test modules. No network calls anywhere in this file."""
from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from market_mvp.db import init_db


@pytest.fixture
def con():
    """In-memory DuckDB with fully initialized schema."""
    c = duckdb.connect(":memory:")
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def sample_prices():
    """150 trading days of synthetic SPY OHLCV data."""
    np.random.seed(42)
    n = 150
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 400 + np.cumsum(np.random.randn(n) * 2)
    close = np.maximum(close, 50)  # keep positive
    return pd.DataFrame({
        "symbol": "SPY",
        "date": dates.strftime("%Y-%m-%d"),
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "adjusted_close": close,
        "volume": np.random.randint(50_000_000, 100_000_000, n).astype(float),
        "dividend_amount": [0.5 if i % 30 == 0 else 0.0 for i in range(n)],
        "split_coefficient": 1.0,
    })


@pytest.fixture
def loaded_prices(con, sample_prices):
    """Prices already inserted into the in-memory DB."""
    con.register("tmp_p", sample_prices)
    con.execute(
        """
        INSERT OR REPLACE INTO prices_daily
        SELECT symbol, CAST(date AS DATE), open, high, low, close,
               adjusted_close, CAST(volume AS BIGINT), dividend_amount, split_coefficient
        FROM tmp_p
        """
    )
    con.unregister("tmp_p")
    return con


@pytest.fixture
def sample_features_df():
    """300 rows of synthetic features_daily data for one symbol/horizon."""
    np.random.seed(7)
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(7)

    data = {
        "symbol": "SPY",
        "date": dates.strftime("%Y-%m-%d"),
        "horizon": 5,
        "y_fwd_return": rng.normal(0.001, 0.02, n),
        "pcr": rng.uniform(0.5, 1.5, n),
        "pcr_chg_5": rng.normal(0, 0.1, n),
        "voi": rng.uniform(0.2, 1.0, n),
        "voi_chg_5": rng.normal(0, 0.05, n),
        "news_sent": rng.normal(0, 0.3, n),
        "news_sent_chg_5": rng.normal(0, 0.1, n),
        "news_count": rng.integers(1, 20, n).astype(float),
        "div_ex_days": rng.integers(0, 90, n).astype(float),
        "div_amount": rng.choice([0.0, 0.5], n),
        "ret_1d": rng.normal(0.001, 0.01, n),
        "ret_5d": rng.normal(0.005, 0.02, n),
        "ret_20d": rng.normal(0.02, 0.04, n),
        "ret_60d": rng.normal(0.06, 0.08, n),
        "vol_10d": rng.uniform(0.005, 0.025, n),
        "vol_20d": rng.uniform(0.005, 0.025, n),
        "price_ma5_ratio": rng.normal(0, 0.01, n),
        "price_ma20_ratio": rng.normal(0, 0.02, n),
        "price_ma60_ratio": rng.normal(0, 0.04, n),
        "rsi_14": rng.uniform(20, 80, n),
        "vol_ratio_20d": rng.uniform(0.5, 2.0, n),
        "intraday_range": rng.uniform(0.002, 0.02, n),
        "eps_surprise_pct": rng.normal(0, 5, n),
        "rev_surprise_pct": rng.normal(0, 3, n),
        "days_to_earnings": rng.integers(0, 90, n).astype(float),
        "fed_hawkish_score": rng.uniform(0, 5, n),
        "fed_net_score": rng.normal(0, 2, n),
        "fed_days_since": rng.integers(0, 45, n).astype(float),
        "stocktwits_bull_ratio": rng.uniform(0.3, 0.7, n),
        "reddit_sentiment": rng.normal(0, 0.3, n),
        "social_volume_ratio": rng.uniform(0.5, 2.0, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def loaded_features(con, sample_features_df):
    """features_daily rows already inserted into in-memory DB."""
    con.register("tmp_f", sample_features_df)
    con.execute(
        """
        INSERT OR REPLACE INTO features_daily
        SELECT symbol, CAST(date AS DATE), horizon, y_fwd_return,
               pcr, pcr_chg_5, voi, voi_chg_5,
               news_sent, news_sent_chg_5, CAST(news_count AS INTEGER),
               CAST(div_ex_days AS INTEGER), div_amount,
               ret_1d, ret_5d, ret_20d, ret_60d,
               vol_10d, vol_20d,
               price_ma5_ratio, price_ma20_ratio, price_ma60_ratio,
               rsi_14, vol_ratio_20d, intraday_range,
               eps_surprise_pct, rev_surprise_pct, CAST(days_to_earnings AS INTEGER),
               fed_hawkish_score, fed_net_score, CAST(fed_days_since AS INTEGER),
               stocktwits_bull_ratio, reddit_sentiment, social_volume_ratio
        FROM tmp_f
        """
    )
    con.unregister("tmp_f")
    return con
