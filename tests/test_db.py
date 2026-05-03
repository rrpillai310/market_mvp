from __future__ import annotations

import duckdb
import pytest

from market_mvp.db import DB, init_db


EXPECTED_TABLES = {
    "api_cache",
    "prices_daily",
    "options_pcr_daily",
    "options_voi_daily",
    "news_sentiment",
    "features_daily",
    "edgar_cik_map",
    "edgar_filings",
    "edgar_earnings",
    "fed_minutes",
    "social_sentiment_daily",
}


def test_all_expected_tables_created(con):
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert EXPECTED_TABLES.issubset(tables), f"Missing: {EXPECTED_TABLES - tables}"


def test_init_db_is_idempotent(con):
    init_db(con)  # second call must not raise
    init_db(con)  # third call too
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert EXPECTED_TABLES.issubset(tables)


def test_features_daily_has_all_feature_columns(con):
    cols = {r[0] for r in con.execute("DESCRIBE features_daily").fetchall()}
    expected_features = {
        "pcr", "pcr_chg_5", "voi", "voi_chg_5",
        "news_sent", "news_sent_chg_5", "news_count",
        "div_ex_days", "div_amount",
        "ret_1d", "ret_5d", "ret_20d", "ret_60d",
        "vol_10d", "vol_20d",
        "price_ma5_ratio", "price_ma20_ratio", "price_ma60_ratio",
        "rsi_14", "vol_ratio_20d", "intraday_range",
        "eps_surprise_pct", "rev_surprise_pct", "days_to_earnings",
        "fed_hawkish_score", "fed_net_score", "fed_days_since",
        "stocktwits_bull_ratio", "reddit_sentiment", "social_volume_ratio",
    }
    missing = expected_features - cols
    assert not missing, f"features_daily is missing columns: {missing}"


def test_prices_daily_primary_key_enforced(con):
    con.execute(
        "INSERT INTO prices_daily(symbol, date, close) VALUES ('TEST', '2024-01-01', 100)"
    )
    con.execute(
        "INSERT OR REPLACE INTO prices_daily(symbol, date, close) VALUES ('TEST', '2024-01-01', 200)"
    )
    count = con.execute(
        "SELECT COUNT(*) FROM prices_daily WHERE symbol='TEST' AND date='2024-01-01'"
    ).fetchone()[0]
    assert count == 1

    close = con.execute(
        "SELECT close FROM prices_daily WHERE symbol='TEST' AND date='2024-01-01'"
    ).fetchone()[0]
    assert close == 200


def test_db_connect_creates_parent_directory(tmp_path):
    db_path = tmp_path / "subdir" / "test.duckdb"
    db = DB(path=db_path)
    con = db.connect()
    con.close()
    assert db_path.exists()
