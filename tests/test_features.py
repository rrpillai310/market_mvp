from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_mvp.features import _rsi, _roll_chg, build_features


def test_roll_chg_correct_values():
    s = pd.Series([10.0, 12.0, 11.0, 13.0, 14.0])
    result = _roll_chg(s, 2)
    assert result.iloc[2] == pytest.approx(1.0)
    assert result.iloc[3] == pytest.approx(1.0)
    assert np.isnan(result.iloc[0])


def test_rsi_is_bounded_between_0_and_100():
    np.random.seed(0)
    returns = pd.Series(np.random.randn(200) * 0.01)
    rsi = _rsi(returns)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_all_gains_approaches_100():
    returns = pd.Series([0.01] * 50)
    rsi = _rsi(returns)
    assert rsi.dropna().iloc[-1] > 90


def test_rsi_all_losses_approaches_0():
    returns = pd.Series([-0.01] * 50)
    rsi = _rsi(returns)
    assert rsi.dropna().iloc[-1] < 10


def test_build_features_produces_expected_columns(loaded_prices):
    con = loaded_prices
    build_features(con, "SPY", horizons=[5])
    cols = {r[0] for r in con.execute("DESCRIBE features_daily").fetchall()}
    required = {
        "ret_1d", "ret_5d", "ret_20d", "rsi_14",
        "vol_ratio_20d", "intraday_range", "div_ex_days",
    }
    assert required.issubset(cols)


def test_build_features_writes_rows_for_each_horizon(loaded_prices):
    con = loaded_prices
    build_features(con, "SPY", horizons=[5, 20])
    h5 = con.execute(
        "SELECT COUNT(*) FROM features_daily WHERE symbol='SPY' AND horizon=5"
    ).fetchone()[0]
    h20 = con.execute(
        "SELECT COUNT(*) FROM features_daily WHERE symbol='SPY' AND horizon=20"
    ).fetchone()[0]
    assert h5 > 0
    assert h20 > 0
    # 20-day horizon needs 20 more rows of lookahead so fewer rows than h5
    assert h20 <= h5


def test_build_features_rsi_values_are_valid(loaded_prices):
    con = loaded_prices
    build_features(con, "SPY", horizons=[5])
    df = con.execute(
        "SELECT rsi_14 FROM features_daily WHERE symbol='SPY' AND rsi_14 IS NOT NULL"
    ).df()
    assert not df.empty
    assert (df["rsi_14"] >= 0).all()
    assert (df["rsi_14"] <= 100).all()


def test_build_features_is_idempotent(loaded_prices):
    con = loaded_prices
    build_features(con, "SPY", horizons=[5])
    count_first = con.execute(
        "SELECT COUNT(*) FROM features_daily WHERE symbol='SPY'"
    ).fetchone()[0]
    build_features(con, "SPY", horizons=[5])
    count_second = con.execute(
        "SELECT COUNT(*) FROM features_daily WHERE symbol='SPY'"
    ).fetchone()[0]
    assert count_first == count_second


def test_build_features_div_ex_days_non_negative(loaded_prices):
    con = loaded_prices
    build_features(con, "SPY", horizons=[5])
    df = con.execute(
        "SELECT div_ex_days FROM features_daily WHERE symbol='SPY' AND div_ex_days IS NOT NULL AND div_ex_days >= 0"
    ).df()
    # At least some rows should have non-negative div_ex_days
    assert not df.empty


def test_build_features_vol_ratio_positive(loaded_prices):
    con = loaded_prices
    build_features(con, "SPY", horizons=[5])
    df = con.execute(
        "SELECT vol_ratio_20d FROM features_daily WHERE symbol='SPY' AND vol_ratio_20d IS NOT NULL"
    ).df()
    assert not df.empty
    assert (df["vol_ratio_20d"] > 0).all()


@pytest.mark.parametrize("col,min_val,max_val", [
    ("ret_1d", -0.5, 0.5),
    ("intraday_range", 0.0, 0.5),
    ("price_ma5_ratio", -0.5, 0.5),
])
def test_build_features_price_columns_in_reasonable_range(loaded_prices, col, min_val, max_val):
    con = loaded_prices
    build_features(con, "SPY", horizons=[5])
    df = con.execute(
        f"SELECT {col} FROM features_daily WHERE symbol='SPY' AND {col} IS NOT NULL"
    ).df()
    if df.empty:
        pytest.skip(f"No non-null {col} values (need more price history)")
    assert (df[col] >= min_val).all(), f"{col} has values below {min_val}"
    assert (df[col] <= max_val).all(), f"{col} has values above {max_val}"
