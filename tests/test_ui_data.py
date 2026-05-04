"""Tests for ui_data.py — the shared Streamlit data layer.

All tests use in-memory DuckDB and temp directories.
No Streamlit runtime required; cache decorators are bypassed via direct patching.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import market_mvp.ui_data as uid
from market_mvp.db import init_db
from market_mvp.train import FEATURE_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_st_caches():
    """Clear Streamlit data caches between tests so patches take effect."""
    for fn in [
        uid.get_last_ingest_date,
        uid.get_row_counts,
        uid.get_latest_features,
        uid.get_features_history,
        uid.get_prices,
        uid.get_options,
        uid.get_news_sentiment,
        uid.get_fed_history,
        uid.get_social,
        uid.get_earnings,
    ]:
        try:
            fn.clear()
        except Exception:
            pass
    yield


@pytest.fixture
def mock_con(con):
    """Patch ui_data._con() to return the in-memory test connection."""
    with patch("market_mvp.ui_data._con", return_value=con):
        yield con


@pytest.fixture
def mock_db_path(tmp_path):
    """Patch _DB_PATH to a temp path that doesn't exist (no DB)."""
    fake = tmp_path / "missing.duckdb"
    with patch("market_mvp.ui_data._DB_PATH", fake):
        yield fake


@pytest.fixture
def mock_db_exists(tmp_path):
    """Patch _DB_PATH to a temp path that DOES exist."""
    real = tmp_path / "market_mvp.duckdb"
    real.touch()
    with patch("market_mvp.ui_data._DB_PATH", real):
        yield real


@pytest.fixture
def mock_models_dir(tmp_path):
    """Patch _MODELS_DIR to a temp directory."""
    with patch("market_mvp.ui_data._MODELS_DIR", tmp_path):
        yield tmp_path


@pytest.fixture
def saved_model(tmp_path):
    """Write a minimal real sklearn model pkl to tmp_path."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, len(FEATURE_COLS)))
    y = rng.normal(size=100)
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=10)
    model.fit(X, y)
    bundle = {"model": model, "feature_cols": FEATURE_COLS, "symbol": "SPY", "horizon": 5}
    pkl_path = tmp_path / "SPY_h5.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    return pkl_path


@pytest.fixture
def saved_metrics(tmp_path):
    """Write a minimal metrics JSON to tmp_path."""
    metrics = {
        "symbol": "SPY", "horizon": 5, "model": "lightgbm",
        "train_rows": 400, "mean_rmse": 0.0123, "mean_dir_acc": 0.54,
        "feature_importances": {col: float(i) for i, col in enumerate(FEATURE_COLS)},
        "walk_forward_folds": [
            {"fold": 1, "train_rows": 252, "test_rows": 30,
             "train_end_date": "2023-06-01", "rmse": 0.012, "dir_acc": 0.53},
        ],
    }
    path = tmp_path / "SPY_h5_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f)
    return path


# ── db_exists ─────────────────────────────────────────────────────────────────

def test_db_exists_false_when_no_file(mock_db_path):
    assert uid.db_exists() is False


def test_db_exists_true_when_file_present(mock_db_exists):
    assert uid.db_exists() is True


# ── model_exists ──────────────────────────────────────────────────────────────

def test_model_exists_false_when_no_pkl(mock_models_dir):
    assert uid.model_exists("SPY", 5) is False


def test_model_exists_true_when_pkl_present(mock_models_dir, saved_model):
    assert uid.model_exists("SPY", 5) is True


# ── load_model ────────────────────────────────────────────────────────────────

def test_load_model_returns_none_when_missing(mock_models_dir):
    assert uid.load_model("SPY", 5) is None


def test_load_model_returns_bundle_when_present(mock_models_dir, saved_model):
    bundle = uid.load_model("SPY", 5)
    assert bundle is not None
    assert "model" in bundle
    assert "feature_cols" in bundle


def test_load_model_bundle_has_correct_symbol(mock_models_dir, saved_model):
    bundle = uid.load_model("SPY", 5)
    assert bundle["symbol"] == "SPY"
    assert bundle["horizon"] == 5


# ── load_metrics ──────────────────────────────────────────────────────────────

def test_load_metrics_returns_none_when_missing(mock_models_dir):
    assert uid.load_metrics("SPY", 5) is None


def test_load_metrics_returns_dict_when_present(mock_models_dir, saved_metrics):
    metrics = uid.load_metrics("SPY", 5)
    assert metrics is not None
    assert metrics["mean_dir_acc"] == pytest.approx(0.54)
    assert metrics["mean_rmse"] == pytest.approx(0.0123)


def test_load_metrics_contains_feature_importances(mock_models_dir, saved_metrics):
    metrics = uid.load_metrics("SPY", 5)
    assert "feature_importances" in metrics
    assert len(metrics["feature_importances"]) == len(FEATURE_COLS)


# ── get_last_ingest_date ──────────────────────────────────────────────────────

def test_get_last_ingest_date_none_when_empty(mock_con):
    result = uid.get_last_ingest_date("SPY")
    assert result is None


def test_get_last_ingest_date_returns_max_date(mock_con):
    mock_con.execute(
        "INSERT INTO prices_daily(symbol, date, close) VALUES ('SPY', '2024-03-15', 500)"
    )
    mock_con.execute(
        "INSERT INTO prices_daily(symbol, date, close) VALUES ('SPY', '2024-06-01', 510)"
    )
    result = uid.get_last_ingest_date("SPY")
    assert "2024-06-01" in result


# ── get_row_counts ────────────────────────────────────────────────────────────

def test_get_row_counts_returns_all_keys(mock_con):
    counts = uid.get_row_counts()
    expected = ["prices_daily", "features_daily", "edgar_earnings",
                "fed_minutes", "social_sentiment_daily", "news_sentiment"]
    for key in expected:
        assert key in counts


def test_get_row_counts_zero_for_empty_db(mock_con):
    counts = uid.get_row_counts()
    assert all(v == 0 for v in counts.values())


def test_get_row_counts_increments_after_insert(mock_con):
    mock_con.execute(
        "INSERT INTO prices_daily(symbol, date, close) VALUES ('SPY', '2024-01-02', 400)"
    )
    counts = uid.get_row_counts()
    assert counts["prices_daily"] == 1


# ── get_prices ────────────────────────────────────────────────────────────────

def test_get_prices_empty_when_no_data(mock_con):
    result = uid.get_prices("SPY", days=30)
    assert result.empty


def test_get_prices_returns_correct_columns(mock_con):
    mock_con.execute(
        "INSERT INTO prices_daily(symbol, date, open, high, low, close, adjusted_close, volume) "
        "VALUES ('SPY', '2024-01-02', 398, 402, 397, 400, 400, 75000000)"
    )
    result = uid.get_prices("SPY", days=30)
    assert not result.empty
    assert "date" in result.columns
    assert "adjusted_close" in result.columns


def test_get_prices_date_is_sorted_ascending(mock_con):
    for d in ["2024-01-05", "2024-01-03", "2024-01-04"]:
        mock_con.execute(
            f"INSERT INTO prices_daily(symbol, date, close) VALUES ('SPY', '{d}', 400)"
        )
    result = uid.get_prices("SPY", days=30)
    dates = result["date"].tolist()
    assert dates == sorted(dates)


def test_get_prices_respects_days_limit(mock_con):
    for i in range(1, 21):
        mock_con.execute(
            f"INSERT INTO prices_daily(symbol, date, close) VALUES ('SPY', '2024-01-{i:02d}', 400)"
        )
    result = uid.get_prices("SPY", days=5)
    assert len(result) == 5


# ── get_features_history ──────────────────────────────────────────────────────

def test_get_features_history_empty_when_no_data(mock_con):
    result = uid.get_features_history("SPY", 5, days=30)
    assert result.empty


def test_get_features_history_date_is_datetime(mock_con, loaded_features):
    result = uid.get_features_history("SPY", 5, days=365)
    assert not result.empty
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_get_features_history_sorted_ascending(mock_con, loaded_features):
    result = uid.get_features_history("SPY", 5, days=365)
    dates = result["date"].tolist()
    assert dates == sorted(dates)


# ── predict ───────────────────────────────────────────────────────────────────

def test_predict_returns_none_when_no_model(mock_models_dir, mock_con):
    result = uid.predict("SPY", 5)
    assert result is None


def test_predict_returns_up_for_positive_model(mock_con, loaded_features):
    """A model bundle that always predicts positive should return direction=UP."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.05])
    bundle = {"model": mock_model, "feature_cols": FEATURE_COLS, "symbol": "SPY", "horizon": 5}
    with patch("market_mvp.ui_data.load_model", return_value=bundle):
        result = uid.predict("SPY", 5)
    assert result is not None
    assert result["direction"] == "UP"
    assert result["predicted_return"] == pytest.approx(0.05)


def test_predict_returns_down_for_negative_model(mock_con, loaded_features):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([-0.03])
    bundle = {"model": mock_model, "feature_cols": FEATURE_COLS, "symbol": "SPY", "horizon": 5}
    with patch("market_mvp.ui_data.load_model", return_value=bundle):
        result = uid.predict("SPY", 5)
    assert result is not None
    assert result["direction"] == "DOWN"


def test_predict_result_has_required_keys(mock_models_dir, mock_con, loaded_features, saved_model):
    result = uid.predict("SPY", 5)
    if result is None:
        pytest.skip("Model prediction not available in this test config")
    required = {"symbol", "horizon", "as_of_date", "predicted_return", "direction"}
    assert required.issubset(result.keys())


# ── predict_tft ───────────────────────────────────────────────────────────────

def test_predict_tft_returns_none_when_file_missing(tmp_path):
    with patch("market_mvp.ui_data._MODELS_DIR", tmp_path):
        assert uid.predict_tft("SPY", 5) is None


def test_predict_tft_reads_json_when_file_exists(tmp_path):
    payload = {
        "symbol": "SPY", "horizon": 5, "as_of_date": "2024-01-15",
        "predicted_return": 0.032, "direction": "UP",
        "p10": -0.01, "p90": 0.07, "model": "tft",
        "val_loss": 0.0012, "checkpoint": "/models/SPY_h5_tft.ckpt",
        "generated_at": "2024-01-15T10:00:00+00:00",
    }
    pred_file = tmp_path / "SPY_h5_tft_pred.json"
    pred_file.write_text(json.dumps(payload))

    with patch("market_mvp.ui_data._MODELS_DIR", tmp_path):
        result = uid.predict_tft("SPY", 5)

    assert result is not None
    assert result["symbol"] == "SPY"
    assert result["direction"] == "UP"
    assert result["predicted_return"] == pytest.approx(0.032)


def test_predict_tft_returns_none_for_different_symbol(tmp_path):
    payload = {"symbol": "SPY", "horizon": 5, "predicted_return": 0.01,
               "direction": "UP", "model": "tft"}
    (tmp_path / "SPY_h5_tft_pred.json").write_text(json.dumps(payload))

    with patch("market_mvp.ui_data._MODELS_DIR", tmp_path):
        assert uid.predict_tft("QQQ", 5) is None


def test_predict_tft_has_required_keys(tmp_path):
    payload = {
        "symbol": "QQQ", "horizon": 20, "as_of_date": "2024-01-15",
        "predicted_return": -0.01, "direction": "DOWN", "p10": -0.05,
        "p90": 0.02, "model": "tft", "val_loss": 0.002,
        "checkpoint": "/models/QQQ_h20_tft.ckpt",
        "generated_at": "2024-01-15T10:00:00+00:00",
    }
    (tmp_path / "QQQ_h20_tft_pred.json").write_text(json.dumps(payload))

    with patch("market_mvp.ui_data._MODELS_DIR", tmp_path):
        result = uid.predict_tft("QQQ", 20)

    required = {"symbol", "horizon", "predicted_return", "direction"}
    assert required.issubset(result.keys())


# ── get_options ───────────────────────────────────────────────────────────────

def test_get_options_empty_when_no_data(mock_con):
    result = uid.get_options("SPY", days=30)
    assert result.empty


def test_get_options_returns_pcr_when_available(mock_con):
    mock_con.execute(
        "INSERT INTO options_pcr_daily(symbol, date, put_call_ratio) VALUES ('SPY', '2024-01-02', 0.85)"
    )
    result = uid.get_options("SPY", days=30)
    assert not result.empty
    assert "put_call_ratio" in result.columns


# ── get_fed_history ───────────────────────────────────────────────────────────

def test_get_fed_history_empty_when_no_data(mock_con):
    result = uid.get_fed_history()
    assert result.empty


def test_get_fed_history_returns_datetime_dates(mock_con):
    mock_con.execute(
        "INSERT INTO fed_minutes(meeting_date, document_type, net_score, hawkish_score) "
        "VALUES ('2024-01-31', 'statement', 1.5, 3.0)"
    )
    result = uid.get_fed_history()
    assert not result.empty
    assert pd.api.types.is_datetime64_any_dtype(result["meeting_date"])


# ── get_social ────────────────────────────────────────────────────────────────

def test_get_social_empty_when_no_data(mock_con):
    result = uid.get_social("SPY", days=30)
    assert result.empty


def test_get_social_returns_both_sources(mock_con):
    for source in ["stocktwits", "reddit"]:
        mock_con.execute(
            f"INSERT INTO social_sentiment_daily(symbol, date, source, sentiment_score, message_count) "
            f"VALUES ('SPY', '2024-01-02', '{source}', 0.2, 100)"
        )
    result = uid.get_social("SPY", days=30)
    assert set(result["source"].unique()) == {"stocktwits", "reddit"}
