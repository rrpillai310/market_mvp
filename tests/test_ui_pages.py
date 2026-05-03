"""Streamlit page tests using AppTest.

AppTest runs each page script in-process with st.cache_data bypassed.
All external data is mocked via unittest.mock.patch so no DB or models are needed.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from market_mvp.train import FEATURE_COLS

_PAGES = Path(__file__).parent.parent / "pages"
_APP = Path(__file__).parent.parent / "app.py"

# ── Shared mock data ──────────────────────────────────────────────────────────

def _price_df(n=60):
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = 400 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "adjusted_close": close,
        "volume": np.random.randint(50_000_000, 100_000_000, n).astype(float),
    })


def _features_df(n=120):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=n)
    data = {"date": pd.to_datetime(dates), "horizon": 5, "symbol": "SPY",
            "y_fwd_return": rng.normal(0.001, 0.02, n)}
    for col in FEATURE_COLS:
        data[col] = rng.normal(0, 1, n)
    data["rsi_14"] = rng.uniform(20, 80, n)
    return pd.DataFrame(data)


def _metrics_dict():
    return {
        "symbol": "SPY", "horizon": 5, "model": "lightgbm",
        "train_rows": 400, "mean_rmse": 0.0120, "mean_dir_acc": 0.56,
        "trained_at": "2026-05-03T10:00:00+00:00",
        "feature_importances": {col: float(i + 1) for i, col in enumerate(FEATURE_COLS)},
        "walk_forward_folds": [
            {"fold": 1, "train_rows": 252, "test_rows": 40,
             "train_end_date": "2023-06-01", "test_start_date": "2023-06-02",
             "rmse": 0.011, "dir_acc": 0.55, "model": "lightgbm"},
            {"fold": 2, "train_rows": 292, "test_rows": 40,
             "train_end_date": "2023-08-15", "test_start_date": "2023-08-16",
             "rmse": 0.013, "dir_acc": 0.57, "model": "lightgbm"},
        ],
    }


def _predict_result(direction="UP"):
    ret = 0.025 if direction == "UP" else -0.018
    return {"symbol": "SPY", "horizon": 5, "as_of_date": "2024-06-01",
            "predicted_return": ret, "direction": direction}


# ── Home page (app.py) ────────────────────────────────────────────────────────

class TestHomePage:
    def test_shows_warning_when_db_missing(self):
        with patch("market_mvp.ui_data.db_exists", return_value=False):
            at = AppTest.from_file(str(_APP)).run()
        assert len(at.warning) > 0

    def test_does_not_crash_when_db_missing(self):
        with patch("market_mvp.ui_data.db_exists", return_value=False):
            at = AppTest.from_file(str(_APP)).run()
        assert not at.exception

    def test_shows_pipeline_status_when_db_exists(self):
        counts = {"prices_daily": 500, "features_daily": 1000, "edgar_earnings": 80,
                  "fed_minutes": 12, "social_sentiment_daily": 60, "news_sentiment": 200}
        with (
            patch("market_mvp.ui_data.db_exists", return_value=True),
            patch("market_mvp.ui_data.get_last_ingest_date", return_value="2024-06-01"),
            patch("market_mvp.ui_data.get_row_counts", return_value=counts),
            patch("market_mvp.ui_data.load_metrics", return_value=None),
        ):
            at = AppTest.from_file(str(_APP)).run()
        assert not at.exception
        metric_values = [m.value for m in at.metric]
        assert any("2024-06-01" in str(v) for v in metric_values)

    def test_shows_model_metrics_when_trained(self):
        counts = {t: 100 for t in ["prices_daily", "features_daily", "edgar_earnings",
                                    "fed_minutes", "social_sentiment_daily", "news_sentiment"]}
        with (
            patch("market_mvp.ui_data.db_exists", return_value=True),
            patch("market_mvp.ui_data.get_last_ingest_date", return_value="2024-06-01"),
            patch("market_mvp.ui_data.get_row_counts", return_value=counts),
            patch("market_mvp.ui_data.load_metrics", return_value=_metrics_dict()),
        ):
            at = AppTest.from_file(str(_APP)).run()
        assert not at.exception
        metric_labels = [m.label for m in at.metric]
        assert any("SPY" in str(lbl) for lbl in metric_labels)

    def test_no_exception_with_zero_counts(self):
        counts = {t: 0 for t in ["prices_daily", "features_daily", "edgar_earnings",
                                   "fed_minutes", "social_sentiment_daily", "news_sentiment"]}
        with (
            patch("market_mvp.ui_data.db_exists", return_value=True),
            patch("market_mvp.ui_data.get_last_ingest_date", return_value=None),
            patch("market_mvp.ui_data.get_row_counts", return_value=counts),
            patch("market_mvp.ui_data.load_metrics", return_value=None),
        ):
            at = AppTest.from_file(str(_APP)).run()
        assert not at.exception


# ── Predictions page ──────────────────────────────────────────────────────────

class TestPredictionsPage:
    _page = str(_PAGES / "1_Predictions.py")

    def test_shows_warning_when_no_model(self):
        with patch("market_mvp.ui_data.model_exists", return_value=False):
            at = AppTest.from_file(self._page).run()
        assert len(at.warning) > 0
        assert not at.exception

    def test_no_exception_when_model_missing(self):
        with patch("market_mvp.ui_data.model_exists", return_value=False):
            at = AppTest.from_file(self._page).run()
        assert not at.exception

    def test_shows_up_prediction(self):
        with (
            patch("market_mvp.ui_data.model_exists", return_value=True),
            patch("market_mvp.ui_data.predict", return_value=_predict_result("UP")),
            patch("market_mvp.ui_data.load_metrics", return_value=_metrics_dict()),
            patch("market_mvp.ui_data.get_latest_features", return_value=_features_df(1)),
        ):
            at = AppTest.from_file(self._page).run()
        assert not at.exception
        page_text = " ".join(e.value for e in at.markdown if hasattr(e, "value"))
        assert "UP" in page_text or any("UP" in str(m.value) for m in at.metric)

    def test_shows_down_prediction(self):
        with (
            patch("market_mvp.ui_data.model_exists", return_value=True),
            patch("market_mvp.ui_data.predict", return_value=_predict_result("DOWN")),
            patch("market_mvp.ui_data.load_metrics", return_value=_metrics_dict()),
            patch("market_mvp.ui_data.get_latest_features", return_value=_features_df(1)),
        ):
            at = AppTest.from_file(self._page).run()
        assert not at.exception

    def test_shows_feature_snapshot_metrics(self):
        feats = _features_df(1)
        with (
            patch("market_mvp.ui_data.model_exists", return_value=True),
            patch("market_mvp.ui_data.predict", return_value=_predict_result("UP")),
            patch("market_mvp.ui_data.load_metrics", return_value=_metrics_dict()),
            patch("market_mvp.ui_data.get_latest_features", return_value=feats),
        ):
            at = AppTest.from_file(self._page).run()
        assert not at.exception
        assert len(at.metric) >= 4

    def test_no_crash_when_predict_returns_none(self):
        with (
            patch("market_mvp.ui_data.model_exists", return_value=True),
            patch("market_mvp.ui_data.predict", return_value=None),
        ):
            at = AppTest.from_file(self._page).run()
        assert not at.exception
        assert len(at.error) > 0


# ── Signals page ──────────────────────────────────────────────────────────────

class TestSignalsPage:
    _page = str(_PAGES / "2_Signals.py")

    def test_no_crash_with_all_empty_data(self):
        empty = pd.DataFrame()
        with (
            patch("market_mvp.ui_data.get_prices", return_value=empty),
            patch("market_mvp.ui_data.get_features_history", return_value=empty),
            patch("market_mvp.ui_data.get_options", return_value=empty),
            patch("market_mvp.ui_data.get_news_sentiment", return_value=empty),
            patch("market_mvp.ui_data.get_fed_history", return_value=empty),
            patch("market_mvp.ui_data.get_social", return_value=empty),
        ):
            at = AppTest.from_file(self._page).run()
        assert not at.exception

    def test_shows_warning_for_missing_prices(self):
        empty = pd.DataFrame()
        with (
            patch("market_mvp.ui_data.get_prices", return_value=empty),
            patch("market_mvp.ui_data.get_features_history", return_value=empty),
            patch("market_mvp.ui_data.get_options", return_value=empty),
            patch("market_mvp.ui_data.get_news_sentiment", return_value=empty),
            patch("market_mvp.ui_data.get_fed_history", return_value=empty),
            patch("market_mvp.ui_data.get_social", return_value=empty),
        ):
            at = AppTest.from_file(self._page).run()
        assert len(at.warning) > 0

    def test_no_crash_with_full_data(self):
        feats = _features_df()
        prices = _price_df()
        options = pd.DataFrame({
            "date": pd.to_datetime(pd.bdate_range("2024-01-02", periods=60)),
            "put_call_ratio": np.random.uniform(0.7, 1.3, 60),
            "volume_oi_ratio": np.random.uniform(0.2, 0.8, 60),
        })
        news = pd.DataFrame({
            "date": pd.to_datetime(pd.bdate_range("2024-01-02", periods=30)),
            "avg_sentiment": np.random.uniform(-0.3, 0.3, 30),
            "article_count": np.random.randint(1, 20, 30),
        })
        fed = pd.DataFrame({
            "meeting_date": pd.to_datetime(["2024-01-31", "2024-03-20"]),
            "document_type": ["statement", "statement"],
            "net_score": [1.5, -0.5],
            "hawkish_score": [3.0, 2.0],
            "dovish_score": [1.5, 2.5],
        })
        social = pd.DataFrame({
            "date": pd.to_datetime(pd.bdate_range("2024-01-02", periods=20)),
            "source": ["stocktwits"] * 20,
            "bull_ratio": np.random.uniform(0.4, 0.7, 20),
            "sentiment_score": np.random.uniform(-0.3, 0.3, 20),
            "message_count": np.random.randint(50, 200, 20),
        })
        with (
            patch("market_mvp.ui_data.get_prices", return_value=prices),
            patch("market_mvp.ui_data.get_features_history", return_value=feats),
            patch("market_mvp.ui_data.get_options", return_value=options),
            patch("market_mvp.ui_data.get_news_sentiment", return_value=news),
            patch("market_mvp.ui_data.get_fed_history", return_value=fed),
            patch("market_mvp.ui_data.get_social", return_value=social),
        ):
            at = AppTest.from_file(self._page).run()
        assert not at.exception

    def test_symbol_selector_present(self):
        empty = pd.DataFrame()
        with (
            patch("market_mvp.ui_data.get_prices", return_value=empty),
            patch("market_mvp.ui_data.get_features_history", return_value=empty),
            patch("market_mvp.ui_data.get_options", return_value=empty),
            patch("market_mvp.ui_data.get_news_sentiment", return_value=empty),
            patch("market_mvp.ui_data.get_fed_history", return_value=empty),
            patch("market_mvp.ui_data.get_social", return_value=empty),
        ):
            at = AppTest.from_file(self._page).run()
        assert len(at.selectbox) >= 1
        options_vals = at.selectbox[0].options
        assert "SPY" in options_vals


# ── Performance page ──────────────────────────────────────────────────────────

class TestPerformancePage:
    _page = str(_PAGES / "3_Performance.py")

    def test_shows_warning_when_no_metrics(self):
        with patch("market_mvp.ui_data.load_metrics", return_value=None):
            at = AppTest.from_file(self._page).run()
        assert len(at.warning) > 0
        assert not at.exception

    def test_no_crash_when_metrics_present(self):
        with (
            patch("market_mvp.ui_data.load_metrics", return_value=_metrics_dict()),
            patch("market_mvp.ui_data.get_features_history", return_value=_features_df()),
            patch("market_mvp.ui_data.load_model", return_value=None),
        ):
            at = AppTest.from_file(self._page).run()
        assert not at.exception

    def test_shows_top_line_metrics(self):
        with (
            patch("market_mvp.ui_data.load_metrics", return_value=_metrics_dict()),
            patch("market_mvp.ui_data.get_features_history", return_value=_features_df()),
            patch("market_mvp.ui_data.load_model", return_value=None),
        ):
            at = AppTest.from_file(self._page).run()
        assert not at.exception
        assert len(at.metric) >= 3

    def test_dir_acc_metric_is_present(self):
        with (
            patch("market_mvp.ui_data.load_metrics", return_value=_metrics_dict()),
            patch("market_mvp.ui_data.get_features_history", return_value=_features_df()),
            patch("market_mvp.ui_data.load_model", return_value=None),
        ):
            at = AppTest.from_file(self._page).run()
        labels = [m.label for m in at.metric]
        assert any("acc" in str(lbl).lower() or "accuracy" in str(lbl).lower()
                   for lbl in labels)

    def test_no_crash_with_no_folds(self):
        metrics = _metrics_dict()
        metrics["walk_forward_folds"] = []
        with (
            patch("market_mvp.ui_data.load_metrics", return_value=metrics),
            patch("market_mvp.ui_data.get_features_history", return_value=_features_df()),
            patch("market_mvp.ui_data.load_model", return_value=None),
        ):
            at = AppTest.from_file(self._page).run()
        assert not at.exception

    def test_no_crash_with_no_feature_importances(self):
        metrics = _metrics_dict()
        metrics["feature_importances"] = {}
        with (
            patch("market_mvp.ui_data.load_metrics", return_value=metrics),
            patch("market_mvp.ui_data.get_features_history", return_value=pd.DataFrame()),
            patch("market_mvp.ui_data.load_model", return_value=None),
        ):
            at = AppTest.from_file(self._page).run()
        assert not at.exception
