"""Tests for train_dgx.py — TFT data-prep helpers and JSON output.

torch and pytorch-forecasting are not required on Mac; tests that need them
are skipped automatically via pytest.importorskip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# train_dgx.py lives at repo root (one level above this file's directory)
sys.path.insert(0, str(Path(__file__).parent.parent))
import train_dgx


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n=200, null_target_rows=0, missing_cols=None):
    """Synthetic features_daily DataFrame matching the real schema."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n)
    targets = rng.normal(0.001, 0.02, n)
    if null_target_rows:
        targets[:null_target_rows] = np.nan

    data = {"symbol": "SPY", "date": dates, "horizon": 5, "y_fwd_return": targets}
    for col in train_dgx.FEATURE_COLS:
        if missing_cols and col in missing_cols:
            continue
        data[col] = rng.normal(0, 1, n)
    return pd.DataFrame(data)


# ── _prep ──────────────────────────────────────────────────────────────────────

def test_prep_adds_contiguous_time_idx():
    df = _make_df(100)
    out = train_dgx._prep(df)
    assert "time_idx" in out.columns
    assert list(out["time_idx"]) == list(range(len(out)))


def test_prep_drops_rows_with_null_target():
    df = _make_df(100, null_target_rows=10)
    out = train_dgx._prep(df)
    assert out["y_fwd_return"].isna().sum() == 0
    assert len(out) == 90


def test_prep_time_idx_is_contiguous_after_null_drop():
    df = _make_df(100, null_target_rows=5)
    out = train_dgx._prep(df)
    assert list(out["time_idx"]) == list(range(len(out)))


def test_prep_fills_missing_feature_column_with_zero():
    df = _make_df(100, missing_cols=["pcr", "voi"])
    out = train_dgx._prep(df)
    assert (out["pcr"] == 0.0).all()
    assert (out["voi"] == 0.0).all()


def test_prep_fills_nan_features_with_ffill_then_zero():
    df = _make_df(100)
    df.loc[0:5, "rsi_14"] = np.nan   # leading NaN → filled with 0.0
    df.loc[50:55, "rsi_14"] = np.nan  # mid NaN → forward-filled
    out = train_dgx._prep(df)
    assert out["rsi_14"].isna().sum() == 0


def test_prep_sorts_ascending_by_date():
    df = _make_df(100)
    df = df.sample(frac=1, random_state=0)  # shuffle
    out = train_dgx._prep(df)
    dates = out["date"].tolist()
    assert dates == sorted(dates)


def test_prep_symbol_is_string():
    df = _make_df(50)
    out = train_dgx._prep(df)
    assert out["symbol"].iloc[0] == "SPY"
    assert isinstance(out["symbol"].iloc[0], str)


def test_prep_all_feature_cols_present_in_output():
    df = _make_df(50)
    out = train_dgx._prep(df)
    for col in train_dgx.FEATURE_COLS:
        assert col in out.columns, f"Missing feature column: {col}"


def test_prep_returns_copy_not_mutating_input():
    df = _make_df(50)
    original_cols = set(df.columns)
    train_dgx._prep(df)
    assert set(df.columns) == original_cols
    assert "time_idx" not in df.columns


# ── _save_prediction ───────────────────────────────────────────────────────────

def _make_quantile_tensor(median=0.05, p10=-0.02, p90=0.12):
    """Build a (1, 1, 7) tensor mimicking TFT quantile output."""
    torch = pytest.importorskip("torch")
    # Quantile indices: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    q = torch.zeros(1, 1, 7)
    q[0, 0, 1] = p10
    q[0, 0, 3] = median
    q[0, 0, 5] = p90
    return q


def test_save_prediction_writes_json_file(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_forecasting")
    train_dgx.MODELS_DIR = tmp_path

    mock_tft = MagicMock()
    mock_tft.predict.return_value = _make_quantile_tensor(0.04)

    df = _make_df(100)
    df["date"] = pd.to_datetime(df["date"])

    with patch("train_dgx.TemporalFusionTransformer" if hasattr(train_dgx, "TemporalFusionTransformer")
               else "pytorch_forecasting.TemporalFusionTransformer"):
        with patch("pytorch_forecasting.TemporalFusionTransformer.load_from_checkpoint",
                   return_value=mock_tft):
            train_dgx._save_prediction("fake.ckpt", MagicMock(), "SPY", 5, df, 0.123)

    pred_path = tmp_path / "SPY_h5_tft_pred.json"
    assert pred_path.exists()


def test_save_prediction_json_has_required_keys(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_forecasting")
    train_dgx.MODELS_DIR = tmp_path

    mock_tft = MagicMock()
    mock_tft.predict.return_value = _make_quantile_tensor(0.04)

    df = _make_df(100)
    df["date"] = pd.to_datetime(df["date"])

    with patch("pytorch_forecasting.TemporalFusionTransformer.load_from_checkpoint",
               return_value=mock_tft):
        train_dgx._save_prediction("fake.ckpt", MagicMock(), "SPY", 5, df, 0.123)

    with open(tmp_path / "SPY_h5_tft_pred.json") as f:
        data = json.load(f)

    required = {"symbol", "horizon", "as_of_date", "predicted_return", "direction",
                "p10", "p90", "model", "val_loss", "generated_at"}
    assert required.issubset(data.keys())


def test_save_prediction_direction_up_for_positive_median(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_forecasting")
    train_dgx.MODELS_DIR = tmp_path

    mock_tft = MagicMock()
    mock_tft.predict.return_value = _make_quantile_tensor(0.06)

    df = _make_df(100)
    df["date"] = pd.to_datetime(df["date"])

    with patch("pytorch_forecasting.TemporalFusionTransformer.load_from_checkpoint",
               return_value=mock_tft):
        train_dgx._save_prediction("fake.ckpt", MagicMock(), "SPY", 5, df, 0.1)

    with open(tmp_path / "SPY_h5_tft_pred.json") as f:
        data = json.load(f)
    assert data["direction"] == "UP"
    assert data["predicted_return"] == pytest.approx(0.06, abs=1e-4)


def test_save_prediction_direction_down_for_negative_median(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_forecasting")
    train_dgx.MODELS_DIR = tmp_path

    mock_tft = MagicMock()
    mock_tft.predict.return_value = _make_quantile_tensor(-0.03)

    df = _make_df(100)
    df["date"] = pd.to_datetime(df["date"])

    with patch("pytorch_forecasting.TemporalFusionTransformer.load_from_checkpoint",
               return_value=mock_tft):
        train_dgx._save_prediction("fake.ckpt", MagicMock(), "SPY", 5, df, 0.1)

    with open(tmp_path / "SPY_h5_tft_pred.json") as f:
        data = json.load(f)
    assert data["direction"] == "DOWN"


def test_save_prediction_does_not_crash_on_inference_error(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_forecasting")
    train_dgx.MODELS_DIR = tmp_path

    df = _make_df(100)
    df["date"] = pd.to_datetime(df["date"])

    with patch("pytorch_forecasting.TemporalFusionTransformer.load_from_checkpoint",
               side_effect=RuntimeError("GPU OOM")):
        # Should log and return without raising
        train_dgx._save_prediction("fake.ckpt", MagicMock(), "SPY", 5, df, 0.1)

    assert not (tmp_path / "SPY_h5_tft_pred.json").exists()


# ── MIN_ROWS constant ─────────────────────────────────────────────────────────

def test_min_rows_equals_three_encoder_lengths():
    assert train_dgx.MIN_ROWS == train_dgx.MAX_ENCODER_LENGTH * 3


def test_feature_cols_matches_train_py():
    from market_mvp.train import FEATURE_COLS as lgbm_cols
    assert train_dgx.FEATURE_COLS == lgbm_cols, (
        "FEATURE_COLS in train_dgx.py and train.py must be identical"
    )
