from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_mvp.train import (
    FEATURE_COLS,
    _dir_acc,
    _rmse,
    train_and_eval,
    train_final,
    walk_forward_eval,
)


def test_rmse_is_zero_for_perfect_predictions():
    y = np.array([0.01, -0.02, 0.03])
    assert _rmse(y, y) == pytest.approx(0.0)


def test_rmse_is_positive_for_imperfect_predictions():
    y_true = np.array([0.01, -0.02, 0.03])
    y_pred = np.array([0.0, 0.0, 0.0])
    assert _rmse(y_true, y_pred) > 0


def test_dir_acc_perfect_predictions():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    assert _dir_acc(y, y) == pytest.approx(1.0)


def test_dir_acc_all_wrong_predictions():
    y_true = np.array([0.01, -0.02, 0.03])
    y_pred = np.array([-0.01, 0.02, -0.03])
    assert _dir_acc(y_true, y_pred) == pytest.approx(0.0)


def test_dir_acc_bounded_between_0_and_1():
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 0.01, 100)
    y_pred = rng.normal(0, 0.01, 100)
    acc = _dir_acc(y_true, y_pred)
    assert 0.0 <= acc <= 1.0


def test_feature_cols_has_no_duplicates():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_walk_forward_eval_returns_correct_fold_count(sample_features_df):
    results = walk_forward_eval(sample_features_df, n_folds=5, min_train_rows=100)
    assert len(results) == 5


def test_walk_forward_eval_fold_metrics_are_valid(sample_features_df):
    results = walk_forward_eval(sample_features_df, n_folds=3, min_train_rows=100)
    for fold in results:
        assert 0.0 <= fold["dir_acc"] <= 1.0
        assert fold["rmse"] >= 0.0
        assert fold["train_rows"] > 0
        assert fold["test_rows"] > 0


def test_walk_forward_eval_train_grows_each_fold(sample_features_df):
    results = walk_forward_eval(sample_features_df, n_folds=4, min_train_rows=100)
    train_sizes = [r["train_rows"] for r in results]
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] > train_sizes[i - 1], "Each fold should have more training data"


def test_walk_forward_eval_returns_empty_for_insufficient_data():
    tiny_df = pd.DataFrame({
        "date": ["2024-01-01"] * 10,
        "y_fwd_return": [0.01] * 10,
        **{col: [0.0] * 10 for col in FEATURE_COLS},
    })
    results = walk_forward_eval(tiny_df, n_folds=5, min_train_rows=252)
    assert results == []


def test_train_final_saves_model_to_disk(tmp_path, sample_features_df):
    result = train_final(
        sample_features_df,
        symbol="SPY",
        horizon=5,
        models_dir=tmp_path,
    )
    model_path = Path(result["model_path"])
    assert model_path.exists()
    assert model_path.suffix == ".pkl"


def test_train_final_saved_model_is_loadable(tmp_path, sample_features_df):
    train_final(sample_features_df, symbol="SPY", horizon=5, models_dir=tmp_path)
    pkl_path = tmp_path / "SPY_h5.pkl"
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    assert "model" in bundle
    assert "feature_cols" in bundle
    assert bundle["symbol"] == "SPY"
    assert bundle["horizon"] == 5


def test_train_final_saved_model_can_predict(tmp_path, sample_features_df):
    train_final(sample_features_df, symbol="SPY", horizon=5, models_dir=tmp_path)
    pkl_path = tmp_path / "SPY_h5.pkl"
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    X = sample_features_df[FEATURE_COLS].ffill().fillna(0.0).to_numpy()
    preds = model.predict(X[:5])
    assert len(preds) == 5


def test_train_and_eval_returns_error_when_no_features(con):
    result = train_and_eval(con, "SPY", 5, models_dir=Path("/tmp"))
    assert "error" in result


def test_train_and_eval_succeeds_with_features(loaded_features, tmp_path):
    result = train_and_eval(loaded_features, "SPY", 5, models_dir=tmp_path)
    assert "error" not in result
    assert result["symbol"] == "SPY"
    assert result["horizon"] == 5
    assert result["total_rows"] > 0
    assert "model_path" in result
