from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from market_mvp.db import DB, init_db


FEATURE_COLS = [
    "pcr", "pcr_chg_5",
    "voi", "voi_chg_5",
    "news_sent", "news_sent_chg_5", "news_count",
    "div_ex_days", "div_amount",
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "vol_10d", "vol_20d",
    "price_ma5_ratio", "price_ma20_ratio", "price_ma60_ratio",
    "rsi_14", "vol_ratio_20d", "intraday_range",
    "eps_surprise_pct", "rev_surprise_pct", "days_to_earnings",
    "fed_hawkish_score", "fed_net_score", "fed_days_since",
    "stocktwits_bull_ratio", "reddit_sentiment", "social_volume_ratio",
]


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _dir_acc(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true > 0) == (y_pred > 0)))


def _build_model():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        ), "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.05, random_state=42
        ), "sklearn_hgbr"


def walk_forward_eval(
    df: pd.DataFrame,
    *,
    n_folds: int = 5,
    min_train_rows: int = 252,
) -> list[dict]:
    """Expanding-window walk-forward cross-validation.

    Each fold trains on all data up to a cutoff, tests on the next slice.
    Returns per-fold metrics.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)

    if n < min_train_rows + 30:
        return []

    usable = n - min_train_rows
    fold_size = max(usable // n_folds, 10)

    results = []
    for fold in range(n_folds):
        train_end = min_train_rows + fold * fold_size
        test_end = min(train_end + fold_size, n)

        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]

        if len(test) < 5:
            continue

        # Fill columns that may not exist yet (new feature columns from later phases)
        for c in FEATURE_COLS:
            if c not in train.columns:
                train = train.copy()
                train[c] = 0.0
            if c not in test.columns:
                test = test.copy()
                test[c] = 0.0

        X_train = train[FEATURE_COLS].ffill().fillna(0.0).to_numpy()
        y_train = train["y_fwd_return"].to_numpy()
        X_test = test[FEATURE_COLS].ffill().fillna(0.0).to_numpy()
        y_test = test["y_fwd_return"].to_numpy()

        model, model_name = _build_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "fold": fold + 1,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_end_date": train.iloc[-1]["date"],
            "test_start_date": test.iloc[0]["date"],
            "rmse": _rmse(y_test, y_pred),
            "dir_acc": _dir_acc(y_test, y_pred),
            "model": model_name,
        })

    return results


def train_final(
    df: pd.DataFrame,
    *,
    symbol: str,
    horizon: int,
    models_dir: Path,
) -> dict:
    """Train on full dataset and save model to disk."""
    df = df.sort_values("date").reset_index(drop=True)

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    X = df[FEATURE_COLS].ffill().fillna(0.0).to_numpy()
    y = df["y_fwd_return"].to_numpy()

    model, model_name = _build_model()
    model.fit(X, y)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{symbol}_h{horizon}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS, "symbol": symbol, "horizon": horizon}, f)

    # Feature importance (LightGBM or sklearn)
    try:
        importances = {col: float(v) for col, v in zip(FEATURE_COLS, model.feature_importances_)}
        top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
    except AttributeError:
        top5 = []

    all_importances = (
        {col: float(v) for col, v in zip(FEATURE_COLS, model.feature_importances_)}
        if hasattr(model, "feature_importances_") else {}
    )

    metrics = {
        "symbol": symbol,
        "horizon": horizon,
        "model": model_name,
        "train_rows": len(df),
        "feature_importances": all_importances,
        "top5_features": top5,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = models_dir / f"{symbol}_h{horizon}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "model": model_name,
        "train_rows": len(df),
        "top5_features": top5,
    }


def train_and_eval(con: duckdb.DuckDBPyConnection, symbol: str, horizon: int, models_dir: Path) -> dict:
    df = con.execute(
        "SELECT * FROM features_daily WHERE symbol=? AND horizon=? ORDER BY date",
        (symbol, int(horizon)),
    ).df()

    if df.empty:
        return {"error": "no features; run ingest + features first"}

    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df.dropna(subset=["y_fwd_return"])

    # Walk-forward evaluation
    print(f"  [train] Walk-forward CV ({symbol}, h={horizon})...")
    fold_results = walk_forward_eval(df, n_folds=5, min_train_rows=252)

    if fold_results:
        mean_rmse = float(np.mean([r["rmse"] for r in fold_results]))
        mean_dir_acc = float(np.mean([r["dir_acc"] for r in fold_results]))
    else:
        mean_rmse = None
        mean_dir_acc = None

    # Train final model on all data
    print(f"  [train] Training final model ({symbol}, h={horizon})...")
    final = train_final(df, symbol=symbol, horizon=horizon, models_dir=models_dir)

    # Persist walk-forward results into metrics JSON
    metrics_path = models_dir / f"{symbol}_h{horizon}_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            saved = json.load(f)
        saved.update({
            "mean_rmse": mean_rmse,
            "mean_dir_acc": mean_dir_acc,
            "walk_forward_folds": fold_results,
            "total_rows": len(df),
        })
        with open(metrics_path, "w") as f:
            json.dump(saved, f, indent=2)

    return {
        "symbol": symbol,
        "horizon": int(horizon),
        "total_rows": len(df),
        "walk_forward_folds": fold_results,
        "mean_rmse": mean_rmse,
        "mean_dir_acc": mean_dir_acc,
        **final,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/market_mvp.duckdb")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--models-dir", default="models")
    ns = ap.parse_args()

    db = DB(path=Path(ns.db))
    con = db.connect()
    init_db(con)
    res = train_and_eval(con, ns.symbol, ns.horizon, models_dir=Path(ns.models_dir))
    con.close()

    print(f"\n=== {res['symbol']} h={res['horizon']} ===")
    if "error" in res:
        print(f"ERROR: {res['error']}")
    else:
        print(f"Total rows:     {res['total_rows']}")
        print(f"Mean RMSE:      {res.get('mean_rmse', 'N/A')}")
        print(f"Mean Dir Acc:   {res.get('mean_dir_acc', 'N/A')}")
        print(f"Model saved:    {res.get('model_path')}")
        if res.get("top5_features"):
            print("Top 5 features:")
            for feat, imp in res["top5_features"]:
                print(f"  {feat}: {imp:.1f}")
        if res.get("walk_forward_folds"):
            print("\nWalk-forward folds:")
            for fold in res["walk_forward_folds"]:
                print(f"  Fold {fold['fold']}: train_end={fold['train_end_date']} "
                      f"rmse={fold['rmse']:.5f} dir_acc={fold['dir_acc']:.3f}")
