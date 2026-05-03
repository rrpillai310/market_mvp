from __future__ import annotations

"""Load a saved model and score the latest available features.

Run:
    python3 -m market_mvp.predict --symbol SPY --horizon 5
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from market_mvp.db import DB, init_db

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_model(models_dir: Path, symbol: str, horizon: int) -> dict:
    path = models_dir / f"{symbol}_h{horizon}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run pipeline.py or train.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_latest(con, symbol: str, horizon: int, models_dir: Path) -> dict:
    bundle = load_model(models_dir, symbol, horizon)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = con.execute(
        "SELECT * FROM features_daily WHERE symbol=? AND horizon=? ORDER BY date DESC LIMIT 1",
        (symbol, int(horizon)),
    ).df()

    if df.empty:
        raise RuntimeError("No features in DB. Run pipeline.py first.")

    row = df.iloc[0]
    as_of_date = str(row["date"])

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_cols].ffill().fillna(0.0).to_numpy()
    pred = float(model.predict(X)[0])

    direction = "UP" if pred > 0 else "DOWN"
    confidence = abs(pred)

    return {
        "symbol": symbol,
        "horizon_days": horizon,
        "as_of_date": as_of_date,
        "predicted_return": round(pred, 6),
        "direction": direction,
        "confidence_proxy": round(confidence, 6),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict forward return for a symbol.")
    ap.add_argument("--db", default="data/market_mvp.duckdb")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--horizon", type=int, default=5)
    ns = ap.parse_args()

    db = DB(path=Path(ns.db))
    con = db.connect()
    init_db(con)

    result = predict_latest(con, ns.symbol, ns.horizon, models_dir=Path(ns.models_dir))
    con.close()

    print(f"\n=== Prediction: {result['symbol']} ({result['horizon_days']}-day horizon) ===")
    print(f"As of:              {result['as_of_date']}")
    print(f"Predicted return:   {result['predicted_return']:+.4%}")
    print(f"Direction:          {result['direction']}")
    print(f"Confidence proxy:   {result['confidence_proxy']:.6f}")
