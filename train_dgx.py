#!/usr/bin/env python3
"""TFT training on DGX Spark GPU (run inside NVIDIA NGC PyTorch container).

Run from inside the container:
    python /workspace/market_mvp/train_dgx.py --symbol SPY --horizon 5
    python /workspace/market_mvp/train_dgx.py --symbols SPY QQQ --horizons 5 20

Outputs to /workspace/models/:
    {symbol}_h{horizon}_tft.ckpt         — best checkpoint
    {symbol}_h{horizon}_tft_metrics.json — val loss + training metadata
    {symbol}_h{horizon}_tft_pred.json    — latest prediction (rsync to Mac for Streamlit)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import warnings

import duckdb
import numpy as np
import pandas as pd
import torch

# pytorch-forecasting's EncoderNormalizer passes numpy arrays to a sklearn
# StandardScaler that was fitted on a DataFrame — the names mismatch is harmless.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

DB_PATH = Path("/workspace/data/market_mvp.duckdb")
MODELS_DIR = Path("/workspace/models")

# Must stay in sync with train.py FEATURE_COLS
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

MAX_ENCODER_LENGTH = 60   # 3-month lookback window
MAX_PREDICTION_LENGTH = 1  # one step ahead (= horizon trading days)
MIN_ROWS = MAX_ENCODER_LENGTH * 3  # need enough data for train/val split


def _load_features(symbol: str, horizon: int) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT * FROM features_daily WHERE symbol=? AND horizon=? ORDER BY date",
        (symbol, int(horizon)),
    ).df()
    con.close()
    return df


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["y_fwd_return"]).reset_index(drop=True)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].ffill().fillna(0.0).astype(float)

    # TFT requires contiguous integer time index — trading gaps are handled by
    # allow_missing_timesteps=True in TimeSeriesDataSet
    df["time_idx"] = range(len(df))
    df["symbol"] = df["symbol"].astype(str)
    return df


def _make_datasets(df: pd.DataFrame):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import EncoderNormalizer

    cutoff_idx = int(len(df) * 0.8)
    training_cutoff = int(df["time_idx"].iloc[cutoff_idx])

    dataset_kwargs = dict(
        time_idx="time_idx",
        target="y_fwd_return",
        group_ids=["symbol"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["symbol"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=FEATURE_COLS,
        target_normalizer=EncoderNormalizer(method="standard"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    train_ds = TimeSeriesDataSet(df[df["time_idx"] <= training_cutoff], **dataset_kwargs)
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        df[df["time_idx"] > training_cutoff - MAX_ENCODER_LENGTH],
        predict=True,
        stop_randomization=True,
    )
    return train_ds, val_ds


def train(symbol: str, horizon: int, max_epochs: int, batch_size: int) -> None:
    # Import lightning before pytorch_forecasting so both share the same LightningModule
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    device = "GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device = f"GPU ({torch.cuda.get_device_name(0)})"
    print(f"[train_dgx] Device: {device}")
    print(f"[train_dgx] Training TFT — {symbol} h={horizon}, epochs={max_epochs}")

    df = _load_features(symbol, horizon)
    if df.empty:
        print(f"[train_dgx] No features for {symbol} h{horizon} — run pipeline and rsync first.")
        return

    df = _prep(df)
    print(f"[train_dgx] {len(df)} usable rows.")

    if len(df) < MIN_ROWS:
        print(f"[train_dgx] Skipping {symbol} h{horizon} — need at least {MIN_ROWS} rows, got {len(df)}.")
        return

    train_ds, val_ds = _make_datasets(df)
    train_loader = train_ds.to_dataloader(
        train=True, batch_size=batch_size, num_workers=4, persistent_workers=True
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=4, persistent_workers=True
    )

    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=0.03,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,  # 7 quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss=QuantileLoss(),
        log_interval=20,
        reduce_on_plateau_patience=4,
    )
    print(f"[train_dgx] Parameters: {sum(p.numel() for p in tft.parameters()):,}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_stem = f"{symbol}_h{horizon}_tft"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=str(MODELS_DIR),
            filename=ckpt_stem,
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = callbacks[1].best_model_path
    val_loss = float(trainer.callback_metrics.get("val_loss", float("nan")))
    print(f"[train_dgx] Best checkpoint: {best_ckpt}")
    print(f"[train_dgx] Val loss: {val_loss:.6f}")

    # Save metrics sidecar
    cutoff_idx = int(len(df) * 0.8)
    metrics = {
        "symbol": symbol,
        "horizon": horizon,
        "model": "tft",
        "train_rows": cutoff_idx,
        "val_rows": len(df) - cutoff_idx,
        "val_loss": val_loss,
        "hidden_size": 64,
        "max_encoder_length": MAX_ENCODER_LENGTH,
        "checkpoint": best_ckpt,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODELS_DIR / f"{ckpt_stem}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Run inference on validation set — take last prediction as "today's" signal
    _save_prediction(best_ckpt, val_loader, symbol, horizon, df, val_loss)


def _save_prediction(
    ckpt_path: str,
    val_loader,
    symbol: str,
    horizon: int,
    df: pd.DataFrame,
    val_loss: float,
) -> None:
    from pytorch_forecasting import TemporalFusionTransformer

    try:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
        best_tft.eval()

        # mode="quantiles" → tensor (N_samples, max_pred_len, n_quantiles)
        quantile_preds = best_tft.predict(val_loader, mode="quantiles")
        # Take last batch-sample, first (only) prediction step, median quantile (index 3)
        pred_median = float(quantile_preds[-1, 0, 3].cpu().numpy())
        pred_p10 = float(quantile_preds[-1, 0, 1].cpu().numpy())
        pred_p90 = float(quantile_preds[-1, 0, 5].cpu().numpy())

        as_of = str(df.iloc[-1]["date"].date())
        pred_json = {
            "symbol": symbol,
            "horizon": horizon,
            "as_of_date": as_of,
            "predicted_return": pred_median,
            "p10": pred_p10,
            "p90": pred_p90,
            "direction": "UP" if pred_median > 0 else "DOWN",
            "model": "tft",
            "val_loss": val_loss,
            "checkpoint": ckpt_path,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        pred_path = MODELS_DIR / f"{symbol}_h{horizon}_tft_pred.json"
        with open(pred_path, "w") as f:
            json.dump(pred_json, f, indent=2)

        print(f"[train_dgx] {symbol} h={horizon}: {pred_json['direction']} "
              f"({pred_median:+.4f}, p10={pred_p10:+.4f}, p90={pred_p90:+.4f})")
        print(f"[train_dgx] Prediction saved → {pred_path}")

    except Exception as e:
        print(f"[train_dgx] Inference skipped ({e}) — checkpoint still saved.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train TFT on DGX Spark GPU.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--symbol", help="Single symbol (use with --horizon)")
    group.add_argument("--symbols", nargs="+", help="Multiple symbols")
    ap.add_argument("--horizon", type=int, default=5, help="Single horizon (used with --symbol)")
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ns = ap.parse_args()

    if ns.symbol:
        symbols = [ns.symbol]
        horizons = [ns.horizon]
    else:
        symbols = ns.symbols or ["SPY", "QQQ", "VXUS", "XSD", "XLK"]
        horizons = ns.horizons

    for sym in symbols:
        for h in horizons:
            train(sym, h, ns.epochs, ns.batch_size)
