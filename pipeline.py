from __future__ import annotations

"""Single-entry-point pipeline: ingest → EDGAR → Fed → social → features → train.

Run:
    python3 -m market_mvp.pipeline --symbols SPY QQQ
    python3 -m market_mvp.pipeline --symbols SPY QQQ --skip-social --skip-fed
"""

import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from market_mvp.alpha_vantage import AlphaVantageClient
from market_mvp.db import DB, init_db
from market_mvp.ingest import (
    ingest_daily_adjusted,
    ingest_news_sentiment,
    ingest_options_pcr_yf,
)
from market_mvp.features import build_features
from market_mvp.train import train_and_eval


def run_pipeline(
    *,
    symbols: list[str],
    db_path: Path,
    models_dir: Path,
    horizons: list[int],
    throttle_secs: float,
    skip_options: bool,
    skip_news: bool,
    skip_edgar: bool,
    skip_fed: bool,
    skip_social: bool,
    use_llm_fed: bool,
    news_limit: int,
) -> None:
    db = DB(path=db_path)
    con = db.connect()
    init_db(con)

    av = AlphaVantageClient(throttle_secs=throttle_secs)

    # --- Step 1: Alpha Vantage ingestion ---
    print("\n=== Step 1: Alpha Vantage ingestion ===")
    for sym in symbols:
        print(f"\n[pipeline] {sym}: prices...")
        ingest_daily_adjusted(con, av, sym)
        if not skip_options:
            print(f"[pipeline] {sym}: options PCR (yfinance)...")
            try:
                ingest_options_pcr_yf(con, sym)
            except Exception as e:
                print(f"[pipeline] {sym}: options PCR skipped ({e})")
        if not skip_news:
            print(f"[pipeline] {sym}: news sentiment...")
            ingest_news_sentiment(con, av, sym, limit=news_limit)
        con.execute("CHECKPOINT")

    # --- Step 2: EDGAR earnings ---
    if not skip_edgar:
        print("\n=== Step 2: EDGAR earnings (XBRL fast-path) ===")
        from market_mvp.edgar import ingest_symbol_earnings
        for sym in symbols:
            ingest_symbol_earnings(con, sym)
        con.execute("CHECKPOINT")

    # --- Step 3: Fed minutes ---
    if not skip_fed:
        print("\n=== Step 3: Fed FOMC minutes ===")
        from market_mvp.fed import ingest_fed_minutes
        ingest_fed_minutes(con, use_llm=use_llm_fed)
        con.execute("CHECKPOINT")

    # --- Step 4: Social sentiment ---
    if not skip_social:
        print("\n=== Step 4: Social sentiment (StockTwits + Reddit) ===")
        from market_mvp.social import ingest_stocktwits, ingest_reddit
        for sym in symbols:
            print(f"[pipeline] {sym}: StockTwits...")
            ingest_stocktwits(con, sym)
            print(f"[pipeline] {sym}: Reddit...")
            ingest_reddit(con, sym)
        con.execute("CHECKPOINT")

    # --- Step 5: Build features ---
    print("\n=== Step 5: Feature engineering ===")
    for sym in symbols:
        print(f"[pipeline] {sym}: building features (horizons={horizons})...")
        build_features(con, sym, horizons=horizons)
    con.execute("CHECKPOINT")

    # --- Step 6: Train ---
    print("\n=== Step 6: Model training (walk-forward CV) ===")
    for sym in symbols:
        for h in horizons:
            print(f"\n[pipeline] Training {sym} h={h}...")
            res = train_and_eval(con, sym, h, models_dir=models_dir)
            if "error" in res:
                print(f"  ERROR: {res['error']}")
            else:
                print(f"  Mean RMSE={res.get('mean_rmse', 'N/A'):.5f}  "
                      f"Dir Acc={res.get('mean_dir_acc', 'N/A'):.3f}  "
                      f"Model={res.get('model_path')}")

    con.close()
    print("\n=== Pipeline complete. ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run full market_mvp pipeline.")
    ap.add_argument("--db", default="data/market_mvp.duckdb")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--throttle-secs", type=float, default=12.5)
    ap.add_argument("--skip-options", action="store_true")
    ap.add_argument("--skip-news", action="store_true")
    ap.add_argument("--skip-edgar", action="store_true")
    ap.add_argument("--skip-fed", action="store_true")
    ap.add_argument("--skip-social", action="store_true")
    ap.add_argument("--use-llm-fed", action="store_true", help="Use Ollama (deepseek-r1:70b) to summarize Fed minutes")
    ap.add_argument("--news-limit", type=int, default=200)
    ns = ap.parse_args()

    run_pipeline(
        symbols=ns.symbols,
        db_path=Path(ns.db),
        models_dir=Path(ns.models_dir),
        horizons=list(ns.horizons),
        throttle_secs=ns.throttle_secs,
        skip_options=ns.skip_options,
        skip_news=ns.skip_news,
        skip_edgar=ns.skip_edgar,
        skip_fed=ns.skip_fed,
        skip_social=ns.skip_social,
        use_llm_fed=ns.use_llm_fed,
        news_limit=ns.news_limit,
    )
