# Market MVP (Alpha Vantage + DuckDB + LightGBM)

Goal: a hobbyist-friendly pipeline that turns text + options aggregates into structured daily features for SPY/QQQ, then trains a walk-forward LightGBM model to predict forward returns.

This MVP is designed for Alpha Vantage's low request quotas by:
- caching every API response in DuckDB
- only fetching missing (endpoint, symbol, date-range) slices

## Setup

1. Install dependencies (add these to your environment):
   - `duckdb`, `pandas`, `requests`, `lightgbm`, `scikit-learn`

2. Export your Alpha Vantage key:
```bash
export ALPHAVANTAGE_API_KEY="..."
```

3. Run ingestion (safe to re-run; it will use cached responses):
```bash
python3 -m market_mvp.ingest --symbols SPY QQQ --db data/market_mvp.duckdb
```

4. Build a daily feature table:
```bash
python3 -m market_mvp.features --symbols SPY QQQ --db data/market_mvp.duckdb
```

5. Train/evaluate:
```bash
python3 -m market_mvp.train --symbol SPY --horizon 5 --db data/market_mvp.duckdb
python3 -m market_mvp.train --symbol QQQ --horizon 5 --db data/market_mvp.duckdb
```

## Notes
- This MVP predicts *probabilities / expected returns*; it does not try to "LLM-predict prices" directly.
- Options data is ingested via Alpha Vantage options endpoints (put/call, volume-to-OI). See `market_mvp/ingest.py`.
- You can extend it later with:
  - intraday events
  - per-stock transcripts/news
  - a local LLM extraction layer (store extracted JSON features)

