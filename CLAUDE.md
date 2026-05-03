# market_mvp — CLAUDE.md

Project context for Claude Code. Read this before touching any file.

## What this is

A hobbyist ML pipeline that turns price, options, earnings, Fed minutes, and social
sentiment into daily features for SPY/QQQ, then trains a walk-forward LightGBM model
to predict N-day forward returns. No live trading. No external DB. Everything lives in
a single DuckDB file.

## Repo layout

```
market_mvp/          ← Python package (run from parent dir)
  alpha_vantage.py   ← throttled Alpha Vantage HTTP client
  db.py              ← DuckDB schema + connection helper
  ingest.py          ← AV ingestion: prices, options PCR/VOI, news sentiment
  edgar.py           ← EDGAR XBRL fast-path: EPS + revenue for top ETF holdings
  fed.py             ← FOMC calendar scraper + hawkish/dovish keyword scoring + LLM
  social.py          ← StockTwits (free API) + Reddit (PRAW) daily sentiment
  llm.py             ← Ollama client via OpenAI-compatible endpoint
  features.py        ← Feature engineering: all sources → features_daily table
  train.py           ← LightGBM, walk-forward CV (5 folds), model persistence
  predict.py         ← Load saved model, score latest features row
  pipeline.py        ← Single orchestrator: runs all 6 steps end-to-end
  ui_data.py         ← Shared Streamlit data layer (cached queries, model loading)
  tests/             ← pytest suite (no network calls, in-memory DuckDB)
  data/              ← DuckDB file lives here (gitignored)
  models/            ← Saved .pkl models + metrics JSON (gitignored)
  requirements.txt
  .env.example

app.py               ← Streamlit home page (pipeline status, model summary cards)
pages/
  1_Predictions.py   ← Big UP/DOWN card, feature importance, feature snapshot
  2_Signals.py       ← Candlestick, RSI, momentum, options, news, Fed, social
  3_Performance.py   ← Walk-forward fold charts, feature importance, actual vs predicted
```

## Data flow

```
Alpha Vantage API  ──→  ingest.py  ──→  DuckDB: prices_daily, options_*, news_sentiment
EDGAR data.sec.gov ──→  edgar.py   ──→  DuckDB: edgar_earnings, edgar_cik_map
federalreserve.gov ──→  fed.py     ──→  DuckDB: fed_minutes
StockTwits / Reddit──→  social.py  ──→  DuckDB: social_sentiment_daily
                                                  │
                                             features.py
                                                  │
                                          DuckDB: features_daily  (21 feature cols + label)
                                                  │
                                             train.py
                                                  │
                                         models/SPY_h5.pkl + SPY_h5_metrics.json
                                                  │
                                     ┌────────────┴────────────┐
                                predict.py  →  CLI output     ui_data.py  →  Streamlit UI
```

## Running the pipeline

Always run from the **parent** of this directory (`/Users/rakeshpillai/`):

```bash
cd /Users/rakeshpillai

# Full pipeline
python3 -m market_mvp.pipeline --symbols SPY QQQ

# With LLM-powered Fed minutes analysis (uses deepseek-r1:70b on DGX Spark)
python3 -m market_mvp.pipeline --symbols SPY QQQ --use-llm-fed

# Fast test run — skip slow steps
python3 -m market_mvp.pipeline --symbols SPY --skip-social --skip-fed --skip-edgar

# Individual steps
python3 -m market_mvp.ingest   --symbols SPY QQQ
python3 -m market_mvp.edgar    --etf SPY QQQ
python3 -m market_mvp.fed      --use-llm
python3 -m market_mvp.social   --symbols SPY QQQ
python3 -m market_mvp.features --symbols SPY QQQ --horizons 5 20
python3 -m market_mvp.train    --symbol SPY --horizon 5
python3 -m market_mvp.predict  --symbol SPY --horizon 5
```

## Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Default | Notes |
|---|---|---|---|
| `ALPHAVANTAGE_API_KEY` | Yes | — | Free tier: 25 req/day |
| `OLLAMA_HOST` | No | `http://spark-1dca.local:11434` | DGX Spark on local WiFi |
| `OLLAMA_MODEL` | No | `qwen2.5:72b` | For JSON extraction tasks |
| `OLLAMA_MODEL_REASONING` | No | `deepseek-r1:70b` | For Fed minutes analysis |
| `REDDIT_CLIENT_ID` | No | — | Free app at reddit.com/prefs/apps |
| `REDDIT_CLIENT_SECRET` | No | — | |
| `REDDIT_USER_AGENT` | No | `market_mvp/1.0` | |

## DGX Spark / Ollama

The project uses a local DGX Spark connected via direct 10GbE Ethernet at `10.0.0.2`.
Sub-millisecond latency. The Ollama client in `llm.py` uses:
- 120s timeout on all requests
- Exponential backoff retry (up to 3 attempts)
- `qwen3.6:latest` for structured JSON extraction (36B, faster than qwen2.5:72b)
- `deepseek-r1:70b` for Fed minutes (chain-of-thought reasoning)

If the DGX is offline, set `OLLAMA_HOST` to any other Ollama instance.
EDGAR and Fed keyword scoring work fully offline.

## Database schema (key tables)

| Table | PK | Purpose |
|---|---|---|
| `api_cache` | provider, endpoint, symbol, params_json | Dedup Alpha Vantage API calls |
| `prices_daily` | symbol, date | OHLCV + adjusted close |
| `options_pcr_daily` | symbol, date | Put/call ratio |
| `options_voi_daily` | symbol, date | Volume-to-OI ratio |
| `news_sentiment` | symbol, time_published, url | AV news feed |
| `edgar_earnings` | symbol, period_end | EPS + revenue from XBRL |
| `fed_minutes` | meeting_date, document_type | FOMC statements + minutes |
| `social_sentiment_daily` | symbol, date, source | StockTwits + Reddit |
| `features_daily` | symbol, date, horizon | All features + forward return label |

## Data sources

| Source | What | Free? | Key needed? |
|---|---|---|---|
| yfinance | Daily OHLCV + adjusted close | Yes | No |
| Alpha Vantage | News sentiment | Yes (25 req/day) | Yes |
| SEC EDGAR | EPS/revenue (top 100 holdings per ETF) | Yes | No |
| CBOE | Put/call ratio (planned) | Yes | No |
| federalreserve.gov | FOMC statements + minutes | Yes | No |
| StockTwits | Bull/bear ratio | Yes | No |
| Reddit (PRAW) | Sentiment | Yes | Yes (registration required) |
| Ollama (DGX Spark) | LLM extraction + Fed analysis | Yes (self-hosted) | No |

## Feature columns (21 total in FEATURE_COLS)

Options: `pcr`, `pcr_chg_5`, `voi`, `voi_chg_5`
News: `news_sent`, `news_sent_chg_5`, `news_count`
Dividends: `div_ex_days`, `div_amount`
Price momentum: `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d`
Volatility: `vol_10d`, `vol_20d`
MA ratios: `price_ma5_ratio`, `price_ma20_ratio`, `price_ma60_ratio`
Technical: `rsi_14`, `vol_ratio_20d`, `intraday_range`
Earnings: `eps_surprise_pct`, `rev_surprise_pct`, `days_to_earnings`
Fed: `fed_hawkish_score`, `fed_net_score`, `fed_days_since`
Social: `stocktwits_bull_ratio`, `reddit_sentiment`, `social_volume_ratio`

## Code conventions

- All modules are importable as `market_mvp.<module>` (run from parent dir).
- DuckDB connections are **not** thread-safe; open one per process.
- API responses are always cached in `api_cache` before parsing. Never parse live responses without caching first.
- `INSERT OR REPLACE` is used everywhere — reruns are safe.
- Feature columns that don't exist yet in the DB (e.g., social not yet ingested) are filled with 0.0 before model training. Do not drop rows for missing features.
- The `FEATURE_COLS` list in `train.py` is the single source of truth for feature order. `features.py` must write exactly these columns to `features_daily`.
- Never hardcode API keys. Always read from env vars.
- The `data/` and `models/` directories are gitignored. Never commit `.duckdb` or `.pkl` files.

## Streamlit UI

Launch with PYTHONPATH set (required — package lives one level above the repo):
```bash
PYTHONPATH=/Users/rakeshpillai streamlit run /Users/rakeshpillai/market_mvp/app.py
```

**Architecture:** `ui_data.py` is the single shared data layer imported by all pages.
- Uses `@st.cache_resource` for the DB connection (one read-only connection per process)
- Uses `@st.cache_data(ttl=300)` for query results (5-minute cache)
- DB opened read-only: `duckdb.connect(str(_DB_PATH), read_only=True)`
- DB path: `Path(__file__).parent.parent / "data" / "market_mvp.duckdb"` (one level above repo)
- Model path: `_MODELS_DIR / f"{symbol}_h{horizon}.pkl"` (also one level above repo)
- Metrics sidecar: `_MODELS_DIR / f"{symbol}_h{horizon}_metrics.json"`

`predict()` in `ui_data.py` calls `load_model()` then scores the latest row from `features_daily`.

**iOS access:** Open `http://<mac-local-ip>:8501` in Safari on the same Wi-Fi.
Find your Mac IP with: `ipconfig getifaddr en0`

## Testing

```bash
cd /Users/rakeshpillai/market_mvp
pytest tests/ -v
```

143 tests total. All use in-memory DuckDB. No network calls. No API keys needed.

Key test files:
- `tests/conftest.py` — shared fixtures (`con`, `loaded_features`, etc.)
- `tests/test_ui_data.py` — 32 tests for `ui_data.py` (patches `_con()`, `_DB_PATH`, `_MODELS_DIR`)
- `tests/test_ui_pages.py` — 21 tests using `streamlit.testing.v1.AppTest` (patches `ui_data.*`)

Cache clearing: `test_ui_data.py` has an `autouse` fixture that calls `fn.clear()` on all
`@st.cache_data` functions between tests so patches take effect.

## Adding a new data source

1. Create `market_mvp/mysource.py` with `ingest_mysource(con, ...)` and `build_mysource_features(con, ...) -> pd.DataFrame`
2. Add new table(s) to `db.py` SCHEMA_SQL
3. Add new feature columns to `features_daily` schema in `db.py`
4. Join the new features in `features.py:build_features()`
5. Add feature column names to `FEATURE_COLS` in `train.py`
6. Add a `--skip-mysource` flag and call in `pipeline.py`
7. Write tests in `tests/test_mysource.py`
