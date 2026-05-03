# market_mvp

A hobbyist ML pipeline for SPY/QQQ: pulls price, options, earnings, Fed minutes, and social sentiment into a DuckDB database, then trains a LightGBM model to predict forward returns.

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

If LightGBM fails to load on macOS (missing `libomp`):
```bash
brew install libomp
```

---

## 2. Add your API keys

Copy the example env file and fill in your keys:
```bash
cp .env.example .env
```

Then open `.env` and set the values:

| Key | Required | How to get it |
|---|---|---|
| `ALPHAVANTAGE_API_KEY` | **Yes** | Free at [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| `REDDIT_CLIENT_ID` | No (social only) | Create a free app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) — requires API registration |
| `REDDIT_CLIENT_SECRET` | No (social only) | Same app page as above |
| `OLLAMA_HOST` | No | Defaults to `http://10.0.0.2:11434` — change to your Ollama host |
| `OLLAMA_MODEL` | No | Defaults to `qwen3.6:latest` (extraction) |
| `OLLAMA_MODEL_REASONING` | No | Defaults to `deepseek-r1:70b` (Fed minutes) |

> The pipeline works without Reddit and Ollama. Those features are opt-in.

---

## 3. Set up PYTHONPATH

The pipeline runs from the **parent folder** of the repo. Add this to `~/.zshrc` once so imports always resolve:

```bash
echo 'export PYTHONPATH=/Users/rakeshpillai:$PYTHONPATH' >> ~/.zshrc
source ~/.zshrc
```

---

## 4. Run the pipeline

Run all commands from the **parent folder** of this repo:

```bash
cd /Users/rakeshpillai

# Full pipeline (ingest → EDGAR → Fed → features → train)
python3 -m market_mvp.pipeline --symbols SPY QQQ

# Skip social (Reddit/StockTwits) if you haven't set Reddit credentials
python3 -m market_mvp.pipeline --symbols SPY QQQ --skip-social

# Add Fed minutes analysis via Ollama (uses deepseek-r1:70b)
python3 -m market_mvp.pipeline --symbols SPY QQQ --skip-social --use-llm-fed

# Rerun just EDGAR + features + train (after expanding holdings list)
python3 -m market_mvp.pipeline --symbols SPY QQQ --skip-social --skip-news --skip-options --skip-fed

# Get a prediction after training
python3 -m market_mvp.predict --symbol SPY --horizon 5
```

Data is written to `../data/market_mvp.duckdb` and models to `../models/` (one level above the repo).

---

## 5. Run individual steps

```bash
cd /Users/rakeshpillai

# 1. Fetch prices via yfinance + news sentiment via Alpha Vantage
python3 -m market_mvp.ingest --symbols SPY QQQ

# 2. Fetch earnings from SEC EDGAR (free, no key — top 100 holdings per ETF)
python3 -m market_mvp.edgar --etf SPY QQQ

# 3. Fetch Fed FOMC minutes (add --use-llm for Ollama summary)
python3 -m market_mvp.fed

# 4. Fetch StockTwits + Reddit sentiment
python3 -m market_mvp.social --symbols SPY QQQ

# 5. Build feature table
python3 -m market_mvp.features --symbols SPY QQQ --horizons 5 20

# 6. Train and evaluate
python3 -m market_mvp.train --symbol SPY --horizon 5
python3 -m market_mvp.train --symbol QQQ --horizon 5
```

---

## 6. Open the dashboard

```bash
PYTHONPATH=/Users/rakeshpillai streamlit run /Users/rakeshpillai/market_mvp/app.py
```

This starts a local server at `http://localhost:8501`.

**Access from iPhone / iPad on the same Wi-Fi:**
1. Find your Mac's local IP: `ipconfig getifaddr en0`
2. Open `http://<mac-ip>:8501` in Safari

---

## Important: pipeline and dashboard can't run at the same time

DuckDB only allows one writer at a time. Stop Streamlit (Ctrl+C) before running the pipeline, then restart it after.

---

## 7. Run tests

```bash
cd /Users/rakeshpillai/market_mvp
pytest tests/ -v
```

143 tests, no API keys needed, all in-memory.

---

## What it predicts

The model predicts the **N-day forward return** for SPY or QQQ. It is a regression model, not a buy/sell signal. Features include:

- **Price**: momentum (1d/5d/20d/60d), volatility, RSI, moving average ratios, volume
- **Options**: put/call ratio, volume-to-OI ratio (premium AV endpoint — zeros until CBOE scraper is added)
- **Earnings**: EPS and revenue surprises from SEC EDGAR (top 100 holdings per ETF)
- **Fed**: FOMC hawkish/dovish keyword score, days since last meeting
- **Social**: StockTwits bull ratio, Reddit sentiment (opt-in, requires Reddit API registration)
- **News**: Alpha Vantage news sentiment score (free tier)

---

## Notes

- Price data comes from **yfinance** (free, no key, proper adjusted closes)
- All data lives in `../data/market_mvp.duckdb` relative to the repo — re-running is always safe
- Models are saved to `../models/` as `.pkl` + `_metrics.json` pairs
- Alpha Vantage free tier: 25 requests/day — used only for news sentiment
- EDGAR is fully free with no meaningful rate limits
- Ollama runs on DGX Spark via direct 10GbE Ethernet at `10.0.0.2` (sub-ms latency)
