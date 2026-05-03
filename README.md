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
| `REDDIT_CLIENT_ID` | No (social only) | Create a free app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) |
| `REDDIT_CLIENT_SECRET` | No (social only) | Same app page as above |
| `OLLAMA_HOST` | No | Pre-set to `spark-1dca.local:11434` — change if your Ollama is elsewhere |
| `OLLAMA_MODEL` | No | Defaults to `qwen2.5:72b` |

> The pipeline works without Reddit and Ollama. Those features are opt-in.

---

## 3. Run the pipeline

Run all commands from the **parent folder** of this repo (i.e. one level up):

```bash
cd /Users/rakeshpillai   # or wherever you cloned this

# Full pipeline (ingest → features → train)
python3 -m market_mvp.pipeline --symbols SPY QQQ

# Skip social (Reddit/StockTwits) if you haven't set Reddit credentials
python3 -m market_mvp.pipeline --symbols SPY QQQ --skip-social

# Add Fed minutes analysis via Ollama (uses deepseek-r1:70b)
python3 -m market_mvp.pipeline --symbols SPY QQQ --skip-social --use-llm-fed

# Get a prediction after training
python3 -m market_mvp.predict --symbol SPY --horizon 5
```

---

## 4. Run individual steps

If you want to run steps separately:

```bash
# 1. Fetch prices, options, and news from Alpha Vantage
python3 -m market_mvp.ingest --symbols SPY QQQ

# 2. Fetch earnings data from EDGAR (free, no key needed)
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

## 5. Open the dashboard

The project includes a Streamlit web UI with four pages: Home, Predictions, Signals, and Performance.

```bash
cd /Users/rakeshpillai/market_mvp
streamlit run app.py
```

This starts a local server at `http://localhost:8501`. Open that URL in any browser.

**Access from iPhone / iPad on the same Wi-Fi:**
1. Find your Mac's local IP address: `ipconfig getifaddr en0`
2. Open `http://<mac-ip>:8501` in Safari on your device

The UI reads from the same `data/market_mvp.duckdb` file and models in `models/`. Run the pipeline first so there is data to display.

---

## 6. Run tests

```bash
cd /Users/rakeshpillai/market_mvp
pytest tests/ -v
```

No API keys needed — all tests use in-memory data. The test suite covers the data pipeline, model training, and all Streamlit pages (143 tests total).

---

## What it predicts

The model predicts the **N-day forward return** for SPY or QQQ. It is a regression model, not a buy/sell signal. Features include:

- **Price**: momentum (1d/5d/20d/60d), volatility, RSI, moving average ratios, volume
- **Options**: put/call ratio, volume-to-OI ratio (via Alpha Vantage)
- **Earnings**: EPS and revenue from SEC EDGAR (top 20 holdings per ETF)
- **Fed**: FOMC hawkish/dovish keyword score, days since last meeting
- **Social**: StockTwits bull ratio, Reddit sentiment (opt-in)
- **News**: Alpha Vantage news sentiment score

---

## Notes

- All data is cached in `data/market_mvp.duckdb` — re-running is safe and won't burn API quota
- Models are saved to `models/` as `.pkl` files
- Alpha Vantage free tier: 25 requests/day — the pipeline caches aggressively to stay within limits
- EDGAR is fully free with no rate limits worth worrying about
