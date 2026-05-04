# market_mvp — What This System Does

A plain-English walkthrough of the data pipeline, from raw inputs to final prediction.

---

## Three types of model — and why you need all three

This system uses three different kinds of AI model. They do completely different jobs and cannot substitute for each other.

| Model | Job | Why the others can't do it |
|---|---|---|
| **Claude / Ollama LLM** (`claude-opus-4-7` or `qwen3.6` / `deepseek-r1`) | Read unstructured text (Fed minutes, filings) and produce a number | LightGBM and TFT only accept numbers — they cannot read prose |
| **LightGBM** | Look at 30 numbers per day across years of history and learn what combinations predicted price moves | Claude has no access to this historical dataset and cannot learn statistical patterns from it |
| **TFT (Temporal Fusion Transformer)** | Same as LightGBM but also learns which *time periods* matter — e.g., that a hawkish Fed score from 3 weeks ago is still relevant today | LightGBM treats each day independently; TFT understands sequence and memory across time |

In short: **the LLM converts words into numbers. The prediction models learn from those numbers — and thousands of other numbers — across years of market history.** Claude could not replace LightGBM or TFT even with access to the same data, because it has no mechanism to run walk-forward training, learn from past prediction errors, or be evaluated on held-out folds.

---

## What it predicts

For any stock or ETF you track, the system answers one question every day:

> **"What will this ticker's price return be over the next N trading days?"**

N is the prediction horizon — currently 5 days (one week) and 20 days (one month). The output is a percentage: e.g., "+2.3% over the next 5 days." It is a regression estimate, not a buy/sell signal.

---

## Step 1 — Pull price data (yfinance, free)

**What:** Daily open, high, low, close, and volume for every tracked ticker going back several years. Prices are adjusted for stock splits and dividends automatically.

**Source:** Yahoo Finance via the `yfinance` Python library. No API key needed. Runs daily.

**What it produces:** A table of daily price bars per ticker stored in the local database.

---

## Step 2 — Pull options market data (yfinance, free)

**What:** Two signals derived from the options market:
- **Put/call ratio (PCR):** the ratio of puts traded to calls traded. A high ratio means more investors are buying downside protection — often a bearish signal.
- **Volume-to-open-interest ratio (VOI):** how much of today's options trading is new activity vs. existing positions. High VOI suggests unusual directional conviction.

**Source:** Yahoo Finance options chains, fetched and computed daily.

**What it produces:** Two daily values per ticker added to the database.

---

## Step 3 — Pull earnings data (SEC EDGAR, free)

**What:** Actual vs. expected earnings per share (EPS) and revenue for each company. The system calculates the "surprise" — how much better or worse the result was versus analyst expectations. It also tracks how many days until the next earnings announcement.

**Source:** SEC EDGAR XBRL filings (the official US government financial disclosure database). No API key needed, no cost.

**How it handles ETFs:** For ETFs like SPY or QQQ, the system looks up the ETF's top holdings and pulls earnings for each of those companies, then aggregates them.

**What it produces:** EPS surprise %, revenue surprise %, and days-to-earnings per ticker stored in the database.

---

## Step 4 — Pull Fed / FOMC data (federalreserve.gov, free)

**What:** FOMC meeting statements and minutes from the Federal Reserve website. The system scores each document for "hawkish" language (rate hikes, inflation concerns, tightening) vs. "dovish" language (rate cuts, growth support, easing). It also tracks how many days have passed since the last Fed meeting.

**Source:** federalreserve.gov — scraped directly. No API key needed.

**Keyword scoring (always on):** The system scans every document for a curated list of hawkish words ("inflation", "tighten", "hike") and dovish words ("accommodate", "ease", "support") and produces a net score. This runs fully offline with no AI needed.

**Optional LLM enhancement (Claude or Ollama):** If enabled, an AI language model reads the full document and produces a richer interpretation — not just word counts but an understanding of hedged language, context, and emphasis. For example, "we remain attentive to inflation risks" scores differently from "inflation is well under control." The LLM turns this nuance into a single refined score that feeds into the same feature column. Claude (Anthropic's API) or a local Ollama model on the DGX Spark can be used interchangeably — same output either way.

**What it produces:** A hawkish score, a net score (hawkish minus dovish), and days-since-last-meeting per date, stored in the database. These are plain numbers — the prediction models never see the raw text.

---

## Step 5 — Pull news sentiment (Alpha Vantage, free tier)

**What:** Financial news articles mentioning each ticker. Each article is pre-scored for sentiment (positive, negative, neutral) by Alpha Vantage's API. The system stores the average daily sentiment and how many articles were published.

**Source:** Alpha Vantage news sentiment API. Free tier gives 25 requests per day — enough for a small set of tickers.

**What it produces:** Average news sentiment score and article count per ticker per day.

---

## Step 6 — Pull social sentiment (StockTwits + Reddit, optional)

**What:** Crowd sentiment from retail traders. StockTwits gives a bull/bear ratio (what fraction of people posting about a ticker are bullish). Reddit gives a sentiment score from relevant investing subreddits.

**Source:** StockTwits free API (no key needed). Reddit via the PRAW library (free, requires a free app registration). This step is opt-in — the rest of the system works fine without it.

**What it produces:** Bull ratio, Reddit sentiment score, and social volume ratio per ticker per day.

---

## Role of Claude / Ollama — text-to-number conversion

Both Claude (Anthropic's API) and Ollama (local self-hosted models like `qwen3.6` for extraction and `deepseek-r1` for reasoning) do exactly the same job in this system. They are interchangeable — you switch between them with one environment variable (`LLM_PROVIDER=claude` or `LLM_PROVIDER=ollama`). The prediction models don't know or care which one was used.

They are **data preprocessing tools**, not prediction models. Their only job is to read unstructured text and turn it into a number that the prediction models can use.

Specifically they are used in two places:

1. **Fed minutes scoring (Step 4):** Reading FOMC documents and producing a more nuanced hawkish/dovish score than keyword counting alone. Ollama uses `deepseek-r1:70b` (a reasoning model) for this; Claude uses adaptive thinking mode. Same output either way.
2. **Structured extraction:** Parsing semi-structured text in documents where the data isn't cleanly formatted (e.g., extracting a figure buried in a paragraph). Ollama uses `qwen3.6:latest` with chain-of-thought disabled for speed; Claude uses `claude-opus-4-7`.

Once a document has been processed, the LLM output is stored as a plain number in the database. The prediction models (LightGBM, TFT) never see the original text — they only ever see numbers.

**Why can't Claude just predict the stock price directly?**

Several reasons:

- **No historical training data.** Claude hasn't seen years of (features → actual return) pairs for SPY or QQQ. LightGBM and TFT are trained on exactly this data — they learn which signal combinations actually correlated with price moves over thousands of trading days.
- **No walk-forward validation.** The models are evaluated on data they've never seen (held-out folds). Claude has no equivalent mechanism — asking it the same question twice can give different answers.
- **No numerical pattern recognition at scale.** Combining 30 signals across hundreds of tickers and thousands of days and finding the statistical relationships between them is what gradient boosting and deep learning are specifically designed for.
- **Cost and speed.** Running an LLM call for every (ticker, horizon, day) row across years of history would be extremely slow and expensive. LightGBM scores 10,000 rows in milliseconds.

Claude's role is narrow but important: it makes one or two of those 30 features *better* than they would be from keyword matching alone.

---

## Step 7 — Build features (feature engineering)

**What:** Combines all the raw data above into 30 numerical signals ("features") for each ticker, for each day. These are the inputs the model learns from.

The 30 features are:

| Category | Signals |
|---|---|
| Price momentum | 1-day, 5-day, 20-day, 60-day return |
| Volatility | 10-day and 20-day rolling volatility |
| Moving averages | Price vs. 5-day, 20-day, 60-day moving average |
| Technical | RSI (14-day), intraday range, volume ratio |
| Options | Put/call ratio, PCR 5-day change, VOI, VOI 5-day change |
| Earnings | EPS surprise %, revenue surprise %, days to next earnings |
| Fed | Hawkish score, net score, days since last FOMC meeting |
| News | Sentiment score, 5-day change in sentiment, article count |
| Dividends | Days to ex-dividend date, dividend amount |
| Social | StockTwits bull ratio, Reddit sentiment, social volume ratio |

It also computes the **label** — the actual forward return over the next N days — which is what the model is trained to predict.

**What it produces:** One row per (ticker, date, horizon) in the `features_daily` table, with all 30 features and the forward return label.

---

## Step 8 — Train the LightGBM model (Mac Studio, daily)

**What:** A gradient-boosted decision tree model (LightGBM) is trained on the historical feature rows. Training uses **walk-forward cross-validation**: the model is trained on older data and tested on more recent data, simulating how it would have performed in real time. Five folds are used.

**Why this model:** Fast to train, works well on tabular data with mixed signal quality, handles missing values gracefully, and produces good feature importance rankings.

**Output:**
- `{ticker}_h{N}.pkl` — the trained model file
- `{ticker}_h{N}_metrics.json` — accuracy metrics: RMSE, R², directional accuracy per fold

Training runs on the Mac Studio every morning after data ingestion.

---

## Step 9 — Train the TFT model (DGX Spark GPU, daily)

**What:** A Temporal Fusion Transformer (TFT) is a deep learning model that understands time series data — it pays attention to which time periods and which features mattered most for each prediction. It is trained on the same features but uses a 60-day look-back window to capture longer patterns.

**Why a second model:** The TFT can learn non-linear temporal patterns the tree model misses. Showing both side by side gives a cross-check.

**Where it runs:** On the DGX Spark (a local NVIDIA Grace Blackwell GPU server). It runs inside a Docker container because the DGX uses an ARM chip for which pre-built PyTorch binaries don't exist — the NVIDIA container handles this automatically.

**Workflow:** After the Mac finishes LightGBM training, it syncs the database to the DGX, runs TFT training there, then syncs the prediction file back.

**Output:**
- `{ticker}_h{N}_tft.ckpt` — the trained model checkpoint (stays on the DGX)
- `{ticker}_h{N}_tft_pred.json` — the latest prediction: median return + 10th/90th percentile confidence interval (synced back to the Mac)

The Mac dashboard reads only the JSON file — no GPU or PyTorch needed on the Mac.

---

## Step 10 — Predict

**What:** Given the latest row of features (today's signals), both models score it and produce a prediction.

- **LightGBM:** Loads the `.pkl` model, scores the feature row, outputs a single predicted return.
- **TFT:** Reads the pre-computed `_tft_pred.json` — median prediction plus 80% confidence interval (p10 to p90).

The Streamlit dashboard shows both predictions side by side so you can see whether they agree.

---

## Step 11 — Claude analyst commentary (opt-in, Claude only)

**What:** An optional step that runs after model training. A Claude agent is given six tools to query the live database and model outputs, then writes a 2-4 sentence qualitative market commentary.

The six tools Claude can call:
- `get_prediction` — LightGBM and TFT predicted returns, directions, confidence intervals, and accuracy metrics
- `get_features` — today's full 30-feature snapshot
- `get_price_history` — recent daily prices and volume
- `get_fed_score` — latest hawkish/dovish score and days since last FOMC meeting
- `get_news_sentiment` — recent news sentiment trend
- `get_options_data` — recent put/call ratio and volume-to-OI ratio

Claude decides which tools to call, calls them in sequence, inspects the results, and synthesises a plain-English commentary — for example:

> "SPY's LightGBM and TFT models both point to +1.8% over 5 days and are in close agreement (+1.6% to +2.0%). The Fed score is modestly hawkish but has been stable for 12 days, and news sentiment ticked up over the past week — both consistent with the bullish model signal. The only diverging note is the put/call ratio, which rose sharply in the last two sessions, suggesting some hedging activity worth watching."

**Why Claude and not Ollama?** Ollama's models don't have reliable structured tool use. Claude's tool-use API is the mechanism that lets the agent *decide what to look at* rather than being handed everything at once.

**Output:** Saved to `{ticker}_h{N}_analyst.json`. The Predictions page in the dashboard displays it automatically when present. If no file exists, the page shows the command to generate it.

**How to run:**
```bash
LLM_PROVIDER=claude python3 -m market_mvp.pipeline --symbols SPY --skip-social --analyst
```

---

## The Streamlit dashboard

A local web UI accessible in a browser (and on your iPhone/iPad over Wi-Fi). Five sections on the Predictions page:

| Page | What you see |
|---|---|
| **Home** | Pipeline status for all tracked tickers; LightGBM + TFT model cards |
| **Manage Tickers** | Add any stock or ETF — validation, data pull, and training all run automatically in the background |
| **Predictions** | LightGBM and TFT predictions side by side; feature importance; feature snapshot; Claude analyst commentary |
| **Signals** | Candlestick chart, RSI, options data, news sentiment, Fed score, social sentiment over time |
| **Performance** | Walk-forward fold accuracy, feature importance chart, actual vs. predicted scatter |

---

## Automation

A scheduled job runs every morning at 7 AM automatically (launchd on Mac, systemd on AWS):

1. Pull fresh price, options, news, EDGAR, and Fed data
2. Rebuild features for all tracked tickers
3. Retrain LightGBM models
4. Sync data to DGX → retrain TFT → sync predictions back
5. (Optional) Run Claude analyst agent — add `--analyst` to the daily script to enable
6. Dashboard reflects new predictions and commentary on next page load (no restart needed)

---

## What it does NOT do

- **No live trading.** It produces predictions only — no orders, no brokerage connection.
- **No real-time data.** All data is end-of-day. Predictions update once per day.
- **No guarantee of accuracy.** Markets are hard to predict; this is a hobbyist research tool.
- **No external database.** Everything lives in a single local DuckDB file — one file, zero infrastructure.
