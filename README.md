# market_mvp

A ML pipeline for any ticker (ETF or single stock): pulls price, options, earnings, Fed minutes, and social sentiment into a DuckDB database, trains a LightGBM model locally and a Temporal Fusion Transformer on the DGX Spark GPU. The Streamlit dashboard shows both model predictions side by side. Add new symbols via the GUI — the pipeline runs automatically in the background.

Runs on **Mac Studio** or **AWS EC2** — same codebase, configured via env vars.

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

```bash
cp .env.example .env
```

| Key | Required | How to get it |
|---|---|---|
| `ALPHAVANTAGE_API_KEY` | **Yes** | Free at [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| `LLM_PROVIDER` | No | `ollama` (default) or `claude` |
| `ANTHROPIC_API_KEY` | If `LLM_PROVIDER=claude` | [console.anthropic.com](https://console.anthropic.com/) |
| `CLAUDE_MODEL` | No | Claude model to use (default: `claude-opus-4-7`) |
| `OLLAMA_HOST` | No | Defaults to `http://spark-1dca.local:11434` |
| `OLLAMA_MODEL` | No | Defaults to `qwen3.6:latest` |
| `OLLAMA_MODEL_REASONING` | No | Defaults to `deepseek-r1:70b` (Fed minutes) |
| `REDDIT_CLIENT_ID` | No | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) — requires API registration |
| `DATA_DIR` | No | Overrides default `../data/` path (useful on AWS with EBS) |
| `MODELS_DIR` | No | Overrides default `../models/` path |
| `S3_BUCKET` | No | S3 bucket name — enables automatic data/model sync on AWS |

---

## 3. Set up PYTHONPATH (Mac Studio)

```bash
echo 'export PYTHONPATH=/Users/rakeshpillai:$PYTHONPATH' >> ~/.zshrc
source ~/.zshrc
```

---

## 4. Run the pipeline (Mac Studio)

Stop Streamlit first if it's running — DuckDB allows only one writer at a time.

```bash
cd /Users/rakeshpillai

# Full pipeline (Ollama LLM, default)
python3 -m market_mvp.pipeline --symbols SPY QQQ VXUS XSD XLK --skip-social

# With LLM Fed minutes analysis — uses whichever provider LLM_PROVIDER selects
python3 -m market_mvp.pipeline --symbols SPY QQQ VXUS XSD XLK --skip-social --use-llm-fed

# Switch to Claude for LLM calls without changing .env
LLM_PROVIDER=claude python3 -m market_mvp.pipeline --symbols SPY --use-llm-fed

# Claude analyst agent — qualitative commentary after training (Claude only)
LLM_PROVIDER=claude python3 -m market_mvp.pipeline --symbols SPY --skip-social --analyst

# EDGAR + features + train only (skip ingestion)
python3 -m market_mvp.pipeline --symbols SPY QQQ VXUS XSD XLK --skip-social --skip-news --skip-options --skip-fed

# CLI prediction
python3 -m market_mvp.predict --symbol SPY --horizon 5
```

### On-demand Claude analyst commentary

After the pipeline has run, call the analyst agent independently for any symbol — no retraining needed. Reads the existing DB and model outputs, writes `{symbol}_h{horizon}_analyst.json` to the models directory (Streamlit picks it up automatically on the Predictions page).

```bash
# Set your Anthropic key first (one-time):
# console.anthropic.com → API Keys → copy key → add to .env:
#   ANTHROPIC_API_KEY=sk-ant-...

# Single symbol, default horizons (5 and 20 days)
python3 -m market_mvp.analyst --symbols SPY

# Multiple symbols
python3 -m market_mvp.analyst --symbols SPY QQQ NVDA

# Specific horizon only
python3 -m market_mvp.analyst --symbols SPY --horizons 5
```

Data: `../data/market_mvp.duckdb` — Models: `../models/`

---

## 5. Open the dashboard (Mac Studio)

```bash
PYTHONPATH=/Users/rakeshpillai /opt/anaconda3/bin/streamlit run /Users/rakeshpillai/market_mvp/app.py
```

Opens at `http://localhost:8501`. Also accessible remotely via **Tailscale** at `http://<tailscale-mac-ip>:8501`.

**iPhone / iPad (same Wi-Fi):** `ipconfig getifaddr en0` → open `http://<mac-ip>:8501` in Safari.

**Pages:**
- **Home** — pipeline status, LightGBM + TFT model cards for all tracked symbols
- **0 · Manage Tickers** — add any ETF or single stock; validates via yfinance; triggers data pull + feature build + model train in background; auto-refreshes until done
- **1 · Predictions** — LightGBM and TFT predictions side by side; Claude analyst commentary (if `--analyst` was run)
- **2 · Signals** — price chart, RSI, options, news sentiment, Fed, social
- **3 · Performance** — walk-forward fold accuracy, feature importance, actual-vs-predicted scatter

> Stop the dashboard before running the pipeline manually (the daily cron does this automatically). The GUI-triggered pipeline can run while the dashboard is open.

---

## 6. Daily automated pipeline

A launchd job runs the pipeline every day at 7am automatically:

```bash
# Loaded via:
launchctl load ~/Library/LaunchAgents/com.market_mvp.daily.plist

# Run manually:
/Users/rakeshpillai/market_mvp/scripts/daily_pipeline.sh

# Check logs:
cat /Users/rakeshpillai/market_mvp/logs/pipeline_$(date +%Y%m%d).log
```

---

## 7. AWS deployment

The same codebase runs on EC2 without code changes — everything is driven by env vars.

### Bootstrap a new EC2 instance

```bash
# On the EC2 instance (Amazon Linux 2023 or Ubuntu):
curl -sO https://raw.githubusercontent.com/rrpillai310/market_mvp/main/scripts/aws_setup.sh
bash aws_setup.sh
```

The script installs Python, AWS CLI, clones the repo, creates a venv, installs deps, and sets up a systemd timer for the 7 AM daily pipeline.

### Key differences vs Mac

| | Mac Studio | AWS EC2 |
|---|---|---|
| Data dir | `~/data/` (or `DATA_DIR`) | `/data/` (EBS mount recommended) |
| Models dir | `~/models/` (or `MODELS_DIR`) | `/data/models/` |
| LLM | Ollama on DGX (`LLM_PROVIDER=ollama`) | Claude API (`LLM_PROVIDER=claude`) |
| Persistence | Local filesystem | S3 sync via `S3_BUCKET` env var |
| Cron | launchd `.plist` | systemd timer |
| GPU training | DGX Spark via rsync | Optional GPU instance or skip |

### S3 sync for durability

When `S3_BUCKET` is set, the pipeline automatically syncs `data/` and `models/` from S3 before each run and back to S3 after — so data survives EC2 stop/start without a persistent EBS volume.

```bash
# In .env on EC2:
S3_BUCKET=my-market-mvp-bucket
DATA_DIR=/data
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
```

### Run on EC2

```bash
# Activate the venv the setup script created
source ~/.venv/market_mvp/bin/activate

PYTHONPATH=$HOME python3 -m market_mvp.pipeline \
  --symbols SPY QQQ VXUS XSD XLK --skip-social --use-llm-fed

# Launch Streamlit (accessible at http://<ec2-public-ip>:8501)
PYTHONPATH=$HOME streamlit run ~/market_mvp/app.py \
  --server.address 0.0.0.0 --server.port 8501
```

---

## 8. DGX Spark GPU setup + TFT training

The DGX Spark (NVIDIA GB10 Grace Blackwell, CUDA 13.2, aarch64) runs GPU training via Docker.
There are no PyTorch CUDA wheels for aarch64 — always use the container.

**One-time setup on DGX (`ssh rrpillai@10.0.0.2`):**

```bash
# Add user to docker group
sudo usermod -aG docker rrpillai
newgrp docker

# Pull NVIDIA PyTorch container (~20GB, one-time)
docker run --gpus all -it --name market_mvp \
  -v ~/market_mvp:/workspace/market_mvp \
  -v ~/data:/workspace/data \
  -v ~/models:/workspace/models \
  nvcr.io/nvidia/pytorch:25.03-py3

# Inside the container:
pip install pytorch-forecasting pytorch-lightning duckdb
python -c "import torch; print(torch.cuda.is_available())"  # must print True
```

**Start existing container after reboot:**
```bash
docker start -ai market_mvp
```

**SSH key for GitHub (run on DGX):**
```bash
ssh-keygen -t ed25519 -C "dgx-spark" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub   # paste into github.com → Settings → SSH keys
git clone git@github.com:rrpillai310/market_mvp.git
```

**One-time: convert container to a persistent daemon (run on DGX once):**

```bash
# Save your pip-installed packages into a new image, then recreate as always-on daemon
docker commit market_mvp market_mvp:with-deps
docker rm -f market_mvp
docker run --gpus all -d --name market_mvp --restart unless-stopped \
  -v ~/market_mvp:/workspace/market_mvp \
  -v ~/data:/workspace/data \
  -v ~/models:/workspace/models \
  market_mvp:with-deps sleep infinity
```

After this, the container stays running across reboots and `docker exec` works without
any manual intervention — which is what the automated pipeline script needs.

**TFT training runs automatically** via `scripts/daily_tft.sh`, called in the background
by `scripts/daily_pipeline.sh` after the LightGBM pipeline finishes each morning.

To run manually:
```bash
bash /Users/rakeshpillai/market_mvp/scripts/daily_tft.sh

# Or directly on DGX inside the container:
docker exec market_mvp python /workspace/market_mvp/train_dgx.py \
    --symbols SPY QQQ VXUS XSD XLK --horizons 5 20
```

The Streamlit Predictions page shows LightGBM and TFT predictions side by side automatically
once the `_tft_pred.json` files are synced back. No GPU or PyTorch needed on the Mac.

---

## 9. Run tests

```bash
cd /Users/rakeshpillai/market_mvp
pytest tests/ -v
```

180 tests, no API keys needed, all in-memory. 5 skipped on Mac (torch/GPU only).

---

## What it predicts

Predicts the **N-day forward return** for any tracked symbol (ETF or single stock). Regression model — not a buy/sell signal.

**30 features per symbol:**
- **Price**: momentum (1d/5d/20d/60d), volatility (10d/20d), RSI, MA ratios, intraday range, volume ratio
- **Options**: put/call ratio and volume-to-OI ratio (computed daily from yfinance options chains)
- **Earnings**: EPS/revenue surprises + days-to-earnings from SEC EDGAR (ETF holdings or direct for single stocks)
- **Fed**: FOMC hawkish/dovish keyword score + net score + days since last meeting
- **Social**: StockTwits bull ratio, Reddit sentiment, social volume ratio (opt-in)
- **News**: Alpha Vantage news sentiment + 5-day change + article count

**Tracked symbols** (in `config/symbols.json`, editable via GUI):
- ETFs: SPY, QQQ, VXUS, XSD, XLK, XLE
- Single stocks: add any via the Manage Tickers page

---

## Hardware setup

| Component | Detail |
|---|---|
| Mac Studio | Pipeline orchestration, Streamlit UI, LLM client (Ollama or Claude) |
| DGX Spark | NVIDIA GB10 Grace Blackwell, CUDA 13.2, GPU model training |
| Connection | Direct 10GbE Ethernet (Mac `10.0.0.1` ↔ DGX `10.0.0.2`) |
| AWS EC2 | Alternative deployment — same code, `LLM_PROVIDER=claude`, S3 for storage |
| Ollama | `qwen3.6:latest` (extraction), `deepseek-r1:70b` (reasoning) |
| Claude API | `claude-opus-4-7` — used when `LLM_PROVIDER=claude` |
| Remote access | Tailscale for dashboard access outside home network |

---

## Notes

- Price data: **yfinance** (free, no key, proper adjusted closes)
- DB: `../data/market_mvp.duckdb` by default — override with `DATA_DIR` env var
- Models: `../models/` by default — override with `MODELS_DIR` env var
- LLM: switch between Ollama and Claude with `LLM_PROVIDER=ollama|claude`
- Alpha Vantage free tier: 25 req/day — used only for news sentiment
- EDGAR: fully free, no rate limits worth worrying about
- DGX Spark connected via direct Ethernet — sub-ms Ollama latency
- On AWS: set `S3_BUCKET` for automatic data/model persistence across instance stop/start
