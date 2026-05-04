# market_mvp

A hobbyist ML pipeline for SPY/QQQ/VXUS/XSD/XLK: pulls price, options, earnings, Fed minutes, and social sentiment into a DuckDB database, trains a LightGBM model locally and a Temporal Fusion Transformer on the DGX Spark GPU. The Streamlit dashboard shows both model predictions side by side.

---

## 1. Install dependencies (Mac Studio)

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
| `REDDIT_CLIENT_ID` | No | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) — requires API registration |
| `REDDIT_CLIENT_SECRET` | No | Same app page |
| `OLLAMA_HOST` | No | Defaults to `http://10.0.0.2:11434` (DGX Spark direct Ethernet) |
| `OLLAMA_MODEL` | No | Defaults to `qwen3.6:latest` |
| `OLLAMA_MODEL_REASONING` | No | Defaults to `deepseek-r1:70b` (Fed minutes) |

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

# Full pipeline
python3 -m market_mvp.pipeline --symbols SPY QQQ VXUS XSD XLK --skip-social

# With Fed LLM analysis
python3 -m market_mvp.pipeline --symbols SPY QQQ VXUS XSD XLK --skip-social --use-llm-fed

# EDGAR + features + train only (skip ingestion)
python3 -m market_mvp.pipeline --symbols SPY QQQ VXUS XSD XLK --skip-social --skip-news --skip-options --skip-fed

# CLI prediction
python3 -m market_mvp.predict --symbol SPY --horizon 5
```

Data: `../data/market_mvp.duckdb` — Models: `../models/`

---

## 5. Open the dashboard (Mac Studio)

```bash
PYTHONPATH=/Users/rakeshpillai /opt/anaconda3/bin/streamlit run /Users/rakeshpillai/market_mvp/app.py
```

Opens at `http://localhost:8501`. Also accessible remotely via **Tailscale** at `http://<tailscale-mac-ip>:8501`.

**iPhone / iPad (same Wi-Fi):** `ipconfig getifaddr en0` → open `http://<mac-ip>:8501` in Safari.

> Stop Streamlit before running the pipeline. Restart it after.

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

## 7. DGX Spark GPU setup + TFT training

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

## 8. Run tests

```bash
cd /Users/rakeshpillai/market_mvp
pytest tests/ -v
```

143 tests, no API keys needed, all in-memory.

---

## What it predicts

Predicts the **N-day forward return** for SPY, QQQ, VXUS, XSD, XLK. Regression model — not a buy/sell signal.

Features:
- **Price**: momentum (1d/5d/20d/60d), volatility, RSI, MA ratios, volume
- **Options**: put/call ratio (computed daily from yfinance options chains)
- **Earnings**: EPS/revenue surprises from SEC EDGAR (top 100 holdings per ETF)
- **Fed**: FOMC hawkish/dovish keyword score, days since last meeting
- **Social**: StockTwits bull ratio, Reddit sentiment (opt-in)
- **News**: Alpha Vantage news sentiment (free tier)

---

## Hardware setup

| Component | Detail |
|---|---|
| Mac Studio | Pipeline orchestration, Streamlit UI, Ollama client |
| DGX Spark | NVIDIA GB10 Grace Blackwell, CUDA 13.2, GPU model training |
| Connection | Direct 10GbE Ethernet (Mac `10.0.0.1` ↔ DGX `10.0.0.2`) |
| Ollama | `qwen3.6:latest` (extraction), `deepseek-r1:70b` (reasoning) |
| Remote access | Tailscale for dashboard access outside home network |

---

## Notes

- Price data: **yfinance** (free, no key, proper adjusted closes)
- DB: `../data/market_mvp.duckdb` — always safe to rerun pipeline
- Models: `../models/` as `.pkl` + `_metrics.json`
- Alpha Vantage free tier: 25 req/day — used only for news sentiment
- EDGAR: fully free, no rate limits worth worrying about
- DGX Spark connected via direct Ethernet — sub-ms Ollama latency
