#!/bin/bash
# Daily market_mvp pipeline — runs ingestion, rebuilds features, retrains models.
# Designed to run at 7am ET via launchd. Stops Streamlit, runs pipeline, restarts.

set -eo pipefail

REPO="/Users/rakeshpillai/market_mvp"
PARENT="/Users/rakeshpillai"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/pipeline_$(date +%Y%m%d).log"
STREAMLIT_PID_FILE="$LOG_DIR/streamlit.pid"
PYTHON="/opt/homebrew/opt/python@3.14/bin/python3.14"

mkdir -p "$LOG_DIR"

# Always restart Streamlit on exit, even if the pipeline errors.
_restart_streamlit() {
    echo "[cron] Restarting Streamlit..." | tee -a "$LOG"
    nohup env PYTHONPATH="$PARENT" "$PYTHON" -m streamlit run "$REPO/app.py" \
        --server.port 8501 \
        --server.headless true \
        >> "$LOG_DIR/streamlit.log" 2>&1 &
    local pid=$!
    disown "$pid"
    echo "$pid" > "$STREAMLIT_PID_FILE"
    echo "[cron] Streamlit started (PID $pid)" | tee -a "$LOG"
}
trap _restart_streamlit EXIT

echo "=== Daily pipeline started: $(date) ===" | tee -a "$LOG"

# Stop Streamlit if running — kill all streamlit processes to release DB lock
echo "[cron] Stopping Streamlit..." | tee -a "$LOG"
pkill -f "streamlit" 2>/dev/null || true
sleep 2
pkill -9 -f "streamlit" 2>/dev/null || true  # Force-kill any stragglers

# Wait until DuckDB lock file is gone (up to 30s)
DB_LOCK="/Users/rakeshpillai/data/market_mvp.duckdb.lock"
for i in $(seq 1 15); do
    [ ! -f "$DB_LOCK" ] && break
    echo "[cron] Waiting for DuckDB lock to release ($i/15)..." | tee -a "$LOG"
    sleep 2
done

# Read symbols from config (falls back to SPY QQQ if config is missing or python fails)
SYMBOLS=$(python3 -c "import json,pathlib; c=json.loads(pathlib.Path('/Users/rakeshpillai/market_mvp/config/symbols.json').read_text()); print(' '.join(s['ticker'] for s in c['symbols']))" 2>/dev/null || echo "SPY QQQ")

# Run pipeline
cd "$PARENT"
echo "[cron] Running pipeline for: $SYMBOLS..." | tee -a "$LOG"
PYTHONPATH="$PARENT" /opt/homebrew/opt/python@3.14/bin/python3.14 \
    -m market_mvp.pipeline \
    --symbols $SYMBOLS \
    --skip-social \
    2>&1 | tee -a "$LOG"

echo "[cron] Pipeline complete: $(date)" | tee -a "$LOG"

# Kick off TFT training on DGX in background (rsync data → train → rsync models back)
# Streamlit restarts immediately; TFT completes in the background and updates pred JSONs.
echo "[cron] Starting TFT training in background..." | tee -a "$LOG"
bash "$REPO/scripts/daily_tft.sh" >> "$LOG_DIR/tft_$(date +%Y%m%d).log" 2>&1 &
echo "[cron] TFT PID: $!" | tee -a "$LOG"

echo "=== Done: $(date) ===" | tee -a "$LOG"
# trap _restart_streamlit fires here on EXIT
