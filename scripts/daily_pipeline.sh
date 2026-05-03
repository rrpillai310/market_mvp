#!/bin/bash
# Daily market_mvp pipeline — runs ingestion, rebuilds features, retrains models.
# Designed to run at 7am ET via launchd. Stops Streamlit, runs pipeline, restarts.

set -e

REPO="/Users/rakeshpillai/market_mvp"
PARENT="/Users/rakeshpillai"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/pipeline_$(date +%Y%m%d).log"
STREAMLIT_PID_FILE="$LOG_DIR/streamlit.pid"

mkdir -p "$LOG_DIR"

echo "=== Daily pipeline started: $(date) ===" | tee -a "$LOG"

# Stop Streamlit if running
if [ -f "$STREAMLIT_PID_FILE" ]; then
    PID=$(cat "$STREAMLIT_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[cron] Stopping Streamlit (PID $PID)..." | tee -a "$LOG"
        kill "$PID"
        sleep 3
    fi
    rm -f "$STREAMLIT_PID_FILE"
fi

# Run pipeline
cd "$PARENT"
echo "[cron] Running pipeline..." | tee -a "$LOG"
PYTHONPATH="$PARENT" /opt/homebrew/opt/python@3.14/bin/python3.14 \
    -m market_mvp.pipeline \
    --symbols SPY QQQ \
    --skip-social \
    2>&1 | tee -a "$LOG"

echo "[cron] Pipeline complete: $(date)" | tee -a "$LOG"

# Restart Streamlit in background
echo "[cron] Restarting Streamlit..." | tee -a "$LOG"
PYTHONPATH="$PARENT" /opt/anaconda3/bin/streamlit run "$REPO/app.py" \
    --server.port 8501 \
    --server.headless true \
    >> "$LOG_DIR/streamlit.log" 2>&1 &
echo $! > "$STREAMLIT_PID_FILE"
echo "[cron] Streamlit started (PID $(cat $STREAMLIT_PID_FILE))" | tee -a "$LOG"

echo "=== Done: $(date) ===" | tee -a "$LOG"
