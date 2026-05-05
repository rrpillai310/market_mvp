#!/bin/bash
# daily_tft.sh — sync DuckDB to DGX, train TFT, sync models back.
# Called by daily_pipeline.sh in background after LightGBM training completes.
#
# One-time DGX setup required to keep the container alive as a daemon:
#
#   ssh rrpillai@10.0.0.2
#   docker commit market_mvp market_mvp:with-deps
#   docker rm -f market_mvp
#   docker run --gpus all -d --name market_mvp --restart unless-stopped \
#     -v ~/market_mvp:/workspace/market_mvp \
#     -v ~/data:/workspace/data \
#     -v ~/models:/workspace/models \
#     market_mvp:with-deps sleep infinity
#
# After that, the container stays running across reboots (--restart unless-stopped)
# and docker exec works reliably without manual intervention.

set -eo pipefail

REPO="/Users/rakeshpillai/market_mvp"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/tft_$(date +%Y%m%d).log"
DGX="rrpillai@10.0.0.2"
DATA_DIR="/Users/rakeshpillai/data"
MODELS_DIR="/Users/rakeshpillai/models"

mkdir -p "$LOG_DIR"
echo "=== TFT training started: $(date) ===" | tee -a "$LOG"

# Check DGX is reachable before attempting anything
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$DGX" true 2>/dev/null; then
    echo "[tft] DGX not reachable — skipping TFT training." | tee -a "$LOG"
    exit 0
fi

# Verify the container is running — try to start it up to 3 times if stopped
_container_running() {
    ssh "$DGX" "docker ps -q -f name=market_mvp | grep -q ." 2>/dev/null
}
if ! _container_running; then
    echo "[tft] Container 'market_mvp' not running — attempting to start..." | tee -a "$LOG"
    for attempt in 1 2 3; do
        ssh "$DGX" "docker start market_mvp" 2>&1 | tee -a "$LOG"
        sleep 5
        if _container_running; then
            echo "[tft] Container started on attempt $attempt." | tee -a "$LOG"
            break
        fi
        echo "[tft] Attempt $attempt failed." | tee -a "$LOG"
        if [ "$attempt" -eq 3 ]; then
            echo "[tft] Container could not be started after 3 attempts — skipping TFT." | tee -a "$LOG"
            exit 0
        fi
    done
fi

# 1. Sync DuckDB to DGX
echo "[tft] Syncing data to DGX ($(du -sh "$DATA_DIR" | cut -f1))..." | tee -a "$LOG"
ssh "$DGX" "mkdir -p ~/data ~/models" 2>&1 | tee -a "$LOG"
rsync -rz --no-perms --no-owner --no-group "$DATA_DIR/" "$DGX:~/data/" 2>&1 | tee -a "$LOG"

# Read symbols from config (falls back to SPY QQQ VXUS XSD XLK if config is missing)
SYMBOLS=$(python3 -c "import json,pathlib; c=json.loads(pathlib.Path('/Users/rakeshpillai/market_mvp/config/symbols.json').read_text()); print(' '.join(s['ticker'] for s in c['symbols']))" 2>/dev/null || echo "SPY QQQ VXUS XSD XLK")

# 2. Pull latest code on DGX then train all symbols/horizons
echo "[tft] Starting GPU training on DGX for: $SYMBOLS..." | tee -a "$LOG"
ssh "$DGX" "
    git -C ~/market_mvp pull --ff-only --quiet 2>&1 || true
    docker exec market_mvp python /workspace/market_mvp/train_dgx.py \
        --symbols $SYMBOLS \
        --horizons 5 20 \
        2>&1
" 2>&1 | tee -a "$LOG"

# 3. Fix ownership of files written by Docker (root inside container → host user)
echo "[tft] Fixing model file permissions on DGX..." | tee -a "$LOG"
ssh "$DGX" "docker exec market_mvp chmod -R a+r /workspace/models/ 2>/dev/null || true" 2>&1 | tee -a "$LOG"

# 4. Sync model checkpoints and prediction JSONs back to Mac
echo "[tft] Syncing models back to Mac..." | tee -a "$LOG"
rsync -rz --no-perms --no-owner --no-group "$DGX:~/models/" "$MODELS_DIR/" 2>&1 | tee -a "$LOG"

echo "=== TFT training done: $(date) ===" | tee -a "$LOG"
