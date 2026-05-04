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

set -e

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

# Verify the container is running
if ! ssh "$DGX" "docker ps -q -f name=market_mvp | grep -q ." 2>/dev/null; then
    echo "[tft] Container 'market_mvp' not running on DGX." | tee -a "$LOG"
    echo "[tft] Run the one-time setup in this script's header comment to fix this." | tee -a "$LOG"
    exit 1
fi

# 1. Sync DuckDB to DGX
echo "[tft] Syncing data to DGX ($(du -sh "$DATA_DIR" | cut -f1))..." | tee -a "$LOG"
rsync -az --info=progress2 "$DATA_DIR/" "$DGX:~/data/" 2>&1 | tee -a "$LOG"

# 2. Pull latest code on DGX then train all symbols/horizons
echo "[tft] Starting GPU training on DGX..." | tee -a "$LOG"
ssh "$DGX" "
    git -C ~/market_mvp pull --ff-only --quiet 2>&1 || true
    docker exec market_mvp python /workspace/market_mvp/train_dgx.py \
        --symbols SPY QQQ VXUS XSD XLK \
        --horizons 5 20 \
        2>&1
" 2>&1 | tee -a "$LOG"

# 3. Sync model checkpoints and prediction JSONs back to Mac
echo "[tft] Syncing models back to Mac..." | tee -a "$LOG"
rsync -az --info=progress2 "$DGX:~/models/" "$MODELS_DIR/" 2>&1 | tee -a "$LOG"

echo "=== TFT training done: $(date) ===" | tee -a "$LOG"
