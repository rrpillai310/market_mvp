#!/usr/bin/env bash
# aws_setup.sh — Bootstrap market_mvp on an Amazon Linux 2023 / Ubuntu EC2 instance.
#
# Usage:
#   ssh ec2-user@<instance-ip>
#   curl -sO https://raw.githubusercontent.com/<you>/market_mvp/main/scripts/aws_setup.sh
#   bash aws_setup.sh
#
# What this does:
#   1. Install system deps (Python 3.11, git, AWS CLI)
#   2. Clone the repo (or pull if already present)
#   3. Install Python deps
#   4. Create .env from .env.example (edit it afterward)
#   5. Create data/ and models/ directories (or mount EBS — see notes)
#   6. Install a systemd cron unit that runs the pipeline daily at 7 AM
#
# Paths on EC2:
#   Repo:    ~/market_mvp/
#   Data:    ~/data/          (or set DATA_DIR to a mounted EBS volume, e.g. /data)
#   Models:  ~/models/        (or set MODELS_DIR=/models)
#   Logs:    ~/market_mvp/logs/
#
# For S3 durability: set S3_BUCKET in .env — the pipeline syncs data/ and models/
# to/from S3 before and after each run automatically.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/rrpillai310/market_mvp.git}"
REPO_DIR="$HOME/market_mvp"
DATA_DIR="${DATA_DIR:-$HOME/data}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
LOG_DIR="$REPO_DIR/logs"
PYTHON="${PYTHON:-python3}"

echo "=== market_mvp AWS setup ==="
echo "Repo:    $REPO_DIR"
echo "Data:    $DATA_DIR"
echo "Models:  $MODELS_DIR"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
if command -v dnf &>/dev/null; then
    # Amazon Linux 2023
    sudo dnf install -y python3.11 python3.11-pip git unzip
    sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 || true
elif command -v apt-get &>/dev/null; then
    # Ubuntu
    sudo apt-get update -q
    sudo apt-get install -y python3.11 python3.11-pip python3.11-venv git unzip
fi

# AWS CLI v2 (if not already installed)
if ! command -v aws &>/dev/null; then
    echo "[setup] Installing AWS CLI v2..."
    ARCH=$(uname -m)
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install
    rm -rf /tmp/aws /tmp/awscliv2.zip
fi

# ── 2. Clone or update repo ───────────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[setup] Pulling latest changes..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[setup] Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

# ── 3. Python virtual environment + deps ─────────────────────────────────────
VENV="$HOME/.venv/market_mvp"
if [ ! -d "$VENV" ]; then
    echo "[setup] Creating venv at $VENV..."
    $PYTHON -m venv "$VENV"
fi

"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$REPO_DIR/requirements.txt"
PYTHON_BIN="$VENV/bin/python3"

echo "[setup] Installed packages:"
"$VENV/bin/pip" list --format=columns | grep -E "anthropic|openai|boto3|lightgbm|duckdb"

# ── 4. Environment file ───────────────────────────────────────────────────────
ENV_FILE="$REPO_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "$REPO_DIR/.env.example" "$ENV_FILE"
    echo ""
    echo "⚠️  Created $ENV_FILE from .env.example."
    echo "   Edit it now to set ALPHAVANTAGE_API_KEY, LLM_PROVIDER, and optionally S3_BUCKET."
    echo "   Then re-run this script or start the pipeline manually."
    echo ""
fi

# ── 5. Directories ────────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$LOG_DIR"

# Write directory overrides into .env if paths differ from repo defaults
# (i.e. when DATA_DIR is an EBS mount like /data rather than ~/data)
if [ "$DATA_DIR" != "$HOME/data" ]; then
    grep -q "^DATA_DIR=" "$ENV_FILE" || echo "DATA_DIR=$DATA_DIR" >> "$ENV_FILE"
fi
if [ "$MODELS_DIR" != "$HOME/models" ]; then
    grep -q "^MODELS_DIR=" "$ENV_FILE" || echo "MODELS_DIR=$MODELS_DIR" >> "$ENV_FILE"
fi

# ── 6. Systemd daily pipeline unit ───────────────────────────────────────────
SERVICE_FILE="/etc/systemd/system/market_mvp_pipeline.service"
TIMER_FILE="/etc/systemd/system/market_mvp_pipeline.timer"

# Read symbols from config
SYMBOLS=$(python3 -c "
import json, pathlib
cfg = json.loads(pathlib.Path('$REPO_DIR/config/symbols.json').read_text())
print(' '.join(s['ticker'] for s in cfg.get('symbols', [])))
" 2>/dev/null || echo "SPY QQQ")

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=market_mvp daily pipeline
After=network-online.target

[Service]
Type=oneshot
User=$(whoami)
WorkingDirectory=$HOME
Environment=PYTHONPATH=$HOME
EnvironmentFile=$ENV_FILE
ExecStart=$PYTHON_BIN -m market_mvp.pipeline --symbols $SYMBOLS --skip-social
StandardOutput=append:$LOG_DIR/pipeline.log
StandardError=append:$LOG_DIR/pipeline.log
EOF

sudo tee "$TIMER_FILE" > /dev/null <<EOF
[Unit]
Description=Run market_mvp pipeline daily at 7 AM UTC

[Timer]
OnCalendar=*-*-* 07:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now market_mvp_pipeline.timer

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit $ENV_FILE"
echo "     • Set ALPHAVANTAGE_API_KEY"
echo "     • Set LLM_PROVIDER=claude and ANTHROPIC_API_KEY  (or keep LLM_PROVIDER=ollama)"
echo "     • Set S3_BUCKET=<bucket>  to enable S3 sync for durability"
echo ""
echo "  2. Run the pipeline manually to verify:"
echo "     PYTHONPATH=\$HOME \\"
echo "       $PYTHON_BIN -m market_mvp.pipeline \\"
echo "       --symbols SPY QQQ --skip-social --skip-fed"
echo ""
echo "  3. Launch Streamlit:"
echo "     PYTHONPATH=\$HOME \\"
echo "       $VENV/bin/streamlit run $REPO_DIR/app.py \\"
echo "       --server.address 0.0.0.0 --server.port 8501"
echo ""
echo "  4. Check the daily timer:"
echo "     systemctl status market_mvp_pipeline.timer"
echo "     journalctl -u market_mvp_pipeline.service -f"
echo ""
echo "  5. (Optional) Mount an EBS volume for data persistence:"
echo "     sudo mkfs.ext4 /dev/nvme1n1"
echo "     sudo mount /dev/nvme1n1 /data"
echo "     echo 'DATA_DIR=/data' >> $ENV_FILE"
echo "     echo 'MODELS_DIR=/data/models' >> $ENV_FILE"
