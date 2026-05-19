#!/usr/bin/env bash
# bat00_runner.sh — Robust experiment runner for bat00 (RTX 2070 Super / future 3090)
#
# Solves the three bat00 problems:
#   1. Dependency install hangs → timeout + retry + verbose logging
#   2. No logging → everything tees to a timestamped log file
#   3. SSH drops → runs under nohup, survives disconnects
#
# Usage (on bat00):
#   chmod +x bat00_runner.sh
#   nohup ./bat00_runner.sh 2>&1 &
#   # or just double-click if on desktop
#
# The script is self-contained: clones repo, installs deps, runs experiment.
# Edit EXPERIMENT below to change what runs.

set -euo pipefail

# ---- Configuration ----
REPO_URL="https://github.com/adpena/comma-lab.git"
REPO_DIR="$HOME/pact"
VENV_DIR="$REPO_DIR/.venv"
LOG_DIR="$HOME/pact-logs"
EXPERIMENT="constrained_gen_smoke"  # or "train_base", "train_raft_only"
PYTHON_VERSION="3.12"

# ---- Logging ----
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${EXPERIMENT}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== bat00_runner.sh | $(date) | experiment=$EXPERIMENT ==="
echo "  Log: $LOG_FILE"
echo "  Host: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
echo ""

# ---- Step 1: Install uv if missing ----
echo "[1/5] Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv: $(uv --version)"

# ---- Step 2: Clone/update repo ----
echo "[2/5] Repo sync..."
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    # Detached-checkout pattern (CLAUDE.md Checks 66-69): preserve any uncommitted
    # local work; the runner consumes the remote tip read-only.
    git fetch origin main --depth=1
    git checkout --detach origin/main
    echo "  Checked out (detached) at $(git rev-parse --short HEAD)"
else
    git clone --depth=1 "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    echo "  Cloned at $(git rev-parse --short HEAD)"
fi

# ---- Step 3: Create venv + install deps (with timeout) ----
echo "[3/5] Dependencies..."
if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating venv..."
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

echo "  Installing tac + deps (timeout 300s)..."
timeout 300 uv pip install -e ".[dev]" --quiet 2>&1 || {
    echo "  WARN: uv pip install timed out or failed. Retrying with verbose..."
    timeout 600 uv pip install -e ".[dev]" -v 2>&1 || {
        echo "  FATAL: dependency install failed after retry"
        exit 1
    }
}

# Install torch with CUDA (if not already present)
echo "  Checking torch CUDA..."
"$VENV_DIR/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(f'  torch {torch.__version__} CUDA {torch.version.cuda}')" 2>/dev/null || {
    echo "  Installing torch with CUDA..."
    timeout 600 uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --quiet 2>&1
}

# Install upstream scorer deps
echo "  Installing scorer deps..."
timeout 120 uv pip install safetensors timm einops segmentation-models-pytorch av --quiet 2>&1

echo "  Dependencies ready."

# ---- Step 4: Setup upstream scorer ----
echo "[4/5] Upstream scorer..."
UPSTREAM="$REPO_DIR/upstream"
if [ ! -d "$UPSTREAM/models" ]; then
    echo "  Cloning upstream..."
    git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git "$UPSTREAM"
    cd "$UPSTREAM" && git lfs pull && cd "$REPO_DIR"
else
    echo "  Upstream already present."
fi

# ---- Step 5: Run experiment ----
echo "[5/5] Running experiment: $EXPERIMENT"
echo "  Started at $(date)"
echo ""

export PYTHONPATH="$REPO_DIR/src:$UPSTREAM"
export PYTHONUNBUFFERED=1

case "$EXPERIMENT" in
    constrained_gen_smoke)
        "$VENV_DIR/bin/python" experiments/smoke_constrained_gen.py \
            --device cuda --steps 100 --n-frames 40
        ;;
    constrained_gen_full)
        # Full 1000-step constrained gen with all 1200 frames
        "$VENV_DIR/bin/python" experiments/smoke_constrained_gen.py \
            --device cuda --steps 1000 --n-frames 1200
        ;;
    train_base)
        "$VENV_DIR/bin/python" experiments/train_renderer_fridrich.py \
            --pair-mode asymmetric --epochs 20000 --batch-size 4 \
            --device cuda --max-hours 8
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        exit 1
        ;;
esac

echo ""
echo "=== Experiment complete | $(date) ==="
echo "  Log: $LOG_FILE"
