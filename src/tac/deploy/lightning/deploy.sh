#!/usr/bin/env bash
# Self-contained Lightning AI deployment — one command, both lanes.
#
# Usage:
#   ./src/tac/deploy/lightning/deploy.sh cpu [--resume-from <checkpoint>]
#   ./src/tac/deploy/lightning/deploy.sh gpu [--resume-from <checkpoint>]
#   ./src/tac/deploy/lightning/deploy.sh setup  # first-time setup only
#
# Requires: Lightning SSH configured (~/.ssh/lightning_rsa)
# Env vars: LIGHTNING_HOST (default: from ~/.ssh/config)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
REMOTE_BASE="/home/zeus/content/pact"

# Discover Lightning SSH host from config or env
LIGHTNING_HOST="${LIGHTNING_HOST:-$(grep -A1 'Host ssh.lightning.ai' ~/.ssh/config 2>/dev/null | head -1 | awk '{print $2}' || echo '')}"
if [ -z "$LIGHTNING_HOST" ]; then
    echo "ERROR: LIGHTNING_HOST not set and no SSH config found."
    echo "Run the Lightning SSH setup command from the Lightning UI first."
    exit 1
fi

# Find the SSH user from the config
LIGHTNING_USER="${LIGHTNING_USER:-$(grep -B5 'ssh.lightning.ai' ~/.ssh/config 2>/dev/null | grep -v '^#' | head -1 || echo '')}"
# Fallback: extract from known_hosts or ask
SSH_TARGET="${LIGHTNING_SSH_TARGET:-$(grep 'ssh.lightning.ai' ~/.ssh/known_hosts 2>/dev/null | head -1 | awk '{print $1}' || echo 'ssh.lightning.ai')}"

# SSH helper
_ssh() {
    ssh -i ~/.ssh/lightning_rsa -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$@"
}
_scp() {
    scp -i ~/.ssh/lightning_rsa -o StrictHostKeyChecking=no "$@"
}

# Find the SSH user by testing connection
_find_user() {
    # Try to get from last successful connection
    local user
    user=$(grep 'IdentityFile.*lightning' ~/.ssh/config -A3 2>/dev/null | grep -v '^$' | head -1 || true)
    # Extract from the setup script output
    if [ -f /tmp/lightning_setup.sh ]; then
        user=$(grep 'ssh s_' /tmp/lightning_setup.sh 2>/dev/null | sed 's/.*ssh \(s_[^@]*\)@.*/\1/' || true)
    fi
    echo "${LIGHTNING_USER:-$user}"
}

REMOTE_USER="${LIGHTNING_USER:-$(grep -oP 's_\w+' ~/.ssh/config 2>/dev/null | head -1 || echo '')}"
REMOTE="${REMOTE_USER}@ssh.lightning.ai"

cmd_setup() {
    echo "=== Lightning Setup ==="
    echo "1. Uploading tac source..."
    tar czf /tmp/tac_lightning.tar.gz -C "$REPO_ROOT/src" tac 2>/dev/null
    _scp /tmp/tac_lightning.tar.gz "$REMOTE:$REMOTE_BASE/" 2>/dev/null
    _ssh "$REMOTE" "cd $REMOTE_BASE && tar xzf tac_lightning.tar.gz -C src/ 2>/dev/null && echo '   tac: $(ls src/tac/*.py | wc -l) modules'"

    echo "2. Uploading submissions..."
    _scp -r "$REPO_ROOT/submissions/robust_current/archive.zip" "$REMOTE:$REMOTE_BASE/submissions/robust_current/" 2>/dev/null

    echo "3. Cloning upstream scorer (if needed)..."
    _ssh "$REMOTE" "
        if [ ! -d /home/zeus/content/upstream/models ]; then
            git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git /home/zeus/content/upstream
            cd /home/zeus/content/upstream && git lfs pull
        else
            echo '   upstream: already present'
        fi
    " 2>/dev/null

    echo "4. Installing dependencies..."
    _ssh "$REMOTE" "
        source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
        conda activate cloudspace 2>/dev/null
        pip install -q safetensors timm einops segmentation-models-pytorch 2>/dev/null
        echo '   deps: installed'
    "

    echo "5. Uploading precomputed data (if not present)..."
    _ssh "$REMOTE" "ls $REMOTE_BASE/precomputed/comp_frames.pt 2>/dev/null && echo '   precomputed: already present'" 2>/dev/null || {
        echo "   Uploading 8.8GB (this takes a few minutes)..."
        _scp "$REPO_ROOT/precomputed_local/"*.pt "$REMOTE:$REMOTE_BASE/precomputed/" 2>/dev/null
    }

    echo ""
    echo "=== Setup complete ==="
}

cmd_cpu() {
    local resume_from="${1:-}"
    local profile="${TAC_PROFILE:-proven_baseline}"
    local epochs="${TAC_EPOCHS:-2500}"
    local tag="${TAC_TAG:-cpu_lightning_$(date +%Y%m%dT%H%M%S)}"

    echo "=== Launching CPU lane on Lightning T4 ==="
    echo "  Profile: $profile"
    echo "  Epochs: $epochs"
    echo "  Tag: $tag"
    [ -n "$resume_from" ] && echo "  Resume: $resume_from"

    # Upload resume checkpoint if specified
    if [ -n "$resume_from" ] && [ -f "$resume_from" ]; then
        echo "  Uploading checkpoint..."
        _scp "$resume_from" "$REMOTE:$REMOTE_BASE/checkpoints/resume.pt" 2>/dev/null
        resume_from="$REMOTE_BASE/checkpoints/resume.pt"
    fi

    _ssh "$REMOTE" "
        source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
        conda activate cloudspace 2>/dev/null
        cd $REMOTE_BASE
        export PYTHONPATH=$REMOTE_BASE/src:/home/zeus/content/upstream
        export PYTHONUNBUFFERED=1
        export TAC_UPSTREAM_DIR=/home/zeus/content/upstream
        export TAC_MODELS_DIR=/home/zeus/content/upstream/models

        nohup python -m tac lossy \\
            --profile $profile \\
            --precomputed $REMOTE_BASE/precomputed \\
            ${resume_from:+--resume-from $resume_from} \\
            --tag $tag \\
            --output-dir $REMOTE_BASE/results \\
            --epochs $epochs \\
            --eval-every 25 > $REMOTE_BASE/training_cpu.log 2>&1 &

        echo \"PID: \$!\"
        sleep 3
        tail -5 $REMOTE_BASE/training_cpu.log
    "
    echo ""
    echo "Monitor: $0 logs cpu"
}

cmd_gpu() {
    local resume_from="${1:-}"
    local profile="${TAC_PROFILE:-dp_sims_smoke}"
    local tag="${TAC_TAG:-gpu_lightning_$(date +%Y%m%dT%H%M%S)}"

    echo "=== Launching GPU lane on Lightning T4 ==="
    echo "  Profile: $profile"
    echo "  Tag: $tag"

    if [ -n "$resume_from" ] && [ -f "$resume_from" ]; then
        echo "  Uploading checkpoint..."
        _scp "$resume_from" "$REMOTE:$REMOTE_BASE/checkpoints/resume_gpu.pt" 2>/dev/null
        resume_from="$REMOTE_BASE/checkpoints/resume_gpu.pt"
    fi

    _ssh "$REMOTE" "
        source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
        conda activate cloudspace 2>/dev/null
        cd $REMOTE_BASE
        export PYTHONPATH=$REMOTE_BASE/src:/home/zeus/content/upstream
        export PYTHONUNBUFFERED=1
        export TAC_UPSTREAM_DIR=/home/zeus/content/upstream
        export TAC_MODELS_DIR=/home/zeus/content/upstream/models

        nohup python -m tac.experiments.train_renderer \\
            --profile $profile \\
            --precomputed $REMOTE_BASE/precomputed \\
            ${resume_from:+--resume-from $resume_from} \\
            --tag $tag \\
            --output-dir $REMOTE_BASE/results > $REMOTE_BASE/training_gpu.log 2>&1 &

        echo \"PID: \$!\"
        sleep 3
        tail -5 $REMOTE_BASE/training_gpu.log
    "
    echo ""
    echo "Monitor: $0 logs gpu"
}

cmd_logs() {
    local lane="${1:-cpu}"
    local logfile="training_${lane}.log"
    echo "=== Lightning $lane lane logs ==="
    _ssh "$REMOTE" "tail -20 $REMOTE_BASE/$logfile 2>/dev/null || echo 'No log found'"
}

cmd_status() {
    echo "=== Lightning Training Status ==="
    _ssh "$REMOTE" "
        echo 'CPU:'
        tail -1 $REMOTE_BASE/training_cpu.log 2>/dev/null || echo '  not running'
        echo ''
        echo 'GPU:'
        tail -1 $REMOTE_BASE/training_gpu.log 2>/dev/null || echo '  not running'
        echo ''
        echo 'GPU info:'
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
    "
}

cmd_download() {
    echo "=== Downloading results from Lightning ==="
    mkdir -p /tmp/lightning_results
    _scp -r "$REMOTE:$REMOTE_BASE/results/" /tmp/lightning_results/ 2>/dev/null
    echo "Downloaded to /tmp/lightning_results/"
    find /tmp/lightning_results -name "*.pt" -o -name "*.json" 2>/dev/null | head -10
}

# Main dispatch
case "${1:-help}" in
    setup)    cmd_setup ;;
    cpu)      cmd_cpu "${2:-}" ;;
    gpu)      cmd_gpu "${2:-}" ;;
    logs)     cmd_logs "${2:-cpu}" ;;
    status)   cmd_status ;;
    download) cmd_download ;;
    *)
        echo "Usage: $0 {setup|cpu|gpu|logs|status|download}"
        echo ""
        echo "  setup              First-time setup (upload tac, deps, upstream)"
        echo "  cpu [checkpoint]   Launch CPU postfilter training"
        echo "  gpu [checkpoint]   Launch GPU renderer training"
        echo "  logs {cpu|gpu}     Show training logs"
        echo "  status             Show both lanes + GPU utilization"
        echo "  download           Download results to local"
        echo ""
        echo "Env vars: TAC_PROFILE, TAC_EPOCHS, TAC_TAG, LIGHTNING_USER"
        ;;
esac
