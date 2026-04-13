#!/usr/bin/env bash
# Self-contained Lightning AI deployment — one command, both lanes.
#
# Training flags are imported from tac.deploy.deploy_config (canonical source).
# Provider-specific: SSH wiring, paths, conda env, nohup launch.
#
# Usage:
#   ./src/tac/deploy/lightning/deploy.sh setup
#   ./src/tac/deploy/lightning/deploy.sh gpu base [--resume-from <checkpoint>]
#   ./src/tac/deploy/lightning/deploy.sh gpu supervised [--resume-from <checkpoint>]
#   ./src/tac/deploy/lightning/deploy.sh gpu raft_only [--resume-from <checkpoint>]
#   ./src/tac/deploy/lightning/deploy.sh cpu [--resume-from <checkpoint>]
#   ./src/tac/deploy/lightning/deploy.sh list-variants
#   ./src/tac/deploy/lightning/deploy.sh logs {cpu|gpu}
#   ./src/tac/deploy/lightning/deploy.sh status
#   ./src/tac/deploy/lightning/deploy.sh download
#
# Requires: Lightning SSH configured (~/.ssh/lightning_rsa)
# Env vars: LIGHTNING_HOST (default: from ~/.ssh/config)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
REMOTE_BASE="/home/zeus/content/pact"
REMOTE_RESULTS="$REMOTE_BASE/results"
REMOTE_SCRIPT="$REMOTE_BASE/experiments/train_renderer_fridrich.py"

# ---------------------------------------------------------------------------
# SSH wiring
# ---------------------------------------------------------------------------
LIGHTNING_HOST="${LIGHTNING_HOST:-$(grep -A1 'Host ssh.lightning.ai' ~/.ssh/config 2>/dev/null | head -1 | awk '{print $2}' || echo '')}"
if [ -z "$LIGHTNING_HOST" ]; then
    echo "ERROR: LIGHTNING_HOST not set and no SSH config found."
    echo "Run the Lightning SSH setup command from the Lightning UI first."
    exit 1
fi

REMOTE_USER="${LIGHTNING_USER:-$(grep -oP 's_\w+' ~/.ssh/config 2>/dev/null | head -1 || echo '')}"
REMOTE="${REMOTE_USER}@ssh.lightning.ai"

_ssh() {
    ssh -i ~/.ssh/lightning_rsa -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$@"
}
_scp() {
    scp -i ~/.ssh/lightning_rsa -o StrictHostKeyChecking=no "$@"
}

# ---------------------------------------------------------------------------
# Import training flags from deploy_config (provider-agnostic source of truth)
#
# Uses environment variables (not shell string interpolation) to pass values
# into Python to avoid injection from user-controlled inputs.
# ---------------------------------------------------------------------------
_build_gpu_cmd() {
    local variant="${1:-base}"
    local resume_from="${2:-}"

    # Pass values via env vars — never interpolate user-controlled strings into
    # Python source code (prevents injection from paths containing quotes/newlines)
    ASYM_VARIANT="$variant" ASYM_RESUME="$resume_from" ASYM_SCRIPT="$REMOTE_SCRIPT" \
    REPO_SRC="$REPO_ROOT/src" "$REPO_ROOT/.venv/bin/python" - <<'PYEOF'
import os, sys, shlex
sys.path.insert(0, os.environ["REPO_SRC"])
from tac.deploy.deploy_config import build_flags
resume = os.environ.get("ASYM_RESUME") or None
cmd = build_flags(
    variant=os.environ["ASYM_VARIANT"],
    provider_script_path=os.environ["ASYM_SCRIPT"],
    resume_from=resume,
)
print(shlex.join(cmd))
PYEOF
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd_setup() {
    echo "=== Lightning Setup ==="
    echo "1. Uploading tac source..."
    tar czf /tmp/tac_lightning.tar.gz -C "$REPO_ROOT/src" tac 2>/dev/null
    _scp /tmp/tac_lightning.tar.gz "$REMOTE:$REMOTE_BASE/" 2>/dev/null
    _ssh "$REMOTE" "cd $REMOTE_BASE && tar xzf tac_lightning.tar.gz -C src/ 2>/dev/null && echo '   tac: ok'"

    echo "2. Uploading experiments/..."
    tar czf /tmp/experiments_lightning.tar.gz \
        -C "$REPO_ROOT" \
        experiments/train_renderer_fridrich.py \
        experiments/compute_raft_flow.py \
        2>/dev/null
    _scp /tmp/experiments_lightning.tar.gz "$REMOTE:$REMOTE_BASE/" 2>/dev/null
    _ssh "$REMOTE" "cd $REMOTE_BASE && tar xzf experiments_lightning.tar.gz 2>/dev/null && echo '   experiments: ok'"

    echo "3. Uploading submissions..."
    _scp -r "$REPO_ROOT/submissions/robust_current/archive.zip" \
        "$REMOTE:$REMOTE_BASE/submissions/robust_current/" 2>/dev/null || true

    echo "4. Cloning upstream scorer (if needed)..."
    _ssh "$REMOTE" "
        if [ ! -d /home/zeus/content/upstream/models ]; then
            git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git /home/zeus/content/upstream
            cd /home/zeus/content/upstream && git lfs pull
        else
            echo '   upstream: already present'
        fi
    " 2>/dev/null

    echo "5. Installing uv + dependencies..."
    _ssh "$REMOTE" "
        source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
        conda activate cloudspace 2>/dev/null
        command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH=\"\$HOME/.cargo/bin:\$PATH\"
        uv pip install -q safetensors timm einops segmentation-models-pytorch av pydantic click 2>/dev/null
        echo '   deps: installed'
    "

    echo "6. Uploading precomputed data (if not present)..."
    _ssh "$REMOTE" "ls $REMOTE_BASE/precomputed/comp_frames.pt 2>/dev/null && echo '   precomputed: already present'" 2>/dev/null || {
        echo "   Uploading (this takes a few minutes)..."
        _scp "$REPO_ROOT/experiments/precomputed_local/"*.pt "$REMOTE:$REMOTE_BASE/precomputed/" 2>/dev/null
    }

    echo ""
    echo "=== Setup complete ==="
}

cmd_list_variants() {
    python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT/src')
from tac.deploy.deploy_config import ALL_VARIANTS, VARIANT_FLAGS
for v in ALL_VARIANTS:
    n = len(VARIANT_FLAGS[v])
    print(f'  {v}  ({n} extra flags)')
"
}

cmd_gpu() {
    local variant="${1:-base}"
    local resume_from=""

    # Parse optional --resume-from flag
    shift || true
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --resume-from) resume_from="$2"; shift 2 ;;
            *) echo "Unknown flag: $1"; exit 1 ;;
        esac
    done

    local tag="${TAC_TAG:-gpu_lightning_${variant}_$(date +%Y%m%dT%H%M%S)}"

    echo "=== Launching GPU renderer (asymmetric warp) on Lightning T4 ==="
    echo "  Variant: $variant"
    echo "  Tag: $tag"
    [ -n "$resume_from" ] && echo "  Resume: $resume_from"

    # Upload resume checkpoint if it's a local path
    if [ -n "$resume_from" ] && [ -f "$resume_from" ]; then
        echo "  Uploading checkpoint..."
        _scp "$resume_from" "$REMOTE:$REMOTE_BASE/checkpoints/resume_gpu.pt" 2>/dev/null
        resume_from="$REMOTE_BASE/checkpoints/resume_gpu.pt"
    fi

    # Build the training command from deploy_config
    local training_cmd
    training_cmd="$(_build_gpu_cmd "$variant" "$resume_from")"

    echo "  Command: $training_cmd"

    # Save deployment manifest locally — use env vars to avoid shell injection
    ASYM_VARIANT="$variant" ASYM_RESUME="$resume_from" ASYM_TAG="$tag" \
    ASYM_SCRIPT="$REMOTE_SCRIPT" REPO_SRC="$REPO_ROOT/src" \
    MANIFEST_DIR="$REPO_ROOT/reports/deployment_manifests" \
    "$REPO_ROOT/.venv/bin/python" - <<'PYEOF'
import os, sys, json, time, socket
sys.path.insert(0, os.environ["REPO_SRC"])
from tac.deploy.deploy_config import build_flags, VARIANT_FLAGS
variant = os.environ["ASYM_VARIANT"]
resume = os.environ.get("ASYM_RESUME") or None
cmd = build_flags(variant=variant, provider_script_path=os.environ["ASYM_SCRIPT"], resume_from=resume)
tag = os.environ["ASYM_TAG"]
manifest = {
    "tag": tag,
    "variant": variant,
    "resume_from": resume,
    "full_command": cmd,
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": socket.gethostname(),
    "provider": "lightning",
    "gpu": "T4",
    "variant_extra_flags": VARIANT_FLAGS[variant],
}
manifest_dir = os.environ["MANIFEST_DIR"]
os.makedirs(manifest_dir, exist_ok=True)
path = os.path.join(manifest_dir, f"{tag}_manifest.json")
json.dump(manifest, open(path, "w"), indent=2)
print(f"  Manifest: {path}")
PYEOF

    _ssh "$REMOTE" "
        source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
        conda activate cloudspace 2>/dev/null
        cd $REMOTE_BASE
        export PYTHONPATH=$REMOTE_BASE/src:/home/zeus/content/upstream
        export PYTHONUNBUFFERED=1
        export UPSTREAM_ROOT=/home/zeus/content/upstream
        export TAC_UPSTREAM_DIR=/home/zeus/content/upstream
        export TAC_MODELS_DIR=/home/zeus/content/upstream/models
        mkdir -p $REMOTE_RESULTS/$tag

        nohup $training_cmd > $REMOTE_BASE/training_gpu.log 2>&1 &

        echo \"PID: \$!\"
        sleep 3
        tail -5 $REMOTE_BASE/training_gpu.log
    "
    echo ""
    echo "Monitor: $0 logs gpu"
    echo "Tag: $tag"
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

cmd_logs() {
    local lane="${1:-gpu}"
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
    setup)         cmd_setup ;;
    gpu)           cmd_gpu "${2:-base}" "${@:3}" ;;
    cpu)           cmd_cpu "${2:-}" ;;
    list-variants) cmd_list_variants ;;
    logs)          cmd_logs "${2:-gpu}" ;;
    status)        cmd_status ;;
    download)      cmd_download ;;
    *)
        echo "Usage: $0 {setup|gpu|cpu|list-variants|logs|status|download}"
        echo ""
        echo "  setup                          First-time setup"
        echo "  gpu [variant] [--resume-from]  Launch GPU asymmetric warp training"
        echo "  cpu [checkpoint]               Launch CPU postfilter training"
        echo "  list-variants                  Show available training variants"
        echo "  logs {cpu|gpu}                 Show training logs"
        echo "  status                         Show both lanes + GPU utilization"
        echo "  download                       Download results to local"
        echo ""
        echo "Variants: base | supervised | raft_only"
        echo "  base:       Pure Lagrangian (no supervision)"
        echo "  supervised: PoseNet + RAFT flow (Layers 1+2)"
        echo "  raft_only:  RAFT flow only (Layer 2 isolation)"
        echo ""
        echo "Env vars: TAC_PROFILE, TAC_EPOCHS, TAC_TAG, LIGHTNING_USER"
        ;;
esac
