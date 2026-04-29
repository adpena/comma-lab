#!/bin/bash
# Lane PSD-STANDARD: Pixel-Shuffle-Downscale architecture controlled training run.
#
# PSD inherits PROVEN_BASELINE + 6 PSD-specific overrides:
#   variant="psd", boundary_weight=50, hard_frame_ratio=0.3,
#   error_replay_every=200, eval_every=5, use_swa=True
# Profile: src/tac/profiles.py:168 PSD_STANDARD_ADAPTIVE
#
# Phase 1 lane 7. Codex Round 1 flagged it as YELLOW — "profile exists, but
# dispatch script gap remains" + "predicted band [1.10, 1.40] standalone but
# high PoseNet risk". This script closes the gap.
#
# Hard kill threshold (Karpathy/Round-1 PSD-specific):
#   - epoch 200 PoseNet > 1.5x Lane G v3 baseline (0.003) → SIGTERM training
#   - epoch 800 final score > 1.40 → don't bother bolting onto base
#
# Strict-scorer-rule + CLAUDE.md non-negotiables:
#   - --device cuda everywhere (no MPS fallback): feedback_mps_cuda_drift_critical
#   - eval_roundtrip=True (default in train_renderer): CLAUDE.md non-negotiable
#   - Auth-eval at end via experiments/contest_auth_eval.py [contest-CUDA]
#   - Tarball-only parity per Check 66 (no git pull / git reset --hard)
#
# Pipeline (4 stages):
#   Stage 0:  NVDEC probe + dead-flag-wiring guard
#   Stage 1:  train_renderer.py --profile psd_standard_adaptive (10-12h on T4)
#   Stage 2:  export FP4A renderer.bin from best fp32 checkpoint
#   Stage 3:  build archive + contest_auth_eval [contest-CUDA]
#
# Cost: ~$8/12h on Modal T4, or ~$13/12h on Modal A10G (A10G 2x faster)
# Predicted band: [1.10, 1.40] [contest-CUDA]
# Bolt-on potential: if PSD lands at <1.20, +0.005-0.015 distortion gain on
# the SC++/q_faithful base.
#
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
# 2026-04-27 fix: train_renderer.py defaults to a stale upstream path.
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LANE_ID="lane_psd_standard"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
TAG="lane_psd_std"

log() { echo "[lane-psd] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps).
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" -c "
import json, time, torch
prov = {
    'lane_id': '$LANE_ID',
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_psd_standard.sh',
    'profile': 'psd_standard_adaptive',
    'output_dir': '$LOG_DIR',
    'predicted_band': [1.10, 1.40],
    'controlled_baseline': 'lane_g_v3 (1.05 contest-CUDA)',
    'hypothesis': 'PSD architecture beats dilated_h64 baseline as a swap-in',
    'risk_2026_04_29': 'Round-1 codex flagged HIGH PoseNet collapse risk',
    'kill_thresholds': {
        'epoch_200_pose_max': 0.0045,
        'epoch_800_score_max': 1.40,
    },
    'cost_estimate_usd': 8.0,
    'cost_cap_usd': 15.0,
    'wall_clock_estimate_hours': 12,
    'wall_clock_cap_hours': 14,
    'lane_tag': '[contest-CUDA]',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"

# Heartbeat loop (memory feedback_remote_code_parity_critical + CLAUDE.md
# remote heartbeat non-negotiable). Background; killed at script exit.
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ' || echo "no-gpu")
    echo "[$(date -u +%FT%TZ)] lane=PSD gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
done ) &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe (memory feedback_vastai_nvdec_host_variation).
log "=== Stage 0: NVDEC probe ==="
if [ -f "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        echo "FATAL: NVDEC probe failed — destroy this instance, pick a different host."
        exit 2
    }
fi

# Stage 0b: dead-flag-wiring guard. Every CLI flag in train_renderer
# invocation below must exist in train_renderer.py argparse.
log "=== Stage 0b: dead-flag-wiring guard ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_psd_standard.sh').read()
tr_src = open('src/tac/experiments/train_renderer.py').read()
m = re.search(r'src/tac/experiments/train_renderer\\.py(.*?)(?=\\n\\s*BEST_FP32=|\\n\\s*log )',
              script, re.DOTALL)
assert m, 'could not locate train_renderer.py invocation in script'
flags = set(re.findall(r'--([\\w-]+)', m.group(1)))
adds  = set(re.findall(r'add_argument\\([\"\\']--([\\w-]+)', tr_src))
invented = flags - adds
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in train_renderer argparse',
          file=sys.stderr)
    sys.exit(2)
print(f'dead-flag check: OK (flags used = {len(flags)}, all present in argparse)')
"

# Stage 0c: write kill_targets.json sidecar AND launch inline watchdog.
# Council Round 3 (project_phase1_dispatch_verdict_20260429.md) found this
# was missing — kill thresholds were declared in comments/provenance but
# NOT enforced during training. The $8/12h spend would proceed even if
# epoch-200 PoseNet > 0.0045 collapsed. This watchdog fixes that.
cat > "$LOG_DIR/kill_targets.json" <<'EOF'
{
  "epoch_200_pose_max": 0.0045,
  "epoch_800_score_max": 1.40,
  "comment": "Hard kill targets per Round-1+3 codex. Watchdog tails train.log + SIGTERMs at threshold."
}
EOF

# Watchdog: tail train.log, parse epoch + PoseNet/score lines, SIGTERM the
# train_renderer.py process when threshold breached. Uses pgrep instead of
# capturing $! through the pipeline (capturing PID through `tee | grep |
# tail` is fragile across shells; pgrep against the script path is robust).
( while true; do
    sleep 30
    [ -f "$LOG_DIR/train.log" ] || continue
    # Format from train_renderer.py: "[eval] epoch=200 ... posenet_distortion=0.0123 ..."
    line=$(tail -50 "$LOG_DIR/train.log" 2>/dev/null | grep -E "epoch=[0-9]+.*posenet" | tail -1)
    if [ -z "$line" ]; then continue; fi
    epoch=$(echo "$line" | grep -oE "epoch=[0-9]+" | head -1 | cut -d= -f2)
    pose=$(echo "$line" | grep -oE "posenet[_a-z]*=[0-9.]+" | head -1 | cut -d= -f2)
    if [ -z "$epoch" ] || [ -z "$pose" ]; then continue; fi
    if [ "$epoch" -ge 200 ] && [ "$epoch" -lt 250 ]; then
        breached=$(awk -v p="$pose" -v t="0.0045" 'BEGIN{print (p>t)?1:0}')
        if [ "$breached" = "1" ]; then
            train_pid=$(pgrep -f "train_renderer.py.*--tag.*$TAG" | head -1)
            if [ -n "$train_pid" ]; then
                echo "[watchdog] PSD HARD KILL: epoch=$epoch posenet=$pose > 0.0045 threshold; SIGTERM pid=$train_pid" \
                    | tee -a "$LOG_DIR/run.log" >&2
                kill -TERM "$train_pid" 2>/dev/null || true
                sleep 10
                kill -KILL "$train_pid" 2>/dev/null || true
            else
                echo "[watchdog] PSD HARD KILL threshold breached but pgrep found no train_renderer.py PID for tag=$TAG" \
                    | tee -a "$LOG_DIR/run.log" >&2
            fi
            exit 0
        fi
    fi
done ) &
WATCHDOG_PID=$!
trap 'kill $HEARTBEAT_PID $WATCHDOG_PID 2>/dev/null || true' EXIT
log "  watchdog PID=$WATCHDOG_PID; will SIGTERM at epoch 200 if PoseNet > 0.0045"

# Stage 1: train PSD with hard kill thresholds.
log "=== Stage 1: train_renderer --profile psd_standard_adaptive ==="
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile psd_standard_adaptive \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch)\]|epoch|Phase|scorer" | tail -200
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: train_renderer exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

# Stage 2: export FP4A renderer.bin from best fp32 checkpoint.
BEST_FP32="$LOG_DIR/train/renderer_${TAG}_best_fp32.pt"
[ -f "$BEST_FP32" ] || {
    echo "FATAL: train_renderer didn't produce ${BEST_FP32}" >&2
    ls -la "$LOG_DIR/train/" >&2
    exit 2
}
log "  best fp32 checkpoint: $BEST_FP32 ($(stat -c '%s' "$BEST_FP32" 2>/dev/null || stat -f '%z' "$BEST_FP32") bytes)"

log "=== Stage 2: export FP4A renderer.bin ==="
mkdir -p "$LOG_DIR/iter_0"
"$PYBIN" -c "
import sys, torch
sys.path.insert(0, 'src')
from tac.renderer import build_renderer
from tac.renderer_export import export_asymmetric_checkpoint_fp4
ckpt = torch.load('$BEST_FP32', map_location='cpu', weights_only=False)
state = ckpt.get('model_state_dict', ckpt)
meta = ckpt.get('__meta__', {}) or {}
def m(key, default):
    return meta.get(key, default)
model = build_renderer(
    embed_dim=m('embed_dim', 6),
    base_ch=m('base_ch', 36),
    mid_ch=m('mid_ch', 60),
    motion_hidden=m('motion_hidden', 32),
    depth=m('depth', 1),
    pose_dim=m('pose_dim', 6),
    use_dsconv=m('use_dsconv', False),
    padding_mode=m('padding_mode', 'zeros'),
    use_dilation=m('use_dilation', False),
    use_zoom_flow=m('use_zoom_flow', False),  # PSD doesn't use zoom_flow
    variant=m('variant', 'psd'),
)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise RuntimeError(
        f'shape mismatch — missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}'
    )
nbytes = export_asymmetric_checkpoint_fp4(
    model, '$LOG_DIR/iter_0/renderer.bin',
    codebook_name=m('fp4_codebook', 'residual'),
    robust_scale=bool(m('fp4_robust_scale', True)),
)
print(f'exported FP4A renderer.bin: {nbytes:,} bytes')
"

# Stage 3: build archive + auth-eval [contest-CUDA].
log "=== Stage 3: contest_auth_eval [contest-CUDA] ==="
ARCHIVE="$LOG_DIR/archive_psd_standard.zip"
"$PYBIN" -u submissions/robust_current/compress_archive.py \
    --renderer-bin "$LOG_DIR/iter_0/renderer.bin" \
    --masks-path "$LOG_DIR/train/masks_${TAG}.mkv" \
    --poses-path "$LOG_DIR/train/optimized_poses_${TAG}.pt" \
    --output "$ARCHIVE" \
    --binary-poses 2>&1 | tee "$LOG_DIR/archive.log" | tail -20

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
log "  archive bytes: $ARCHIVE_BYTES"

"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

log "=== LANE_PSD_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
