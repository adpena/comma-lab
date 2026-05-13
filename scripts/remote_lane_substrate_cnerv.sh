#!/bin/bash
# Remote lane script: substrate cnerv first-anchor dispatch (TRADITION 2).
#
# Trainer: experiments/train_cnerv_as_renderer.py.
# Lane: lane_substrate_cnerv_20260512 (recipe lane id;
#       implementation lane id is lane_cnerv_as_renderer per registry).
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Recipe: .omx/operator_authorize_recipes/substrate_cnerv_modal_a100_dispatch.yaml
#
# Substrate notes:
#   - TRADITION 2 (per .omx/research/substrate_tradition_taxonomy_20260512.md).
#   - CNeRV replaces vanilla NeRV's `Linear(D, C*H*W)` stem with a
#     convolutional stem (tile-replicate + positional bias + 1x1 conv).
#   - Trainer is currently scaffold-only: non-smoke training raises
#     SystemExit at experiments/train_cnerv_as_renderer.py:131. This driver
#     wires the canonical bootstrap+heartbeat+harvest path so the moment
#     non-smoke gating lands, dispatch is one operator-flag away.
#   - The trainer has NO --upstream-dir flag (unlike sane_hnerv); upstream
#     access is via implicit REPO_ROOT/upstream/ resolution.
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_CNERV_DONE marker) per the CLAUDE.md
# score-tag rule and preflight completion-tag check.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity — non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_substrate_cnerv_20260512"
TAG="${TAG:-substrate_cnerv}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_cnerv_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags — aligned with experiments/train_cnerv_as_renderer.py argparse.
# (The trainer predates Catalog #151 TIER_<N>_OPERATOR_REQUIRED_FLAGS; envelope
# stays env-driven until the manifest lands per the recipe reactivation
# criteria #2.)
CNERV_VIDEO_PATH="${CNERV_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
CNERV_OUTPUT_DIR="${CNERV_OUTPUT_DIR:-$OUTPUT_DIR}"
CNERV_EPOCHS="${CNERV_EPOCHS:-2000}"
CNERV_BATCH_SIZE="${CNERV_BATCH_SIZE:-8}"
CNERV_DEVICE="${CNERV_DEVICE:-cuda}"
CNERV_LATENT_DIM="${CNERV_LATENT_DIM:-16}"
CNERV_BASE_CHANNELS="${CNERV_BASE_CHANNELS:-36}"
CNERV_N_PAIRS="${CNERV_N_PAIRS:-600}"
CNERV_LAMBDA_SEG="${CNERV_LAMBDA_SEG:-100.0}"
CNERV_LAMBDA_POSE="${CNERV_LAMBDA_POSE:-288.6751345948129}"
CNERV_LEARNING_RATE="${CNERV_LEARNING_RATE:-1e-3}"
CNERV_EMA_DECAY="${CNERV_EMA_DECAY:-0.997}"
CNERV_SEED="${CNERV_SEED:-20260511}"
CNERV_SMOKE="${CNERV_SMOKE:-0}"

DISPATCH_INSTANCE_JOB_ID="${CNERV_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${CNERV_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-cnerv] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification (mirrors remote_lane_substrate_sane_hnerv.sh).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: CNERV_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi
if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    log "FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py"
    exit 21
fi
if [ ! -f "$DISPATCH_CLAIMS_PATH" ]; then
    log "FATAL: dispatch-claims ledger missing at $DISPATCH_CLAIMS_PATH"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

# Stage 0a: required-input file validation per Catalog #152.
# Cheap (~100ms) local existence check BEFORE GPU meter starts. The cnerv
# trainer dereferences --video-path during data prep; missing file yields a
# 5-30s wasted-spend crash on the GPU host. Validating here costs $0.
if [ ! -f "$CNERV_VIDEO_PATH" ]; then
    log "FATAL: required input --video-path does not exist on remote host: $CNERV_VIDEO_PATH"
    log "  (regenerate via 'git lfs pull' or 'rsync upstream/videos/0.mkv' from local)"
    exit 25
fi

# Stage 0b: NVDEC probe (per CLAUDE.md `feedback_vastai_nvdec_host_variation`).
# Early failure-class probe BEFORE any uv/torch install or training spend.
# Failure rate on cold-pool hosts is ~30-50%; this saves $0.05-0.10 per bad host.
# Skip cleanly on Modal containers (NVDEC always available in CUDA images);
# the probe itself is idempotent and tolerates the no-DALI scenario by exiting 0
# when torchvision-only path is acceptable.
if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    log "stage_0b_nvdec_probe_begin"
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        log "FATAL: NVDEC probe failed (host hardware/driver/DALI mismatch); refusing dispatch"
        exit 2
    }
    log "stage_0b_nvdec_probe_done"
else
    log "WARN: scripts/probe_nvdec.sh missing — skipping NVDEC early-fail probe"
fi

# Stage 1: bootstrap runtime deps (canonical, per CLAUDE.md).
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap script missing at scripts/remote_archive_only_eval.sh"
    exit 22
fi
# shellcheck source=/dev/null
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 prevents the sourced script's main
# flow (which expects a pre-built archive.zip at /workspace/pact/iter_0/) from
# running. We only need its bootstrap_runtime_deps() function. Per Catalog #163
# (check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap) — without
# this sentinel the sourced main flow would fail with "FATAL: archive missing"
# before this script's stages can start.
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"

# bootstrap_runtime_deps installs uv + torch (driver-version-pinned per CLAUDE.md
# "Forbidden uv torch install without driver-version pin"). It also exports PYBIN.
if declare -f bootstrap_runtime_deps > /dev/null 2>&1; then
    log "stage_1_bootstrap_runtime_deps_begin"
    bootstrap_runtime_deps
    log "stage_1_bootstrap_runtime_deps_done PYBIN=${PYBIN:-unset}"
else
    log "FATAL: bootstrap_runtime_deps function not found after sourcing scripts/remote_archive_only_eval.sh"
    exit 23
fi

if [ -z "${PYBIN:-}" ] || [ ! -x "$PYBIN" ]; then
    log "FATAL: PYBIN not set or not executable after bootstrap (PYBIN=${PYBIN:-unset})"
    exit 24
fi

# Stage 2: provenance + remote code parity (per CLAUDE.md "Remote code parity").
log "stage_2_provenance_begin"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'lane_id': '$LANE_ID',
    'dispatch_instance_job_id': '$DISPATCH_INSTANCE_JOB_ID',
    'dispatch_platform': '$DISPATCH_PLATFORM',
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'video_path': '$CNERV_VIDEO_PATH',
    'epochs': $CNERV_EPOCHS,
    'batch_size': $CNERV_BATCH_SIZE,
    'device': '$CNERV_DEVICE',
    'latent_dim': $CNERV_LATENT_DIM,
    'base_channels': $CNERV_BASE_CHANNELS,
    'n_pairs': $CNERV_N_PAIRS,
    'lambda_seg': $CNERV_LAMBDA_SEG,
    'lambda_pose': $CNERV_LAMBDA_POSE,
    'learning_rate': $CNERV_LEARNING_RATE,
    'ema_decay': $CNERV_EMA_DECAY,
    'seed': $CNERV_SEED,
    'smoke': bool(int('$CNERV_SMOKE')),
    # predicted_band per recipe substrate_cnerv_modal_a100_dispatch.yaml.
    # The recipe declares predicted_delta as 'unknown — TRADITION 2 substrate
    # predates Catalog #124 design-time prediction discipline'; the band is
    # therefore reported as null until the first empirical anchor lands.
    'predicted_band': None,
    'predicted_basis': 'recipe_substrate_cnerv_modal_a100_dispatch_unknown_pending_first_anchor',
    'substrate_tradition': 'TRADITION_2',
    'format_id_hex': '0x67',
}
import pathlib
pathlib.Path('$PROVENANCE').write_text(json.dumps(prov, indent=2, sort_keys=True))
print('[provenance]', json.dumps(prov, indent=2, sort_keys=True))
"
log "stage_2_provenance_done"

# Stage 3: heartbeat watchdog (every 5 min per CLAUDE.md).
(
    while true; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID alive" >> "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!
trap 'if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi' EXIT

# Stage 4: invoke trainer.
#
# Trainer argparse (verified against experiments/train_cnerv_as_renderer.py:46-64):
#   --output-dir (required)        --device (cuda|cpu)        --epochs
#   --batch-size                   --learning-rate            --ema-decay
#   --latent-dim                   --base-channels            --n-pairs
#   --lambda-seg                   --lambda-pose              --video-path
#   --seed                         --smoke (bool)             --auth-eval (bool)
#   --phase-b-auth-memo (str)
#
# Note: the trainer is currently SCAFFOLD-ONLY per
# experiments/train_cnerv_as_renderer.py:131 — non-smoke runs raise
# SystemExit("[cnerv] non-smoke training is operator-gated per CLAUDE.md
# scaffold-only directive."). To exercise this driver end-to-end before the
# operator-gating lands, set CNERV_SMOKE=1.
#
# Auth-eval is behind Catalog #150 phase-b-auth-memo gate (mandatory if
# --auth-eval is enabled); not threaded by default — operator opts in via
# CNERV_PHASE_B_AUTH_MEMO env var which adds --auth-eval --phase-b-auth-memo
# explicitly.
log "stage_4_trainer_invoke_begin video=$CNERV_VIDEO_PATH epochs=$CNERV_EPOCHS device=$CNERV_DEVICE smoke=$CNERV_SMOKE"
# R4 finding Z-3.2 (2026-05-13): operator-visible scaffold acknowledgement.
# The trainer experiments/train_cnerv_as_renderer.py:131 raises SystemExit on
# non-smoke; this log line ensures dispatch harvest can detect scaffold-only
# state without parsing trainer source. See feedback_review_zeta_r4_LANDED_20260513.md.
if [ "${CNERV_SMOKE:-0}" != "1" ]; then
    log "stage_4_scaffold_only_acknowledged trainer=experiments/train_cnerv_as_renderer.py:131 scaffold_only=true non_smoke_path=raises_SystemExit"
fi
TRAIN_START_UTC=$(date -u +%FT%TZ)

# Build the optional flag list dynamically.
TRAIN_ARGS=(
    --output-dir "$CNERV_OUTPUT_DIR"
    --device "$CNERV_DEVICE"
    --epochs "$CNERV_EPOCHS"
    --batch-size "$CNERV_BATCH_SIZE"
    --learning-rate "$CNERV_LEARNING_RATE"
    --ema-decay "$CNERV_EMA_DECAY"
    --latent-dim "$CNERV_LATENT_DIM"
    --base-channels "$CNERV_BASE_CHANNELS"
    --n-pairs "$CNERV_N_PAIRS"
    --lambda-seg "$CNERV_LAMBDA_SEG"
    --lambda-pose "$CNERV_LAMBDA_POSE"
    --video-path "$CNERV_VIDEO_PATH"
    --seed "$CNERV_SEED"
)
if [ "$CNERV_SMOKE" = "1" ]; then
    TRAIN_ARGS+=(--smoke)
fi
if [ -n "${CNERV_PHASE_B_AUTH_MEMO:-}" ]; then
    TRAIN_ARGS+=(--auth-eval --phase-b-auth-memo "$CNERV_PHASE_B_AUTH_MEMO")
fi

set +e
"$PYBIN" experiments/train_cnerv_as_renderer.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: cost-band anchor emission (best-effort; non-fatal on missing tool
# per CLAUDE.md no-signal-loss). The trainer itself does not currently emit
# the cost-band anchor (sane_hnerv's _full_main does so at stage 14; cnerv
# trainer is scaffold-only). When the trainer's _full_main lands, this stage
# becomes redundant; leaving it in place preserves the wire-in until then.
ANCHOR_TOOL="$WORKSPACE/tools/append_cost_band_anchor.py"
if [ -x "$ANCHOR_TOOL" ] || [ -f "$ANCHOR_TOOL" ]; then
    log "stage_5_cost_band_anchor_begin"
    set +e
    "$PYBIN" "$ANCHOR_TOOL" \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --gpu "${MODAL_GPU:-A100}" \
        --started-utc "$TRAIN_START_UTC" \
        --ended-utc "$TRAIN_END_UTC" \
        --rc "$TRAIN_RC" \
        --provenance "$PROVENANCE" \
        2>&1 | tee -a "$LOG_DIR/run.log" || true
    set -e
    log "stage_5_cost_band_anchor_done"
else
    log "stage_5_cost_band_anchor_skipped tool_missing=$ANCHOR_TOOL"
fi

# Stage 6: completion record (the auth-eval JSON would be written by the
# trainer if --auth-eval is enabled and Catalog #150 is satisfied). We
# surface the path here for harvest. Per CLAUDE.md "Modal `.spawn()`
# HARVEST OR LOSE", the operator MUST register a harvest hook within 24h.
AUTH_EVAL_JSON="$OUTPUT_DIR/auth_eval.json"
ARCHIVE_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    log "LANE_CNERV_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
elif [ -f "$ARCHIVE_PATH" ]; then
    log "archive_artifact_present path=$ARCHIVE_PATH (no auth-eval JSON; trainer scaffold-only or --auth-eval not enabled)"
    log "LANE_CNERV_DONE [no-score] archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON archive_missing path=$ARCHIVE_PATH (trainer may have failed before export)"
fi

# Stage 7: lane-claim terminal-status update (best-effort; non-fatal). Per
# CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION" — leave no phantom active
# claims.
if [ -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
    set +e
    if [ "$TRAIN_RC" -eq 0 ]; then
        TERMINAL_STATUS="completed_first_anchor"
    else
        TERMINAL_STATUS="failed_rc_${TRAIN_RC}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --force --lane-id "$LANE_ID" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --status "$TERMINAL_STATUS" \
        --notes "remote_driver_terminal rc=$TRAIN_RC archive=$ARCHIVE_PATH" \
        2>&1 | tee -a "$LOG_DIR/run.log" || true
    set -e
fi

exit "$TRAIN_RC"
