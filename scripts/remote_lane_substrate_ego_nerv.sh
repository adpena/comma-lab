#!/bin/bash
# Remote lane script: substrate ego_nerv first-anchor dispatch (TRADITION 2).
#
# Trainer: experiments/train_ego_nerv_as_renderer.py.
# Lane: lane_substrate_ego_nerv_20260512 (recipe lane id;
#       implementation lane id is lane_ego_nerv_as_renderer per registry).
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Recipe: .omx/operator_authorize_recipes/substrate_ego_nerv_modal_a100_dispatch.yaml
#
# Substrate notes:
#   - TRADITION 2 (per .omx/research/substrate_tradition_taxonomy_20260512.md).
#   - EgoNeRV — pose-conditioned NeRV with FiLM modulation. Adds an
#     EgoNeRVPoseTable (per-pair pose embedding) and FiLMModulator on top
#     of the latent table for pose-aware conditioning.
#   - Trainer is currently SCAFFOLD-ONLY: non-smoke training raises SystemExit
#     at experiments/train_ego_nerv_as_renderer.py:140. This driver wires the
#     canonical bootstrap+heartbeat+harvest path so the moment non-smoke
#     gating lands, dispatch is one operator-flag away.
#   - Architectural note: HNeRV parity discipline lesson 4 (inflate.py ≤ 100
#     LOC; ≤ 200 with explicit waiver). submissions/ego_nerv_substrate/inflate.py
#     measured at 207 LOC per .omx/research/full_stack_integration_audit_v4
#     (advisory finding). Resolution path is during reactivation work.
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_EGO_NERV_DONE marker) per the CLAUDE.md
# score-tag rule and preflight completion-tag check.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity — non-negotiable".
set -euo pipefail

# === Catalog #244 / D1 incident anchor (commit 611495f26): canonical Modal/CUDA env hygiene ===
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + standing directive
# 2026-05-15 ("all possible should be pulled into the decorator or similar reusable
# and shareable tools and helpers and such"). Sister substrates D1/D4/Z3/Z4/Z5 carry
# this block; backfilled to all 31 sister drivers via Catalog #244 strict-flip wave.
# Future drivers should be auto-generated via tac.substrate_registry.driver_generator
# (which AUTO-EMITS the block from canonical constants in tac.deploy.modal.runtime).
# D1 dispatch 2026-05-15 (Modal T4 smoke) crashed at NVML 999 because the lane script
# did not export DALI_DISABLE_NVML=1 before DALI imported NVML.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_substrate_ego_nerv_20260512"
TAG="${TAG:-substrate_ego_nerv}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_ego_nerv_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${EGO_NERV_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$EGO_NERV_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags — aligned with experiments/train_ego_nerv_as_renderer.py argparse.
# (The trainer predates Catalog #151 TIER_<N>_OPERATOR_REQUIRED_FLAGS; envelope
# stays env-driven until the manifest lands per the recipe reactivation
# criteria #2.)
EGO_NERV_VIDEO_PATH="${EGO_NERV_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
EGO_NERV_OUTPUT_DIR="${EGO_NERV_OUTPUT_DIR:-$OUTPUT_DIR}"
EGO_NERV_EPOCHS="${EGO_NERV_EPOCHS:-2000}"
EGO_NERV_BATCH_SIZE="${EGO_NERV_BATCH_SIZE:-8}"
EGO_NERV_DEVICE="${EGO_NERV_DEVICE:-cuda}"
EGO_NERV_LATENT_DIM="${EGO_NERV_LATENT_DIM:-16}"
EGO_NERV_POSE_DIM="${EGO_NERV_POSE_DIM:-6}"
EGO_NERV_FILM_HIDDEN_DIM="${EGO_NERV_FILM_HIDDEN_DIM:-64}"
EGO_NERV_BASE_CHANNELS="${EGO_NERV_BASE_CHANNELS:-36}"
EGO_NERV_N_PAIRS="${EGO_NERV_N_PAIRS:-600}"
EGO_NERV_LAMBDA_SEG="${EGO_NERV_LAMBDA_SEG:-100.0}"
EGO_NERV_LAMBDA_POSE="${EGO_NERV_LAMBDA_POSE:-288.6751345948129}"
EGO_NERV_LEARNING_RATE="${EGO_NERV_LEARNING_RATE:-1e-3}"
EGO_NERV_EMA_DECAY="${EGO_NERV_EMA_DECAY:-0.997}"
EGO_NERV_SEED="${EGO_NERV_SEED:-20260511}"
EGO_NERV_SMOKE="${EGO_NERV_SMOKE:-0}"

DISPATCH_INSTANCE_JOB_ID="${EGO_NERV_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${EGO_NERV_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-ego-nerv] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification (mirrors remote_lane_substrate_sane_hnerv.sh).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: EGO_NERV_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
# Cheap (~100ms) local existence check BEFORE GPU meter starts. The ego_nerv
# trainer dereferences --video-path during data prep; missing file yields a
# 5-30s wasted-spend crash on the GPU host. Validating here costs $0.
if [ ! -f "$EGO_NERV_VIDEO_PATH" ]; then
    log "FATAL: required input --video-path does not exist on remote host: $EGO_NERV_VIDEO_PATH"
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
    'video_path': '$EGO_NERV_VIDEO_PATH',
    'epochs': $EGO_NERV_EPOCHS,
    'batch_size': $EGO_NERV_BATCH_SIZE,
    'device': '$EGO_NERV_DEVICE',
    'latent_dim': $EGO_NERV_LATENT_DIM,
    'pose_dim': $EGO_NERV_POSE_DIM,
    'film_hidden_dim': $EGO_NERV_FILM_HIDDEN_DIM,
    'base_channels': $EGO_NERV_BASE_CHANNELS,
    'n_pairs': $EGO_NERV_N_PAIRS,
    'lambda_seg': $EGO_NERV_LAMBDA_SEG,
    'lambda_pose': $EGO_NERV_LAMBDA_POSE,
    'learning_rate': $EGO_NERV_LEARNING_RATE,
    'ema_decay': $EGO_NERV_EMA_DECAY,
    'seed': $EGO_NERV_SEED,
    'smoke': bool(int('$EGO_NERV_SMOKE')),
    # predicted_band per recipe substrate_ego_nerv_modal_a100_dispatch.yaml.
    # The recipe declares predicted_delta as 'unknown — TRADITION 2 substrate
    # predates Catalog #124 design-time prediction discipline'; the band is
    # therefore reported as null until the first empirical anchor lands.
    'predicted_band': None,
    'predicted_basis': 'recipe_substrate_ego_nerv_modal_a100_dispatch_unknown_pending_first_anchor',
    'substrate_tradition': 'TRADITION_2',
    'format_id_hex': '0x68',
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
# Trainer argparse (verified against experiments/train_ego_nerv_as_renderer.py:48-65):
#   --output-dir (required)        --device (cuda|cpu)        --epochs
#   --batch-size                   --learning-rate            --ema-decay
#   --latent-dim                   --pose-dim                 --film-hidden-dim
#   --base-channels                --n-pairs                  --lambda-seg
#   --lambda-pose                  --video-path               --seed
#   --smoke (bool)                 --auth-eval (bool)         --phase-b-auth-memo (str)
#
# Note: the trainer is currently SCAFFOLD-ONLY per
# experiments/train_ego_nerv_as_renderer.py:140 — non-smoke runs raise
# SystemExit("[ego_nerv] non-smoke training is operator-gated..."). To exercise
# this driver end-to-end before the operator-gating lands, set EGO_NERV_SMOKE=1.
#
# Auth-eval is behind Catalog #150 phase-b-auth-memo gate (mandatory if
# --auth-eval is enabled); not threaded by default — operator opts in via
# EGO_NERV_PHASE_B_AUTH_MEMO env var which adds --auth-eval --phase-b-auth-memo
# explicitly.
log "stage_4_trainer_invoke_begin video=$EGO_NERV_VIDEO_PATH epochs=$EGO_NERV_EPOCHS device=$EGO_NERV_DEVICE smoke=$EGO_NERV_SMOKE"
# R4 finding Z-3.2 (2026-05-13): operator-visible scaffold acknowledgement.
# The trainer experiments/train_ego_nerv_as_renderer.py:140 raises SystemExit on
# non-smoke; this log line ensures dispatch harvest can detect scaffold-only
# state without parsing trainer source. See feedback_review_zeta_r4_LANDED_20260513.md.
if [ "${EGO_NERV_SMOKE:-0}" != "1" ]; then
    log "stage_4_scaffold_only_acknowledged trainer=experiments/train_ego_nerv_as_renderer.py:140 scaffold_only=true non_smoke_path=raises_SystemExit"
fi
TRAIN_START_UTC=$(date -u +%FT%TZ)

# Build the optional flag list dynamically.
TRAIN_ARGS=(
    --output-dir "$EGO_NERV_OUTPUT_DIR"
    --device "$EGO_NERV_DEVICE"
    --epochs "$EGO_NERV_EPOCHS"
    --batch-size "$EGO_NERV_BATCH_SIZE"
    --learning-rate "$EGO_NERV_LEARNING_RATE"
    --ema-decay "$EGO_NERV_EMA_DECAY"
    --latent-dim "$EGO_NERV_LATENT_DIM"
    --pose-dim "$EGO_NERV_POSE_DIM"
    --film-hidden-dim "$EGO_NERV_FILM_HIDDEN_DIM"
    --base-channels "$EGO_NERV_BASE_CHANNELS"
    --n-pairs "$EGO_NERV_N_PAIRS"
    --lambda-seg "$EGO_NERV_LAMBDA_SEG"
    --lambda-pose "$EGO_NERV_LAMBDA_POSE"
    --video-path "$EGO_NERV_VIDEO_PATH"
    --seed "$EGO_NERV_SEED"
)
if [ "$EGO_NERV_SMOKE" = "1" ]; then
    TRAIN_ARGS+=(--smoke)
fi
if [ -n "${EGO_NERV_PHASE_B_AUTH_MEMO:-}" ]; then
    TRAIN_ARGS+=(--auth-eval --phase-b-auth-memo "$EGO_NERV_PHASE_B_AUTH_MEMO")
fi

set +e
"$PYBIN" experiments/train_ego_nerv_as_renderer.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: cost-band anchor emission (best-effort; non-fatal on missing tool
# per CLAUDE.md no-signal-loss). The trainer itself does not currently emit
# the cost-band anchor (sane_hnerv's _full_main does so at stage 14; ego_nerv
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
    log "LANE_EGO_NERV_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
elif [ -f "$ARCHIVE_PATH" ]; then
    log "archive_artifact_present path=$ARCHIVE_PATH (no auth-eval JSON; trainer scaffold-only or --auth-eval not enabled)"
    log "LANE_EGO_NERV_DONE [no-score] archive=$ARCHIVE_PATH rc=$TRAIN_RC"
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
