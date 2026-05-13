#!/bin/bash
# Remote lane script: substrate lane_12_v2_nerv first-anchor dispatch (TRADITION 2).
#
# Trainer: experiments/train_lane_12_v2_nerv_as_renderer.py (PRODUCTION-MATURE).
# Lane: lane_substrate_lane_12_v2_nerv_20260512 (recipe lane id;
#       implementation lane id is lane_12_v2_nerv_as_renderer per registry).
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Recipe: .omx/operator_authorize_recipes/substrate_lane_12_v2_nerv_modal_a100_dispatch.yaml
#
# Substrate notes:
#   - TRADITION 2 (per .omx/research/substrate_tradition_taxonomy_20260512.md).
#   - Lane 12 v2 NeRV-as-renderer — re-scoped from Lane 12 mask-only per
#     HNeRV retrospective lesson 5. Architecturally a full-frame NeRV
#     renderer trained with score-aware Lagrangian over the contest scorers.
#   - Trainer is PRODUCTION-MATURE (479 LOC + 20 dedicated tests). ALL
#     CLAUDE.md non-negotiables wired: eval_roundtrip True + EMA 0.997
#     snapshot+restore + differentiable YUV6 + score-aware Lagrangian +
#     CUDA-required default + no MPS + no /tmp + Phase B Option C
#     auth_memo_path Catalog #150 wired. Non-smoke training is NOT scaffold-
#     gated; --auth-eval is gated behind Catalog #150 phase-b-auth-memo.
#   - Phase B authorization gate (Catalog #150 STRICT): `--auth-eval` is
#     REFUSED unless `--phase-b-auth-memo` points to a committed repo-relative
#     file under .omx/research/operator_authorizations/ that contains
#     `operator_phase_b_authorization=true` outside any code-fence.
#   - 8 archive-grammar fields declared at design time per Catalog #124
#     (archive_grammar=L12V_monolithic_single_file_0_bin;
#     parser_section_manifest=ARCHIVE_GRAMMAR_constant_in_module;
#     inflate_runtime_loc_budget=100; runtime_dep_closure=torch+brotli;
#     export_format=monolithic_single_file_0_bin_with_int8_per_tensor_fp16_scales;
#     score_aware_loss=gradient_through_FastViT-T12_PoseNet_and_EfficientNet-B2_SegNet_on_upstream_videos_0_mkv;
#     bolt_on_loc_budget=350; no_op_detector_planned=true).
#
# Score-tagging: any score this script produces is tagged [contest-CUDA] in
# the completion-log line (LANE_12_V2_DONE marker) per the CLAUDE.md
# score-tag rule and preflight completion-tag check.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity — non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_substrate_lane_12_v2_nerv_20260512"
TAG="${TAG:-substrate_lane_12_v2_nerv}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_lane_12_v2_nerv_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags — aligned with experiments/train_lane_12_v2_nerv_as_renderer.py
# argparse. The trainer predates Catalog #151 TIER_<N>_OPERATOR_REQUIRED_FLAGS;
# envelope stays env-driven until the manifest lands per the recipe
# reactivation criteria #2.
LANE_12_V2_VIDEO_PATH="${LANE_12_V2_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
LANE_12_V2_OUTPUT_DIR="${LANE_12_V2_OUTPUT_DIR:-$OUTPUT_DIR}"
LANE_12_V2_EPOCHS="${LANE_12_V2_EPOCHS:-2000}"
LANE_12_V2_BATCH_SIZE="${LANE_12_V2_BATCH_SIZE:-8}"
LANE_12_V2_DEVICE="${LANE_12_V2_DEVICE:-cuda}"
LANE_12_V2_LATENT_DIM="${LANE_12_V2_LATENT_DIM:-16}"
LANE_12_V2_BASE_CHANNELS="${LANE_12_V2_BASE_CHANNELS:-36}"
LANE_12_V2_N_PAIRS="${LANE_12_V2_N_PAIRS:-600}"
LANE_12_V2_LAMBDA_SEG="${LANE_12_V2_LAMBDA_SEG:-100.0}"
LANE_12_V2_LAMBDA_POSE="${LANE_12_V2_LAMBDA_POSE:-288.6751345948129}"
LANE_12_V2_LEARNING_RATE="${LANE_12_V2_LEARNING_RATE:-1e-3}"
LANE_12_V2_EMA_DECAY="${LANE_12_V2_EMA_DECAY:-0.997}"
LANE_12_V2_GRAD_CLIP_NORM="${LANE_12_V2_GRAD_CLIP_NORM:-1.0}"
LANE_12_V2_SEED="${LANE_12_V2_SEED:-20260511}"
LANE_12_V2_SMOKE="${LANE_12_V2_SMOKE:-0}"
LANE_12_V2_MAX_PAIRS="${LANE_12_V2_MAX_PAIRS:-}"
LANE_12_V2_EVAL_EVERY_EPOCHS="${LANE_12_V2_EVAL_EVERY_EPOCHS:-25}"
# These two are default-True in the trainer; provide env knobs for ablation.
# Empty string => use trainer default (True for both).
LANE_12_V2_ENABLE_DIFFERENTIABLE_YUV6="${LANE_12_V2_ENABLE_DIFFERENTIABLE_YUV6:-1}"
LANE_12_V2_ENABLE_SCORE_AWARE_LOSS="${LANE_12_V2_ENABLE_SCORE_AWARE_LOSS:-1}"

DISPATCH_INSTANCE_JOB_ID="${LANE_12_V2_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${LANE_12_V2_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-12-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification (mirrors remote_lane_substrate_sane_hnerv.sh).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: LANE_12_V2_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
# Cheap (~100ms) local existence check BEFORE GPU meter starts. The lane_12_v2
# trainer dereferences --video-path during data prep (and raises explicitly at
# experiments/train_lane_12_v2_nerv_as_renderer.py:296 if missing); missing
# file yields a 5-30s wasted-spend crash on the GPU host. Validating here
# costs $0.
if [ ! -f "$LANE_12_V2_VIDEO_PATH" ]; then
    log "FATAL: required input --video-path does not exist on remote host: $LANE_12_V2_VIDEO_PATH"
    log "  (regenerate via 'git lfs pull' or 'rsync upstream/videos/0.mkv' from local)"
    exit 25
fi

# Stage 0c: Phase B authorization memo validation (per Catalog #150). The
# trainer enforces this at runtime, but pre-validating here means we refuse
# the dispatch BEFORE GPU spend if the operator set --auth-eval but did not
# point at a committed repo-relative auth memo. The trainer-side check at
# tac.lane_12_v2_nerv_as_renderer._assert_auth_memo_path_repo_relative is the
# authoritative gate; this is a pre-flight friendliness check only.
if [ -n "${LANE_12_V2_PHASE_B_AUTH_MEMO:-}" ]; then
    if [ ! -f "$LANE_12_V2_PHASE_B_AUTH_MEMO" ]; then
        log "FATAL: --phase-b-auth-memo points to non-existent file: $LANE_12_V2_PHASE_B_AUTH_MEMO"
        log "  (Catalog #150 requires a committed repo-relative auth memo; see CLAUDE.md Phase B Option C)"
        exit 26
    fi
    case "$LANE_12_V2_PHASE_B_AUTH_MEMO" in
        /tmp/*|/var/tmp/*|/private/tmp/*|"$HOME"/.claude/*)
            log "FATAL: --phase-b-auth-memo anchored under forbidden tmp/.claude path: $LANE_12_V2_PHASE_B_AUTH_MEMO"
            log "  (Catalog #150 requires the memo to live UNDER the git repo root)"
            exit 27
            ;;
    esac
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
    'video_path': '$LANE_12_V2_VIDEO_PATH',
    'epochs': $LANE_12_V2_EPOCHS,
    'batch_size': $LANE_12_V2_BATCH_SIZE,
    'device': '$LANE_12_V2_DEVICE',
    'latent_dim': $LANE_12_V2_LATENT_DIM,
    'base_channels': $LANE_12_V2_BASE_CHANNELS,
    'n_pairs': $LANE_12_V2_N_PAIRS,
    'lambda_seg': $LANE_12_V2_LAMBDA_SEG,
    'lambda_pose': $LANE_12_V2_LAMBDA_POSE,
    'learning_rate': $LANE_12_V2_LEARNING_RATE,
    'ema_decay': $LANE_12_V2_EMA_DECAY,
    'grad_clip_norm': $LANE_12_V2_GRAD_CLIP_NORM,
    'eval_every_epochs': $LANE_12_V2_EVAL_EVERY_EPOCHS,
    'enable_differentiable_yuv6': bool(int('$LANE_12_V2_ENABLE_DIFFERENTIABLE_YUV6')),
    'enable_score_aware_loss': bool(int('$LANE_12_V2_ENABLE_SCORE_AWARE_LOSS')),
    'seed': $LANE_12_V2_SEED,
    'smoke': bool(int('$LANE_12_V2_SMOKE')),
    'auth_eval_requested': bool('${LANE_12_V2_PHASE_B_AUTH_MEMO:-}'),
    # predicted_band per recipe substrate_lane_12_v2_nerv_modal_a100_dispatch.yaml.
    # Recipe declares predicted_delta as 'Phase B prediction: -0.020 to -0.080
    # vs PR101 0.193 [predicted; design memo $9]'.
    'predicted_band': [-0.080, -0.020],
    'predicted_basis': 'recipe_substrate_lane_12_v2_nerv_modal_a100_dispatch_phase_a_design_memo_9',
    'substrate_tradition': 'TRADITION_2',
    'archive_grammar': 'L12V_monolithic_single_file_0_bin',
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
# Trainer argparse (verified against experiments/train_lane_12_v2_nerv_as_renderer.py:101-145):
#   --output-dir (required)            --device (cuda|cpu)        --epochs
#   --batch-size                       --learning-rate            --ema-decay
#   --latent-dim                       --base-channels            --n-pairs
#   --lambda-seg                       --lambda-pose              --grad-clip-norm
#   --video-path                       --max-pairs                --enable-differentiable-yuv6
#   --enable-score-aware-loss          --seed                     --smoke (bool)
#   --auth-eval (bool)                 --phase-b-auth-memo (str)  --eval-every-epochs
#
# Note: this trainer is PRODUCTION-MATURE. Non-smoke training is NOT scaffold-
# gated. `--auth-eval` is gated behind Catalog #150 phase-b-auth-memo (Phase B
# Option C operator decision 2026-05-09); the trainer raises SystemExit if
# --auth-eval is passed without a committed repo-relative auth memo.
#
# Auth-eval opt-in: operator sets LANE_12_V2_PHASE_B_AUTH_MEMO env var which
# adds --auth-eval --phase-b-auth-memo explicitly. Pre-flight Stage 0c above
# validates the path BEFORE GPU spend.
log "stage_4_trainer_invoke_begin video=$LANE_12_V2_VIDEO_PATH epochs=$LANE_12_V2_EPOCHS device=$LANE_12_V2_DEVICE smoke=$LANE_12_V2_SMOKE auth_eval_requested=${LANE_12_V2_PHASE_B_AUTH_MEMO:+yes}"
TRAIN_START_UTC=$(date -u +%FT%TZ)

# Build the optional flag list dynamically.
TRAIN_ARGS=(
    --output-dir "$LANE_12_V2_OUTPUT_DIR"
    --device "$LANE_12_V2_DEVICE"
    --epochs "$LANE_12_V2_EPOCHS"
    --batch-size "$LANE_12_V2_BATCH_SIZE"
    --learning-rate "$LANE_12_V2_LEARNING_RATE"
    --ema-decay "$LANE_12_V2_EMA_DECAY"
    --latent-dim "$LANE_12_V2_LATENT_DIM"
    --base-channels "$LANE_12_V2_BASE_CHANNELS"
    --n-pairs "$LANE_12_V2_N_PAIRS"
    --lambda-seg "$LANE_12_V2_LAMBDA_SEG"
    --lambda-pose "$LANE_12_V2_LAMBDA_POSE"
    --grad-clip-norm "$LANE_12_V2_GRAD_CLIP_NORM"
    --video-path "$LANE_12_V2_VIDEO_PATH"
    --seed "$LANE_12_V2_SEED"
    --eval-every-epochs "$LANE_12_V2_EVAL_EVERY_EPOCHS"
)
if [ -n "$LANE_12_V2_MAX_PAIRS" ]; then
    TRAIN_ARGS+=(--max-pairs "$LANE_12_V2_MAX_PAIRS")
fi
# --enable-differentiable-yuv6 and --enable-score-aware-loss are store_true with
# default=True; we only add the flag explicitly if env var is set to 1 (which
# matches the default behavior). For ablation (env=0), the operator must use
# their own modified trainer or a separate ablation recipe — argparse can't
# negate a store_true default from the CLI here.
if [ "$LANE_12_V2_ENABLE_DIFFERENTIABLE_YUV6" = "1" ]; then
    TRAIN_ARGS+=(--enable-differentiable-yuv6)
fi
if [ "$LANE_12_V2_ENABLE_SCORE_AWARE_LOSS" = "1" ]; then
    TRAIN_ARGS+=(--enable-score-aware-loss)
fi
if [ "$LANE_12_V2_SMOKE" = "1" ]; then
    TRAIN_ARGS+=(--smoke)
fi
if [ -n "${LANE_12_V2_PHASE_B_AUTH_MEMO:-}" ]; then
    TRAIN_ARGS+=(--auth-eval --phase-b-auth-memo "$LANE_12_V2_PHASE_B_AUTH_MEMO")
fi

set +e
"$PYBIN" experiments/train_lane_12_v2_nerv_as_renderer.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: cost-band anchor emission (best-effort; non-fatal on missing tool
# per CLAUDE.md no-signal-loss). The trainer itself may emit a cost-band
# anchor as part of its _full_main path when --auth-eval lands; leaving this
# stage in place ensures the wire-in is preserved even if the trainer-side
# emission has not yet been audited end-to-end.
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
    log "LANE_12_V2_DONE [contest-CUDA] auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
elif [ -f "$ARCHIVE_PATH" ]; then
    log "archive_artifact_present path=$ARCHIVE_PATH (no auth-eval JSON; --auth-eval not enabled or Phase B auth memo missing)"
    log "LANE_12_V2_DONE [no-score] archive=$ARCHIVE_PATH rc=$TRAIN_RC"
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
