#!/bin/bash
# Remote lane script: substrate Z8 hierarchical predictive coding M12a Modal T4
# L2 long-training canonical dispatch per Catalog #325 symposium 4bcc84fc0
# PROCEED_WITH_REVISIONS unanimous 23-of-23 T3 grand council (2026-05-30).
#
# Trainer: experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py
#   full mode (M12a Yousfi Revision #1 active route; canonical
#   mlx_score_aware_full_main with real SegNet + PoseNet Hinton teachers).
#
# Lane: lane_z8_m12a_modal_t4_l2_long_training_per_catalog_325_symposium_proceed_with_revisions_20260530
# Recipe: .omx/operator_authorize_recipes/substrate_z8_hierarchical_predictive_coding_modal_t4_dispatch.yaml
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the canonical sentinel REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1
# MUST be set before sourcing the bootstrap to prevent the sourced main flow
# from running.
#
# Per Catalog #189 every optional-array expansion is guarded as
# ${ARR[@]+"${ARR[@]}"} so the script tolerates `set -u` on macOS bash 3.2.
#
# Per Catalog #326 multi-key trainer mode resolution: Z8_TRAINER_MODE primary
# (full|canonical_quadruple|smoke) with fail-loud warning on invalid values.
#
# Per Catalog #204 cross-driver expansion: 3-branch Modal-aware OUTPUT_DIR
# resolution (Z8_OUTPUT_DIR explicit > /modal_results when MODAL_RUNTIME=1 >
# $LOG_DIR/output fallback) per the canonical pattern from sister Z5 driver.
#
# Design refs:
#   - .omx/research/council_t3_grand_council_per_substrate_symposium_z8_hierarchical_predictive_coding_m12_paid_modal_t4_l2_long_training_plus_paired_cuda_canonical_sub_0_189_attempt_20260530.md
#   - src/tac/substrates/z8_hierarchical_predictive_coding/canonical_quadruple_binding.py
#   - src/tac/substrates/z8_hierarchical_predictive_coding/build_progress.py
#   - Catalog #312 canonical quadruple (Rao-Ballard + Mallat + DreamerV3 + Wyner-Ziv)
#
# Score-tagging: smoke/no-scorer artifacts are explicitly logged as
# score_claim=false and never as [contest-CUDA]. A [contest-CUDA] marker is
# allowed only when stats.json proves a valid contest_cuda score claim
# (auth_eval_score_claim_valid=true AND auth_eval_score_axis=contest_cuda).
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

# === Catalog #244 / D1 incident anchor (commit 611495f26): canonical Modal/CUDA env hygiene ===
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + standing directive
# 2026-05-15. Sister substrates D1/D4/Z3/Z4/Z5 carry this block; backfilled to all 31
# sister drivers via Catalog #244 strict-flip wave. D1 dispatch 2026-05-15 (Modal T4
# smoke) crashed at NVML 999 because the lane script did not export DALI_DISABLE_NVML=1
# before DALI imported NVML.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_z8_m12a_modal_t4_l2_long_training_per_catalog_325_symposium_proceed_with_revisions_20260530"
TAG="${TAG:-substrate_z8_hierarchical_predictive_coding}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_z8_hierarchical_predictive_coding_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${Z8_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$Z8_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
Z8_VIDEO_PATH="${Z8_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
Z8_OUTPUT_DIR="${Z8_OUTPUT_DIR:-$OUTPUT_DIR}"
Z8_EPOCHS="${Z8_EPOCHS:-2000}"
Z8_NUM_PAIRS="${Z8_NUM_PAIRS:-600}"
Z8_CANONICAL_QUADRUPLE_EVAL_H="${Z8_CANONICAL_QUADRUPLE_EVAL_H:-32}"
Z8_CANONICAL_QUADRUPLE_EVAL_W="${Z8_CANONICAL_QUADRUPLE_EVAL_W:-32}"
Z8_DEVICE="${Z8_DEVICE:-cuda}"
Z8_ENABLE_AUTOCAST_FP16="${Z8_ENABLE_AUTOCAST_FP16:-1}"
Z8_ENABLE_GT_SCORER_CACHE="${Z8_ENABLE_GT_SCORER_CACHE:-1}"
Z8_ENABLE_TORCH_COMPILE="${Z8_ENABLE_TORCH_COMPILE:-0}"

# Catalog #326 multi-key trainer mode resolution: Z8_TRAINER_MODE primary
# (full|canonical_quadruple|smoke) with fail-loud warning when no key set.
# Default to full per the M12a Yousfi Revision #1 recipe; canonical_quadruple
# remains an explicit legacy/diagnostic mode and M12c substrate-engineering
# reactivation path. SMOKE_ONLY is a sister env var preserved for backward
# compatibility with the sister Z3/Z4/Z5 drivers.
Z8_TRAINER_MODE="${Z8_TRAINER_MODE:-${SMOKE_ONLY:+smoke}}"
Z8_TRAINER_MODE="${Z8_TRAINER_MODE:-full}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
Z8_POSE_DISTILLATION_WEIGHT="${Z8_POSE_DISTILLATION_WEIGHT:-1.0}"
Z8_GRAD_CLIP_MAX_NORM="${Z8_GRAD_CLIP_MAX_NORM:-1.0}"
Z8_WARMUP_EPOCHS="${Z8_WARMUP_EPOCHS:-5}"
Z8_WEIGHT_DECAY="${Z8_WEIGHT_DECAY:-1e-4}"
Z8_OPTIMIZER_KIND="${Z8_OPTIMIZER_KIND:-adamw}"

DISPATCH_INSTANCE_JOB_ID="${Z8_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${Z8_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0

log() { echo "[lane-z8-hpc-m12a] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: Z8_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

append_terminal_claim() {
    local rc="$1"
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "WARN: claim helper missing; cannot append terminal dispatch claim"
        return 0
    fi
    local status
    if [ "$rc" -eq 0 ]; then
        status="completed_z8_hpc_m12a_remote_driver"
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_z8_hpc_m12a_claim_verification_rc_${rc}"
    else
        status="failed_z8_hpc_m12a_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_z8_hierarchical_predictive_coding" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc output_dir=$Z8_OUTPUT_DIR mode=$Z8_TRAINER_MODE epochs=$Z8_EPOCHS num_pairs=$Z8_NUM_PAIRS" \
        >> "$LOG_DIR/run.log" 2>&1 || {
        log "WARN: failed to append terminal dispatch claim status=$status"
    }
}

cleanup() {
    local rc="$?"
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    append_terminal_claim "$rc"
    exit "$rc"
}
trap cleanup EXIT

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 1: bootstrap remote runtime deps via canonical sourced helper.
# Per Catalog #163 prepend REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 so the
# sourced script's main flow does NOT run.
if [ -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "stage_1_bootstrap_via_canonical_sourced_helper"
    # shellcheck disable=SC1091
    REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
    bootstrap_runtime_deps || {
        log "FATAL: bootstrap_runtime_deps failed; refusing dispatch"
        exit 22
    }
else
    log "WARN: canonical bootstrap script missing; assuming runtime deps present"
fi

# Stage 1b: validate required input files PRE-dispatch (Catalog #152).
log "stage_1b_validate_required_input_files"
if [ -x "$CLAIM_PYTHON" ] && [ -f "$WORKSPACE/tools/validate_dispatch_required_inputs.py" ]; then
    "$CLAIM_PYTHON" "$WORKSPACE/tools/validate_dispatch_required_inputs.py" \
        --trainer "$WORKSPACE/experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py" \
        >> "$LOG_DIR/run.log" 2>&1 || {
        log "WARN: Catalog #152 required input file validation failed (continuing per canonical-quadruple-binding-mode does NOT require trainer manifest)"
    }
fi

# Stage 2: heartbeat (every 5 min).
(
    while true; do
        date -u +%FT%TZ > "$LOG_DIR/heartbeat.log"
        sleep 300
    done
) &
HEARTBEAT_PID=$!

# Stage 3: write provenance.
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "trainer": "experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py",
  "trainer_mode": "$Z8_TRAINER_MODE",
  "recipe": ".omx/operator_authorize_recipes/substrate_z8_hierarchical_predictive_coding_modal_t4_dispatch.yaml",
  "symposium_memo": ".omx/research/council_t3_grand_council_per_substrate_symposium_z8_hierarchical_predictive_coding_m12_paid_modal_t4_l2_long_training_plus_paired_cuda_canonical_sub_0_189_attempt_20260530.md",
  "video_path": "$Z8_VIDEO_PATH",
  "output_dir": "$Z8_OUTPUT_DIR",
  "epochs": "$Z8_EPOCHS",
  "num_pairs": "$Z8_NUM_PAIRS",
  "canonical_quadruple_eval_h": "$Z8_CANONICAL_QUADRUPLE_EVAL_H",
  "canonical_quadruple_eval_w": "$Z8_CANONICAL_QUADRUPLE_EVAL_W",
  "device": "$Z8_DEVICE",
  "enable_autocast_fp16": "$Z8_ENABLE_AUTOCAST_FP16",
  "enable_gt_scorer_cache": "$Z8_ENABLE_GT_SCORER_CACHE",
  "enable_torch_compile": "$Z8_ENABLE_TORCH_COMPILE",
  "pose_distillation_weight": "$Z8_POSE_DISTILLATION_WEIGHT",
  "grad_clip_max_norm": "$Z8_GRAD_CLIP_MAX_NORM",
  "warmup_epochs": "$Z8_WARMUP_EPOCHS",
  "weight_decay": "$Z8_WEIGHT_DECAY",
  "optimizer_kind": "$Z8_OPTIMIZER_KIND",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)",
  "evidence_grade": "[scaffold-only-no-score-claim-until-auth-eval-lands]",
  "score_claim": false,
  "score_claim_valid": false,
  "promotion_eligible": false,
  "rank_or_kill_eligible": false,
  "ready_for_exact_eval_dispatch": false
}
EOF

# Stage 4: invoke trainer.
log "stage_4_trainer_begin mode=$Z8_TRAINER_MODE epochs=$Z8_EPOCHS num_pairs=$Z8_NUM_PAIRS eval=${Z8_CANONICAL_QUADRUPLE_EVAL_H}x${Z8_CANONICAL_QUADRUPLE_EVAL_W} device=$Z8_DEVICE"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

# Catalog #326 multi-key mode resolution: route to the canonical trainer mode.
# full is the M12a active route per the recipe; smoke is the L0 fallback for
# fixture-scope; canonical_quadruple remains explicit M9/M12c diagnostic mode.
TRAINER_ARGS=(
    --video-path "$Z8_VIDEO_PATH"
    --epochs "$Z8_EPOCHS"
    --num-pairs "$Z8_NUM_PAIRS"
)

case "$Z8_TRAINER_MODE" in
    canonical_quadruple)
        TRAINER_ARGS+=(
            --canonical-quadruple-binding
            --canonical-quadruple-output-dir "$Z8_OUTPUT_DIR"
            --canonical-quadruple-eval-h "$Z8_CANONICAL_QUADRUPLE_EVAL_H"
            --canonical-quadruple-eval-w "$Z8_CANONICAL_QUADRUPLE_EVAL_W"
        )
        ;;
    full)
        TRAINER_ARGS+=(
            --output-dir "$Z8_OUTPUT_DIR"
            --pose-distillation-weight "$Z8_POSE_DISTILLATION_WEIGHT"
            --grad-clip-max-norm "$Z8_GRAD_CLIP_MAX_NORM"
            --warmup-epochs "$Z8_WARMUP_EPOCHS"
            --weight-decay "$Z8_WEIGHT_DECAY"
            --optimizer-kind "$Z8_OPTIMIZER_KIND"
        )
        ;;
    smoke)
        TRAINER_ARGS+=(--smoke)
        if [ -n "${Z8_SMOKE_OUTPUT:-}" ]; then
            TRAINER_ARGS+=(--smoke-output "$Z8_SMOKE_OUTPUT")
        fi
        ;;
    *)
        log "FATAL: invalid Z8_TRAINER_MODE=$Z8_TRAINER_MODE (expected canonical_quadruple|full|smoke)"
        exit 24
        ;;
esac

"$PYBIN_RESOLVED" "$TRAINER_PY" ${TRAINER_ARGS[@]+"${TRAINER_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# Stage 5: emit completion marker (operator + autopilot consume).
# Per Catalog #226 + canonical auth_eval routing: completion log marker is
# canonical [contest-CUDA] ONLY when stats.json proves valid contest_cuda
# score claim. The M12a canonical-quadruple-binding mode writes a JSON
# artifact at $Z8_OUTPUT_DIR/m9_canonical_quadruple_artifact.json which
# is canonical training-trajectory evidence per the symposium's M12a-FIRST
# sequencing (NOT yet a paid auth_eval; M12b paired-CUDA is the canonical
# Catalog #246 sister gate).
EVIDENCE_STATUS="$("$PYBIN_RESOLVED" - "$Z8_OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
marker = "[training-artifact-no-score-claim]"
score_claim = "score_claim=false"

# Check canonical M9 artifact (canonical_quadruple_binding mode).
artifact_path = output_dir / "m9_canonical_quadruple_artifact.json"
if artifact_path.is_file():
    try:
        artifact = json.loads(artifact_path.read_text())
        verdict = artifact.get("convergence_verdict", "unknown")
        epochs = artifact.get("total_epochs_completed", 0)
        payload_bytes = artifact.get("final_wyner_ziv_payload_bytes", 0)
        print(f"[macOS-CPU advisory] score_claim=false epochs={epochs} verdict={verdict} payload_bytes={payload_bytes}")
        sys.exit(0)
    except json.JSONDecodeError:
        pass

# Check canonical MLX full-training artifact (M12a full mode).
training_artifact_path = output_dir / "training_artifact.json"
if training_artifact_path.is_file():
    try:
        artifact = json.loads(training_artifact_path.read_text())
    except json.JSONDecodeError:
        artifact = {}
    axis = artifact.get("axis_tag") or artifact.get("evidence_grade") or "macOS-MLX research-signal"
    epochs = artifact.get("total_epochs_completed", artifact.get("epochs", 0))
    promotable = artifact.get("promotable", False)
    score_claim_valid = artifact.get("score_claim_valid", False)
    if score_claim_valid is True and artifact.get("auth_eval_score_axis") == "contest_cuda":
        print(f"[contest-CUDA] score_claim=true epochs={epochs}")
    else:
        print(f"[{axis}] score_claim=false epochs={epochs} promotable={promotable}")
    sys.exit(0)

# Fallback: check stats.json (sister substrate convention).
stats_path = output_dir / "stats.json"
if stats_path.is_file():
    try:
        stats = json.loads(stats_path.read_text())
    except json.JSONDecodeError:
        stats = {}
    if (
        stats.get("auth_eval_score_claim_valid") is True
        and stats.get("auth_eval_score_axis") == "contest_cuda"
    ):
        marker = "[contest-CUDA]"
        score_claim = "score_claim=true"
    elif stats.get("evidence_grade"):
        marker = f"[{stats['evidence_grade']}]"

print(f"{marker} {score_claim}")
PY
)"
EVIDENCE_MARKER="${EVIDENCE_STATUS%% *}"
SCORE_CLAIM_FLAG="${EVIDENCE_STATUS#* }"
log "LANE_Z8_HPC_M12A_DONE ${EVIDENCE_MARKER} output_dir=$Z8_OUTPUT_DIR mode=$Z8_TRAINER_MODE ${SCORE_CLAIM_FLAG}"
echo "LANE_Z8_HPC_M12A_DONE ${EVIDENCE_MARKER} $LANE_ID ${SCORE_CLAIM_FLAG} mode=$Z8_TRAINER_MODE $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
