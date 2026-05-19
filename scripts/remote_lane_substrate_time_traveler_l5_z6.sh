#!/bin/bash
# Remote lane script: substrate Z6 Time-Traveler L5 predictive-coding smoke + full dispatch.
#
# Trainer: experiments/train_substrate_time_traveler_l5_z6.py
# Lane: lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516
# Recipe: .omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the sentinel ``REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1`` is
# prepended to the source line so the sourced script's main flow does NOT
# run.
#
# Per Catalog #189 every optional-array expansion is guarded as
# ``${ARR[@]+"${ARR[@]}"}`` so the script tolerates `set -u` on macOS bash 3.2.
#
# Per Catalog #244 the canonical 3-export NVML/CUDA env block is emitted
# IMMEDIATELY after `set -euo pipefail` so DALI does not crash with
# `nvml error (999)` and CUBLAS produces deterministic results.
#
# Design refs:
#   - .omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md
#   - lane_z5_predictive_coding_world_model_step3_20260514 (sister Z5 L1 scaffold pattern)
#   - Rao & Ballard (1999) "Predictive coding in the visual cortex"
#   - Atick-Redlich (1990) cooperative-receiver theorem
#   - Perez et al. (2017) FiLM modulation
#
# Score-tagging: smoke/no-scorer artifacts are explicitly logged as
# score_claim=false and never as [contest-CUDA]. A [contest-CUDA] marker is
# allowed only when stats.json proves a valid contest_cuda score claim
# (auth_eval_score_claim_valid=true AND auth_eval_score_axis=contest_cuda).
# Per Catalog #204 the output is written to
# /modal_results/${DISPATCH_INSTANCE_JOB_ID}/output for durable provider
# custody when MODAL_RUNTIME=1.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

# Catalog #244 canonical 3-export NVML/CUDA env block. Emitted IMMEDIATELY
# after `set -euo pipefail` per the canonical helper at
# tac.deploy.modal.runtime so DALI does not crash with `nvml error (999)`
# inside fn.experimental.inputs.video and so CUBLAS produces deterministic
# matmul outputs. Sister D1/D4/Z3/Z4/Z5 substrate drivers carry the same
# block; commit 611495f26 (the canonical anchor).
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
DEFAULT_LANE_ID="lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516"
LANE_ID="${Z6_LANE_ID:-${PACT_DISPATCH_LANE_ID:-$DEFAULT_LANE_ID}}"
TAG="${TAG:-substrate_time_traveler_l5_z6}"
RECIPE_PATH="${Z6_RECIPE_PATH:-.omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_time_traveler_l5_z6_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${Z6_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$Z6_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
# Smoke vs full ladder per Catalog #326 (DRIVER-FIX wave 2026-05-18).
#
# Z6_TRAINER_MODE is the canonical env var ("smoke" or "full"); SMOKE_ONLY
# is preserved for back-compat. Z6-v2 Wave 2 DEFER 2026-05-18 (call_ids
# fc-01KRW7RHFHP640BHTQ0FZM3M38 + fc-01KRW7ZCYK5XF6MSHD24R71A46) anchored
# this bug class: the Wave 2 recipe env_overrides did NOT set SMOKE_ONLY
# so the driver defaulted to smoke and the trainer ran _smoke_main with
# synthetic cfg, producing 27,850-param outputs INSTEAD of the council-
# binding ~300K depth=3 spec. Per CLAUDE.md "Bugs must be permanently
# fixed AND self-protected against": precedence is Z6_TRAINER_MODE >
# SMOKE_ONLY; if neither is set, default to smoke + emit a loud WARN
# banner so future operators see the implicit smoke-mode at log time.
Z6_TRAINER_MODE="${Z6_TRAINER_MODE:-}"
SMOKE_ONLY="${SMOKE_ONLY:-}"
if [ -n "$Z6_TRAINER_MODE" ]; then
    case "$Z6_TRAINER_MODE" in
        smoke|SMOKE|Smoke)
            SMOKE_ONLY="1"
            ;;
        full|FULL|Full)
            SMOKE_ONLY="0"
            ;;
        *)
            echo "[lane-z6-pcwm] FATAL: invalid Z6_TRAINER_MODE=$Z6_TRAINER_MODE; expected smoke|full" >&2
            exit 29
            ;;
    esac
elif [ -z "$SMOKE_ONLY" ]; then
    echo "[lane-z6-pcwm] WARN: neither Z6_TRAINER_MODE nor SMOKE_ONLY set; defaulting to smoke. Per Catalog #326 recipes SHOULD declare Z6_TRAINER_MODE=full or SMOKE_ONLY=0 for full-mode dispatch." >&2
    SMOKE_ONLY="1"
fi
Z6_VIDEO_PATH="${Z6_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
Z6_OUTPUT_DIR="${Z6_OUTPUT_DIR:-$OUTPUT_DIR}"
Z6_EPOCHS="${Z6_EPOCHS:-}"
if [ -z "$Z6_EPOCHS" ]; then
    if [ "$SMOKE_ONLY" = "1" ]; then
        Z6_EPOCHS="3"
    else
        Z6_EPOCHS="300"
    fi
fi
Z6_BATCH_SIZE="${Z6_BATCH_SIZE:-4}"
Z6_LR="${Z6_LR:-5e-4}"
Z6_LAMBDA_RESIDUAL_ENTROPY="${Z6_LAMBDA_RESIDUAL_ENTROPY:-1.0}"
Z6_PREDICTOR_HIDDEN_DIM="${Z6_PREDICTOR_HIDDEN_DIM:-64}"
Z6_PREDICTOR_FILM_MLP_HIDDEN_DIM="${Z6_PREDICTOR_FILM_MLP_HIDDEN_DIM:-32}"
Z6_PREDICTOR_KERNEL_SIZE="${Z6_PREDICTOR_KERNEL_SIZE:-3}"
Z6_PREDICTOR_EGO_MOTION_DIM="${Z6_PREDICTOR_EGO_MOTION_DIM:-8}"
Z6_IDENTITY_PREDICTOR="${Z6_IDENTITY_PREDICTOR:-false}"
Z6_PREDICTOR_ARCHITECTURE="${Z6_PREDICTOR_ARCHITECTURE:-single_layer_film_75k}"
Z6_PREDICTOR_PARAM_COUNT_TARGET="${Z6_PREDICTOR_PARAM_COUNT_TARGET:-300000}"
Z6_EGO_SOURCE="${Z6_EGO_SOURCE:-posenet_projection}"
Z6_ENABLE_PAIRED_CONTROL_INITIALIZATION="${Z6_ENABLE_PAIRED_CONTROL_INITIALIZATION:-shared_modules_seed_order_matched_v2}"
Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE="${Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE:-false}"
Z6_PAIRED_CONTROL_DISAMBIGUATOR_DECISION_CRITERION_DELTA_S="${Z6_PAIRED_CONTROL_DISAMBIGUATOR_DECISION_CRITERION_DELTA_S:-0.005}"
Z6_DEVICE="${Z6_DEVICE:-cuda}"
Z6_UPSTREAM_DIR="${Z6_UPSTREAM_DIR:-$WORKSPACE/upstream}"
Z6_ENABLE_AUTOCAST_FP16="${Z6_ENABLE_AUTOCAST_FP16:-false}"
Z6_MAX_PAIRS="${Z6_MAX_PAIRS:-}"
Z6_SKIP_AUTH_EVAL="${Z6_SKIP_AUTH_EVAL:-}"

DISPATCH_INSTANCE_JOB_ID="${Z6_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${Z6_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""
CLAIM_VERIFIED=0
EVIDENCE_MARKER="[not-yet-classified]"
SCORE_CLAIM_FLAG="score_claim=unknown"

log() { echo "[lane-z6-pcwm] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: Z6_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
    exit 21
fi

CLAIM_PYTHON="${PYBIN:-}"
if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
    CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
fi
if [ -z "$CLAIM_PYTHON" ]; then
    CLAIM_PYTHON="python3"
fi

verify_active_dispatch_claim() {
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "FATAL: claim helper missing; cannot verify active dispatch claim"
        exit 26
    fi
    local claim_summary_json="$LOG_DIR/dispatch_claim_summary.json"
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --live-only \
        --format json \
        > "$claim_summary_json" || {
        log "FATAL: claim summary failed; refusing remote driver startup"
        exit 26
    }
    "$CLAIM_PYTHON" - "$claim_summary_json" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" <<'PY' || {
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
lane_id = sys.argv[2]
job_id = sys.argv[3]
payload = json.loads(summary_path.read_text(encoding="utf-8"))
for row in payload.get("active", []):
    if row.get("lane_id") == lane_id and row.get("instance_job_id") == job_id:
        raise SystemExit(0)
print(
    f"no active dispatch claim for lane_id={lane_id} instance_job_id={job_id}",
    file=sys.stderr,
)
raise SystemExit(1)
PY
        log "FATAL: no active dispatch claim for lane=$LANE_ID instance/job=$DISPATCH_INSTANCE_JOB_ID"
        exit 27
    }
    CLAIM_VERIFIED=1
    log "Stage 0 DONE: active dispatch claim verified"
}

append_terminal_claim() {
    local rc="$1"
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "WARN: claim helper missing; cannot append terminal dispatch claim"
        return 0
    fi
    local status
    if [ "$rc" -eq 0 ]; then
        if [ "${EVIDENCE_MARKER:-}" = "[contest-CUDA]" ] && [ "${SCORE_CLAIM_FLAG:-}" = "score_claim=true" ]; then
            status="completed_z6_pcwm_remote_driver_contest_cuda_score_claim"
        else
            status="completed_z6_pcwm_remote_driver_no_score_claim"
        fi
    elif [ "${CLAIM_VERIFIED:-0}" != "1" ]; then
        status="failed_z6_pcwm_claim_verification_rc_${rc}"
    else
        status="failed_z6_pcwm_remote_driver_rc_${rc}"
    fi
    "$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --force \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "remote_lane_substrate_time_traveler_l5_z6" \
        --status "$status" \
        --notes "remote_driver_terminal rc=$rc evidence_marker=${EVIDENCE_MARKER:-unknown} ${SCORE_CLAIM_FLAG:-score_claim=unknown} stats_json=$Z6_OUTPUT_DIR/stats.json output_dir=$Z6_OUTPUT_DIR smoke=$SMOKE_ONLY identity_predictor=$Z6_IDENTITY_PREDICTOR predictor_architecture=$Z6_PREDICTOR_ARCHITECTURE emit_disambiguator=$Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE predictor_kernel=$Z6_PREDICTOR_KERNEL_SIZE" \
        >> "$LOG_DIR/run.log" 2>&1 || {
        log "WARN: failed to append terminal dispatch claim status=$status"
    }
}

cleanup() {
    local rc="$?"
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    append_terminal_claim "$rc"
    exit "$rc"
}
trap cleanup EXIT
verify_active_dispatch_claim

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

# Stage 2: heartbeat (every 5 min).
(
    HEARTBEAT_SLEEP_PID=""
    trap 'if [ -n "$HEARTBEAT_SLEEP_PID" ]; then kill "$HEARTBEAT_SLEEP_PID" 2>/dev/null || true; fi; exit 0' TERM INT EXIT
    while true; do
        date -u +%FT%TZ > "$LOG_DIR/heartbeat.log"
        sleep 300 &
        HEARTBEAT_SLEEP_PID="$!"
        wait "$HEARTBEAT_SLEEP_PID" || true
        HEARTBEAT_SLEEP_PID=""
    done
) &
HEARTBEAT_PID=$!

# Stage 3: write provenance.
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "trainer": "experiments/train_substrate_time_traveler_l5_z6.py",
  "recipe": "$RECIPE_PATH",
  "video_path": "$Z6_VIDEO_PATH",
  "output_dir": "$Z6_OUTPUT_DIR",
  "epochs": "$Z6_EPOCHS",
  "batch_size": "$Z6_BATCH_SIZE",
  "lr": "$Z6_LR",
  "device": "$Z6_DEVICE",
  "lambda_residual_entropy": "$Z6_LAMBDA_RESIDUAL_ENTROPY",
  "predictor_hidden_dim": "$Z6_PREDICTOR_HIDDEN_DIM",
  "predictor_film_mlp_hidden_dim": "$Z6_PREDICTOR_FILM_MLP_HIDDEN_DIM",
  "predictor_kernel_size": "$Z6_PREDICTOR_KERNEL_SIZE",
  "predictor_ego_motion_dim": "$Z6_PREDICTOR_EGO_MOTION_DIM",
  "identity_predictor": "$Z6_IDENTITY_PREDICTOR",
  "predictor_architecture": "$Z6_PREDICTOR_ARCHITECTURE",
  "predictor_param_count_target": "$Z6_PREDICTOR_PARAM_COUNT_TARGET",
  "ego_source": "$Z6_EGO_SOURCE",
  "paired_control_initialization": "$Z6_ENABLE_PAIRED_CONTROL_INITIALIZATION",
  "emit_identity_predictor_disambiguator_archive": "$Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE",
  "paired_control_disambiguator_decision_criterion_delta_s": "$Z6_PAIRED_CONTROL_DISAMBIGUATOR_DECISION_CRITERION_DELTA_S",
  "enable_autocast_fp16": "$Z6_ENABLE_AUTOCAST_FP16",
  "max_pairs": "$Z6_MAX_PAIRS",
  "skip_auth_eval": "$Z6_SKIP_AUTH_EVAL",
  "smoke_only": "$SMOKE_ONLY",
  "dispatch_instance_job_id": "$DISPATCH_INSTANCE_JOB_ID",
  "started_at_utc": "$(date -u +%FT%TZ)"
}
EOF

# Stage 4: invoke trainer.
log "stage_4_trainer_begin epochs=$Z6_EPOCHS lambda_res=$Z6_LAMBDA_RESIDUAL_ENTROPY kernel=$Z6_PREDICTOR_KERNEL_SIZE identity=$Z6_IDENTITY_PREDICTOR smoke=$SMOKE_ONLY"
TRAINER_PY="$WORKSPACE/experiments/train_substrate_time_traveler_l5_z6.py"
if [ ! -f "$TRAINER_PY" ]; then
    log "FATAL: trainer missing at $TRAINER_PY"
    exit 23
fi

PYBIN_RESOLVED="$CLAIM_PYTHON"

SMOKE_FLAG_ARGS=()
if [ "$SMOKE_ONLY" = "1" ]; then
    SMOKE_FLAG_ARGS+=(--smoke)
fi

IDENTITY_PREDICTOR_ARGS=()
case "$Z6_IDENTITY_PREDICTOR" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        IDENTITY_PREDICTOR_ARGS+=(--identity-predictor)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z6_IDENTITY_PREDICTOR=$Z6_IDENTITY_PREDICTOR; expected 0/1/true/false"
        exit 24
        ;;
esac

AUTOCAST_FLAG_ARGS=()
case "$Z6_ENABLE_AUTOCAST_FP16" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        AUTOCAST_FLAG_ARGS+=(--enable-autocast-fp16)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z6_ENABLE_AUTOCAST_FP16=$Z6_ENABLE_AUTOCAST_FP16; expected 0/1/true/false"
        exit 25
        ;;
esac

DISAMBIGUATOR_FLAG_ARGS=()
case "$Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        DISAMBIGUATOR_FLAG_ARGS+=(--emit-identity-predictor-disambiguator-archive)
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE=$Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE; expected 0/1/true/false"
        exit 28
        ;;
esac

MAX_PAIRS_ARGS=()
if [ -n "$Z6_MAX_PAIRS" ]; then
    case "$Z6_MAX_PAIRS" in
        *[!0-9]*|"")
            log "FATAL: invalid Z6_MAX_PAIRS=$Z6_MAX_PAIRS; expected positive integer"
            exit 30
            ;;
    esac
    if [ "$Z6_MAX_PAIRS" -le 0 ]; then
        log "FATAL: invalid Z6_MAX_PAIRS=$Z6_MAX_PAIRS; expected positive integer"
        exit 30
    fi
    MAX_PAIRS_ARGS+=(--max-pairs "$Z6_MAX_PAIRS")
fi

SKIP_AUTH_EVAL=0
case "$Z6_SKIP_AUTH_EVAL" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        SKIP_AUTH_EVAL=1
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
        ;;
    *)
        log "FATAL: invalid Z6_SKIP_AUTH_EVAL=$Z6_SKIP_AUTH_EVAL; expected 0/1/true/false"
        exit 31
        ;;
esac
if [ -n "$Z6_MAX_PAIRS" ] && [ "$Z6_MAX_PAIRS" -lt 600 ]; then
    SKIP_AUTH_EVAL=1
    log "stage_4_pair_capped_smoke_skips_auth_eval max_pairs=$Z6_MAX_PAIRS full_pairs=600"
fi
AUTH_EVAL_ARGS=()
if [ "$SKIP_AUTH_EVAL" = "1" ]; then
    AUTH_EVAL_ARGS+=(--skip-auth-eval)
fi

PREEXISTING_STATS_JSON="$Z6_OUTPUT_DIR/stats.json"
if [ -f "$PREEXISTING_STATS_JSON" ]; then
    STALE_STATS_DIR="$LOG_DIR/stale_stats_quarantine"
    mkdir -p "$STALE_STATS_DIR"
    STALE_STATS_QUARANTINE="$STALE_STATS_DIR/stats.before_${DISPATCH_INSTANCE_JOB_ID:-unknown}.$(date -u +%Y%m%dT%H%M%SZ).$$.json"
    mv "$PREEXISTING_STATS_JSON" "$STALE_STATS_QUARANTINE"
    log "quarantined_preexisting_stats_json path=$STALE_STATS_QUARANTINE"
fi

REMOTE_DRIVER_STAGE4_STARTED_UNIX="$(date +%s)"
"$PYBIN_RESOLVED" "$TRAINER_PY" \
    --video-path "$Z6_VIDEO_PATH" \
    --output-dir "$Z6_OUTPUT_DIR" \
    --epochs "$Z6_EPOCHS" \
    --batch-size "$Z6_BATCH_SIZE" \
    --lr "$Z6_LR" \
    --device "$Z6_DEVICE" \
    --upstream-dir "$Z6_UPSTREAM_DIR" \
    --lambda-residual-entropy "$Z6_LAMBDA_RESIDUAL_ENTROPY" \
    --predictor-hidden-dim "$Z6_PREDICTOR_HIDDEN_DIM" \
    --predictor-film-mlp-hidden-dim "$Z6_PREDICTOR_FILM_MLP_HIDDEN_DIM" \
    --predictor-kernel-size "$Z6_PREDICTOR_KERNEL_SIZE" \
    --predictor-ego-motion-dim "$Z6_PREDICTOR_EGO_MOTION_DIM" \
    --predictor-architecture "$Z6_PREDICTOR_ARCHITECTURE" \
    --predictor-param-count-target "$Z6_PREDICTOR_PARAM_COUNT_TARGET" \
    --ego-source "$Z6_EGO_SOURCE" \
    --enable-paired-control-initialization "$Z6_ENABLE_PAIRED_CONTROL_INITIALIZATION" \
    --paired-control-disambiguator-decision-criterion-delta-s "$Z6_PAIRED_CONTROL_DISAMBIGUATOR_DECISION_CRITERION_DELTA_S" \
    ${SMOKE_FLAG_ARGS[@]+"${SMOKE_FLAG_ARGS[@]}"} \
    ${IDENTITY_PREDICTOR_ARGS[@]+"${IDENTITY_PREDICTOR_ARGS[@]}"} \
    ${AUTOCAST_FLAG_ARGS[@]+"${AUTOCAST_FLAG_ARGS[@]}"} \
    ${DISAMBIGUATOR_FLAG_ARGS[@]+"${DISAMBIGUATOR_FLAG_ARGS[@]}"} \
    ${MAX_PAIRS_ARGS[@]+"${MAX_PAIRS_ARGS[@]}"} \
    ${AUTH_EVAL_ARGS[@]+"${AUTH_EVAL_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/trainer.log"

# Stage 5: emit completion marker (operator + autopilot consume).
EVIDENCE_STATUS="$("$PYBIN_RESOLVED" - "$Z6_OUTPUT_DIR/stats.json" "$REMOTE_DRIVER_STAGE4_STARTED_UNIX" <<'PY'
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
stage4_started_unix = float(sys.argv[2])
marker = "[training-artifact-no-score-claim]"
score_claim = "score_claim=false"
if not stats_path.is_file():
    print(f"missing required stats.json at {stats_path}", file=sys.stderr)
    raise SystemExit(31)
stats_mtime = stats_path.stat().st_mtime
if stats_mtime < stage4_started_unix:
    print(
        f"stale stats.json at {stats_path}: "
        f"mtime={stats_mtime:.6f} < stage4_started_unix={stage4_started_unix:.6f}",
        file=sys.stderr,
    )
    raise SystemExit(33)
try:
    stats = json.loads(stats_path.read_text())
except json.JSONDecodeError as exc:
    print(f"malformed stats.json at {stats_path}: {exc}", file=sys.stderr)
    raise SystemExit(32)
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
log "LANE_Z6_PCWM_DONE ${EVIDENCE_MARKER} output_dir=$Z6_OUTPUT_DIR smoke=$SMOKE_ONLY identity_predictor=$Z6_IDENTITY_PREDICTOR ${SCORE_CLAIM_FLAG}"
echo "LANE_Z6_PCWM_DONE ${EVIDENCE_MARKER} $LANE_ID ${SCORE_CLAIM_FLAG} identity_predictor=$Z6_IDENTITY_PREDICTOR $(date -u +%FT%TZ)" >> "$LOG_DIR/completion.log"
