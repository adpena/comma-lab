#!/bin/bash
# Remote lane script: Time-Traveler L5 Autonomy first-anchor dispatch (PAIR T).
#
# Trainer: experiments/train_substrate_time_traveler_l5_autonomy.py
# Lane:    lane_time_traveler_l5_autonomy_substrate_20260513
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per Catalog #163 the canonical sentinel REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1
# MUST be set before sourcing the bootstrap to prevent the sourced main flow
# (which expects a pre-built archive.zip) from running.
#
# Design memo:
#   .omx/research/time_traveler_architecture_reverse_engineered_20260513.md
#
# Score-tagging: completion logs report the auth-eval JSON's own score_axis /
# lane_tag. Modal A100 training is diagnostic unless the auth-eval contract
# itself says score_axis=contest_cuda and score_claim_valid=true.
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
LANE_ID="lane_time_traveler_l5_autonomy_substrate_20260513"
TAG="${TAG:-substrate_tt5l}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_time_traveler_l5_autonomy_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags — Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
TT5L_VIDEO_PATH="${TT5L_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
TT5L_OUTPUT_DIR="${TT5L_OUTPUT_DIR:-$OUTPUT_DIR}"
TT5L_EPOCHS="${TT5L_EPOCHS:-3000}"
TT5L_UPSTREAM_DIR="${TT5L_UPSTREAM_DIR:-$WORKSPACE/upstream}"
TT5L_DEVICE="${TT5L_DEVICE:-cuda}"
TT5L_HIDDEN_DIM="${TT5L_HIDDEN_DIM:-64}"
TT5L_PER_PAIR_BYTES="${TT5L_PER_PAIR_BYTES:-45}"

DISPATCH_INSTANCE_JOB_ID="${TT5L_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${TT5L_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-tt5l] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv

# Stage 0: dispatch claim verification.
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: TT5L_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
CLAIM_SUMMARY_JSON="$LOG_DIR/dispatch_claim_summary.json"
"$CLAIM_PYTHON" "$WORKSPACE/tools/claim_lane_dispatch.py" summary \
    --claims-path "$DISPATCH_CLAIMS_PATH" \
    --format json > "$CLAIM_SUMMARY_JSON" || {
    log "FATAL: claim summary failed for $DISPATCH_CLAIMS_PATH"
    exit 21
}
set +e
"$CLAIM_PYTHON" -c 'import json, sys
summary_path, lane_id, job_id = sys.argv[1:4]
payload = json.loads(open(summary_path, encoding="utf-8").read())
for row in payload.get("active", []):
    if row.get("lane_id") == lane_id and row.get("instance_job_id") == job_id:
        raise SystemExit(0)
print(f"missing active claim lane={lane_id} job={job_id}", file=sys.stderr)
raise SystemExit(1)' "$CLAIM_SUMMARY_JSON" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID"
CLAIM_MATCH_RC=$?
set -e
if [ "$CLAIM_MATCH_RC" -ne 0 ]; then
    log "FATAL: no active dispatch claim for lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID"
    exit 21
fi
log "stage_0_dispatch_claim_verified lane=$LANE_ID job=$DISPATCH_INSTANCE_JOB_ID"

# Stage 0b: NVDEC probe.
if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    log "stage_0b_nvdec_probe_begin"
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        log "FATAL: NVDEC probe failed; refusing dispatch"
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
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel per Catalog #163.
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"

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

# Stage 2: provenance + remote code parity.
log "stage_2_provenance_begin"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" - "$PROVENANCE" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" "$DISPATCH_PLATFORM" "$GIT_HASH" "$GPU_NAME" "$DRIVER_VER" "$TT5L_VIDEO_PATH" "$TT5L_UPSTREAM_DIR" "$TT5L_EPOCHS" "$TT5L_DEVICE" "$TT5L_HIDDEN_DIM" "$TT5L_PER_PAIR_BYTES" <<'PY'
import json, sys, time, torch
(
    provenance_path,
    lane_id,
    dispatch_instance_job_id,
    dispatch_platform,
    git_hash,
    gpu_name,
    driver_version,
    video_path,
    upstream_dir,
    epochs,
    device,
    hidden_dim,
    per_pair_bytes,
) = sys.argv[1:14]
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'lane_id': lane_id,
    'dispatch_instance_job_id': dispatch_instance_job_id,
    'dispatch_platform': dispatch_platform,
    'git_hash': git_hash,
    'gpu_name': gpu_name,
    'driver_version': driver_version,
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'video_path': video_path,
    'upstream_dir': upstream_dir,
    'epochs': int(epochs),
    'device': device,
    'hidden_dim': int(hidden_dim),
    'per_pair_bytes': int(per_pair_bytes),
    'substrate_kind': 'time_traveler_l5_autonomy_tt5l',
    'design_memo': '.omx/research/time_traveler_architecture_reverse_engineered_20260513.md',
    # Retired planning band only. Keep the null active field so generic
    # calibration tooling sees that TT5L currently has no rank-authoritative
    # prediction band.
    'predicted_band': None,
    'retired_predicted_band': [0.150, 0.170],
    'prediction_band_rank_reward_suppressed': True,
    'predicted_band_evidence_grade': 'retired_time_traveler_prediction_not_score_evidence',
    'predicted_basis': 'time_traveler_architecture_reverse_engineered_20260513',
    'prediction_band_blockers': [
        'requires_c1_z5_tt5l_probe_disambiguator_before_architecture_lock',
        'requires_paired_cpu_cuda_axis_plan_before_promotion',
        'requires_l5_v2_empirical_anchor',
    ],
}
import pathlib
pathlib.Path(provenance_path).write_text(json.dumps(prov, indent=2, sort_keys=True))
print('[provenance]', json.dumps(prov, indent=2, sort_keys=True))
PY
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
log "stage_4_trainer_invoke_begin video=$TT5L_VIDEO_PATH epochs=$TT5L_EPOCHS device=$TT5L_DEVICE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_time_traveler_l5_autonomy.py \
    --video-path "$TT5L_VIDEO_PATH" \
    --output-dir "$TT5L_OUTPUT_DIR" \
    --epochs "$TT5L_EPOCHS" \
    --upstream-dir "$TT5L_UPSTREAM_DIR" \
    --device "$TT5L_DEVICE" \
    --hidden-dim "$TT5L_HIDDEN_DIM" \
    --per-pair-side-info-bytes "$TT5L_PER_PAIR_BYTES" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record.
AUTH_EVAL_JSON="$OUTPUT_DIR/auth_eval.json"
ARCHIVE_PATH="$OUTPUT_DIR/archive.zip"
PAYLOAD_PATH="$OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    AUTH_EVAL_TAG_FILE="$LOG_DIR/auth_eval_tag.txt"
    if "$PYBIN" - "$AUTH_EVAL_JSON" > "$AUTH_EVAL_TAG_FILE" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print("score_axis=unknown lane_tag=unknown score_claim_valid=false")
else:
    print(
        "score_axis={axis} lane_tag={tag} score_claim_valid={valid}".format(
            axis=payload.get("score_axis", "unknown"),
            tag=payload.get("lane_tag", "unknown"),
            valid=str(payload.get("score_claim_valid") is True).lower(),
        )
    )
PY
    then
        AUTH_EVAL_TAG="$(cat "$AUTH_EVAL_TAG_FILE")"
    else
        AUTH_EVAL_TAG="score_axis=unknown lane_tag=unknown score_claim_valid=false"
    fi
    log "LANE_TT5L_DONE $AUTH_EVAL_TAG auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH payload=$PAYLOAD_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before auth-eval stage)"
fi

exit "$TRAIN_RC"
