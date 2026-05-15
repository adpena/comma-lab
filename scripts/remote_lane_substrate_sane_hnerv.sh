#!/bin/bash
# Remote lane script: substrate sane_hnerv first-anchor dispatch.
#
# Trainer: experiments/train_substrate_sane_hnerv.py (commit c9d5aae7+).
# Lane: lane_substrate_sane_hnerv_20260512
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Council memo refs:
#   - feedback_substrate_sane_hnerv_full_main_wired_landed_20260512.md (α
#     16-stage _full_main wire including auth-eval at stage 12 + continual-
#     learning posterior update at stage 13 + cost-band anchor at stage 14)
#   - feedback_substrate_sane_hnerv_first_anchor_dispatch_landed_20260512.md
#     (the prior Wave 3 DEFERRAL — STOP-PRECONDITIONS surfaced; this rerun
#     uses Modal A100 per operator directive 2026-05-12 "reroute all to
#     modal and lightning free tier")
#
# Score-tagging: this Modal training wrapper is not the contest-CUDA eval
# surface. It may produce advisory CPU auth-eval JSON when Modal lacks NVDEC,
# but exact paired CPU/CUDA eval must be launched separately after harvest.
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
LANE_ID="lane_substrate_sane_hnerv_20260512"
TAG="${TAG:-substrate_sane_hnerv}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_sane_hnerv_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags — Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
SANE_HNERV_VIDEO_PATH="${SANE_HNERV_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
SANE_HNERV_OUTPUT_DIR="${SANE_HNERV_OUTPUT_DIR:-$OUTPUT_DIR}"
SANE_HNERV_EPOCHS="${SANE_HNERV_EPOCHS:-2000}"
SANE_HNERV_UPSTREAM_DIR="${SANE_HNERV_UPSTREAM_DIR:-$WORKSPACE/upstream}"
SANE_HNERV_DEVICE="${SANE_HNERV_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${SANE_HNERV_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${SANE_HNERV_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
HEARTBEAT_PID=""

log() { echo "[lane-sane-hnerv] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0: dispatch claim verification (mirrors remote_lane_t1_balle_endtoend.sh).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "FATAL: SANE_HNERV_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID is required for active lane-claim verification"
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
# running. We only need its bootstrap_runtime_deps() function. Precedent:
# scripts/remote_track1_phase_a1_score_gradient_pr101.sh:170. Without this
# sentinel the source triggers Stage 0+ archive-only-eval main flow which
# fails with "FATAL: archive missing" before any training can start
# (observed 2026-05-12 fc-01KREXK209TRX7ED5ZRVXHY1VT rc=1 in 12.87s).
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
    'video_path': '$SANE_HNERV_VIDEO_PATH',
    'upstream_dir': '$SANE_HNERV_UPSTREAM_DIR',
    'epochs': $SANE_HNERV_EPOCHS,
    'device': '$SANE_HNERV_DEVICE',
    # predicted_band per council calibration (CLAUDE.md no-signal-loss).
    # Source: .omx/research/grand_council_fields_medal_substrate_design_20260512.md
    # + recipe substrate_sane_hnerv_modal_a100_dispatch.yaml::predicted_delta
    # = '-0.030 to -0.050' [predicted; council substrate design memo].
    # Band convention: [LOW, HIGH] in score-space (lower = better contest score).
    'predicted_band': [-0.050, -0.030],
    'predicted_basis': 'grand_council_fields_medal_substrate_design_20260512',
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
# The trainer's _full_main wires 16 stages including auth eval (stage 12),
# continual-learning posterior update (stage 13), and cost-band anchor
# emission (stage 14). All required flags are threaded per Catalog #151
# TIER_1_OPERATOR_REQUIRED_FLAGS manifest.
#
# Per CLAUDE.md "Auth eval EVERYWHERE" + "Submission auth eval — BOTH CPU
# AND CUDA": this Modal train-lane path validates trainer/archive wiring only.
# The shared Modal wrapper forces AUTH_EVAL_DEVICE=cpu and
# MODAL_AUTH_EVAL_ADVISORY_ONLY=1; paired exact eval is a separate dispatcher.
log "stage_4_trainer_invoke_begin video=$SANE_HNERV_VIDEO_PATH epochs=$SANE_HNERV_EPOCHS device=$SANE_HNERV_DEVICE"
TRAIN_START_UTC=$(date -u +%FT%TZ)
set +e
"$PYBIN" experiments/train_substrate_sane_hnerv.py \
    --video-path "$SANE_HNERV_VIDEO_PATH" \
    --output-dir "$SANE_HNERV_OUTPUT_DIR" \
    --epochs "$SANE_HNERV_EPOCHS" \
    --upstream-dir "$SANE_HNERV_UPSTREAM_DIR" \
    --device "$SANE_HNERV_DEVICE" \
    2>&1 | tee -a "$LOG_DIR/run.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END_UTC=$(date -u +%FT%TZ)
log "stage_4_trainer_invoke_done rc=$TRAIN_RC start=$TRAIN_START_UTC end=$TRAIN_END_UTC"

# Stage 5: completion record. The trainer writes artifacts under
# SANE_HNERV_OUTPUT_DIR, not necessarily OUTPUT_DIR when the recipe overrides
# it. Only a JSON that explicitly validates as contest-CUDA gets a
# contest-CUDA marker; advisory/fallback JSON remains training-artifact-only.
AUTH_EVAL_JSON="$SANE_HNERV_OUTPUT_DIR/contest_auth_eval_cuda.json"
AUTH_EVAL_TAG="[training-artifact]"
if [ -f "$AUTH_EVAL_JSON" ]; then
    if AUTH_EVAL_TAG="$("$PYBIN" - "$AUTH_EVAL_JSON" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
if (
    payload.get("score_axis") == "contest_cuda"
    and payload.get("score_claim_valid") is True
    and payload.get("exact_cuda_eval_complete") is True
):
    print("[contest-CUDA]")
else:
    print("[training-artifact]")
PY
    )"; then
        :
    else
        AUTH_EVAL_TAG="[training-artifact]"
    fi
elif [ -f "$SANE_HNERV_OUTPUT_DIR/auth_eval.json" ]; then
    AUTH_EVAL_JSON="$SANE_HNERV_OUTPUT_DIR/auth_eval.json"
    AUTH_EVAL_TAG="[training-artifact]"
fi
ARCHIVE_PATH="$SANE_HNERV_OUTPUT_DIR/0.bin"
if [ -f "$AUTH_EVAL_JSON" ]; then
    log "auth_eval_artifact_present path=$AUTH_EVAL_JSON"
    log "LANE_SANE_HNERV_DONE $AUTH_EVAL_TAG auth_eval=$AUTH_EVAL_JSON archive=$ARCHIVE_PATH rc=$TRAIN_RC"
else
    log "auth_eval_artifact_missing path=$AUTH_EVAL_JSON (trainer may have failed before stage 12)"
fi

exit "$TRAIN_RC"
