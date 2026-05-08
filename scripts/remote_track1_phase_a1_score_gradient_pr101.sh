#!/bin/bash
# track1_phase_a1_score_gradient — full-chain remote driver.
#
# Stages on remote (Lightning T4 g4dn.2xlarge or equivalent CUDA host):
#   Stage 0:  bootstrap (uv + ffmpeg + cu124 torch via canonical wrapper)
#   Stage 0a: NVDEC probe (auth eval requires NVDEC)
#   Stage 1:  TRAIN — experiments/train_score_gradient_pr101_finetune.py
#             (Phase A1 score-gradient supervision, 200 epochs, ~3h on T4)
#   Stage 2:  BUILD — re-encode fine-tuned state_dict into PR101 archive via
#             tools/build_pr101_finetuned_archive.py
#   Stage 3:  EVAL — contest_auth_eval.py --device cuda on the rebuilt archive
#   Stage 4:  REPORT — write build_manifest.json with score components
#
# Required env (set by dispatcher before script runs):
#   PR101_ARCHIVE_PATH:     repo-relative path to source PR101 archive.zip
#   PR101_SOURCE_DIR:       repo-relative path to PR101 source/submissions/hnerv_ft_microcodec/src
#   VIDEO_PATH:             repo-relative path to upstream/videos/0.mkv
#   LANE_ID:                track1_phase_a1_score_gradient (or timestamped variant)
#   PRED_LOW / PRED_HIGH:   predicted score band (recorded for audit)
#
# Cost: Lightning T4 ~$0.66/hr × ~3.5h = ~$2.30 (well under $8 cap)
#
# Per CLAUDE.md non-negotiables:
#   - eval_roundtrip=True (script enforces via training default)
#   - EMA decay 0.997 (training default)
#   - noise_std=0.5 (training default)
#   - --device cuda (NO MPS fallback)
#   - dual eval mandate: this script runs CUDA only; CPU eval queued separately
#     via tools/dispatch_cpu_eval_via_github_actions.py per CLAUDE.md
#   - bootstrap via scripts/remote_archive_only_eval.sh::bootstrap_runtime_deps
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=20
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LANE_ID="${LANE_ID:-track1_phase_a1_score_gradient}"
PR101_ARCHIVE_PATH="${PR101_ARCHIVE_PATH:-experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/archive.zip}"
PR101_SOURCE_DIR="${PR101_SOURCE_DIR:-experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/src}"
VIDEO_PATH="${VIDEO_PATH:-upstream/videos/0.mkv}"
PRED_LOW="${PRED_LOW:-0.150}"
PRED_HIGH="${PRED_HIGH:-0.220}"
EPOCHS="${EPOCHS:-200}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
MAX_FRAMES="${MAX_FRAMES:-1200}"  # Cap loaded frame pairs for CUDA memory

LOG_DIR="${LOG_DIR:-$WORKSPACE/${LANE_ID}_results}"
mkdir -p "$LOG_DIR"
TRAIN_OUTPUT="$LOG_DIR/train"
ARCHIVE_OUTPUT="$LOG_DIR/finetuned_archive"
EVAL_WORK="$LOG_DIR/eval_work"
mkdir -p "$TRAIN_OUTPUT" "$ARCHIVE_OUTPUT" "$EVAL_WORK"

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
BUILD_MANIFEST="$LOG_DIR/build_manifest.json"

log() { echo "[track1-a1] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

GIT_HASH="$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo no-git)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)"

# Stage 0: bootstrap via canonical wrapper (CLAUDE.md FORBIDDEN re-implementing
# remote bootstrap inline). Source the canonical script in explicit source-only
# mode so its function definitions load without running archive-only eval.
log "=== Stage 0: bootstrap deps via remote_archive_only_eval.sh ==="
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: scripts/remote_archive_only_eval.sh missing; abort"
    exit 7
fi
# shellcheck disable=SC1090
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
cd "$WORKSPACE"
log() { echo "[track1-a1] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }
bootstrap_runtime_deps
require_uv_and_ffmpeg_contract
log "tooling: uv=$UV_BIN INFLATE_TORCH_SPEC=$INFLATE_TORCH_SPEC FFMPEG_BIN=$FFMPEG_BIN"

# Stage 0a: NVDEC probe.
log "=== Stage 0a: NVDEC probe ==="
if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
        exit 2
    }
else
    log "WARN: scripts/probe_nvdec.sh missing; continuing without NVDEC pre-check"
fi

# Provenance + heartbeat.
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
    'lane_script': 'scripts/remote_track1_phase_a1_score_gradient_pr101.sh',
    'output_dir': '$LOG_DIR',
    'pr101_archive_path': '$PR101_ARCHIVE_PATH',
    'pr101_source_dir': '$PR101_SOURCE_DIR',
    'video_path': '$VIDEO_PATH',
    'epochs': $EPOCHS,
    'steps_per_epoch': $STEPS_PER_EPOCH,
    'batch_size': $BATCH_SIZE,
    'lr': $LR,
    'max_frames': $MAX_FRAMES,
    'predicted_band': [$PRED_LOW, $PRED_HIGH],
    'council_memo_ref': '.omx/research/grand_council_extreme_rigor_track_1_20260508.md',
    'council_decision': 'A1 — score-gradient supervision (UNANIMOUS HIGHEST PRIORITY)',
    'cost_estimate_usd': 2.30,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
"
log "provenance written: $PROVENANCE"

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=$LANE_ID gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 1: TRAIN (Phase A1 score-gradient).
log "=== Stage 1: TRAIN train_score_gradient_pr101_finetune.py ==="
log "  --epochs $EPOCHS --steps-per-epoch $STEPS_PER_EPOCH --batch-size $BATCH_SIZE --lr $LR"

# CLAUDE.md: NEVER use --device mps. NEVER pass --no-eval-roundtrip (default True).
# eval_roundtrip + EMA + noise_std are ALL training defaults per train_score_gradient_pr101_finetune.py.
"$PYBIN" "$WORKSPACE/experiments/train_score_gradient_pr101_finetune.py" \
    --device cuda \
    --epochs "$EPOCHS" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --pr101-archive "$WORKSPACE/$PR101_ARCHIVE_PATH" \
    --video-path "$WORKSPACE/$VIDEO_PATH" \
    --max-frames "$MAX_FRAMES" \
    --output "$TRAIN_OUTPUT" \
    2>&1 | tee "$LOG_DIR/train.log"
TRAIN_RC=${PIPESTATUS[0]}
if [ "$TRAIN_RC" -ne 0 ]; then
    log "FATAL: training failed rc=$TRAIN_RC; abort before archive build"
    exit 3
fi

CHECKPOINT_PATH="$TRAIN_OUTPUT/checkpoint_ema.pt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    log "FATAL: checkpoint_ema.pt missing at $CHECKPOINT_PATH"
    exit 4
fi
log "training OK; checkpoint: $CHECKPOINT_PATH ($(stat -c%s "$CHECKPOINT_PATH" 2>/dev/null || stat -f%z "$CHECKPOINT_PATH") B)"

# Stage 2: BUILD archive from fine-tuned state_dict.
log "=== Stage 2: BUILD finetuned PR101 archive ==="
"$PYBIN" "$WORKSPACE/tools/build_pr101_finetuned_archive.py" \
    --state-dict "$CHECKPOINT_PATH" \
    --source-archive "$WORKSPACE/$PR101_ARCHIVE_PATH" \
    --pr101-source-dir "$WORKSPACE/$PR101_SOURCE_DIR" \
    --output-dir "$ARCHIVE_OUTPUT" \
    --lane-id "$LANE_ID" \
    2>&1 | tee "$LOG_DIR/build.log"
BUILD_RC=${PIPESTATUS[0]}
if [ "$BUILD_RC" -ne 0 ]; then
    log "FATAL: archive build failed rc=$BUILD_RC; abort before eval"
    exit 5
fi

ARCHIVE_PATH="$ARCHIVE_OUTPUT/archive.zip"
INFLATE_SH_PATH="$ARCHIVE_OUTPUT/submission_dir/inflate.sh"
if [ ! -f "$ARCHIVE_PATH" ] || [ ! -f "$INFLATE_SH_PATH" ]; then
    log "FATAL: archive build incomplete (archive=$ARCHIVE_PATH inflate=$INFLATE_SH_PATH)"
    exit 5
fi
ARCHIVE_BYTES=$(stat -c%s "$ARCHIVE_PATH" 2>/dev/null || stat -f%z "$ARCHIVE_PATH")
ARCHIVE_SHA=$(sha256sum "$ARCHIVE_PATH" 2>/dev/null | cut -d ' ' -f 1 || shasum -a 256 "$ARCHIVE_PATH" | cut -d ' ' -f 1)
log "archive built: bytes=$ARCHIVE_BYTES sha=$ARCHIVE_SHA"

# Stage 3: EVAL (contest_auth_eval.py --device cuda).
log "=== Stage 3: EVAL contest_auth_eval.py --device cuda ==="
ensure_scorer_runtime_deps
"$PYBIN" "$WORKSPACE/experiments/contest_auth_eval.py" \
    --archive "$ARCHIVE_PATH" \
    --inflate-sh "$INFLATE_SH_PATH" \
    --upstream-dir "$WORKSPACE/upstream" \
    --device cuda \
    --work-dir "$EVAL_WORK" \
    --keep-work-dir \
    2>&1 | tee "$LOG_DIR/eval.log"
EVAL_RC=${PIPESTATUS[0]}
log "eval rc=$EVAL_RC"

# Stage 4: REPORT — write build_manifest.json with score components.
log "=== Stage 4: REPORT ==="
EVAL_JSON="$EVAL_WORK/contest_auth_eval.json"
"$PYBIN" -c "
import json, hashlib, time, os
manifest = {
    'lane_id': '$LANE_ID',
    'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'archive_path': '$ARCHIVE_PATH',
    'archive_bytes': $ARCHIVE_BYTES,
    'archive_sha256': '$ARCHIVE_SHA',
    'predicted_band': [$PRED_LOW, $PRED_HIGH],
    'train_log_path': '$LOG_DIR/train.log',
    'build_log_path': '$LOG_DIR/build.log',
    'eval_log_path': '$LOG_DIR/eval.log',
    'eval_work_dir': '$EVAL_WORK',
    'eval_rc': $EVAL_RC,
}
ej = '$EVAL_JSON'
if os.path.isfile(ej):
    with open(ej) as f:
        eval_data = json.load(f)
    manifest['eval_data'] = eval_data
    sc = eval_data.get('score_components') or {}
    manifest['score'] = eval_data.get('score') or eval_data.get('total_score')
    manifest['pose_avg'] = sc.get('pose') or sc.get('pose_avg')
    manifest['seg_avg'] = sc.get('seg') or sc.get('seg_avg')
    manifest['rate'] = sc.get('rate')
    manifest['evidence_grade'] = '[contest-CUDA]' if $EVAL_RC == 0 else '[contest-CUDA failed]'
    manifest['score_claim'] = $EVAL_RC == 0
    manifest['ready_for_exact_eval_dispatch'] = True
    manifest['promotion_eligible'] = False  # Requires CPU dual-eval per CLAUDE.md
    manifest['rank_or_kill_eligible'] = False  # Requires CPU dual-eval + adversarial review
    manifest['dispatch_blockers'] = ['contest_cpu_eval_pending'] if $EVAL_RC == 0 else ['eval_failed']
else:
    manifest['evidence_grade'] = '[contest-CUDA failed: eval JSON missing]'
    manifest['score_claim'] = False
    manifest['ready_for_exact_eval_dispatch'] = False
    manifest['promotion_eligible'] = False
    manifest['rank_or_kill_eligible'] = False
    manifest['dispatch_blockers'] = ['eval_json_missing']
with open('$BUILD_MANIFEST', 'w') as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2)[:1500])
"

log "=== TRACK1_A1_DONE [contest-CUDA] (rc=$EVAL_RC) ==="
exit "$EVAL_RC"
