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
#   DISPATCH_INSTANCE_JOB_ID: active provider job id from claim_lane_dispatch.py
#   DISPATCH_CLAIMS_PATH:   path to copied active claim ledger on the remote host
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
DISPATCH_INSTANCE_JOB_ID="${T1_A1_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${T1_A1_DISPATCH_CLAIMS_PATH:-${DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-lightning}"
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
CLAIM_VERIFIED=0
TERMINAL_CLAIM_CLOSED=0
HB_PID=""

log() { echo "[track1-a1] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

require_active_dispatch_claim() {
    if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
        log "FATAL: DISPATCH_INSTANCE_JOB_ID is required before PR101 CUDA train/eval dispatch"
        exit 21
    fi
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        log "FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py"
        exit 21
    fi
    if [ ! -f "$DISPATCH_CLAIMS_PATH" ]; then
        log "FATAL: dispatch-claims ledger missing at $DISPATCH_CLAIMS_PATH; copy the active claim ledger to the remote host before dispatch"
        exit 21
    fi
    "$PYBIN" - "$WORKSPACE" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" "$DISPATCH_CLAIMS_PATH" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
lane_id = sys.argv[2]
instance_job_id = sys.argv[3]
claims_path = Path(sys.argv[4])
cmd = [
    sys.executable,
    str(workspace / "tools" / "claim_lane_dispatch.py"),
    "summary",
    "--claims-path",
    str(claims_path),
    "--format",
    "json",
]
proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
if proc.returncode != 0:
    print(proc.stderr or proc.stdout, file=sys.stderr)
    raise SystemExit(21)
payload = json.loads(proc.stdout)
for row in payload.get("active", []):
    if row.get("lane_id") == lane_id and row.get("instance_job_id") == instance_job_id:
        raise SystemExit(0)
print(
    f"no active dispatch claim for lane_id={lane_id!r} instance_job_id={instance_job_id!r}",
    file=sys.stderr,
)
raise SystemExit(21)
PY
}

close_dispatch_claim() {
    local status="$1"
    local notes="$2"
    if [ "$CLAIM_VERIFIED" != "1" ] || [ "$TERMINAL_CLAIM_CLOSED" = "1" ]; then
        return 0
    fi
    if [ -z "$DISPATCH_INSTANCE_JOB_ID" ] || [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        return 0
    fi
    "$PYBIN" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "codex:remote_track1_phase_a1_score_gradient_pr101" \
        --predicted-eta-utc "$(date -u +%FT%TZ)" \
        --status "$status" \
        --notes "$notes" \
        --force >/dev/null 2>>"$LOG_DIR/run.log" || true
    TERMINAL_CLAIM_CLOSED=1
}

cleanup() {
    local rc=$?
    if [ -n "${HB_PID:-}" ]; then
        kill "$HB_PID" 2>/dev/null || true
    fi
    if [ "$rc" -ne 0 ]; then
        close_dispatch_claim "failed_remote_script_rc_${rc}" "Phase A1 remote script exited rc=${rc} before terminal stage-specific closure"
    fi
}
trap cleanup EXIT

require_active_dispatch_claim
CLAIM_VERIFIED=1

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

# Stage 0b: $PYBIN torch CUDA verification (closes A1 review R1-3 advisory).
# $PYBIN is the conda-managed Python (separate from the uv-managed inflate
# env where INFLATE_TORCH_SPEC pins cu124). Lightning Studio conda images
# *should* ship a CUDA-compatible torch, but if they don't, training would
# silently run on CPU per CLAUDE.md "Forbidden cu13-vs-cu124 trap" — every
# score becomes [advisory only] and the dispatch is wasted. Fail fast here
# so the operator sees the issue at Stage 0b, not after 4 hours of CPU run.
log "=== Stage 0b: \$PYBIN torch CUDA verification ==="
"$PYBIN" -c "
import sys, torch
ok = torch.cuda.is_available()
ver = getattr(torch.version, 'cuda', None) or 'unknown'
print(f'[stage0b] torch={torch.__version__} cuda={ver} cuda_available={ok}', file=sys.stderr)
if not ok:
    print('[stage0b] FATAL: \$PYBIN torch reports cuda_available=False; would train on CPU and produce only [advisory only] scores. Fix: rebuild conda env with CUDA-compatible torch wheel; do NOT proceed.', file=sys.stderr)
    sys.exit(2)
# Sanity: a concrete CUDA op succeeds (catches torch-built-with-cuda but no driver).
try:
    _ = torch.zeros(1, device='cuda') + torch.ones(1, device='cuda')
except Exception as e:
    print(f'[stage0b] FATAL: torch.cuda is_available()==True but a 1-element CUDA op failed: {e!r}', file=sys.stderr)
    sys.exit(2)
print('[stage0b] OK', file=sys.stderr)
" || {
    log "FATAL: \$PYBIN torch CUDA verification failed — see stage0b output above. Aborting before any GPU spend."
    exit 2
}

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=$LANE_ID gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!

# Stage 1: TRAIN (Phase A1 score-gradient).
log "=== Stage 1: TRAIN train_score_gradient_pr101_finetune.py ==="
log "  --epochs $EPOCHS --steps-per-epoch $STEPS_PER_EPOCH --batch-size $BATCH_SIZE --lr $LR"

# CLAUDE.md: NEVER use --device mps. NEVER pass --no-eval-roundtrip (default True).
# eval_roundtrip + EMA + noise_std are ALL training defaults per train_score_gradient_pr101_finetune.py.
set +e
"$PYBIN" "$WORKSPACE/experiments/train_score_gradient_pr101_finetune.py" \
    --device cuda \
    --epochs "$EPOCHS" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --pr101-archive "$WORKSPACE/$PR101_ARCHIVE_PATH" \
    --pr101-source-dir "$WORKSPACE/$PR101_SOURCE_DIR" \
    --video-path "$WORKSPACE/$VIDEO_PATH" \
    --max-frames "$MAX_FRAMES" \
    --output "$TRAIN_OUTPUT" \
    2>&1 | tee "$LOG_DIR/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
if [ "$TRAIN_RC" -ne 0 ]; then
    log "FATAL: training failed rc=$TRAIN_RC; abort before archive build"
    exit 3
fi

BEST_PROXY_CHECKPOINT="$TRAIN_OUTPUT/checkpoint_best_proxy.pt"
FINAL_EMA_CHECKPOINT="$TRAIN_OUTPUT/checkpoint_ema.pt"
if [ -f "$BEST_PROXY_CHECKPOINT" ]; then
    CHECKPOINT_PATH="$BEST_PROXY_CHECKPOINT"
    CHECKPOINT_KIND="best_proxy"
else
    CHECKPOINT_PATH="$FINAL_EMA_CHECKPOINT"
    CHECKPOINT_KIND="final_ema"
fi
if [ ! -f "$CHECKPOINT_PATH" ]; then
    log "FATAL: selected checkpoint missing; checked best_proxy=$BEST_PROXY_CHECKPOINT final_ema=$FINAL_EMA_CHECKPOINT"
    exit 4
fi
log "training OK; selected checkpoint_kind=$CHECKPOINT_KIND path=$CHECKPOINT_PATH ($(stat -c%s "$CHECKPOINT_PATH" 2>/dev/null || stat -f%z "$CHECKPOINT_PATH") B)"

# Stage 2: BUILD archive from fine-tuned state_dict.
log "=== Stage 2: BUILD finetuned PR101 archive ==="
set +e
"$PYBIN" "$WORKSPACE/tools/build_pr101_finetuned_archive.py" \
    --state-dict "$CHECKPOINT_PATH" \
    --source-archive "$WORKSPACE/$PR101_ARCHIVE_PATH" \
    --pr101-source-dir "$WORKSPACE/$PR101_SOURCE_DIR" \
    --output-dir "$ARCHIVE_OUTPUT" \
    --lane-id "$LANE_ID" \
    2>&1 | tee "$LOG_DIR/build.log"
BUILD_RC=${PIPESTATUS[0]}
set -e
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
set +e
"$PYBIN" "$WORKSPACE/experiments/contest_auth_eval.py" \
    --archive "$ARCHIVE_PATH" \
    --inflate-sh "$INFLATE_SH_PATH" \
    --upstream-dir "$WORKSPACE/upstream" \
    --device cuda \
    --work-dir "$EVAL_WORK" \
    --keep-work-dir \
    2>&1 | tee "$LOG_DIR/eval.log"
EVAL_RC=${PIPESTATUS[0]}
set -e
log "eval rc=$EVAL_RC"

# Stage 4: REPORT — write build_manifest.json with score components.
log "=== Stage 4: REPORT ==="
EVAL_JSON="$EVAL_WORK/contest_auth_eval.json"
set +e
"$PYBIN" -c "
import json, time, os, sys
from tac.auth_eval_schema import eval_metric_summary, required_contest_cuda_evidence_blockers
manifest = {
    'lane_id': '$LANE_ID',
    'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'archive_path': '$ARCHIVE_PATH',
    'archive_bytes': $ARCHIVE_BYTES,
    'archive_sha256': '$ARCHIVE_SHA',
    'predicted_band': [$PRED_LOW, $PRED_HIGH],
    'checkpoint_kind': '$CHECKPOINT_KIND',
    'checkpoint_path': '$CHECKPOINT_PATH',
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
    metrics = eval_metric_summary(eval_data)
    manifest['score'] = metrics['score']
    manifest['canonical_score'] = metrics['score']
    manifest['canonical_score_source'] = metrics['canonical_score_source']
    manifest['pose_avg'] = metrics['pose_avg']
    manifest['seg_avg'] = metrics['seg_avg']
    manifest['rate'] = metrics['rate']
    manifest['rate_unscaled'] = metrics['rate_unscaled']
    manifest['eval_archive_size_bytes'] = metrics['archive_size_bytes']
    manifest['n_samples'] = metrics['n_samples']
    manifest['score_axis'] = eval_data.get('score_axis')
    manifest['evidence_semantics'] = eval_data.get('evidence_semantics')
    manifest['lane_tag'] = eval_data.get('lane_tag')
    metric_blockers = required_contest_cuda_evidence_blockers(
        eval_data,
        metrics,
        expected_archive_bytes=$ARCHIVE_BYTES,
        expected_n_samples=600,
    )
    score_claim = $EVAL_RC == 0 and not metric_blockers
    manifest['evidence_grade'] = '[contest-CUDA]' if score_claim else '[exact-eval incomplete]'
    manifest['score_claim'] = score_claim
    manifest['score_claim_valid'] = score_claim
    manifest['ready_for_exact_eval_dispatch'] = False
    manifest['exact_cuda_eval_complete'] = score_claim
    manifest['promotion_eligible'] = False
    manifest['rank_or_kill_eligible'] = False
    if $EVAL_RC == 0:
        manifest['dispatch_blockers'] = ['contest_cpu_eval_pending'] if score_claim else metric_blockers
    else:
        manifest['dispatch_blockers'] = ['eval_failed']
else:
    manifest['evidence_grade'] = '[exact-eval failed: eval JSON missing]'
    manifest['score_claim'] = False
    manifest['score_claim_valid'] = False
    manifest['ready_for_exact_eval_dispatch'] = False
    manifest['exact_cuda_eval_complete'] = False
    manifest['promotion_eligible'] = False
    manifest['rank_or_kill_eligible'] = False
    manifest['dispatch_blockers'] = ['eval_json_missing']
with open('$BUILD_MANIFEST', 'w') as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2)[:1500])
if manifest.get('eval_rc') == 0 and manifest.get('score_claim') is not True:
    print(
        '[track1-a1] FATAL: eval rc=0 but manifest is not score-claimable: '
        + ','.join(manifest.get('dispatch_blockers') or []),
        file=sys.stderr,
    )
    raise SystemExit(6)
"
REPORT_RC=$?
set -e

if [ "$REPORT_RC" -ne 0 ]; then
    close_dispatch_claim "failed_manifest_adjudication_rc_${REPORT_RC}" "Phase A1 manifest adjudication refused score claim; see $BUILD_MANIFEST"
    log "=== TRACK1_A1_INCOMPLETE exact-eval evidence refused (report_rc=$REPORT_RC eval_rc=$EVAL_RC) ==="
    exit "$REPORT_RC"
fi
if [ "$EVAL_RC" -ne 0 ]; then
    close_dispatch_claim "failed_exact_eval_rc_${EVAL_RC}" "Phase A1 contest_auth_eval failed rc=${EVAL_RC}; no score claim"
    log "=== TRACK1_A1_FAILED exact eval rc=$EVAL_RC; no score claim ==="
    exit "$EVAL_RC"
fi
close_dispatch_claim "completed_contest_cuda_exact_eval" "Phase A1 completed with claimable contest-CUDA exact eval; manifest=$BUILD_MANIFEST"
log "=== TRACK1_A1_DONE [contest-CUDA] (rc=0) ==="
exit 0
