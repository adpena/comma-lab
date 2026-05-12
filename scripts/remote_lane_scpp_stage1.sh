#!/bin/bash
# Remote lane script for SC++ Stage 1 anchor smoke dispatch.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson
# 7: SC++ is `lane_class=substrate_engineering` (>350 LOC budget OK once per
# architecture class). This Stage 1 dispatch produces the FIRST empirical
# anchor for SC++ block-FP self-compression on the contest-CUDA axis.
#
# Cost target: ~$3 Modal T4 (Stage 1 smoke = 3 epochs).
# Predicted Δ: -0.005 to -0.010 [predicted; SC++ block-FP per HNeRV parity
# discipline lesson 7]. NO score claim until exact contest_auth_eval custody
# artifact lands separately.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity — non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_scpp_stage1_smoke_anchor"
TAG="${TAG:-scpp_stage1}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_scpp_stage1_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"
RUN_RECORD="$LOG_DIR/run_record.json"
ALLOW_REMOTE_SCAFFOLD_SMOKE="${SCPP_ALLOW_REMOTE_SCAFFOLD_SMOKE:-0}"
ALLOW_SCORE_DOMAIN_TRAINING="${SCPP_ALLOW_SCORE_DOMAIN_TRAINING:-0}"
RUN_CONTEST_CUDA_AUTH_EVAL="${SCPP_RUN_CONTEST_CUDA_AUTH_EVAL:-0}"
DISPATCH_INSTANCE_JOB_ID="${SCPP_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${SCPP_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"
DECLARED_MOUNTED_GIT_HEAD="${SCPP_MOUNTED_CODE_GIT_HEAD:-}"
DECLARED_MOUNTED_GIT_BRANCH="${SCPP_MOUNTED_CODE_GIT_BRANCH:-}"
STAGE_1_EPOCHS="${SCPP_STAGE_1_EPOCHS:-3}"
TARGET_ARCHIVE_BYTES="${SCPP_TARGET_ARCHIVE_BYTES:-200000}"
LATENT_DIM="${SCPP_LATENT_DIM:-32}"
BASE_CHANNELS="${SCPP_BASE_CHANNELS:-24}"
N_PAIRS="${SCPP_N_PAIRS:-50}"
CLAIM_VERIFIED=0
TERMINAL_CLAIM_CLOSED=0
HEARTBEAT_PID=""

require_active_dispatch_claim() {
    if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
        echo "[lane-scpp] FATAL: SCPP_ALLOW_SCORE_DOMAIN_TRAINING=1 requires SCPP_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID for active lane-claim verification." >&2
        exit 21
    fi
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        echo "[lane-scpp] FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py" >&2
        exit 21
    fi
    if [ ! -f "$DISPATCH_CLAIMS_PATH" ]; then
        echo "[lane-scpp] FATAL: dispatch-claims ledger missing at $DISPATCH_CLAIMS_PATH; copy the active claim ledger to the remote host before training" >&2
        exit 21
    fi
    local claim_python="${PYBIN:-}"
    if [ -z "$claim_python" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
        claim_python="$WORKSPACE/.venv/bin/python"
    fi
    if [ -z "$claim_python" ]; then
        claim_python="python3"
    fi
    "$claim_python" - "$WORKSPACE" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" "$DISPATCH_CLAIMS_PATH" <<'PY'
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
    f"no active dispatch claim for lane_id={lane_id!r} "
    f"instance_job_id={instance_job_id!r}",
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
    local claim_python="${PYBIN:-}"
    if [ -z "$claim_python" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
        claim_python="$WORKSPACE/.venv/bin/python"
    fi
    if [ -z "$claim_python" ]; then
        claim_python="python3"
    fi
    "$claim_python" "$WORKSPACE/tools/claim_lane_dispatch.py" claim \
        --claims-path "$DISPATCH_CLAIMS_PATH" \
        --lane-id "$LANE_ID" \
        --platform "$DISPATCH_PLATFORM" \
        --instance-job-id "$DISPATCH_INSTANCE_JOB_ID" \
        --agent "claude:remote_lane_scpp_stage1" \
        --predicted-eta-utc "$(date -u +%FT%TZ)" \
        --status "$status" \
        --notes "$notes" \
        --force >/dev/null 2>>"$LOG_DIR/run.log" || true
    TERMINAL_CLAIM_CLOSED=1
}

cleanup() {
    local rc=$?
    if [ -n "${HEARTBEAT_PID:-}" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    if [ "$rc" -ne 0 ]; then
        close_dispatch_claim "failed_remote_script_rc_${rc}" "SC++ Stage 1 remote script exited rc=${rc} before terminal stage-specific closure"
    fi
}

if [ "$ALLOW_SCORE_DOMAIN_TRAINING" != "1" ] && [ "$ALLOW_REMOTE_SCAFFOLD_SMOKE" != "1" ]; then
    echo "[lane-scpp] FATAL: remote SC++ Stage 1 refused by default. Set SCPP_ALLOW_SCORE_DOMAIN_TRAINING=1 for real scorer-domain training, or SCPP_ALLOW_REMOTE_SCAFFOLD_SMOKE=1 only for synthetic smoke verification." >&2
    exit 20
fi

cd "$WORKSPACE"
if [ "$ALLOW_SCORE_DOMAIN_TRAINING" = "1" ]; then
    require_active_dispatch_claim
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
if [ "$ALLOW_SCORE_DOMAIN_TRAINING" = "1" ]; then
    CLAIM_VERIFIED=1
fi
trap cleanup EXIT
rm -f upstream/videos/._*.mkv 2>/dev/null || true
export PYTHONHASHSEED=20
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

log() { echo "[lane-scpp] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log" ; }

resolve_python_bin() {
    if [ -n "$PYBIN" ]; then
        if command -v "$PYBIN" >/dev/null 2>&1; then command -v "$PYBIN"; return 0; fi
        if [ -x "$PYBIN" ]; then echo "$PYBIN"; return 0; fi
        return 1
    fi
    for candidate in /opt/conda/bin/python /usr/local/bin/python python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then command -v "$candidate"; return 0; fi
        if [ -x "$candidate" ]; then echo "$candidate"; return 0; fi
    done
    return 1
}

log "===== Remote lane SC++ Stage 1 anchor starting ====="
log "lane_id: $LANE_ID"
log "workspace: $WORKSPACE"
log "log_dir: $LOG_DIR"
log "output_dir: $OUTPUT_DIR"

# Stage 0: Git parity check.
LOCAL_BRANCH="$(git -C "$WORKSPACE" branch --show-current 2>/dev/null | tr -d '\r' | awk 'NF{print; exit}' || echo unknown)"
LOCAL_HEAD="$(git -C "$WORKSPACE" rev-parse HEAD 2>/dev/null | tr -d '\r' | awk 'NF{print; exit}' || echo unknown)"
log "local branch: $LOCAL_BRANCH"
log "local HEAD: $LOCAL_HEAD"
log "declared mounted git branch: ${DECLARED_MOUNTED_GIT_BRANCH:-unset}"
log "declared mounted git HEAD: ${DECLARED_MOUNTED_GIT_HEAD:-unset}"
if [ -z "$DECLARED_MOUNTED_GIT_HEAD" ] || [ -z "$DECLARED_MOUNTED_GIT_BRANCH" ]; then
    log "FATAL: missing SCPP_MOUNTED_CODE_GIT_HEAD/SCPP_MOUNTED_CODE_GIT_BRANCH custody metadata"
    exit 10
fi
if [ "$DECLARED_MOUNTED_GIT_BRANCH" != "main" ]; then
    log "FATAL: refusing SC++ remote work for non-main mounted branch ($DECLARED_MOUNTED_GIT_BRANCH)"
    exit 10
fi
if [ "$LOCAL_HEAD" != "unknown" ] && [ "$LOCAL_HEAD" != "$DECLARED_MOUNTED_GIT_HEAD" ]; then
    log "FATAL: local git HEAD mismatch (observed=$LOCAL_HEAD declared=$DECLARED_MOUNTED_GIT_HEAD)"
    exit 10
fi

cat > "$PROVENANCE" <<EOF
{
  "schema_version": "remote_run_provenance.v1",
  "lane_id": "${LANE_ID}",
  "tag": "${TAG}",
  "lane_class": "substrate_engineering",
  "script": "scripts/remote_lane_scpp_stage1.sh",
  "workspace": "${WORKSPACE}",
  "log_dir": "${LOG_DIR}",
  "output_dir": "${OUTPUT_DIR}",
  "git_branch": "${LOCAL_BRANCH}",
  "git_head": "${LOCAL_HEAD}",
  "declared_mounted_git_branch": "${DECLARED_MOUNTED_GIT_BRANCH}",
  "declared_mounted_git_head": "${DECLARED_MOUNTED_GIT_HEAD}",
  "pythonhashseed": "${PYTHONHASHSEED}",
  "allow_remote_scaffold_smoke_env": "${ALLOW_REMOTE_SCAFFOLD_SMOKE}",
  "allow_score_domain_training_env": "${ALLOW_SCORE_DOMAIN_TRAINING}",
  "run_contest_cuda_auth_eval_env": "${RUN_CONTEST_CUDA_AUTH_EVAL}",
  "stage_1_epochs": ${STAGE_1_EPOCHS},
  "target_archive_bytes": ${TARGET_ARCHIVE_BYTES},
  "latent_dim": ${LATENT_DIM},
  "base_channels": ${BASE_CHANNELS},
  "n_pairs": ${N_PAIRS},
  "predicted_band": [-0.010, -0.005],
  "prediction_scope": "SC++ Stage 1 smoke anchor; HNeRV parity discipline lesson 7 substrate engineering; no score claim until [contest-CUDA] contest_auth_eval custody",
  "score_claim": false
}
EOF
cat > "$RUN_RECORD" <<EOF
{
  "schema_version": "remote_run_record.v1",
  "status": "started",
  "started_at_utc": "$(date -u +%FT%TZ)",
  "provenance_json": "${PROVENANCE}",
  "heartbeat_log": "${LOG_DIR}/heartbeat.log"
}
EOF
if [ "$LOCAL_BRANCH" != "main" ] && [ "$LOCAL_BRANCH" != "unknown" ]; then
    log "FATAL: refusing SC++ remote work outside main (branch=$LOCAL_BRANCH)"
    exit 10
fi

# Stage 1: Bootstrap deps via canonical wrapper (NEVER inline uv install
# per CLAUDE.md "Forbidden re-implementing remote bootstrap inline").
log "Stage 1: bootstrap_runtime_deps"
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap wrapper missing"
    exit 11
fi
# shellcheck disable=SC1091
source <(awk '/^bootstrap_runtime_deps\(\)/,/^}/' "$WORKSPACE/scripts/remote_archive_only_eval.sh")
bootstrap_runtime_deps
log "Stage 1: bootstrap_runtime_deps OK"

PYBIN="$(resolve_python_bin)" || { log "FATAL: python executable not found; set PYBIN explicitly"; exit 12; }
export PYBIN
log "python_bin: $PYBIN"

# Stage 2: NVDEC probe (per CLAUDE.md remote_scripts_have_nvdec_probe rule).
log "Stage 2: NVDEC probe"
if [[ "${LOCAL_CUDA_WORKER:-0}" != "1" ]]; then
    log "FATAL: LOCAL_CUDA_WORKER=1 must be set on the remote CUDA provider before NVDEC/CUDA probing"
    exit 13
fi
if [ -f "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        log "FATAL: probe_nvdec.sh failed; refusing CUDA training on this host"
        exit 13
    }
fi
"$PYBIN" -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" \
    | tee "$LOG_DIR/nvdec_probe.log"

# Stage 3: heartbeat
log "Stage 3: heartbeat started"
(
    while sleep 300; do
        echo "[heartbeat] $(date -u +%FT%TZ) lane=$LANE_ID pid=$$" >> "$LOG_DIR/heartbeat.log"
    done
) &
HEARTBEAT_PID=$!
log "Stage 3: heartbeat pid=$HEARTBEAT_PID"

# Stage 4: SC++ Stage 1 anchor training (smoke if SCPP_ALLOW_REMOTE_SCAFFOLD_SMOKE only).
log "Stage 4: SC++ Stage 1 anchor training (epochs=${STAGE_1_EPOCHS})"
TRAIN_CMD=(
    "$PYBIN" -u "$WORKSPACE/experiments/train_scpp_self_compression.py"
    --output-dir "$OUTPUT_DIR"
    --device cuda
    --video-path "$WORKSPACE/upstream/videos/0.mkv"
    --target-archive-bytes "$TARGET_ARCHIVE_BYTES"
    --stage-1-epochs "$STAGE_1_EPOCHS"
    --stage-2-epochs 0
    --stage-3-epochs 0
    --stage-4-epochs 0
    --stage-5-iters 0
    --latent-dim "$LATENT_DIM"
    --base-channels "$BASE_CHANNELS"
    --n-pairs "$N_PAIRS"
    --eval-roundtrip
    --seed 20
)

if [ "$ALLOW_REMOTE_SCAFFOLD_SMOKE" = "1" ] && [ "$ALLOW_SCORE_DOMAIN_TRAINING" != "1" ]; then
    TRAIN_CMD+=(--smoke)
    log "Stage 4: SMOKE-ONLY mode (synthetic data; no score claim)"
fi

if [ "$RUN_CONTEST_CUDA_AUTH_EVAL" = "1" ]; then
    log "FATAL: SCPP_RUN_CONTEST_CUDA_AUTH_EVAL=1 requested, but SC++ Stage 1 currently emits training/build custody only. Run exact contest-CUDA auth eval via the canonical claimed dispatcher after a byte-closed packet exists."
    close_dispatch_claim "refused_dispatch_scpp_exact_eval_not_implemented" "SC++ Stage 1 refused exact-eval request; trainer emits build artifact only"
    exit 12
fi

log "Stage 4: training command: ${TRAIN_CMD[*]}"
set +e
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_DIR/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
if [ "$TRAIN_RC" -ne 0 ]; then
    log "FATAL: SC++ Stage 1 training failed (rc=$TRAIN_RC)"
    close_dispatch_claim "failed_training_rc_${TRAIN_RC}" "SC++ Stage 1 trainer exited rc=$TRAIN_RC"
    exit "$TRAIN_RC"
fi

log "Stage 4: SC++ Stage 1 training OK"

# Stage 5: terminal claim close
log "Stage 5: closing dispatch claim"
close_dispatch_claim "completed_stage_1_anchor_smoke" "SC++ Stage 1 smoke anchor completed; stage_1_epochs=${STAGE_1_EPOCHS}; eval_roundtrip=true; auth_eval_on_best=${RUN_CONTEST_CUDA_AUTH_EVAL}; output_dir=${OUTPUT_DIR}; no score claim until exact [contest-CUDA] contest_auth_eval custody artifact lands separately"

# Update run record on success.
cat > "$RUN_RECORD" <<EOF
{
  "schema_version": "remote_run_record.v1",
  "status": "completed",
  "started_at_utc": "$(cat "$RUN_RECORD" | grep started_at_utc | awk -F'"' '{print $4}')",
  "completed_at_utc": "$(date -u +%FT%TZ)",
  "provenance_json": "${PROVENANCE}",
  "heartbeat_log": "${LOG_DIR}/heartbeat.log",
  "train_log": "${LOG_DIR}/train.log",
  "score_claim": false,
  "promotion_eligible": false,
  "evidence_grade": "[predicted; SC++ block-FP Stage 1 smoke; HNeRV parity discipline lesson 7]"
}
EOF

log "===== Remote lane SC++ Stage 1 anchor complete ====="
