#!/bin/bash
# Remote lane script for T1 — Ballé hyperprior + 128K decoder end-to-end.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline", this
# script DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function rather than re-implementing uv install,
# ffmpeg install, or torch CUDA pinning.
#
# Council memo refs:
#   - feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md
#     (Track 1 design at $80, predicted -0.030 to -0.040 score band)
#   - feedback_t1_balle_hyperprior_endtoend_BLOCKED_scaffold_missing_20260509.md
#     (the blocker that drove this scaffold; reactivation criteria 1-9 met)
#
# Cost: $80 hard cap. Vast.ai 4090 ~$0.42/hr × ~24h = ~$10 budget; the rest
# is reserved for retries, post-eval CPU dispatch, and recovery dispatches.
#
# Predicted score band: [predicted; Phase 1 scorer-domain training; not yet
# empirical]. This script may train the real A1 substrate with differentiable
# PoseNet/SegNet scorer-domain losses, but it MUST NOT print a [contest-CUDA]
# completion tag until a separate exact contest_auth_eval custody artifact lands.
# Default behavior is fail-closed: set T1_ALLOW_SCORE_DOMAIN_TRAINING=1 for the
# real training path, or T1_ALLOW_REMOTE_SCAFFOLD_SMOKE=1 for a synthetic
# build-only probe.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity — non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="t1_balle_128k_endtoend"
TAG="${TAG:-t1_balle_endtoend}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_t1_balle_endtoend_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
PROVENANCE="$LOG_DIR/provenance.json"
RUN_RECORD="$LOG_DIR/run_record.json"
ALLOW_REMOTE_SCAFFOLD_SMOKE="${T1_ALLOW_REMOTE_SCAFFOLD_SMOKE:-0}"
ALLOW_SCORE_DOMAIN_TRAINING="${T1_ALLOW_SCORE_DOMAIN_TRAINING:-0}"
if [ -z "${T1_RUN_CONTEST_CUDA_AUTH_EVAL+x}" ]; then
    if [ "$ALLOW_SCORE_DOMAIN_TRAINING" = "1" ]; then
        RUN_CONTEST_CUDA_AUTH_EVAL="1"
    else
        RUN_CONTEST_CUDA_AUTH_EVAL="0"
    fi
else
    RUN_CONTEST_CUDA_AUTH_EVAL="$T1_RUN_CONTEST_CUDA_AUTH_EVAL"
fi
DISPATCH_INSTANCE_JOB_ID="${T1_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_CLAIMS_PATH="${T1_DISPATCH_CLAIMS_PATH:-$WORKSPACE/.omx/state/active_lane_dispatch_claims.md}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-vastai}"
DECLARED_MOUNTED_GIT_HEAD="${T1_MOUNTED_CODE_GIT_HEAD:-}"
DECLARED_MOUNTED_GIT_BRANCH="${T1_MOUNTED_CODE_GIT_BRANCH:-}"
PR95_PARITY_PROFILE="${T1_PR95_PARITY_PROFILE:-$WORKSPACE/.omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json}"
CLAIM_VERIFIED=0
TERMINAL_CLAIM_CLOSED=0
HEARTBEAT_PID=""

require_active_dispatch_claim() {
    if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
        echo "[lane-t1] FATAL: T1_ALLOW_SCORE_DOMAIN_TRAINING=1 requires T1_DISPATCH_INSTANCE_JOB_ID or DISPATCH_INSTANCE_JOB_ID for active lane-claim verification." >&2
        exit 21
    fi
    if [ ! -f "$WORKSPACE/tools/claim_lane_dispatch.py" ]; then
        echo "[lane-t1] FATAL: claim helper missing at $WORKSPACE/tools/claim_lane_dispatch.py" >&2
        exit 21
    fi
    if [ ! -f "$DISPATCH_CLAIMS_PATH" ]; then
        echo "[lane-t1] FATAL: dispatch-claims ledger missing at $DISPATCH_CLAIMS_PATH; copy the active claim ledger to the remote host before training" >&2
        exit 21
    fi
    CLAIM_PYTHON="${PYBIN:-}"
    if [ -z "$CLAIM_PYTHON" ] && [ -x "$WORKSPACE/.venv/bin/python" ]; then
        CLAIM_PYTHON="$WORKSPACE/.venv/bin/python"
    fi
    if [ -z "$CLAIM_PYTHON" ]; then
        CLAIM_PYTHON="python3"
    fi
    "$CLAIM_PYTHON" - "$WORKSPACE" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID" "$DISPATCH_CLAIMS_PATH" <<'PY'
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
        --agent "codex:remote_lane_t1_balle_endtoend" \
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
        close_dispatch_claim "failed_remote_script_rc_${rc}" "T1 remote script exited rc=${rc} before terminal stage-specific closure"
    fi
}

if [ "$ALLOW_SCORE_DOMAIN_TRAINING" != "1" ] && [ "$ALLOW_REMOTE_SCAFFOLD_SMOKE" != "1" ]; then
    echo "[lane-t1] FATAL: remote T1 refused by default. Set T1_ALLOW_SCORE_DOMAIN_TRAINING=1 for real scorer-domain training, or T1_ALLOW_REMOTE_SCAFFOLD_SMOKE=1 only for explicit synthetic smoke verification." >&2
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
rm -f upstream/videos/._*.mkv
export PYTHONHASHSEED=20
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

log() { echo "[lane-t1] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log" ; }

resolve_python_bin() {
    if [ -n "$PYBIN" ]; then
        if command -v "$PYBIN" >/dev/null 2>&1; then
            command -v "$PYBIN"
            return 0
        fi
        if [ -x "$PYBIN" ]; then
            echo "$PYBIN"
            return 0
        fi
        return 1
    fi
    for candidate in /opt/conda/bin/python /usr/local/bin/python python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

log "===== Remote lane T1 (Ballé end-to-end) starting ====="
log "lane_id: $LANE_ID"
log "workspace: $WORKSPACE"
log "log_dir: $LOG_DIR"
log "output_dir: $OUTPUT_DIR"

# Stage 0: Git parity — required before any work per CLAUDE.md
# "Remote code parity — non-negotiable".
normalize_git_probe_output() {
    local kind="$1"
    local raw="${2:-}"
    local cleaned line_count
    cleaned="$(printf '%s\n' "$raw" | tr -d '\r' | awk 'NF {print}')"
    if [ -z "$cleaned" ]; then
        echo unknown
        return 0
    fi
    line_count="$(printf '%s\n' "$cleaned" | wc -l | tr -d '[:space:]')"
    if [ "$line_count" != "1" ]; then
        echo unknown
        return 0
    fi
    if [ "$kind" = "head" ] && ! [[ "$cleaned" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo unknown
        return 0
    fi
    if [ "$kind" = "branch" ] && [ "$cleaned" = "HEAD" ]; then
        echo unknown
        return 0
    fi
    echo "$cleaned"
}

read_git_probe_value() {
    local kind="$1"
    shift
    local raw rc
    set +e
    raw="$("$@" 2>/dev/null)"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo unknown
        return 0
    fi
    normalize_git_probe_output "$kind" "$raw"
}

LOCAL_BRANCH="$(read_git_probe_value branch git -C "$WORKSPACE" branch --show-current)"
LOCAL_HEAD="$(read_git_probe_value head git -C "$WORKSPACE" rev-parse HEAD)"
log "local branch: $LOCAL_BRANCH"
log "local HEAD: $LOCAL_HEAD"
log "declared mounted git branch: ${DECLARED_MOUNTED_GIT_BRANCH:-unset}"
log "declared mounted git HEAD: ${DECLARED_MOUNTED_GIT_HEAD:-unset}"
if [ -z "$DECLARED_MOUNTED_GIT_HEAD" ] || [ -z "$DECLARED_MOUNTED_GIT_BRANCH" ]; then
    log "FATAL: missing T1_MOUNTED_CODE_GIT_HEAD/T1_MOUNTED_CODE_GIT_BRANCH custody metadata"
    exit 10
fi
if [ "$DECLARED_MOUNTED_GIT_BRANCH" != "main" ]; then
    log "FATAL: refusing T1 remote work for non-main mounted branch ($DECLARED_MOUNTED_GIT_BRANCH)"
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
  "script": "scripts/remote_lane_t1_balle_endtoend.sh",
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
  "archive_in_loop_env": "${T1_ARCHIVE_IN_LOOP:-1}",
  "archive_every_epochs_env": "${T1_ARCHIVE_EVERY_EPOCHS:-${EVAL_EVERY_EPOCHS:-100}}",
  "max_candidate_archive_bytes_env": "${T1_MAX_CANDIDATE_ARCHIVE_BYTES:-190000}",
  "max_exact_cuda_candidates_env": "${T1_MAX_EXACT_CUDA_CANDIDATES:-1}",
  "pr95_parity_profile": "${PR95_PARITY_PROFILE}",
  "predicted_band": [0.155, 0.180],
  "prediction_scope": "Phase 1 scorer-domain A1 substrate training when T1_ALLOW_SCORE_DOMAIN_TRAINING=1; no score claim unless packet compiler plus contest-CUDA auth-eval adjudication both pass",
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
    log "FATAL: refusing T1 remote work outside main (branch=$LOCAL_BRANCH)"
    exit 10
fi

# Stage 1: Bootstrap deps via canonical wrapper (NEVER inline uv install
# per CLAUDE.md "Forbidden re-implementing remote bootstrap inline").
log "Stage 1: bootstrap_runtime_deps"
if [ ! -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    log "FATAL: canonical bootstrap wrapper missing"
    exit 11
fi
# Source the bootstrap function only.
# shellcheck disable=SC1091
source <(awk '/^bootstrap_runtime_deps\(\)/,/^}/' "$WORKSPACE/scripts/remote_archive_only_eval.sh")
bootstrap_runtime_deps
log "Stage 1: bootstrap_runtime_deps OK"

PYBIN="$(resolve_python_bin)" || { log "FATAL: python executable not found; set PYBIN explicitly"; exit 12; }
export PYBIN
log "python_bin: $PYBIN"

# Stage 2: Install scaffold-specific deps. compressai is the only T1-
# specific dep beyond what bootstrap_runtime_deps installs.
log "Stage 2: install compressai==1.2.8"
uv pip install --system compressai==1.2.8 \
    > "$LOG_DIR/uv_install_compressai.log" 2>&1 \
    || { log "FATAL: compressai install failed"; cat "$LOG_DIR/uv_install_compressai.log"; exit 12; }

# Stage 3: NVDEC probe (per CLAUDE.md remote_scripts_have_nvdec_probe rule).
log "Stage 3: NVDEC probe"
if [[ "${LOCAL_CUDA_WORKER:-0}" != "1" ]]; then
    log "FATAL: LOCAL_CUDA_WORKER=1 must be set on the remote CUDA provider before NVDEC/CUDA probing"
    exit 13
fi
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: probe_nvdec.sh failed; refusing CUDA training on this host"
    exit 13
}
"$PYBIN" -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" \
    | tee "$LOG_DIR/nvdec_probe.log"
if ! grep -q "cuda: True" "$LOG_DIR/nvdec_probe.log"; then
    log "FATAL: cuda not available; refusing to dispatch (would produce [advisory only] score)"
    exit 13
fi

# Stage 4: Heartbeat watchdog (every 5 min per CLAUDE.md "Remote code parity").
HEARTBEAT="$LOG_DIR/heartbeat.log"
(
    while true; do
        sleep 300
        echo "$(date -u +%FT%TZ) HB pid=$$" >> "$HEARTBEAT"
    done
) &
HEARTBEAT_PID=$!
log "Stage 4: heartbeat started (pid=$HEARTBEAT_PID)"

# Stage 5: Run either real score-domain training or the legacy synthetic smoke
# probe. The real path trains on frozen A1 latents + upstream/videos/0.mkv
# under PR95 eval-roundtrip/YUV6, T8 Sinkhorn-by-default, T13 sqrt(n) budget,
# and T19 adaptive rho. Auth eval remains separate so scored evidence gets its
# own dispatch claim, archive SHA, runtime custody, hardware tag, and logs.
TRAIN_CMD=(
    "$PYBIN" -u "$WORKSPACE/experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
    --output-dir "$OUTPUT_DIR"
    --device cuda
    --epochs "${EPOCHS:-3000}"
    --batch-size "${BATCH_SIZE:-16}"
    --learning-rate "${LR:-1e-4}"
    --aux-learning-rate "${AUX_LR:-1e-3}"
    --ema-decay 0.997
    --rate-target-bytes "${RATE_TARGET_BYTES:-80000}"
    --noise-std 0.5
    --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}"
    --eval-every-epochs "${EVAL_EVERY_EPOCHS:-100}"
    --eval-batch-size "${EVAL_BATCH_SIZE:-${BATCH_SIZE:-16}}"
    --archive-in-loop
    --archive-every-epochs "${T1_ARCHIVE_EVERY_EPOCHS:-${EVAL_EVERY_EPOCHS:-100}}"
    --max-candidate-archive-bytes "${T1_MAX_CANDIDATE_ARCHIVE_BYTES:-190000}"
    --max-exact-cuda-candidates "${T1_MAX_EXACT_CUDA_CANDIDATES:-1}"
    --baseline-archive-sha256 "${T1_BASELINE_ARCHIVE_SHA256:-87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5}"
    --baseline-archive-size-bytes "${T1_BASELINE_ARCHIVE_SIZE_BYTES:-178262}"
    --pr95-parity-profile "$PR95_PARITY_PROFILE"
    --enable-t13-sqrt-n-budget
    --enable-t19-adaptive-rho
    --no-auth-eval
    --seed 20
)

if [ "$ALLOW_SCORE_DOMAIN_TRAINING" = "1" ]; then
    if [ ! -f "$PR95_PARITY_PROFILE" ]; then
        log "FATAL: missing PR95 parity profile at $PR95_PARITY_PROFILE; set T1_PR95_PARITY_PROFILE or run experiments/profile_pr95_hnerv_muon_intake.py before dispatch"
        exit 22
    fi
    log "Stage 5: score-domain training (epochs=${EPOCHS:-3000})"
    TRAIN_CMD+=(
        --enable-scorer-domain-loss
        # Default aligned with trainer argparse and PR95/T1 parity.
        # Set SEGMENTATION_SURROGATE=soft_cosine explicitly for speed probes.
        --segmentation-surrogate "${SEGMENTATION_SURROGATE:-sinkhorn}"
        --sinkhorn-max-positions-per-chunk "${SINKHORN_MAX_POSITIONS_PER_CHUNK:-2048}"
        --pixel-l1-anchor-weight "${PIXEL_L1_ANCHOR_WEIGHT:-0.0}"
    )
    # Tier-1 env→CLI wiring (per 2026-05-12 fresh-eyes adversarial NF1):
    # Without these branches the Tier-1 wins landed in the trainer but the
    # operator wrapper that sets T1_ENABLE_* never reached the trainer flags.
    if [ "${T1_ENABLE_AUTOCAST_FP16:-0}" = "1" ]; then
        TRAIN_CMD+=(--enable-autocast-fp16)
    fi
    if [ "${T1_ENABLE_MP4_CODEC_SIM:-0}" = "1" ]; then
        TRAIN_CMD+=(--enable-mp4-codec-sim)
        TRAIN_CMD+=(--mp4-codec-sim-noise-std "${T1_MP4_CODEC_SIM_NOISE_STD:-0.0}")
    fi
    if [ "${T1_ENABLE_T20_KL_POSE_DISTILL:-0}" = "1" ]; then
        TRAIN_CMD+=(--enable-t20-kl-pose-distill)
        if [ -n "${T1_T20_MODE:-}" ]; then
            TRAIN_CMD+=(--t20-mode "$T1_T20_MODE")
        fi
        if [ -n "${T1_T20_WEIGHT_POSE:-}" ]; then
            TRAIN_CMD+=(--t20-weight-pose "$T1_T20_WEIGHT_POSE")
        fi
    fi
    if [ "${T1_ENABLE_T22_TEMPORAL_CONSISTENCY:-0}" = "1" ]; then
        TRAIN_CMD+=(--enable-t22-temporal-consistency)
        if [ -n "${T1_T22_LAMBDA_WEIGHT:-}" ]; then
            TRAIN_CMD+=(--t22-lambda-weight "$T1_T22_LAMBDA_WEIGHT")
        fi
    fi
    if [ -n "${MAX_TARGET_PAIRS:-}" ]; then
        TRAIN_CMD+=(--max-target-pairs "$MAX_TARGET_PAIRS")
    fi
else
    log "Stage 5: scaffold smoke training (epochs=1)"
    TRAIN_CMD=(
        "$PYBIN" -u "$WORKSPACE/experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
        --output-dir "$OUTPUT_DIR"
        --device cuda
        --epochs 1
        --batch-size "${BATCH_SIZE:-16}"
        --learning-rate "${LR:-1e-4}"
        --aux-learning-rate "${AUX_LR:-1e-3}"
        --ema-decay 0.997
        --rate-target-bytes "${RATE_TARGET_BYTES:-80000}"
        --noise-std 0.5
        --eval-every-epochs "${EVAL_EVERY_EPOCHS:-100}"
        --eval-batch-size "${EVAL_BATCH_SIZE:-${BATCH_SIZE:-16}}"
        --pr95-parity-profile "$PR95_PARITY_PROFILE"
        --smoke
        --allow-missing-canonical-a1
        --no-auth-eval
        --seed 20
    )
fi

set +e
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_DIR/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
log "Stage 5: train exited with rc=$TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
    log "FATAL: trainer failed"
    exit "$TRAIN_RC"
fi

# Stage 6: Surface results, then optionally compile + exact-auth-eval the
# emitted packet. The trainer keeps --no-auth-eval so score custody is owned by
# this stage: exact archive bytes, runtime tree SHA, durable work dir/json, and
# dispatch-claim closure are recorded together.
ARCHIVE="$OUTPUT_DIR/archive.zip"
if [ ! -f "$ARCHIVE" ]; then
    log "FATAL: expected scaffold archive missing: $ARCHIVE"
    exit 14
fi
ARCHIVE_BYTES=$(stat -f%z "$ARCHIVE" 2>/dev/null || stat -c%s "$ARCHIVE")
ARCHIVE_SHA=$(sha256sum "$ARCHIVE" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)
log "ARCHIVE bytes=$ARCHIVE_BYTES sha=$ARCHIVE_SHA"
if [ "$ALLOW_SCORE_DOMAIN_TRAINING" = "1" ]; then
    if [ "$RUN_CONTEST_CUDA_AUTH_EVAL" != "1" ]; then
        log "AUTH_EVAL_JSON=not_applicable_training_only_no_score_claim"
        echo "[lane-t1] LANE_T1_SCORE_DOMAIN_TRAINING_DONE [archive produced; no contest score claim]"
        close_dispatch_claim "completed_t1_score_domain_training_no_score_claim" "T1 score-domain training completed and produced archive; auth eval disabled; score_claim=false"
    else
        SUBMISSION_DIR="$OUTPUT_DIR/submission_dir"
        if [ ! -d "$SUBMISSION_DIR" ]; then
            log "FATAL: expected submission_dir missing: $SUBMISSION_DIR"
            close_dispatch_claim "failed_t1_missing_submission_dir" "trainer produced archive but no submission_dir runtime for exact eval"
            exit 15
        fi
        PACKET_DIR="${T1_PACKET_DIR:-$LOG_DIR/packet_compiled}"
        SKIP_PACKET_COMPILE=0
        ARCHIVE_BUILDS_MANIFEST="$OUTPUT_DIR/archive_builds_manifest.json"
        EXACT_CUDA_CANDIDATES_JSONL="$LOG_DIR/exact_cuda_candidates.jsonl"
        SELECTED_EXACT_CUDA_CANDIDATE_JSON="$LOG_DIR/selected_exact_cuda_candidate.json"
        if [ -f "$ARCHIVE_BUILDS_MANIFEST" ]; then
            SELECTED_PACKET_DIR=$("$PYBIN" - "$ARCHIVE_BUILDS_MANIFEST" "$EXACT_CUDA_CANDIDATES_JSONL" "$SELECTED_EXACT_CUDA_CANDIDATE_JSON" "${T1_MAX_EXACT_CUDA_CANDIDATES:-1}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
jsonl_path = Path(sys.argv[2])
selected_path = Path(sys.argv[3])
limit = int(sys.argv[4])
payload = json.loads(manifest_path.read_text())
rows = payload.get("rows", [])
eligible = []
for row in rows:
    raw_packet_dir = row.get("compiled_packet_dir")
    if not isinstance(raw_packet_dir, str) or not raw_packet_dir:
        continue
    packet_dir = Path(raw_packet_dir)
    blockers = list(row.get("compiler_blockers") or [])
    if (
        row.get("exact_cuda_eligible") is True
        and row.get("rate_cap_passed") is True
        and not blockers
        and (packet_dir / "archive.zip").is_file()
        and (packet_dir / "inflate.sh").is_file()
        and (packet_dir / "build_manifest.json").is_file()
    ):
        eligible.append(row)

def score_key(row):
    metrics = row.get("proxy_metrics") if isinstance(row.get("proxy_metrics"), dict) else {}
    proxy = metrics.get("ema_proxy_pixel_l1")
    rate_bits = metrics.get("ema_proxy_rate_bits")
    return (
        float(proxy) if isinstance(proxy, (int, float)) else float("inf"),
        float(rate_bits) if isinstance(rate_bits, (int, float)) else float("inf"),
        int(row.get("archive_bytes") or 10**18),
        int(row.get("epoch") or 10**18),
    )

selected_rows = sorted(eligible, key=score_key)[:max(1, limit)]
jsonl_path.parent.mkdir(parents=True, exist_ok=True)
with jsonl_path.open("w") as f:
    for row in selected_rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")
selected = selected_rows[0] if selected_rows else {
    "selected": False,
    "reason": "no_archive_in_loop_exact_cuda_eligible_candidates",
    "manifest": manifest_path.as_posix(),
}
selected_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
print(selected.get("compiled_packet_dir") or "")
PY
)
            if [ -n "$SELECTED_PACKET_DIR" ]; then
                PACKET_DIR="$SELECTED_PACKET_DIR"
                SKIP_PACKET_COMPILE=1
                log "Stage 6a: selected archive-in-loop packet for exact eval: $PACKET_DIR"
                log "Stage 6a: candidate selection json: $SELECTED_EXACT_CUDA_CANDIDATE_JSON"
            else
                log "Stage 6a: no archive-in-loop exact-CUDA-eligible candidate; falling back to terminal archive packet compile"
            fi
        fi
        BASELINE_ARCHIVE_SHA="${T1_BASELINE_ARCHIVE_SHA256:-87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5}"
        BASELINE_ARCHIVE_SIZE="${T1_BASELINE_ARCHIVE_SIZE_BYTES:-178262}"
        PACKET_CMD=(
            "$PYBIN" -u "$WORKSPACE/tools/build_phase1_packet_compiler.py"
            --input-packet "$SUBMISSION_DIR"
            --output-dir "$PACKET_DIR"
            --mode optimize
            --target-mode contest_one_video_replay
            --runtime-dep-closure torch brotli compressai
            --export-format phase1_three_member_x_decoder_bin_balle_bin
            --bolt-on-loc-budget 400
            --allow-existing-output-dir
            --score-affecting-payload-changed
            --baseline-archive-sha256 "$BASELINE_ARCHIVE_SHA"
            --baseline-archive-size-bytes "$BASELINE_ARCHIVE_SIZE"
            --print-result-json
        )
        if [ "$SKIP_PACKET_COMPILE" != "1" ]; then
            log "Stage 6a: compile Phase 1 packet for exact eval"
            set +e
            "${PACKET_CMD[@]}" 2>&1 | tee "$LOG_DIR/packet_compiler.log"
            PACKET_RC=${PIPESTATUS[0]}
            set -e
            log "Stage 6a: packet compiler exited with rc=$PACKET_RC"
            if [ "$PACKET_RC" -ne 0 ]; then
                close_dispatch_claim "failed_t1_packet_compile_rc_${PACKET_RC}" "T1 packet compiler refused exact-eval packet; see packet_compiler.log"
                exit "$PACKET_RC"
            fi
        fi
        if [ ! -f "$PACKET_DIR/build_manifest.json" ] || [ ! -f "$PACKET_DIR/archive.zip" ] || [ ! -x "$PACKET_DIR/inflate.sh" ]; then
            log "FATAL: packet compiler did not emit required archive/runtime custody files"
            close_dispatch_claim "failed_t1_packet_compile_missing_custody" "missing build_manifest/archive.zip/executable inflate.sh after compile"
            exit 16
        fi
        PACKET_COMPILER_RUNTIME_TREE_SHA=$("$PYBIN" - "$PACKET_DIR/build_manifest.json" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
print(manifest.get("pre_manifest_runtime_tree_sha256") or manifest.get("runtime_tree_sha256") or "")
PY
)
        AUTH_EVAL_EXPECTED_RUNTIME_TREE_SHA=$("$PYBIN" - "$PACKET_DIR/inflate.sh" "$WORKSPACE/upstream" "$WORKSPACE" <<'PY'
import sys
from pathlib import Path
from experiments.contest_auth_eval import _runtime_dependency_manifest

manifest = _runtime_dependency_manifest(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    repo_root=Path(sys.argv[3]),
)
print(manifest["runtime_tree_sha256"])
PY
)
        log "packet compiler runtime tree sha: $PACKET_COMPILER_RUNTIME_TREE_SHA"
        log "contest auth-eval expected runtime tree sha: $AUTH_EVAL_EXPECTED_RUNTIME_TREE_SHA"
        AUTH_EVAL_JSON="${T1_AUTH_EVAL_JSON:-$LOG_DIR/contest_auth_eval.json}"
        AUTH_EVAL_WORK_DIR="${T1_AUTH_EVAL_WORK_DIR:-$LOG_DIR/auth_eval_work}"
        AUTH_EVAL_ADJUDICATION_JSON="${T1_AUTH_EVAL_ADJUDICATION_JSON:-$LOG_DIR/auth_eval_adjudication.json}"
        AUTH_EVAL_CMD=(
            "$PYBIN" -u "$WORKSPACE/experiments/contest_auth_eval.py"
            --archive "$PACKET_DIR/archive.zip"
            --inflate-sh "$PACKET_DIR/inflate.sh"
            --upstream-dir "$WORKSPACE/upstream"
            --device cuda
            --work-dir "$AUTH_EVAL_WORK_DIR"
            --json-out "$AUTH_EVAL_JSON"
            --keep-work-dir
            --expected-runtime-tree-sha256 "$AUTH_EVAL_EXPECTED_RUNTIME_TREE_SHA"
        )
        log "Stage 6b: contest-CUDA auth eval"
        set +e
        "${AUTH_EVAL_CMD[@]}" 2>&1 | tee "$LOG_DIR/contest_auth_eval.log"
        AUTH_EVAL_RC=${PIPESTATUS[0]}
        set -e
        log "Stage 6b: contest_auth_eval exited with rc=$AUTH_EVAL_RC"
        if [ "$AUTH_EVAL_RC" -ne 0 ]; then
            close_dispatch_claim "failed_t1_contest_auth_eval_rc_${AUTH_EVAL_RC}" "contest_auth_eval.py failed; no score claim"
            exit "$AUTH_EVAL_RC"
        fi
        set +e
        "$PYBIN" - "$AUTH_EVAL_JSON" "$PACKET_DIR/build_manifest.json" "$PACKET_DIR/archive.zip" "$AUTH_EVAL_ADJUDICATION_JSON" "$AUTH_EVAL_EXPECTED_RUNTIME_TREE_SHA" <<'PY'
import json
import sys
from pathlib import Path

from tac.auth_eval_schema import eval_metric_summary, required_contest_cuda_evidence_blockers

auth_json = Path(sys.argv[1])
manifest_json = Path(sys.argv[2])
archive_path = Path(sys.argv[3])
out_json = Path(sys.argv[4])
expected_runtime_tree_sha256 = sys.argv[5]


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )

eval_data = json.loads(auth_json.read_text())
manifest = json.loads(manifest_json.read_text())
runtime_manifest = eval_data.get("provenance", {}).get("inflate_runtime_manifest", {})
if not isinstance(runtime_manifest, dict):
    runtime_manifest = {}
runtime_tree_sha256 = runtime_manifest.get("runtime_tree_sha256")
metrics = eval_metric_summary(eval_data)
blockers = required_contest_cuda_evidence_blockers(
    eval_data,
    metrics,
    expected_archive_bytes=archive_path.stat().st_size,
    expected_n_samples=600,
)
if eval_data.get("provenance", {}).get("archive_sha256") != manifest.get("archive_sha256"):
    blockers.append("provenance_archive_sha256_mismatch_packet_manifest")
if eval_data.get("provenance", {}).get("archive_size_bytes") != manifest.get("archive_size_bytes"):
    blockers.append("provenance_archive_size_mismatch_packet_manifest")
if manifest.get("score_claim") is not False:
    blockers.append("packet_manifest_unexpected_score_claim")
if manifest.get("no_op_proof", {}).get("runtime_consumption_proof") is not True:
    blockers.append("packet_no_op_runtime_consumption_not_proven")
if manifest.get("blockers"):
    blockers.append("packet_manifest_has_blockers")
if not _is_sha256(runtime_tree_sha256):
    blockers.append("contest_auth_eval_runtime_tree_sha256_missing_or_invalid")
if runtime_tree_sha256 != expected_runtime_tree_sha256:
    blockers.append("contest_auth_eval_runtime_tree_sha256_mismatch_expected")

payload = {
    "schema_version": "t1_contest_cuda_auth_eval_adjudication.v1",
    "auth_eval_json": auth_json.as_posix(),
    "packet_build_manifest": manifest_json.as_posix(),
    "packet_archive": archive_path.as_posix(),
    "packet_archive_sha256": manifest.get("archive_sha256"),
    "packet_archive_size_bytes": manifest.get("archive_size_bytes"),
    "runtime_tree_sha256": runtime_tree_sha256,
    "runtime_tree_sha256_source": (
        "contest_auth_eval.provenance.inflate_runtime_manifest.runtime_tree_sha256"
    ),
    "expected_runtime_tree_sha256": expected_runtime_tree_sha256,
    "packet_manifest_runtime_tree_sha256": manifest.get("runtime_tree_sha256"),
    "packet_pre_manifest_runtime_tree_sha256": manifest.get("pre_manifest_runtime_tree_sha256"),
    "metrics": metrics,
    "blockers": blockers,
    "score_claim": not blockers,
    "promotion_eligible": False,
    "promotion_blockers": [
        "paired_contest_cpu_reproduction_required",
        "registry_level_promotion_required",
        "operator_submission_policy_required",
    ],
    "evidence_semantics": "contest_cuda_exact_auth_eval" if not blockers else "blocked_contest_cuda_auth_eval",
}
out_json.write_text(json.dumps(payload, indent=2) + "\n")
if blockers:
    print("[lane-t1-adjudication] BLOCKERS: " + ", ".join(blockers), file=sys.stderr)
    raise SystemExit(17)
print("[lane-t1-adjudication] contest-CUDA exact auth-eval evidence accepted")
PY
        ADJUDICATION_RC=$?
        set -e
        if [ "$ADJUDICATION_RC" -ne 0 ]; then
            close_dispatch_claim "failed_t1_auth_eval_adjudication" "contest_auth_eval completed but evidence semantics/custody failed; see auth_eval_adjudication.json"
            exit "$ADJUDICATION_RC"
        fi
        log "AUTH_EVAL_JSON=$AUTH_EVAL_JSON"
        log "AUTH_EVAL_ADJUDICATION_JSON=$AUTH_EVAL_ADJUDICATION_JSON"
        echo "[lane-t1] LANE_T1_CONTEST_CUDA_AUTH_EVAL_DONE [contest-CUDA]"
        close_dispatch_claim "completed_t1_contest_cuda_auth_eval" "T1 packet compiler + contest-CUDA auth eval completed with exact archive/runtime custody"
    fi
else
    echo "[lane-t1] LANE_T1_LOCAL_SCAFFOLD_SMOKE_DONE [scaffold-smoke only; no provider score/eval claim]"
fi
log "===== Remote lane T1 DONE ====="
