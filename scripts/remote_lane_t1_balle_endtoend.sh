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
# Predicted score band: [predicted; Phase 1 scaffold; not yet empirical].
# This script is scaffold/build verification only. It MUST NOT print a
# [contest-CUDA] completion tag until Phase 2 lands a hermetic inflate
# contract and exact contest_auth_eval custody.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity — non-negotiable".
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="t1_balle_128k_endtoend"
TAG="${TAG:-t1_balle_endtoend}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_t1_balle_endtoend_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/output}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

cd "$WORKSPACE"
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
LOCAL_BRANCH="$(git -C "$WORKSPACE" branch --show-current 2>/dev/null || echo unknown)"
LOCAL_HEAD="$(git -C "$WORKSPACE" rev-parse HEAD 2>/dev/null || echo unknown)"
log "local branch: $LOCAL_BRANCH"
log "local HEAD: $LOCAL_HEAD"
if [ "$LOCAL_BRANCH" != "main" ]; then
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
trap "kill $HEARTBEAT_PID 2>/dev/null || true" EXIT

# Stage 5: Run trainer in scaffold-smoke mode. Full training requires real
# frame targets and a closed export contract; auth-eval is deliberately
# refused by the trainer in Phase 1.
log "Stage 5: scaffold smoke training (epochs=1)"
"$PYBIN" -u "$WORKSPACE/experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --epochs 1 \
    --batch-size "${BATCH_SIZE:-16}" \
    --learning-rate "${LR:-1e-4}" \
    --aux-learning-rate "${AUX_LR:-1e-3}" \
    --ema-decay 0.997 \
    --rate-target-bytes "${RATE_TARGET_BYTES:-80000}" \
    --noise-std 0.5 \
    --eval-every-epochs "${EVAL_EVERY_EPOCHS:-100}" \
    --smoke \
    --allow-missing-canonical-a1 \
    --no-auth-eval \
    --seed 20 \
    2>&1 | tee "$LOG_DIR/train.log"

TRAIN_RC=${PIPESTATUS[0]}
log "Stage 5: train exited with rc=$TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ]; then
    log "FATAL: trainer failed"
    exit "$TRAIN_RC"
fi

# Stage 6: Surface results.
ARCHIVE="$OUTPUT_DIR/archive.zip"
if [ ! -f "$ARCHIVE" ]; then
    log "FATAL: expected scaffold archive missing: $ARCHIVE"
    exit 14
fi
ARCHIVE_BYTES=$(stat -f%z "$ARCHIVE" 2>/dev/null || stat -c%s "$ARCHIVE")
ARCHIVE_SHA=$(sha256sum "$ARCHIVE" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)
log "ARCHIVE bytes=$ARCHIVE_BYTES sha=$ARCHIVE_SHA"
log "AUTH_EVAL_JSON=not_applicable_phase1_research_only_no_export"
echo "[lane-t1] LANE_T1_SCAFFOLD_SMOKE_DONE [predicted; Phase 1 scaffold; not yet empirical]"
log "===== Remote lane T1 DONE ====="
