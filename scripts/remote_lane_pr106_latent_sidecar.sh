#!/bin/bash
# CUDA_REQUIRED — tensor-side codec build plus contest auth eval.
# Lane PR106 + latent sidecar — 28-dim x 600-pair latents corrected via per-pair (dim, delta_q)
#
# Planning target: reproduce the PR100-style sidecar gain on PR106. The default
# path is now score_table: build a CUDA scorer table over latent perturbation
# candidates, reduce measured improvements into charged sidecar bytes, then run
# exact CUDA auth eval on the emitted archive.
#
# Pipeline (4 stages, single Vast.ai 4090 ~$0.30/hr × 30min ≈ $0.30 OR
# Lightning T4 final auth eval ~$0.22/hr × 30min ≈ $0.11):
#
#   Stage 0 (CPU): Provenance + CUDA preflight + heartbeat
#   Stage 1a (CUDA): Build latent candidate score table when mode=score_table
#   Stage 1b (CUDA): Build PR106 + sidecar archive from score table
#   Stage 2 (CPU): Local parser-roundtrip sanity check
#   Stage 3 (CUDA-T4): contest_auth_eval — score must be < 0.20945 (PR106) to ship
#
# Strict-scorer-rule: scorer is loaded only by Stage 1a compress-time table
# generation and Stage 3 contest auth eval. Inflate-time has NO scorer
# dependency. Inflate-time only needs HNeRVDecoder + brotli.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
LANE_ID="lane_pr106_latent_sidecar"
PR106_ARCHIVE="${PR106_ARCHIVE:-experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip}"
SIDECAR_TOP_K="${SIDECAR_TOP_K:-600}"
PR106_LATENT_MODE="${PR106_LATENT_MODE:-score_table}"
PR106_LATENT_DELTA_RADIUS="${PR106_LATENT_DELTA_RADIUS:-1}"
PR106_LATENT_N_PAIRS="${PR106_LATENT_N_PAIRS:-600}"
PR106_LATENT_DIM="${PR106_LATENT_DIM:-28}"
PR106_LATENT_SCORE_TABLE_BATCH_PAIRS="${PR106_LATENT_SCORE_TABLE_BATCH_PAIRS:-2}"
PR106_LATENT_SCORE_TABLE_CANDIDATE_BATCH_SIZE="${PR106_LATENT_SCORE_TABLE_CANDIDATE_BATCH_SIZE:-8}"
PR106_LATENT_SCORE_TABLE_RESUME="${PR106_LATENT_SCORE_TABLE_RESUME:-1}"
PR106_LATENT_SCORE_TABLE_NPY="${PR106_LATENT_SCORE_TABLE_NPY:-}"
PR106_LATENT_SCORE_TABLE_MANIFEST="${PR106_LATENT_SCORE_TABLE_MANIFEST:-}"
PR106_LATENT_SCORE_TABLE_LANE_ID="${PR106_LATENT_SCORE_TABLE_LANE_ID:-$LANE_ID}"
PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID="${PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID:-${INSTANCE_JOB_ID:-}}"

[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"

cd "$WORKSPACE"

LOG_DIR="${PR106_LATENT_LOG_DIR:-$WORKSPACE/experiments/results/${LANE_ID}_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"
log() { echo "[lane-pr106-sidecar] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Stage 0: NVDEC probe — required by preflight check_remote_scripts_have_nvdec_probe.
# probe MUST come before any GPU-work marker including bare `nvidia-smi`.
if [ "${SKIP_NVDEC_PROBE:-0}" != "1" ] && [ -f "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
        log "FATAL: NVDEC/DALI probe failed; exact CUDA eval is not trustworthy on this host."
        exit 2
    }
fi

# Heartbeat (per CLAUDE.md remote-code-parity rule)
HEARTBEAT="$LOG_DIR/heartbeat.log"
( while true; do echo "$(date -u +%FT%TZ) lane-pr106-sidecar alive" >> "$HEARTBEAT"; sleep 60; done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

# ── Stage 0: Provenance + CUDA preflight ──────────────────────────────────
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
"$PYBIN" -c "
import json, time, sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: --device cuda required per CLAUDE.md MPS-auth-eval-is-NOISE')
prov = {
    'lane_id': '$LANE_ID',
    'predicted_band': [0.205, 0.208],
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'pr106_archive': '$PR106_ARCHIVE',
    'latent_mode': '$PR106_LATENT_MODE',
    'latent_delta_radius': int('$PR106_LATENT_DELTA_RADIUS'),
    'latent_score_table_lane_id': '$PR106_LATENT_SCORE_TABLE_LANE_ID',
    'latent_score_table_instance_job_id': '$PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID',
    'sidecar_top_k': int('$SIDECAR_TOP_K'),
}
with open('$LOG_DIR/provenance.json', 'w') as f:
    json.dump(prov, f, indent=2)
print(f'[stage-0] provenance written; CUDA={torch.cuda.is_available()}; top_k={prov[\"sidecar_top_k\"]}')
"

# ── Stage 1: Build score table + PR106 sidecar archive ────────────────────
log "=== Stage 1: build PR106 + latent-correction sidecar (mode=$PR106_LATENT_MODE, top_k=$SIDECAR_TOP_K) ==="
BUILD_DIR="$LOG_DIR/build"
mkdir -p "$BUILD_DIR"

if [ "$PR106_LATENT_MODE" = "score_table" ] && [ -z "$PR106_LATENT_SCORE_TABLE_NPY" ]; then
    log "=== Stage 1a: generate CUDA latent score table ==="
    if [ -z "$PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID" ]; then
        log "FATAL: PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID is required for score_table mode"
        exit 3
    fi
    SCORE_TABLE_DIR="$LOG_DIR/score_table"
    mkdir -p "$SCORE_TABLE_DIR"
    PR106_LATENT_SCORE_TABLE_NPY="$SCORE_TABLE_DIR/score_table.npy"
    PR106_LATENT_SCORE_TABLE_MANIFEST="$SCORE_TABLE_DIR/score_table_manifest.json"
    if [ "$PR106_LATENT_SCORE_TABLE_RESUME" = "1" ] && [ -f "$PR106_LATENT_SCORE_TABLE_NPY" ] && [ -f "$PR106_LATENT_SCORE_TABLE_MANIFEST" ]; then
        log "Stage 1a RESUME: validating completed latent score table at $PR106_LATENT_SCORE_TABLE_NPY"
    fi
    SCORE_TABLE_ARGS=(
        experiments/build_pr106_latent_score_table.py
        --pr106-archive "$PR106_ARCHIVE"
        --out-dir "$SCORE_TABLE_DIR"
        --delta-radius "$PR106_LATENT_DELTA_RADIUS"
        --latent-dim "$PR106_LATENT_DIM"
        --n-pairs "$PR106_LATENT_N_PAIRS"
        --batch-pairs "$PR106_LATENT_SCORE_TABLE_BATCH_PAIRS"
        --candidate-batch-size "$PR106_LATENT_SCORE_TABLE_CANDIDATE_BATCH_SIZE"
        --lane-id "$PR106_LATENT_SCORE_TABLE_LANE_ID"
        --instance-job-id "$PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID"
    )
    if [ "$PR106_LATENT_SCORE_TABLE_RESUME" = "1" ]; then
        SCORE_TABLE_ARGS+=(--resume-checkpoint)
    fi
    "$PYBIN" -u "${SCORE_TABLE_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/run.log"
    if [ ! -f "$PR106_LATENT_SCORE_TABLE_NPY" ] || [ ! -f "$PR106_LATENT_SCORE_TABLE_MANIFEST" ]; then
        log "FATAL: latent score-table generation did not produce table+manifest"
        exit 3
    fi
fi

BUILD_ARGS=(
    experiments/build_pr106_latent_sidecar.py
    --source-archive "$PR106_ARCHIVE" \
    --output-dir "$BUILD_DIR" \
    --top-k "$SIDECAR_TOP_K" \
    --device cuda \
    --search-mode "$PR106_LATENT_MODE" \
    --delta-radius "$PR106_LATENT_DELTA_RADIUS"
)
if [ "$PR106_LATENT_MODE" = "score_table" ]; then
    BUILD_ARGS+=(--score-table-npy "$PR106_LATENT_SCORE_TABLE_NPY")
    if [ -n "$PR106_LATENT_SCORE_TABLE_MANIFEST" ]; then
        BUILD_ARGS+=(--score-table-manifest "$PR106_LATENT_SCORE_TABLE_MANIFEST")
    fi
fi
"$PYBIN" -u "${BUILD_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/run.log"
SIDECAR_ARCHIVE="$BUILD_DIR/sidecar_archive.zip"
if [ ! -f "$SIDECAR_ARCHIVE" ]; then
    log "FATAL: stage 1 did not produce $SIDECAR_ARCHIVE"
    exit 3
fi
ARCHIVE_BYTES=$(stat -c '%s' "$SIDECAR_ARCHIVE" 2>/dev/null || stat -f '%z' "$SIDECAR_ARCHIVE")
log "stage 1 OK: archive bytes=$ARCHIVE_BYTES"

# ── Stage 2: Local parser-roundtrip sanity ────────────────────────────────
log "=== Stage 2: parser-roundtrip sanity (no GPU forward) ==="
"$PYBIN" -c "
import sys, zipfile
sys.path.insert(0, '$WORKSPACE/submissions/pr106_latent_sidecar')
from inflate import parse_sidecar_archive, decode_sidecar_corrections
sys.path.insert(0, '$WORKSPACE/submissions/pr106_latent_sidecar/src')
from codec import parse_packed_archive
with zipfile.ZipFile('$SIDECAR_ARCHIVE') as z:
    bin_bytes = z.read('0.bin')
pr106_b, sidecar_b = parse_sidecar_archive(bin_bytes)
sd, lat, meta = parse_packed_archive(pr106_b)
dim, delta_q = decode_sidecar_corrections(sidecar_b)
assert lat.shape == (600, 28), f'lat shape {lat.shape} != (600, 28)'
assert dim.shape == (600,), f'dim shape {dim.shape} != (600,)'
assert delta_q.shape == (600,), f'delta_q shape {delta_q.shape} != (600,)'
n_corr = int((dim != 255).sum())
print(f'parse OK: {len(sd)} tensors, latents shape {tuple(lat.shape)}, '
      f'sidecar {n_corr}/600 pairs corrected, sidecar bytes {len(sidecar_b)}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Stage 3: Contest auth eval (CUDA T4) ─────────────────────────────────
log "=== Stage 3: contest auth eval (CUDA) ==="
INFLATE_SH="$WORKSPACE/submissions/pr106_latent_sidecar/inflate.sh"
EVAL_DIR="$LOG_DIR/eval"
mkdir -p "$EVAL_DIR"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$SIDECAR_ARCHIVE" \
    --inflate-sh "$INFLATE_SH" \
    --work-dir "$EVAL_DIR" \
    --keep-work-dir \
    --device cuda 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Final summary ─────────────────────────────────────────────────────────
SCORE_JSON="$EVAL_DIR/contest_auth_eval.json"
if [ -f "$SCORE_JSON" ]; then
    SCORE=$("$PYBIN" -c "import json; print(json.load(open('$SCORE_JSON'))['final_score'])" 2>/dev/null || echo "PARSE_FAIL")
    log "DONE: lane=$LANE_ID archive_bytes=$ARCHIVE_BYTES contest_cuda_score=$SCORE [contest-CUDA]"
    log "  beats PR106 baseline 0.20946? $("$PYBIN" -c "
s = $SCORE
print('YES — new public-frontier candidate' if isinstance(s, (int, float)) and s < 0.20946 else f'no (score {s} >= 0.20946)')
" 2>/dev/null || echo "?")"
else
    log "FATAL: stage 3 did not produce $SCORE_JSON"
    exit 4
fi
