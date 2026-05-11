#!/bin/bash
# NO_NVDEC_NEEDED — pure tensor-side codec + scorer-forward; no DALI/NVDEC video pipeline.
# Lane PR106 + latent sidecar — 28-dim x 600-pair latents corrected via per-pair (dim, delta_q)
#
# Planning target: reproduce the PR100-style sidecar gain on PR106. The current
# builder emits a heuristic nonzero smoke sidecar, so any exact eval from this
# script is exploratory until a scorer-backed selector replaces the heuristic.
#
# Pipeline (3 stages, single Vast.ai 4090 ~$0.30/hr × 30min ≈ $0.30 OR
# Lightning T4 final auth eval ~$0.22/hr × 30min ≈ $0.11):
#
#   Stage 0 (CPU): Provenance + CUDA preflight + heartbeat
#   Stage 1 (CUDA): Build PR106 + sidecar archive (heuristic per-pair selector;
#                   score_claim=false until a scorer-backed selector lands)
#   Stage 2 (CPU): Local parser-roundtrip sanity check
#   Stage 3 (CUDA-T4): contest_auth_eval — score must be < 0.20945 (PR106) to ship
#
# Strict-scorer-rule: scorer is loaded only by future score-aware build modes
# and by Stage 3 contest auth eval. Inflate-time has NO scorer dependency.
# Inflate-time only needs HNeRVDecoder + brotli.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
PR106_ARCHIVE="${PR106_ARCHIVE:-experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip}"
SIDECAR_TOP_K="${SIDECAR_TOP_K:-600}"

[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"

# Stage 0: NVDEC probe — required by preflight check_remote_scripts_have_nvdec_probe.
# probe MUST come before any GPU-work marker including bare `nvidia-smi`.
if [ "${SKIP_NVDEC_PROBE:-0}" != "1" ] && [ -f "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
        log "FATAL: NVDEC/DALI probe failed; exact CUDA eval is not trustworthy on this host."
        exit 2
    }
fi

cd "$WORKSPACE"

LANE_ID="lane_pr106_latent_sidecar"
LOG_DIR="$WORKSPACE/experiments/results/${LANE_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
log() { echo "[lane-pr106-sidecar] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'sidecar_top_k': int('$SIDECAR_TOP_K'),
}
with open('$LOG_DIR/provenance.json', 'w') as f:
    json.dump(prov, f, indent=2)
print(f'[stage-0] provenance written; CUDA={torch.cuda.is_available()}; top_k={prov[\"sidecar_top_k\"]}')
"

# ── Stage 1: Build PR106 + sidecar archive ────────────────────────────────
log "=== Stage 1: build PR106 + latent-correction sidecar (top_k=$SIDECAR_TOP_K) ==="
BUILD_DIR="$LOG_DIR/build"
mkdir -p "$BUILD_DIR"
"$PYBIN" -u experiments/build_pr106_latent_sidecar.py \
    --source-archive "$PR106_ARCHIVE" \
    --output-dir "$BUILD_DIR" \
    --top-k "$SIDECAR_TOP_K" \
    --device cuda 2>&1 | tee -a "$LOG_DIR/run.log"
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
