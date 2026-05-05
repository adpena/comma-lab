#!/bin/bash
# NO_NVDEC_NEEDED — pure tensor-side codec + scorer-forward; no DALI/NVDEC video pipeline.
# Lane #04 generic intN — PR106 HNeRV decoder repacked via signed intN block-FP
#
# Operator picks bits via env var (4..8). Magic byte encoding:
#   APOGEE_INTN_BITS=4 → magic 0xA4, HIGH risk, predicted [0.155, 0.180]
#   APOGEE_INTN_BITS=5 → magic 0xA5, MEDIUM risk, predicted [0.180, 0.196] ← sweet spot
#   APOGEE_INTN_BITS=6 → magic 0xA6, LOW risk, predicted [0.190, 0.204]
#   APOGEE_INTN_BITS=7 → magic 0xA7, VERY LOW, predicted [0.198, 0.208]
#   APOGEE_INTN_BITS=8 → magic 0xA8, almost lossless, predicted [0.196, 0.207]
#
# Pipeline (3 stages, single Vast.ai 4090 ~$0.30/hr × 30min ≈ $0.30 OR
# Lightning T4 final auth eval ~$0.22/hr × 30min ≈ $0.11):
#
#   Stage 1 (CPU): Repack PR106 → apogee_intN archive (uses
#                  experiments/repack_pr106_with_intN_block_fp.py --bits N)
#   Stage 2 (CPU): Local parser-roundtrip verification (cheap sanity check)
#   Stage 3 (CUDA-T4): contest_auth_eval — score must be < 0.20945 (PR106 baseline) to ship
#
# Stub-mode previews already verified locally:
#   bits=5: 154,555-byte archive (-31,684 vs PR106 / -17.0%) → rate Δ -0.021097
#   bits=6: 170,450-byte archive (-15,789 vs PR106 / -8.5%)  → rate Δ -0.010513
#
# Strict-scorer-rule: scorer is loaded ONLY at Stage 3 (contest auth eval).
# No scorer at archive-build time. No scorer at inflate time per CLAUDE.md
# feedback_strict_scorer_rule.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
APOGEE_INTN_BITS="${APOGEE_INTN_BITS:-5}"
PR106_ARCHIVE="${PR106_ARCHIVE:-experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip}"
PR106_STATE_DICT="${PR106_STATE_DICT:-experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt}"

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

if ! [[ "$APOGEE_INTN_BITS" =~ ^[4-8]$ ]]; then
    echo "FATAL: APOGEE_INTN_BITS must be 4..8 (got: $APOGEE_INTN_BITS)" >&2
    exit 2
fi

LANE_ID="lane_apogee_int${APOGEE_INTN_BITS}_pr106"
LOG_DIR="$WORKSPACE/experiments/results/${LANE_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
log() { echo "[lane-apogee-int${APOGEE_INTN_BITS}] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Heartbeat (per CLAUDE.md remote-code-parity rule)
HEARTBEAT="$LOG_DIR/heartbeat.log"
( while true; do echo "$(date -u +%FT%TZ) lane-apogee-int${APOGEE_INTN_BITS} alive" >> "$HEARTBEAT"; sleep 60; done ) &
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
    'predicted_band': [0.155, 0.207],
    'apogee_intn_bits': int('$APOGEE_INTN_BITS'),
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'pr106_archive': '$PR106_ARCHIVE',
    'pr106_state_dict': '$PR106_STATE_DICT',
    'magic_byte_hex': hex(0xA0 | int('$APOGEE_INTN_BITS')),
}
with open('$LOG_DIR/provenance.json', 'w') as f:
    json.dump(prov, f, indent=2)
print(f'[stage-0] provenance written; CUDA={torch.cuda.is_available()}; bits={prov[\"apogee_intn_bits\"]}; magic={prov[\"magic_byte_hex\"]}')
"

# ── Stage 1: Repack PR106 → apogee_intN ───────────────────────────────────
log "=== Stage 1: repack PR106 with intN block-FP (bits=$APOGEE_INTN_BITS) ==="
REPACK_DIR="$LOG_DIR/repack"
mkdir -p "$REPACK_DIR"
"$PYBIN" -u experiments/repack_pr106_with_intN_block_fp.py \
    --state-dict "$PR106_STATE_DICT" \
    --pr106-archive "$PR106_ARCHIVE" \
    --bits "$APOGEE_INTN_BITS" \
    --out-dir "$REPACK_DIR" 2>&1 | tee -a "$LOG_DIR/run.log"
APOGEE_ARCHIVE="$REPACK_DIR/apogee_int${APOGEE_INTN_BITS}_archive.zip"
if [ ! -f "$APOGEE_ARCHIVE" ]; then
    log "FATAL: stage 1 did not produce $APOGEE_ARCHIVE"
    exit 3
fi
ARCHIVE_BYTES=$(stat -c '%s' "$APOGEE_ARCHIVE" 2>/dev/null || stat -f '%z' "$APOGEE_ARCHIVE")
log "stage 1 OK: archive bytes=$ARCHIVE_BYTES"

# ── Stage 2: Local parser-roundtrip sanity ────────────────────────────────
log "=== Stage 2: parser-roundtrip sanity (no GPU forward) ==="
"$PYBIN" -c "
import sys, zipfile
sys.path.insert(0, '$WORKSPACE/submissions/apogee_intN')
from inflate import parse_apogee_intn_archive
with zipfile.ZipFile('$APOGEE_ARCHIVE') as z:
    bin_bytes = z.read('0.bin')
sd, lat, meta = parse_apogee_intn_archive(bin_bytes)
print(f'parse OK: {len(sd)} tensors, bits={meta[\"bits\"]}, latents shape={tuple(lat.shape)}')
assert meta['bits'] == int('$APOGEE_INTN_BITS'), 'bits mismatch'
" 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Stage 3: Contest auth eval (CUDA T4) ─────────────────────────────────
log "=== Stage 3: contest auth eval (CUDA) ==="
INFLATE_SH="$WORKSPACE/submissions/apogee_intN/inflate.sh"
EVAL_DIR="$LOG_DIR/eval"
mkdir -p "$EVAL_DIR"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$APOGEE_ARCHIVE" \
    --inflate-sh "$INFLATE_SH" \
    --work-dir "$EVAL_DIR" \
    --keep-work-dir \
    --device cuda 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Final summary ─────────────────────────────────────────────────────────
SCORE_JSON="$EVAL_DIR/contest_auth_eval.json"
if [ -f "$SCORE_JSON" ]; then
    SCORE=$("$PYBIN" -c "import json; print(json.load(open('$SCORE_JSON'))['final_score'])" 2>/dev/null || echo "PARSE_FAIL")
    log "DONE: lane=$LANE_ID bits=$APOGEE_INTN_BITS archive_bytes=$ARCHIVE_BYTES contest_cuda_score=$SCORE [contest-CUDA]"
    log "  beats PR106 baseline 0.20946? $("$PYBIN" -c "
s = $SCORE
print('YES — new public-frontier candidate' if isinstance(s, (int, float)) and s < 0.20946 else f'no (score {s} >= 0.20946)')
" 2>/dev/null || echo "?")"
else
    log "FATAL: stage 3 did not produce $SCORE_JSON"
    exit 4
fi
