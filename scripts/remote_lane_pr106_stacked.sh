#!/bin/bash
# NO_NVDEC_NEEDED — pure tensor-side codec + scorer-forward; no DALI/NVDEC video pipeline.
# Lane #pr106_stacked — META-COMPOSITION of all 3 score-aware sidechannels
# (latent + yshift + lrl1) layered into a single archive.
#
# DISPATCH GATE: only run AFTER ALL 3 sister sidechannel lanes
# (lane_pr106_latent_sidecar + lane_pr106_yshift_sidechannel +
# lane_pr106_lrl1_sidechannel) AND apogee_intN have landed empirical
# contest-CUDA scores beating PR106 0.20945. Per
# tools/sidechannel_stack_predictor.py --bits 5 --all, the int4+full-stack
# predicted score is 0.163 (-0.046 vs PR106). The stacked composition is
# the single-dispatch payoff of all the sister lanes.
#
# Operator picks which sister archives to compose via env vars (each path is
# OPTIONAL — composition with any subset is supported):
#   STACKED_LATENT_ARCHIVE   path to pr106_latent_sidecar archive.zip
#   STACKED_YSHIFT_ARCHIVE   path to pr106_yshift_sidechannel archive.zip
#   STACKED_LRL1_ARCHIVE     path to pr106_lrl1_sidechannel archive.zip
#
# Operator builds each sister via its own remote_lane_*.sh runbook FIRST,
# then runs this composition to test the joint payoff in a SINGLE eval.
#
# Pipeline (3 stages, single Vast.ai 4090 ~$0.30/hr × 30min ≈ $0.30):
#
#   Stage 1 (CPU): Compose pr106_stacked archive from sister inputs
#                  (uses experiments/build_pr106_stacked.py — pure bytes work,
#                   no GPU forward pass needed at compose time)
#   Stage 2 (CPU): Local parser-roundtrip verification (catches wire-format drift)
#   Stage 3 (CUDA-T4 or 4090): contest_auth_eval — score must beat the
#                              best individual sister to ship as a stack-on
#                              improvement, OR beat predicted band 0.16-0.18
#                              for the full 4-element (apogee + 3 sidechannel) stack
#
# Predicted (per docs/INDEX_score_aware_sidechannel_thread_20260504.md +
# tools/sidechannel_stack_predictor.py):
#   3-sidechannel-only stack on PR106: ~0.18-0.20 score (paradigm-additive)
#   3-sidechannel + apogee_int4 stack: ~0.163 score [PRE-DISPATCH PREDICTION]
#
# Strict-scorer-rule: scorer is loaded ONLY at sister-build time + Stage 3
# (contest auth eval). Composition is bytes-only; inflate is CUDA but no scorer.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
PR106_ARCHIVE="${PR106_ARCHIVE:-experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip}"
STACKED_LATENT_ARCHIVE="${STACKED_LATENT_ARCHIVE:-}"
STACKED_YSHIFT_ARCHIVE="${STACKED_YSHIFT_ARCHIVE:-}"
STACKED_LRL1_ARCHIVE="${STACKED_LRL1_ARCHIVE:-}"

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

LANE_ID="lane_pr106_stacked"
LOG_DIR="$WORKSPACE/experiments/results/${LANE_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
log() { echo "[lane-pr106-stacked] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Heartbeat (per CLAUDE.md remote-code-parity rule)
HEARTBEAT="$LOG_DIR/heartbeat.log"
( while true; do echo "$(date -u +%FT%TZ) lane-pr106-stacked alive" >> "$HEARTBEAT"; sleep 60; done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

# ── Stage 0: Provenance + CUDA preflight + sister-archive sanity ─────────
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
"$PYBIN" -c "
import json, time, sys, os, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: --device cuda required for Stage 3 per CLAUDE.md MPS-auth-eval-is-NOISE')
sister_paths = {
    'latent': '$STACKED_LATENT_ARCHIVE' or None,
    'yshift': '$STACKED_YSHIFT_ARCHIVE' or None,
    'lrl1':   '$STACKED_LRL1_ARCHIVE'   or None,
}
present = {k: v for k, v in sister_paths.items() if v}
if not present:
    sys.exit('FATAL: at least one of STACKED_{LATENT,YSHIFT,LRL1}_ARCHIVE must be set; '
             'this lane composes pre-built sister archives, NOT trains them. '
             'Build sisters via their own remote_lane_*.sh runbooks first.')
for name, p in present.items():
    if not os.path.isfile(p):
        sys.exit(f'FATAL: sister archive {name} not found at {p}')
prov = {
    'lane_id': '$LANE_ID',
    'predicted_band': [0.15, 0.18],
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'pr106_archive': '$PR106_ARCHIVE',
    'sister_archives': present,
    'n_sidechannels': len(present),
}
with open('$LOG_DIR/provenance.json', 'w') as f:
    json.dump(prov, f, indent=2)
print(f'[stage-0] provenance written; CUDA={torch.cuda.is_available()}; '
      f'sidechannels=[{\",\".join(present.keys())}] (n={len(present)})')
"

# ── Stage 1: Compose pr106_stacked archive ────────────────────────────────
log "=== Stage 1: compose pr106_stacked from sister archives ==="
BUILD_DIR="$LOG_DIR/build"
mkdir -p "$BUILD_DIR"
COMPOSE_OPTIONAL=()
[ -n "$STACKED_LATENT_ARCHIVE" ] && COMPOSE_OPTIONAL+=(--latent "$STACKED_LATENT_ARCHIVE")
[ -n "$STACKED_YSHIFT_ARCHIVE" ] && COMPOSE_OPTIONAL+=(--yshift "$STACKED_YSHIFT_ARCHIVE")
[ -n "$STACKED_LRL1_ARCHIVE" ]   && COMPOSE_OPTIONAL+=(--lrl1   "$STACKED_LRL1_ARCHIVE")
"$PYBIN" -u experiments/build_pr106_stacked.py \
    --pr106-archive "$PR106_ARCHIVE" \
    --output-dir "$BUILD_DIR" \
    "${COMPOSE_OPTIONAL[@]}" \
    2>&1 | tee -a "$LOG_DIR/run.log"
STACKED_ARCHIVE="$BUILD_DIR/pr106_stacked_archive.zip"
if [ ! -f "$STACKED_ARCHIVE" ]; then
    log "FATAL: stage 1 did not produce $STACKED_ARCHIVE"
    exit 3
fi
ARCHIVE_BYTES=$(stat -c '%s' "$STACKED_ARCHIVE" 2>/dev/null || stat -f '%z' "$STACKED_ARCHIVE")
log "stage 1 OK: archive bytes=$ARCHIVE_BYTES"

# ── Stage 2: Local parser-roundtrip sanity ────────────────────────────────
log "=== Stage 2: parser-roundtrip sanity (no GPU forward) ==="
"$PYBIN" -c "
import sys, zipfile
sys.path.insert(0, '$WORKSPACE/submissions/pr106_stacked')
sys.path.insert(0, '$WORKSPACE/submissions/pr106_latent_sidecar/src')
from inflate import parse_stacked_archive, SECTION_LATENT, SECTION_YSHIFT, SECTION_LRL1
with zipfile.ZipFile('$STACKED_ARCHIVE') as z:
    bin_bytes = z.read('0.bin')
sd, lat, meta, sections = parse_stacked_archive(bin_bytes)
sec_names = []
if SECTION_LATENT in sections: sec_names.append('latent')
if SECTION_YSHIFT in sections: sec_names.append('yshift')
if SECTION_LRL1 in sections:   sec_names.append('lrl1')
print(f'parse OK: {len(sd)} tensors, latents shape={tuple(lat.shape)}, '
      f'sections=[{\",\".join(sec_names)}] (n={len(sections)})')
assert len(sd) == 28, f'expected 28 PR106 tensors, got {len(sd)}'
assert tuple(lat.shape) == (600, 28), f'expected (600, 28) latents, got {tuple(lat.shape)}'
assert len(sections) >= 1, 'no sidechannel sections present (composition is empty)'
" 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Stage 3: Contest auth eval (CUDA T4 or 4090) ──────────────────────────
log "=== Stage 3: contest auth eval (CUDA) ==="
INFLATE_SH="$WORKSPACE/submissions/pr106_stacked/inflate.sh"
EVAL_DIR="$LOG_DIR/eval"
mkdir -p "$EVAL_DIR"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$STACKED_ARCHIVE" \
    --inflate-sh "$INFLATE_SH" \
    --work-dir "$EVAL_DIR" \
    --keep-work-dir \
    --device cuda 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Final summary ─────────────────────────────────────────────────────────
SCORE_JSON="$EVAL_DIR/contest_auth_eval.json"
if [ -f "$SCORE_JSON" ]; then
    SCORE=$("$PYBIN" -c "import json; print(json.load(open('$SCORE_JSON'))['final_score'])" 2>/dev/null || echo "PARSE_FAIL")
    log "DONE: lane=$LANE_ID archive_bytes=$ARCHIVE_BYTES contest_cuda_score=$SCORE [contest-CUDA]"
    log "  beats best individual sister landed score? Operator must compare via tools/score_dashboard.py."
    log "  predicted band: ~0.18 (3-sidechannel only) / ~0.16 (full apogee+3-sidechannel stack)"
    log "  cross-ref docs/INDEX_score_aware_sidechannel_thread_20260504.md + "
    log "             tools/sidechannel_stack_predictor.py --bits 5 --all"
else
    log "FATAL: stage 3 did not produce $SCORE_JSON"
    exit 4
fi
