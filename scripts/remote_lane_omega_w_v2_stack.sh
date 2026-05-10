#!/bin/bash
# Lane Ω-W-V2 Stack — Lane G v3 + OWV2 renderer.bin re-encoding
#
# Highest-EV alternative dispatch from .omx/research/council_chain_integrity_audit_20260430.md
# Part G1. Re-encodes Lane G v3's ASYM renderer.bin via the OWV2 (water-fill +
# arithmetic terminal) codec, swaps it into the otherwise-identical Lane G v3
# archive (masks.mkv + optimized_poses.pt unchanged), and runs contest-CUDA
# auth eval against the rebuilt archive.
#
# This is NOT a training lane. It is an export-time codec swap. No GPU is
# needed for the build (Stage 2); GPU only fires at Stage 3 for the auth eval.
# Total expected wallclock: ~10 min on a Vast.ai 4090 (~$0.04 at $0.25/hr +
# instance overhead).
#
# Predicted band [0.95, 1.02] [contest-CUDA] [derivation].
#   Local empirical build:
#     Lane G v3 archive: 694,074B
#     Lane G v3 + Ω-W-V2 archive: 643,089B (-50,985B, -7.34%)
#     Δ rate term: 25 × -50,985 / 37,545,489 = -0.0339
#     Predicted contest-CUDA score: 1.05 + (-0.0339) = ~1.016
#     Variance from FP16 fallback round-trip + OWV2 per-channel block-FP
#     algebra (max_abs / 4 L_inf) bumps the band ±0.02.
#
# Anchor: experiments/results/lane_g_v3_landed/ (Lane G v3 = 1.05 [contest-CUDA]).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# Strict-scorer-rule compliance: this script never loads PoseNet or SegNet
# weights at any stage; the OWV2 codec is pure-CPU byte-to-tensor arithmetic.
#
# E2E_SMOKE_OPT_OUT: backed by 11 round-trip tests in
# src/tac/tests/test_owv2_renderer_archive_inflate.py + 9 single-tensor tests
# in src/tac/tests/test_omega_w_v2_real_archive.py — full local end-to-end
# build + inflate-side dispatch verified before deployment (commit eff3020c).
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_g_v3_omega_w_v2_stack"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-owv2-stack] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
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
    'lane_script': 'scripts/remote_lane_omega_w_v2_stack.sh',
    'output_dir': '$LOG_DIR',
    'paradigm': 'export_time_codec_swap_owv2_water_fill_arithmetic',
    'codec_module': 'tac.owv2_renderer_archive',
    'codec_inner': 'tac.water_filling_codec_v2',
    'inflate_dispatch_magic': 'OWV2',
    'inflate_dispatch_module': 'submissions/robust_current/inflate_renderer.py',
    'bit_budget_ratio': 0.7,
    'predicted_band': [0.95, 1.02],
    'predicted_score_central': 1.016,
    'baseline_lane': 'lane_g_v3_landed',
    'baseline_score_contest_cuda': 1.05,
    'expected_archive_bytes': 643089,
    'expected_savings_bytes': 50985,
    'controlled_baseline': 'lane_g_v3_landed (only renderer.bin codec swapped: ASYM -> OWV2)',
    'no_training': True,
    'strict_scorer_rule_compliant': True,
    'audit_doc': '.omx/research/council_chain_integrity_audit_20260430.md (Part G1)',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=OWV2-STACK gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- destroy this Vast.ai instance."
    exit 2
}

ANCHOR_DIR="experiments/results/lane_g_v3_landed"
ANCHOR_RENDERER="$ANCHOR_DIR/iter_0/renderer.bin"
ANCHOR_MASKS="$ANCHOR_DIR/iter_0/masks.mkv"
ANCHOR_POSES="$ANCHOR_DIR/iter_0/optimized_poses.pt"
ANCHOR_ARCHIVE="$ANCHOR_DIR/archive_lane_g_v3.zip"

log "=== Stage 1: anchor checks (Lane G v3 + OWV2 inflate handler verification) ==="
for f in "$ANCHOR_RENDERER" "$ANCHOR_MASKS" "$ANCHOR_POSES" "$ANCHOR_ARCHIVE" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         submissions/robust_current/inflate_renderer.py \
         submissions/robust_current/inflate.sh \
         src/tac/owv2_renderer_archive.py \
         src/tac/water_filling_codec_v2.py \
         experiments/build_lane_g_v3_omega_w_v2_stack.py \
         experiments/contest_auth_eval.py \
         scripts/adjudicate_contest_auth_eval.py; do
    [ -f "$f" ] || { echo "FATAL: missing anchor $f" >&2; exit 1; }
done

# Verify the OWV2 dispatch case is in the deployed inflate_renderer.py.
# This is a 'remote_code_parity'-style guard: if the tarball was built from
# a stale tree, fail loud at Stage 1 instead of producing a corrupt archive.
if ! grep -q 'magic == b"OWV2"' "$WORKSPACE/submissions/robust_current/inflate_renderer.py"; then
    echo "FATAL: deployed inflate_renderer.py does NOT have the OWV2 magic dispatch case." >&2
    echo "       The tarball was built before commit 232b24ec; rebuild + redeploy." >&2
    exit 1
fi
log "  OWV2 dispatch verified in deployed inflate_renderer.py"

# Verify the ASYM renderer's magic byte (Lane G v3 ships ASYM).
ANCHOR_MAGIC=$("$PYBIN" -c "print(open('$ANCHOR_RENDERER','rb').read(4).decode('utf-8','replace'))")
if [ "$ANCHOR_MAGIC" != "ASYM" ]; then
    echo "FATAL: $ANCHOR_RENDERER magic is '$ANCHOR_MAGIC', expected 'ASYM'." >&2
    exit 1
fi
log "  Lane G v3 renderer.bin magic = ASYM (expected)"

log "=== Stage 2: build Lane G v3 + Ω-W-V2 stacked archive (CPU-only, ~1s) ==="
STACKED_ARCHIVE="$LOG_DIR/archive_lane_g_v3_omega_w_v2.zip"
BUILD_PROVENANCE="$LOG_DIR/build_provenance.json"
set +e
"$PYBIN" -u experiments/build_lane_g_v3_omega_w_v2_stack.py \
    --output "$STACKED_ARCHIVE" \
    --bit-budget-ratio 0.7 \
    --provenance-json "$BUILD_PROVENANCE" 2>&1 | tee "$LOG_DIR/build.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: build_lane_g_v3_omega_w_v2_stack.py rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

[ -f "$STACKED_ARCHIVE" ] || { echo "FATAL: stacked archive not written" >&2; exit 2; }
ARCHIVE_BYTES=$(stat -c '%s' "$STACKED_ARCHIVE" 2>/dev/null || stat -f '%z' "$STACKED_ARCHIVE")
ARCHIVE_SHA=$("$PYBIN" -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())" "$STACKED_ARCHIVE")
log "  stacked archive: $STACKED_ARCHIVE = $ARCHIVE_BYTES bytes"
log "  stacked archive sha256: $ARCHIVE_SHA"

# Sanity: archive must be < Lane G v3 baseline (the entire premise of this lane).
LANE_G_V3_BYTES=$(stat -c '%s' "$ANCHOR_ARCHIVE" 2>/dev/null || stat -f '%z' "$ANCHOR_ARCHIVE")
if [ "$ARCHIVE_BYTES" -ge "$LANE_G_V3_BYTES" ]; then
    echo "FATAL: stacked archive ($ARCHIVE_BYTES B) >= Lane G v3 baseline ($LANE_G_V3_BYTES B)." >&2
    echo "       OWV2 codec produced a regression; investigate before continuing." >&2
    exit 3
fi
log "  archive delta: $((ARCHIVE_BYTES - LANE_G_V3_BYTES)) bytes vs Lane G v3"

log "=== Stage 3: contest_auth_eval [contest-CUDA] on Lane G v3 + Ω-W-V2 archive ==="
EVAL_LOG="$LOG_DIR/auth_eval.log"
EVAL_WORK_DIR="$LOG_DIR/eval_work"
rm -rf "$EVAL_WORK_DIR"
set +e
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$STACKED_ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$EVAL_WORK_DIR" 2>&1 | tee "$EVAL_LOG" | tail -30
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

log "=== Stage 4: JSON adjudication + final provenance ==="
RESULT_JSON="$LOG_DIR/RESULT_JSON"
ADJUDICATION_LOG="$LOG_DIR/adjudication.log"
"$PYBIN" -u scripts/adjudicate_contest_auth_eval.py \
    --contest-json "$EVAL_WORK_DIR/contest_auth_eval.json" \
    --provenance "$PROVENANCE" \
    --archive "$STACKED_ARCHIVE" \
    --result-copy "$RESULT_JSON" \
    --baseline-score 1.05 \
    --baseline-archive-bytes "$LANE_G_V3_BYTES" \
    --predicted-band 0.95 1.02 \
    --regression-threshold 1.05 \
    --delta-key score_delta_vs_lane_g_v3 | tee "$ADJUDICATION_LOG"
SCORE=$(grep '^SCORE_RECOMPUTED=' "$ADJUDICATION_LOG" | tail -1 | cut -d= -f2)
LANE_STATUS=$(grep '^LANE_STATUS=' "$ADJUDICATION_LOG" | tail -1 | cut -d= -f2)
log "Lane G v3 + Ω-W-V2 stack score_recomputed = $SCORE [contest-CUDA] status=$LANE_STATUS"
log "=== LANE_OWV2_STACK_DONE — score_recomputed=$SCORE [contest-CUDA] archive_bytes=$ARCHIVE_BYTES ==="
