#!/bin/bash
# Lane PFP16 Stack — Lane G v3 + pose fp16 cast
#
# Lane GP v4 scoped smooth-basis retirement review
# (.omx/research/council_lane_gp_v4_design_20260430.md) surfaced Hotz's
# dominant-strategy successor: re-encode Lane G v3's
# `optimized_poses.pt` (fp32 pickle, 15,620 B) as a raw fp16 binary
# (`optimized_poses.bin`, 7,200 B). The inflate path already supports the
# raw fp16 binary via Branch B of `tac.submission_archive.load_optimized_poses`
# (content-detect by absence of pickle magic), so this is a build-time
# concept with NO inflate-side wire-format changes.
#
# This is NOT a training lane. It is an export-time pose cast. No GPU is
# needed for the build (Stage 2); GPU only fires at Stage 3 for the auth
# eval. Total expected wallclock: ~10 min on a Vast.ai 4090 (~$0.04 at
# $0.25/hr + instance overhead). [predicted_band: 1.04 - 1.05]
#
# Predicted band [1.04, 1.05] [contest-CUDA] [derivation].
#   Local empirical build (verified 2026-04-30):
#     Lane G v3 archive:           694,074 B
#     Lane G v3 + PFP16 archive:   686,635 B (-7,439 B, -1.07%)
#     Δ rate term: 25 × -7,439 / 37,545,489 = -0.00495
#     Predicted contest-CUDA score: 1.05 + (-0.00495) = ~1.045
#     Variance from PoseNet's intrinsic fp16 forward pass: ±0.005 (well
#     below noise floor; PoseNet already runs in fp16 on contest CUDA).
#
# Anchor: experiments/results/lane_g_v3_landed/ (Lane G v3 = 1.05 [contest-CUDA]).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# Strict-scorer-rule compliance: this script never loads PoseNet or SegNet
# weights at any stage; the PFP16 codec is pure-CPU tensor cast.
#
# E2E_SMOKE_OPT_OUT: backed by 19 tests in
# src/tac/tests/test_pfp16_codec.py — full local end-to-end build verified
# before deployment.
set -euo pipefail
# LIGHTNING STUDIO VARIANT (L40S 48GB AWS). Vast.ai-specific stages adapted:
#   - WORKSPACE=/home/zeus/pact, PYBIN=venv python
#   - Stage 0 NVDEC probe → simple GPU presence check
#   - DALI installed lazily before Stage 3 contest_auth_eval
WORKSPACE="${WORKSPACE:-/home/zeus/pact}"
PYBIN="${PYBIN:-$WORKSPACE/.venv/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="$WORKSPACE/upstream"

LANE_ID="lane_g_v3_pfp16_stack"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-pfp16-stack] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/lightning_lane_pfp16_stack.sh',
    'output_dir': '$LOG_DIR',
    'paradigm': 'export_time_pose_cast_fp32_pickle_to_fp16_raw',
    'codec_module': 'tac.pfp16_codec',
    'codec_inner': 'torch.Tensor.half + torch.frombuffer',
    'inflate_dispatch_magic': 'NONE (raw fp16 buffer detected by absence of pickle magic in load_optimized_poses Branch B)',
    'inflate_dispatch_module': 'submissions/robust_current/inflate_renderer.py',
    'predicted_band': [1.04, 1.05],
    'predicted_score_central': 1.045,
    'baseline_lane': 'lane_g_v3_landed',
    'baseline_score_contest_cuda': 1.05,
    'expected_archive_bytes': 686635,
    'expected_savings_bytes': 7439,
    'controlled_baseline': 'lane_g_v3_landed (only optimized_poses.{pt|bin} swapped: fp32 pickle -> fp16 raw)',
    'no_training': True,
    'strict_scorer_rule_compliant': True,
    'audit_doc': '.omx/research/council_lane_gp_v4_design_20260430.md (Hotz successor option)',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=PFP16-STACK gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: GPU-presence check (Lightning L40S, no Vast.ai NVDEC roulette) ==="
"$PYBIN" -c "
import torch, sys
if not torch.cuda.is_available():
    print('FATAL: no CUDA on Lightning Studio', file=sys.stderr); sys.exit(2)
print(f'OK: GPU={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB cuda_version={torch.version.cuda}')
"

ANCHOR_DIR="experiments/results/lane_g_v3_landed"
ANCHOR_RENDERER="$ANCHOR_DIR/iter_0/renderer.bin"
ANCHOR_MASKS="$ANCHOR_DIR/iter_0/masks.mkv"
ANCHOR_POSES="$ANCHOR_DIR/iter_0/optimized_poses.pt"
ANCHOR_ARCHIVE="$ANCHOR_DIR/archive_lane_g_v3.zip"

log "=== Stage 1: anchor checks (Lane G v3 + PFP16 inflate compatibility) ==="
for f in "$ANCHOR_RENDERER" "$ANCHOR_MASKS" "$ANCHOR_POSES" "$ANCHOR_ARCHIVE" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         submissions/robust_current/inflate_renderer.py \
         submissions/robust_current/inflate.sh \
         src/tac/pfp16_codec.py \
         src/tac/submission_archive.py \
         experiments/build_lane_g_v3_pfp16_stack.py \
         experiments/contest_auth_eval.py \
         scripts/adjudicate_contest_auth_eval.py; do
    [ -f "$f" ] || { echo "FATAL: missing anchor $f" >&2; exit 1; }
done

# Verify the inflate path supports the raw-fp16 .bin route. This is a
# 'remote_code_parity'-style guard: if the tarball was built from a stale
# tree without the Branch B logic, fail loud at Stage 1 instead of producing
# an archive that the inflate path can't decode.
if ! grep -q 'optimized_bin_path' "$WORKSPACE/submissions/robust_current/inflate_renderer.py"; then
    echo "FATAL: deployed inflate_renderer.py does NOT have the optimized_poses.bin route." >&2
    echo "       The tarball was built before the canonical loader landed; rebuild + redeploy." >&2
    exit 1
fi
if ! grep -q 'load_optimized_poses' "$WORKSPACE/src/tac/submission_archive.py"; then
    echo "FATAL: deployed submission_archive.py missing load_optimized_poses." >&2
    exit 1
fi
log "  PFP16 inflate-side compatibility verified (load_optimized_poses Branch B + optimized_poses.bin route)"

# Verify the ASYM renderer's magic byte (Lane G v3 ships ASYM).
ANCHOR_MAGIC=$("$PYBIN" -c "print(open('$ANCHOR_RENDERER','rb').read(4).decode('utf-8','replace'))")
if [ "$ANCHOR_MAGIC" != "ASYM" ]; then
    echo "FATAL: $ANCHOR_RENDERER magic is '$ANCHOR_MAGIC', expected 'ASYM'." >&2
    exit 1
fi
log "  Lane G v3 renderer.bin magic = ASYM (expected)"

log "=== Stage 2: build Lane G v3 + PFP16 stacked archive (CPU-only, ~1s) ==="
STACKED_ARCHIVE="$LOG_DIR/archive_lane_g_v3_pfp16.zip"
BUILD_PROVENANCE="$LOG_DIR/build_provenance.json"
set +e
"$PYBIN" -u experiments/build_lane_g_v3_pfp16_stack.py \
    --output "$STACKED_ARCHIVE" \
    --provenance-json "$BUILD_PROVENANCE" 2>&1 | tee "$LOG_DIR/build.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: build_lane_g_v3_pfp16_stack.py rc=${PIPE_RC[0]}" >&2
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
    echo "       PFP16 cast produced a regression; investigate before continuing." >&2
    exit 3
fi
log "  archive delta: $((ARCHIVE_BYTES - LANE_G_V3_BYTES)) bytes vs Lane G v3"

# Sanity: archive size assertion (Check 47 contract).
# Expected band: 686,000 - 687,000 B based on local empirical build.
if [ "$ARCHIVE_BYTES" -lt 685000 ] || [ "$ARCHIVE_BYTES" -gt 690000 ]; then
    echo "FATAL: stacked archive ($ARCHIVE_BYTES B) outside expected band [685000, 690000]." >&2
    echo "       The empirical local build was 686,635 B; large drift = corrupt build." >&2
    exit 4
fi
EXPECTED_ARCHIVE_SHA="0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f"
if [ "$ARCHIVE_SHA" != "$EXPECTED_ARCHIVE_SHA" ]; then
    echo "FATAL: stacked archive sha256 drift: got $ARCHIVE_SHA expected $EXPECTED_ARCHIVE_SHA." >&2
    echo "       PFP16 is a deterministic export-only lane; hash drift means the exact artifact changed." >&2
    exit 5
fi

log "=== Stage 3: contest_auth_eval [contest-CUDA] on Lane G v3 + PFP16 archive ==="
# Lightning-specific: DALI not installed by default. Install lazily.
log "  Ensuring nvidia-dali-cuda130 is installed for upstream/evaluate.py..."
"$PYBIN" -c "
import importlib.util, subprocess, sys
if importlib.util.find_spec('nvidia.dali') is None:
    print('Installing nvidia-dali-cuda130...', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                    '--extra-index-url', 'https://pypi.nvidia.com',
                    'nvidia-dali-cuda130'], check=True)
import nvidia.dali as dali
print(f'DALI version: {dali.__version__}')
"
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
    --predicted-band 1.04 1.05 \
    --regression-threshold 1.05 \
    --delta-key score_delta_vs_lane_g_v3 | tee "$ADJUDICATION_LOG"
SCORE=$(grep '^SCORE_RECOMPUTED=' "$ADJUDICATION_LOG" | tail -1 | cut -d= -f2)
LANE_STATUS=$(grep '^LANE_STATUS=' "$ADJUDICATION_LOG" | tail -1 | cut -d= -f2)
log "Lane G v3 + PFP16 stack score_recomputed = $SCORE [contest-CUDA] status=$LANE_STATUS"
log "=== LANE_PFP16_STACK_DONE — score_recomputed=$SCORE [contest-CUDA] archive_bytes=$ARCHIVE_BYTES ==="
