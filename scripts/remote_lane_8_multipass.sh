#!/bin/bash
# Lane 8 — Multi-pass compress with score-feedback on Lane G v3 anchor.
#
# Council 2026-04-30 design memo:
# .omx/research/council_lane_8_multipass_design_20260430.md
#
# This lane wraps step_archive + step_eval inside a MultiPassCompressor
# outer loop that iterates encoder parameters (mask_crf, pose_q_bits,
# block_fp_block_size, residual_gain) based on per-pass score deltas.
# Compress-time only — strict-scorer-rule per CLAUDE.md (no scorers loaded
# at inflate time; the inflate path is unchanged from Lane G v3).
#
# Cost: 3-pass on Lane G v3 anchor (88K renderer, 600 frames):
#       ~10min/pass × 3 = ~30min on Vast.ai 4090 = ~$0.13. UNDER $10 cap.
#
# Predicted band [empirical:reports/lane_8_multipass_real_archive.json]:
#   `[+5, +15] bp` improvement over Lane G v3 1.05 baseline.
#   Expected landing: 1.035-1.045 [contest-CUDA].
#
# Anchor: experiments/results/lane_g_v3_landed/ (Lane G v3 = 1.05 [contest-CUDA]).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# Strict-scorer-rule compliance: this script invokes auth_eval_renderer at
# COMPRESS time inside MultiPassCompressor's scorer callback. The inflate
# side remains unchanged. Preflight Check 92 enforces the symbol firewall.
#
# E2E_SMOKE_OPT_OUT: backed by 23 unit tests in
# src/tac/tests/test_multipass_compressor.py + 7 preflight tests in
# src/tac/tests/test_check_no_inflate_time_multipass.py + an offline real-
# archive smoke (experiments/lane_8_multipass_real_archive_smoke.py).
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_8_multipass_g_v3"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-8-multipass] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_8_multipass.sh',
    'output_dir': '$LOG_DIR',
    'paradigm': 'multipass_compress_with_score_feedback',
    'codec_module': 'tac.multipass_compressor',
    'wraps_module': 'experiments.pipeline.step_multipass',
    'multipass_max_passes': 3,
    'multipass_eps': 1e-3,
    'multipass_target_score': 1.045,
    'predicted_band': [1.035, 1.045],
    'predicted_score_central': 1.040,
    'baseline_lane': 'lane_g_v3_landed',
    'baseline_score_contest_cuda': 1.05,
    'controlled_baseline': 'lane_g_v3_landed (compress-time multipass; same renderer/masks/poses)',
    'no_training': True,
    'strict_scorer_rule_compliant': True,
    'design_memo': '.omx/research/council_lane_8_multipass_design_20260430.md',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=LANE-8-MULTIPASS gpu=$GPU" >> "$HEARTBEAT"
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

log "=== Stage 1: anchor checks ==="
for f in "$ANCHOR_RENDERER" "$ANCHOR_MASKS" "$ANCHOR_POSES" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         submissions/robust_current/inflate_renderer.py \
         submissions/robust_current/inflate.sh \
         src/tac/multipass_compressor.py \
         experiments/pipeline.py \
         experiments/auth_eval_renderer.py; do
    [ -f "$f" ] || { echo "FATAL: missing anchor $f" >&2; exit 1; }
done
log "  all anchors verified"

# Strict-scorer-rule preflight: ensure inflate-side files do NOT reference
# the multipass codec (Check 92).
"$PYBIN" -c "
from tac.preflight import check_no_inflate_time_multipass
v = check_no_inflate_time_multipass(strict=True, verbose=True)
print(f'check 92: {len(v)} violations')
" || { echo "FATAL: Check 92 (no-inflate-time-multipass) failed." >&2; exit 1; }
log "  Check 92 (no-inflate-time-multipass) clean"

log "=== Stage 2: dispatch experiments/pipeline.py compress --multipass ==="
PIPELINE_OUT="$LOG_DIR/pipeline_out"
mkdir -p "$PIPELINE_OUT"

# Use the canonical pipeline standard with --profile multipass_lane_g_v3.
# The profile sets multipass=True + max_passes=3 + eps=1e-3. The video, masks,
# checkpoint, and poses come from the Lane G v3 anchor.
"$PYBIN" -u experiments/pipeline.py compress \
    --profile multipass_lane_g_v3 \
    --video upstream/videos/0.mkv \
    --checkpoint "$ANCHOR_RENDERER" \
    --masks "$ANCHOR_MASKS" \
    --output-dir "$PIPELINE_OUT" \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --upstream upstream \
    --max-iterations 1 \
    --multipass \
    --multipass-max-passes 3 \
    --multipass-target-score 1.045 \
    --multipass-eps 1e-3 \
    2>&1 | tee "$LOG_DIR/pipeline.log" | tail -50
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: pipeline compress rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

# The MultiPass loop wrote per-pass JSONL history + summary alongside the
# final archive at iter_0/.
ITER_DIR="$PIPELINE_OUT/iter_0"
FINAL_ARCHIVE="$ITER_DIR/archive.zip"
HISTORY_LOG="$ITER_DIR/multipass_history.jsonl"
SUMMARY_JSON="$ITER_DIR/multipass_summary.json"

[ -f "$FINAL_ARCHIVE" ] || { echo "FATAL: final archive not written" >&2; exit 2; }
[ -f "$HISTORY_LOG" ]  || { echo "FATAL: pass history not written" >&2; exit 2; }
[ -f "$SUMMARY_JSON" ] || { echo "FATAL: pass summary not written" >&2; exit 2; }

ARCHIVE_BYTES=$(stat -c '%s' "$FINAL_ARCHIVE" 2>/dev/null || stat -f '%z' "$FINAL_ARCHIVE")
log "  final archive: $FINAL_ARCHIVE = $ARCHIVE_BYTES bytes"

log "=== Stage 3: extract score from multipass summary ==="
SCORE=$("$PYBIN" -c "
import json
data = json.load(open('$SUMMARY_JSON'))
print(data['final_score'])
")
log "Lane 8 multipass final score = $SCORE [contest-CUDA]"

log "=== Stage 4.5: defense-in-depth canonical contest_auth_eval on FINAL archive ==="
# Round 1 Yousfi finding: the multipass-internal score IS contest-CUDA, but
# we want a SEPARATE canonical contest_auth_eval invocation on the final
# archive bytes to match other lane scripts and provide a redundant
# attestation that the score we report came from the EXACT bytes we ship.
EVAL_LOG="$LOG_DIR/auth_eval.log"
EVAL_WORK_DIR="$LOG_DIR/eval_work"
rm -rf "$EVAL_WORK_DIR"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$FINAL_ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$EVAL_WORK_DIR" 2>&1 | tee "$EVAL_LOG" | tail -30
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: redundant contest_auth_eval rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

CANONICAL_SCORE=$("$PYBIN" -c "
import re
text = open('$EVAL_LOG').read()
m = re.search(r'\"score\"\s*:\s*([0-9.]+)', text)
if not m:
    m = re.search(r'final[_ ]?score[\s:=]+([0-9.]+)', text, re.IGNORECASE)
print(m.group(1) if m else 'NA')
")
log "  canonical contest_auth_eval score = $CANONICAL_SCORE [contest-CUDA]"

# Sanity: canonical score should match multipass-internal score within noise.
"$PYBIN" -c "
mp = float('$SCORE')
canon_str = '$CANONICAL_SCORE'
if canon_str == 'NA':
    raise SystemExit('canonical score parse failed')
canon = float(canon_str)
diff = abs(mp - canon)
if diff > 0.01:
    raise SystemExit(f'multipass score $SCORE vs canonical $CANONICAL_SCORE differ by {diff:.4f} > 0.01 — investigate')
print(f'multipass-vs-canonical agreement: |{mp:.4f} - {canon:.4f}| = {diff:.4f}')
"

log "=== Stage 5: tag result + write final provenance ==="
"$PYBIN" -c "
import json, time
prov = json.load(open('$PROVENANCE'))
summary = json.load(open('$SUMMARY_JSON'))
prov['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
prov['final_score'] = summary['final_score']
prov['final_archive_bytes'] = int('$ARCHIVE_BYTES')
prov['n_passes'] = summary['n_passes']
prov['converged'] = summary['converged']
prov['target_hit'] = summary['target_hit']
prov['reverted'] = summary['reverted']
prov['multipass_summary'] = summary
prov['result_tag'] = '[contest-CUDA]'
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"

log "Lane 8 multipass complete. Score: $SCORE [contest-CUDA]. Archive: $ARCHIVE_BYTES bytes."
log "Provenance: $PROVENANCE"
log "Pass history: $HISTORY_LOG"
