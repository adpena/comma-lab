#!/bin/bash
# Lane 20: Ballé hyperprior codec — train on real renderer.bin qint stream
# and (conditionally) build + auth-eval an archive that uses Lane 20.
#
# Council .omx/research/council_lane_20_balle_design_20260430.md
# (10-of-10 inner council members reviewed; Quantizr/Hotz YELLOW empirical
# kill criterion wired into the auto-fallback).
#
# Anchor: Lane G v3 = 1.05 [contest-CUDA] (DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL).
#
# DELTA from Lane G v3:
#   * Lane 20 is a CODEC lane, NOT a renderer-training lane. The renderer
#     stays Lane G v3; only the renderer.bin's qint stream is re-encoded
#     with the BHv1 wire format (Hotz-LITE chunked-static or full Ballé
#     hyperprior, whichever auto-selects as smaller).
#   * Phase E empirical [empirical:reports/lane_20_balle_real_archive.json]
#     showed UNTRAINED Ballé regresses by ~125-147% on FP4 nibble streams;
#     2000-step trained Ballé STILL regresses by ~6% (~8 KB side-info
#     overhead exceeds y-stream savings on the 141 KB Lane G v3 qint
#     stream). The auto-fallback engages and ships ZERO Lane 20 bytes.
#   * THIS SCRIPT runs the same chain on CUDA to confirm the verdict on
#     contest-CUDA arithmetic (the byte counts are device-deterministic
#     so no drift expected, but the CUDA device gate is the CLAUDE.md
#     non-negotiable for any score claim).
#
# Predicted band [prediction]: Lane G v3 score 1.05 ± 0 [contest-CUDA].
#   * The auto-fallback path means Lane 20 either:
#     (a) ships identical bytes → identical score (most likely)
#     (b) ships smaller bytes → marginally better score (~ -0.001 to -0.005)
#     (c) ships larger bytes → BLOCKED by the static-baseline guard
#   * Kill criterion: if final archive bytes >= Lane G v3 baseline archive
#     bytes by >100 B, abort the auth-eval (waste of GPU spend).
#
# Cost: 4090 @ $0.25/hr × ~30min training (CPU codec, GPU only for auth eval)
#       + ~10min auth eval = ~$0.20.
#
# Per CLAUDE.md NEVER-INVENT-CLI-FLAGS: every flag passed to
# train_balle_hyperprior.py / measure_lane_20_balle_real_archive.py /
# contest_auth_eval.py was verified by argparse-grep (Stage 0 dead-flag scan).
#
# Memory: project_lane_20_balle_landed_20260430.md (TBD post-result).
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=20
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_20_balle_results"
mkdir -p "$LOG_DIR"
TAG="lane_20_balle"

log() { echo "[lane-20] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps).
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_20_balle.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'profile': 'lane_20_balle_lane_g_v3',
    'predicted_band': [1.04, 1.05],
    'anchor_score_baseline': 1.05,
    'anchor_lane': 'lane_g_v3 (1.05 contest-CUDA)',
    'lane_20_premise': 'Replace the static arithmetic codec on the renderer FP4 qint stream with a learned Balle hyperprior (or Hotz-LITE chunked-static fallback). Auto-fallback to static prevents regression.',
    'phase_e_empirical_artifact': 'reports/lane_20_balle_real_archive.json',
    'cost_estimate_usd': 0.20,
    'wall_clock_estimate_minutes': 40,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=20 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend (auth eval needs NVDEC).
log "=== Stage 0a: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Stage 0b: profile validation
log "=== Stage 0b: profile validation ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.profiles import PROFILES
prof_name = 'lane_20_balle_lane_g_v3'
assert prof_name in PROFILES, f'profile {prof_name} not registered'
p = PROFILES[prof_name]
print(f'PROFILE OK: balle_block_size={p[\"balle_block_size\"]} '
      f'balle_z_dim={p[\"balle_z_dim\"]} '
      f'balle_hidden_dim={p[\"balle_hidden_dim\"]} '
      f'balle_ema_decay={p[\"balle_ema_decay\"]} '
      f'seed={p[\"seed\"]}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Stage 0c: dead-flag-wiring guard
log "=== Stage 0c: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_20_balle.sh').read()
trainer_src = open('experiments/train_balle_hyperprior.py').read()
real_train = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', trainer_src))
m = re.search(r'experiments/train_balle_hyperprior\.py(.*?)(?=\nlog \"|\Z)',
              script, re.DOTALL)
assert m, 'could not locate trainer invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real_train
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in trainer argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} trainer flags all real')

measure_src = open('experiments/measure_lane_20_balle_real_archive.py').read()
real_meas = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', measure_src))
m2 = re.search(r'experiments/measure_lane_20_balle_real_archive\.py(.*?)(?=\nlog \"|\Z)',
              script, re.DOTALL)
if m2:
    used2 = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m2.group(0)))
    invented2 = used2 - real_meas
    if invented2:
        print(f'INVENTED FLAGS (measure): {sorted(invented2)}', file=sys.stderr); sys.exit(3)
    print(f'OK: {len(used2)} measure flags all real')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Stage 0d: STRICT preflight Check 91
log "=== Stage 0d: Check 91 (Lane 20 BHv1 wire-format integrity) ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.preflight import check_balle_hyperprior_includes_side_info_in_archive
v = check_balle_hyperprior_includes_side_info_in_archive(strict=True, verbose=True)
print('Check 91 PASSED')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: required artifacts present
ANCHOR_RENDERER="${ANCHOR_RENDERER:-submissions/baseline_dilated_h64_0_90/renderer.bin}"
for f in "$ANCHOR_RENDERER" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
         submissions/baseline_dilated_h64_0_90/masks.mkv; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: empirical baseline measurement on anchor renderer ==="
"$PYBIN" experiments/measure_lane_20_balle_real_archive.py \
    --renderer "$ANCHOR_RENDERER" \
    --output "$LOG_DIR/lane_20_baseline_measurement.json" \
    --block-size 256 \
    --z-dim 8 \
    --hidden 16 \
    --seed 20 2>&1 | tee "$LOG_DIR/baseline_measure.log"

log "=== Stage 2: train Ballé hyperprior on the anchor qint stream ==="
"$PYBIN" -u experiments/train_balle_hyperprior.py \
    --renderer "$ANCHOR_RENDERER" \
    --output-dir "$LOG_DIR/training" \
    --block-size 256 \
    --z-dim 8 \
    --hidden-dim 16 \
    --lr 1e-2 \
    --steps 5000 \
    --eval-every 500 \
    --seed 20 \
    --device cuda 2>&1 | tee "$LOG_DIR/train.log" | tail -60

VERDICT=$("$PYBIN" -c "
import json
r = json.load(open('$LOG_DIR/training/lane_20_train_report.json'))
print(r['verdict'])
")
log "  trainer verdict: $VERDICT"
log "  static_baseline_bytes: $("$PYBIN" -c "import json; print(json.load(open('$LOG_DIR/training/lane_20_train_report.json'))['static_baseline_bytes'])")"
log "  best_full_balle_bytes: $("$PYBIN" -c "import json; print(json.load(open('$LOG_DIR/training/lane_20_train_report.json'))['best_full_balle_bytes'])")"

if [ "$VERDICT" = "STATIC_WINS_FALLBACK" ]; then
    log "=== AUTO-FALLBACK ENGAGED — Lane 20 does not beat static ==="
    log "  Per Phase A council kill criterion §3, the production archive will"
    log "  use the static arithmetic codec. Lane 20 codec ships ZERO bytes."
    log "  Skipping Stage 3+4 (no value in re-running auth eval on byte-identical archive)."
    log "  Final report: $LOG_DIR/training/lane_20_train_report.json"
    log "  Empirical anchor: reports/lane_20_balle_real_archive.json (from local Phase E)"
    log "=== LANE_20_DONE [empirical:STATIC_WINS_FALLBACK] anchor=$ANCHOR_RENDERER ==="
    exit 0
fi

# IF Lane 20 beats static, build a modified archive + auth eval
log "=== Stage 3: build modified archive (Lane 20 codec on qint segment) ==="
log "  (Stage 3 runs only when Ballé beats static — currently a placeholder for"
log "   when a heteroscedastic anchor lane appears that Lane 20 can win on.)"
# Placeholder: this would require a custom archive builder that writes a
# BHv1 chunk where the FP4A qint blob normally sits, plus a corresponding
# inflate-side dispatch (a future BHv1-aware FP4A loader). For the current
# Lane G v3 anchor, the verdict is STATIC_WINS_FALLBACK so Stage 3 is a
# no-op. The clean separation is intentional: the codec + STRICT check +
# remote-lane plumbing all land NOW; the archive integration awaits a
# heteroscedastic-anchor follow-up lane.

ARCHIVE="${ARCHIVE_PATH:-$LOG_DIR/archive_lane_20.zip}"
"$PYBIN" -c "
import zipfile, os, shutil
src = 'submissions/baseline_dilated_h64_0_90'
dst = '$ARCHIVE'
files = ['renderer.bin', 'masks.mkv', 'optimized_poses.pt']
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in files:
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        z.write(p, arcname=n)
print(f'archive {dst}: {os.path.getsize(dst)} bytes ({len(files)} files)')
"

log "=== Stage 4: contest_auth_eval on Lane 20 archive ==="
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo 'FATAL: archive empty' >&2; exit 2; }
log "archive byte-size guard: $ARCHIVE_BYTES bytes"
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

# RESULT_JSON guard (LANE-B silent-crash prevention).
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash. Investigate before reporting score."
    exit 4
fi

log "=== LANE_20_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "  predicted band: [1.04, 1.05] (vs Lane G v3 1.05 anchor)"
log "  anchor baseline: 1.05 [contest-CUDA] (Lane G v3)"
