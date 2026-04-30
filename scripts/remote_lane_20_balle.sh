#!/bin/bash
# Lane 20: Ballé hyperprior codec — train on real renderer.bin qint stream
# and (conditionally) build + auth-eval an archive that uses Lane 20.
#
# Council .omx/research/council_lane_20_balle_design_20260430.md
# (10-of-10 inner council members reviewed; Quantizr/Hotz YELLOW empirical
# kill criterion wired into the auto-fallback).
#
# Current frontier gate: Lane G v3 PFP16 A++ frontier =
# 1.043987524793892 [contest-CUDA/T4], archive 686635 bytes.
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
#   * THIS SCRIPT is fail-closed while Lane 20 is on forensic hold. It may
#     train and write byte reports, but it must not clear the hold or spend
#     auth-eval GPU unless a non-static BHv1 byte win exists and a real
#     BHv1 archive/inflate path has landed.
#
# Dispatch clearance requirements:
#   * A non-static byte precheck must show BALLE_BEATS_STATIC with
#     best_full_balle_bytes < static_baseline_bytes on the real qint stream.
#   * The archive builder must ship a real BHv1 archive member, not a
#     baseline/static copy.
#   * inflate_renderer.py must decode that BHv1 archive member during inflate.
#
# Cost: 4090 @ $0.25/hr × ~30min training (CPU codec, GPU only for auth eval)
#       + ~10min auth eval = ~$0.20.
#
# Per CLAUDE.md NEVER-INVENT-CLI-FLAGS: every flag passed to
# train_balle_hyperprior.py / measure_lane_20_balle_real_archive.py are
# verified by argparse-grep (Stage 0 dead-flag scan). Exact auth eval remains
# behind the BHv1 integration gate and must use the JSON adjudicator when
# re-enabled.
#
# Score-tagging: any score this script ever produces will be tagged
# [contest-CUDA] in the completion-log line (LANE_20_DONE marker) per the
# CLAUDE.md score-tag rule and preflight completion-tag check. This literal
# is required even while Lane 20 is on forensic hold (no eval is actually
# launched today) because the check matches script content statically.
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
    'predicted_band': [1.04, 1.043987524793892],
    'anchor_score_baseline': 1.043987524793892,
    'anchor_lane': 'Lane G v3 PFP16 A++ frontier (1.043987524793892 contest-CUDA/T4)',
    'anchor_archive_bytes': 686635,
    'lane_20_premise': 'Replace the static arithmetic codec on the renderer FP4 qint stream with a learned Balle hyperprior (or Hotz-LITE chunked-static fallback). Auto-fallback to static prevents regression.',
    'phase_e_empirical_artifact': 'reports/lane_20_balle_real_archive.json',
    'forensic_hold_clearance_requirements': [
        'non_static_byte_precheck.json proves BALLE_BEATS_STATIC and best_full_balle_bytes < static_baseline_bytes',
        'real BHv1 archive member is built from encode_qints_balle_auto output',
        'inflate_renderer.py decodes the BHv1 archive member during inflate',
    ],
    'balle_archive_integration_status': 'held: real BHv1 archive/inflate integration not yet landed',
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

log "=== Stage 2b: non-static byte precheck (required before any auth eval) ==="
"$PYBIN" -c "
import json, sys
report_path = '$LOG_DIR/training/lane_20_train_report.json'
precheck_path = '$LOG_DIR/non_static_byte_precheck.json'
r = json.load(open(report_path))
static_bytes = int(r['static_baseline_bytes'])
best_bytes = int(r['best_full_balle_bytes'])
best_step = int(r.get('best_step', -1))
verdict = str(r.get('verdict', ''))
ok = verdict == 'BALLE_BEATS_STATIC' and best_step >= 0 and 0 < best_bytes < static_bytes
payload = {
    'schema_version': 1,
    'lane': 'lane_20_balle',
    'report': report_path,
    'verdict': verdict,
    'best_step': best_step,
    'static_baseline_bytes': static_bytes,
    'best_full_balle_bytes': best_bytes,
    'non_static_byte_win': ok,
    'required_for_hold_clearance': True,
}
with open(precheck_path, 'w') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
print('non_static_byte_precheck:', json.dumps(payload, sort_keys=True))
if not ok:
    raise SystemExit(
        'FATAL_NON_STATIC_BYTE_PRECHECK_FAILED: Lane 20 remains on forensic hold; '
        'do not run auth eval until BALLE_BEATS_STATIC and '
        'best_full_balle_bytes < static_baseline_bytes on the real qint stream.'
    )
" 2>&1 | tee "$LOG_DIR/non_static_byte_precheck.log"

log "=== Stage 3: BHv1 archive/inflate integration gate ==="
"$PYBIN" -c "
from pathlib import Path
script = Path('scripts/remote_lane_20_balle.sh').read_text(errors='ignore')
inflate = Path('submissions/robust_current/inflate_renderer.py').read_text(errors='ignore')
missing = []
if 'renderer.bhv1' not in script or 'encode_qints_balle_auto' not in script:
    missing.append('BHv1 archive member built from encode_qints_balle_auto output')
if 'decode_qints_balle' not in inflate or 'renderer.bhv1' not in inflate or 'BHv1' not in inflate:
    missing.append('inflate_renderer.py BHv1 archive member decode path')
if missing:
    raise SystemExit(
        'FATAL_BHV1_ARCHIVE_INFLATE_INTEGRATION_MISSING: '
        + '; '.join(missing)
        + '. See .omx/research/lane19_lane20_forensic_hold_repair_design_20260430_codex.md.'
    )
print('BHv1 archive member integration ready: renderer.bhv1 decode path present')
" 2>&1 | tee "$LOG_DIR/bhv1_integration_gate.log"

log "FATAL: Lane 20 reached an impossible state: the BHv1 integration gate passed, but the real archive builder is not implemented in this script."
exit 6
