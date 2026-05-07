#!/bin/bash
# Lane α-wavelet-mask (PARADIGM-α HNeRV-wavelet mask encoder).
#
# Module: src/tac/wavelet_mask_codec.py + hnerv_wavelet_apply_transform.py +
# hnerv_wavelet_sidechannel.py. Adversarial review fixes 2026-05-06:
# decoder for/else raise (CRITICAL #2), REPACKABLE_SECTIONS string constants
# (#3), strength_numerator>0 guard (#5), slug-filename race-mode safety
# (commit 28201ee7), WR01 schema branch in cross_paradigm_atoms.py
# (commit 0abfd60e + regression test f8975eaa).
#
# REGISTERED-BUT-NOT-WIRED in step_extract_masks as of 2026-05-06; the
# NotImplementedError gate (commit 80455cf8) raises when cfg.mask_codec=
# 'wavelet'. Reactivation requires:
#   1. Compress-time wavelet training harness (not yet built)
#   2. Bit-identical decode roundtrip test against contest scorer
#
# DELTA from av1_monochrome:
#   * Wavelet residual sidechannel adds atoms to the mask payload at decode
#     time. Bytes target the masks.mkv slot (~412KB → smaller). Predicted
#     band needs empirical baseline before commitment.
#
# Cost: T4 @ ~$0.50/hr × ~30min wavelet training + 30min auth eval = ~$0.50
# (after compress-time harness lands).
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_alpha_wavelet_mask_results"
mkdir -p "$LOG_DIR"
TAG="lane_alpha_wavelet_mask"
log() { echo "[lane-α-wavelet] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_alpha_wavelet_mask.sh',
    'tag': '$TAG',
    'paradigm': 'alpha_wavelet_mask',
    'predicted_band': 'TBD-needs-empirical-baseline',
    'lane_registry_id': 'lane_alpha_wavelet_mask',
    'cross_paradigm_wiring_status': 'NotImplementedError gate at step_extract_masks (commit 80455cf8); compress-time wavelet training harness not yet built.',
    'audit_fixes_applied': ['decoder_for_else_raise (CRITICAL #2)', 'REPACKABLE_SECTIONS_string_constants (#3)', 'strength_numerator_gt_0 (#5)', 'slug_filename_race_safety (commit 28201ee7)', 'WR01_schema_branch (commit 0abfd60e)'],
    'pre_dispatch_blockers': ['compress-time wavelet training harness not built', 'bit-identical decode roundtrip test against contest scorer not landed'],
    'cost_estimate_usd': 0.50,
}
with open('$PROVENANCE', 'w') as f: json.dump(prov, f, indent=2)
print(json.dumps(prov))
"

( while true; do sleep 60; echo "[$(date -u +%FT%TZ)] lane=α-wavelet" >> "$HEARTBEAT"; done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC + module check ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || { log "FATAL: NVDEC failed"; exit 2; }
"$PYBIN" -c "
import sys; sys.path.insert(0, 'src')
from tac.wavelet_mask_codec import encode_wavelet_codec, decode_wavelet_codec
from tac.hnerv_wavelet_apply_transform import apply_wr01_atoms_to_raw
print('α-wavelet modules importable')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Skipped phase 1: compress-time wavelet training (BLOCKER) ==="
log "WARN: compress-time wavelet training harness not yet built."
log "WARN: experiments/train_wavelet_mask.py is the proposed entry point."
log "WARN: Required: WR01 atom planning + per-archive bit-budget allocation."

log "=== Skipped phase 2: decode roundtrip test (BLOCKER) ==="
log "WARN: bit-identical decode roundtrip test against contest scorer not landed."
log "WARN: Reference: src/tac/tests/test_hnerv_wavelet_sidechannel.py covers"
log "WARN:   sidechannel encode/decode but not full mask-roundtrip."

log "=== Skipped phase 3: archive build (skipped pending Phases 1-2) ==="
log "=== Skipped phase 4: contest-CUDA auth eval (skipped) ==="
log "LANE_ALPHA_WAVELET_MASK_DONE score=N/A [contest-CUDA] paradigm=α-wavelet blocked=compress_time_training_harness_missing"
