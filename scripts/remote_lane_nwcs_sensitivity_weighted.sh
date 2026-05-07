#!/bin/bash
# Lane NWCS-sensitivity-weighted (PARADIGM-β β-variant of Lane J-NWC).
#
# Module: src/tac/neural_weight_codec_sensitivity.py — per-block sensitivity
# bucketing + variable-K VQ codebook (codebook_sizes per bucket).
#
# DELTA from Lane J-NWC:
#   * Standard NWC uses uniform codebook size K across all weights.
#   * NWCS uses sensitivity-bucketed codebooks: high-sensitivity buckets
#     get K=256 (more precision), low-sensitivity get K=4 (more aggressive
#     compression). Predicted -50 to -150B vs J-NWC at fixed distortion.
#   * Predicted band [contest-CUDA] [0.78, 0.92] vs J-NWC-EC stack 0.78-0.92.
#
# Wiring status as of 2026-05-06:
#   * cfg.weight_compression='nwcs_sensitivity' branch lands a two-stage gate
#     in pipeline.step_compress_weights (commit 9922335c): stage 1 raises
#     NotImplementedError with explicit blocker list when codec_path /
#     sensitivity_map_path are missing or non-existent; stage 2 raises after
#     the per-tensor encoding loop entry point as a structural gate so an
#     operator cannot accidentally ship a stub. Silent no-op trap is closed.
#   * tac.neural_weight_codec_sensitivity has the codec primitives
#     (SensitivityAwareWeightCodec, export_nwcs_renderer_container,
#     load_nwcs_renderer_container, compute_per_block_sensitivity,
#     encode_with_variable_codebook, decode_with_per_block_codebook).
# Remaining blockers (operator-gated):
#   1. Per-tensor encoding loop in step_compress_weights that calls
#      SensitivityAwareWeightCodec.encode → export_nwcs_renderer_container
#      with the appropriate sensitivity buckets.
#   2. Trained NWCS codec checkpoint produced by a (not yet built)
#      experiments/train_neural_weight_codec_sensitivity.py harness.
#   3. CUDA sensitivity-map artifact (per-channel) at
#      cfg.sensitivity_map_path. Reference: tac.sensitivity_map (already
#      lands the schema + validators).
#
# Cost: T4 @ ~$0.50/hr × ~30min sweep + 30min auth eval = ~$0.50.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_nwcs_sensitivity_weighted_results"
mkdir -p "$LOG_DIR"
TAG="lane_nwcs_sensitivity_weighted"

log() { echo "[lane-nwcs-β] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_nwcs_sensitivity_weighted.sh',
    'tag': '$TAG',
    'paradigm': 'beta_nwcs_sensitivity_weighted',
    'predicted_band': [0.78, 0.92],
    'lane_registry_id': 'lane_nwcs_sensitivity_weighted',
    'cross_paradigm_wiring_status': 'WARN-guard wired in step_compress_weights (commit 9bdd3d56); dispatch branch DEFERRED — needs operator landing.',
    'pre_dispatch_blocker': 'NWCS dispatch branch not yet wired in step_compress_weights (mirror of β commit 107f6fea required).',
    'cost_estimate_usd': 0.50,
}
with open('$PROVENANCE', 'w') as f: json.dump(prov, f, indent=2)
print(json.dumps(prov))
"

( while true; do sleep 60; echo "[$(date -u +%FT%TZ)] lane=nwcs-β" >> "$HEARTBEAT"; done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC + wiring check ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || { log "FATAL: NVDEC failed"; exit 2; }
"$PYBIN" -c "
import sys; sys.path.insert(0, 'src')
from tac.neural_weight_codec_sensitivity import SensitivityAwareWeightCodec
print('NWCS module importable')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Skipped phase 1: NWCS codec training (BLOCKER: harness not built) ==="
log "WARN: experiments/train_neural_weight_codec_sensitivity.py is the proposed"
log "WARN:   entry point but has not yet been built. Without a trained codec"
log "WARN:   checkpoint, cfg.weight_codec_path is unsatisfiable and the gate"
log "WARN:   raises NotImplementedError on the first invocation."

log "=== Skipped phase 2: per-tensor encoding loop wiring (BLOCKER: pipeline.step_compress_weights) ==="
log "INFO: pipeline.step_compress_weights at mode='nwcs_sensitivity' (commit 9922335c)"
log "INFO:   has the gate but NOT the encoding loop. Reference β branch (commit 107f6fea)"
log "INFO:   for the production-shape pattern: load model, walk tensors, encode each via"
log "INFO:   SensitivityAwareWeightCodec, package via export_nwcs_renderer_container."

log "=== Skipped phase 3: contest-CUDA auth eval (skipped pending Phases 1-2) ==="
log "LANE_NWCS_SENSITIVITY_WEIGHTED_DONE score=N/A [contest-CUDA] paradigm=β-nwcs blocked=codec_training_harness_missing+per_tensor_encoding_loop_missing"
