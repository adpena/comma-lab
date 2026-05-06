#!/bin/bash
# Lane Joint-Codec-Stack (PARADIGM-γ).
#
# Module: src/tac/joint_codec_stack_orchestrator.py — JCSP wire format +
# magic byte + ADMM coordinator across {repr, predict, quant, entropy}.
# Per-stream Pareto-gated codec selection (codex-applied audit fixes
# 2026-05-06: score-cap inversion + _gauss_cdf vectorize + pad-with-mean +
# JCSP static_wins codec_kind override).
#
# DELTA from per-stream codec selection:
#   * ADMM coordinator jointly optimizes byte budget across multiple streams
#     with shared Lagrange multiplier (λ broadcast). Per-stream codec_kind
#     override correctly routes static_wins to KIND_ARITHMETIC_STATIC even
#     when the stream is configured as KIND_BALLE_HYPERPRIOR.
#   * Predicted band [contest-CUDA] +150-500bp on stack-eligible archives.
#
# REGISTERED-BUT-NOT-WIRED in step_compress_weights as of 2026-05-06; the
# WARN guard (commit 9bdd3d56) prevents silent no-op. Operator must build a
# `model_to_jcsp_streams(model)` decomposition helper before this runbook
# can produce a real archive.
#
# Cost: T4 @ ~$0.50/hr × ~1hr ADMM convergence + 30min auth eval = ~$0.75.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_joint_codec_stack_results"
mkdir -p "$LOG_DIR"
TAG="lane_joint_codec_stack"
log() { echo "[lane-γ-jcsp] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_joint_codec_stack.sh',
    'tag': '$TAG',
    'paradigm': 'gamma_joint_codec_stack',
    'predicted_band_bp': [150, 500],
    'lane_registry_id': 'lane_joint_codec_stack',
    'cross_paradigm_wiring_status': 'WARN-guard wired in step_compress_weights (commit 9bdd3d56); full dispatch DEFERRED — needs model_to_jcsp_streams(model) helper.',
    'audit_fixes_applied': ['score_cap_inversion (commit 721770d8)', '_gauss_cdf_vectorization (commit 13e809ae)', 'balle_pad_with_mean (commit 3c87e5e2)', 'static_wins_codec_kind_override (already in code)'],
    'pre_dispatch_blocker': 'model_to_jcsp_streams decomposition helper not yet built',
    'cost_estimate_usd': 0.75,
}
with open('$PROVENANCE', 'w') as f: json.dump(prov, f, indent=2)
print(json.dumps(prov))
"

( while true; do sleep 60; echo "[$(date -u +%FT%TZ)] lane=γ-jcsp" >> "$HEARTBEAT"; done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC + module import check ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || { log "FATAL: NVDEC failed"; exit 2; }
"$PYBIN" -c "
import sys; sys.path.insert(0, 'src')
from tac.joint_codec_stack_orchestrator import run_joint_codec_stack, StreamSource
print('JCSP module importable')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Stage 1: model→JCSP-streams decomposition (BLOCKER) ==="
log "WARN: model_to_jcsp_streams(model) helper not yet built."
log "WARN: Required to convert a renderer state_dict into list[StreamSource]."
log "WARN: Reference: tac.joint_codec_stack_orchestrator.StreamSource fields"
log "WARN:   (name, codec_kind, qints, num_symbols, offset, balle_codec?, score_per_byte_marginal)."

log "=== Stage 2: ADMM coordinator (skipped pending Stage 1) ==="
log "=== Stage 3: contest-CUDA auth eval (skipped) ==="
log "LANE_JOINT_CODEC_STACK_DONE score=N/A [contest-CUDA] paradigm=γ-jcsp blocked=streams_decomposition_helper_missing"
