#!/bin/bash
# Lane Joint-Codec-Stack (PARADIGM-gamma).
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
# Wiring status as of commit f6c0035a (2026-05-06):
#   * tac.joint_codec_stack_orchestrator.model_to_jcsp_streams -> planning
#     specs (codec_kind, shape, dtype, byte estimates)
#   * tac.jcsp_stream_builder.model_to_stream_sources -> bridges specs to
#     runnable StreamSource objects (symmetric int8 quantization +
#     RAW_PASSTHROUGH payload routing); 16-test suite.
#   * tac.jcsp_score_marginals.{derive,save,load}_marginals -> per-tensor
#     dScore/dByte custody artifact with strict envelope schema; 18-test
#     suite. CLI: experiments/build_jcsp_score_marginals.py.
#   * tac.jcsp_stream_builder.jcsp_stream_source_local_archive_member ->
#     deterministic byte-closed JCSK skeleton archive (verified 4373 B
#     archive, byte-identical across builds, PK\x03\x04 magic).
#   * pipeline.step_compress_weights with cfg.use_joint_codec_stack=True
#     writes <iter_dir>/jcsp_local_skeleton_archive.zip and raises
#     NotImplementedError to prevent silent dispatch.
# Remaining blockers (operator-gated, not code-side):
#   1. Submission runtime does not consume the JCSK magic byte.
#   2. Strict preflight proof of submission-runtime closure not yet landed.
#   3. No contest-CUDA replay attempted.
#
# Cost: T4 @ ~$0.50/hr x ~1hr ADMM convergence + 30min auth eval = ~$0.75.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_joint_codec_stack_results"
mkdir -p "$LOG_DIR"
TAG="lane_joint_codec_stack"
log() { echo "[lane-gamma-jcsp] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'cross_paradigm_wiring_status': 'WARN-guard wired in step_compress_weights (commit 9bdd3d56); decomposition path complete via model_to_jcsp_streams + jcsp_stream_builder.model_to_stream_sources; remaining blocker is pipeline.py step_compress_weights dispatch wiring.',
    'audit_fixes_applied': ['score_cap_inversion (commit 721770d8)', '_gauss_cdf_vectorization (commit 13e809ae)', 'balle_pad_with_mean (commit 3c87e5e2)', 'static_wins_codec_kind_override (already in code)', 'jcsp_stream_builder bridge module (this commit, 16 tests)'],
    'pre_dispatch_blocker': 'pipeline.py step_compress_weights mode=jcsp dispatch wiring not yet landed (model_to_stream_sources -> run_admm -> archive packaging)',
    'cost_estimate_usd': 0.75,
}
with open('$PROVENANCE', 'w') as f: json.dump(prov, f, indent=2)
print(json.dumps(prov))
"

( while true; do sleep 60; echo "[$(date -u +%FT%TZ)] lane=gamma-jcsp" >> "$HEARTBEAT"; done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC + module import check ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || { log "FATAL: NVDEC failed"; exit 2; }
"$PYBIN" -c "
import sys; sys.path.insert(0, 'src')
from tac.joint_codec_stack_orchestrator import run_joint_codec_stack, StreamSource
print('JCSP module importable')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Stage 1: model-to-StreamSources decomposition (UNBLOCKED) ==="
"$PYBIN" -c "
import sys; sys.path.insert(0, 'src')
from tac.jcsp_stream_builder import model_to_stream_sources, quantize_tensor_symmetric
from tac.joint_codec_stack_orchestrator import model_to_jcsp_streams
print('jcsp_stream_builder.model_to_stream_sources importable')
print('quantize_tensor_symmetric importable')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Stage 2: local-skeleton archive build (UNBLOCKED at commit f6c0035a) ==="
log "INFO: pipeline.step_compress_weights with cfg.use_joint_codec_stack=True"
log "INFO:   now calls jcsp_stream_source_local_archive_member which produces"
log "INFO:   a deterministic byte-closed JCSK skeleton archive at"
log "INFO:   <iter_dir>/jcsp_local_skeleton_archive.zip plus a manifest JSON."
log "INFO: Operator drives via:"
log "INFO:   1. python experiments/build_jcsp_score_marginals.py --model X.pt"
log "INFO:        --out marginals.json --mode uniform --evidence '...'"
log "INFO:   2. python experiments/pipeline.py with use_joint_codec_stack=True"
log "INFO:        + jcsp_score_marginals_path=marginals.json"
log "INFO:   The pipeline raises NotImplementedError after the archive write,"
log "INFO:   so the bytes exist but no contest-CUDA submission is attempted."

log "=== Stage 3: contest-CUDA auth eval (BLOCKER: submission runtime does not consume jcsp.bin/JCSK) ==="
log "WARN: The submission runtime in submissions/exact_current/inflate.{sh,py}"
log "WARN:   does not parse the JCSK skeleton magic byte. Until a runtime"
log "WARN:   consumer ships, the local-skeleton archive cannot be evaluated"
log "WARN:   on the contest scorer. STRICT preflight proof is also pending."
log "LANE_JOINT_CODEC_STACK_DONE score=N/A [contest-CUDA] paradigm=gamma-jcsp blocked=submission_runtime_jcsp_consumer_missing+strict_preflight_proof_missing"
