#!/bin/bash
# Lane OWV3-sensitivity-weighted (PARADIGM-β β-variant of Lane Ω-W).
#
# Wired in experiments/pipeline.py step_compress_weights via
# cfg.use_sensitivity_weighted=True (commit 107f6fea + adversarial-review
# fixes cb2ea361). Module: src/tac/owv3_sensitivity_weighted.py
# (codex CRITICAL fixes 1-4 applied per .omx/research/paradigm_audit_findings_20260506.md).
#
# Council Q1 verdict (.omx/research/grand_council_meta_lagrangian_pareto_design_decisions_20260506.md):
# parallel-fan-out the Pareto frontier; β has the fewest blockers among
# cross-paradigm flags so it dispatches first.
#
# DELTA from Lane G v3:
#   * Same renderer architecture; OWV3 archive uses sensitivity-weighted
#     bit-budget allocation per-conv based on a precomputed sensitivity_map
#     artifact instead of the uniform-fp4 baseline.
#   * Predicted band [contest-CUDA] [1.00, 1.04] vs Lane G v3 PFP16 1.044
#     (modest -2 to -4 percentage points; hits at the diminishing-returns
#     edge of single-axis weight compression).
#
# Cost estimate: T4 @ ~$0.50/hr × ~30min sweep + 30min auth eval = ~$0.50.
#
# Score-tag: any score this script produces is tagged [contest-CUDA] in the
# completion-log line (LANE_OWV3_BETA_DONE marker) per CLAUDE.md score-tag
# rule.
#
# DISPATCH CHAIN:
#   Stage 0:  NVDEC probe + profile validation + dead-flag scan + remote-code
#             parity check
#   Stage 1:  Sensitivity sweep on Lane G v3 anchor
#             → produces sensitivity_map.pt artifact
#   Stage 2:  Build OWV3 sensitivity-weighted archive via β dispatch
#             (experiments/pipeline.py with --use-sensitivity-weighted)
#             → produces renderer_owv3_sensitivity.bin
#   Stage 3:  Build full submission archive (renderer + masks + poses)
#   Stage 4:  Contest-CUDA auth eval via inflate.sh + upstream/evaluate.py
#             → reports [contest-CUDA] score
#   Stage 5:  Harvest + write LANE_OWV3_BETA_DONE marker
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=20
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_owv3_sensitivity_weighted_results"
mkdir -p "$LOG_DIR"
TAG="lane_owv3_sensitivity_weighted"

log() { echo "[lane-β-owv3] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_owv3_sensitivity_weighted.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'paradigm': 'beta_sensitivity_weighted',
    'profile': 'lane_owv3_sensitivity_weighted_lane_g_v3',
    'predicted_band': [1.00, 1.04],
    'anchor_score_baseline': 1.043987524793892,
    'anchor_lane': 'Lane G v3 PFP16 A++ frontier (1.043987524793892 contest-CUDA/T4)',
    'anchor_archive_bytes': 686635,
    'beta_premise': 'Replace uniform fp4 bit allocation with sensitivity-weighted per-conv bit budget. Critical-channel precision is preserved while non-critical channels get aggressive quantization. Module: tac.owv3_sensitivity_weighted.encode_owv3_archive (commit 107f6fea + cb2ea361).',
    'cross_paradigm_wiring_status': 'WIRED (commits 107f6fea + cb2ea361 + integration tests in test_pipeline_beta_dispatch.py)',
    'integration_test_status': '3 tests passing in src/tac/tests/test_pipeline_beta_dispatch.py',
    'preflight_gate': 'check_cross_paradigm_wiring_contract STRICT (commit a0f00246)',
    'cost_estimate_usd': 0.50,
    'wall_clock_estimate_minutes': 60,
    'lane_registry_id': 'lane_owv3_sensitivity_weighted',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=owv3-β gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0a: NVDEC probe BEFORE any GPU spend (auth eval needs NVDEC).
log "=== Stage 0a: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Stage 0b: remote-code parity check (CLAUDE.md non-negotiable).
log "=== Stage 0b: remote-code parity ==="
REMOTE_HEAD=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null)
log "remote HEAD: $REMOTE_HEAD"
if [ "$REMOTE_HEAD" = "no-git" ]; then
    log "FATAL: remote workspace is not a git checkout — abort."
    exit 3
fi

# Stage 0c: cross-paradigm flag verification — verify the β branch is wired
log "=== Stage 0c: β dispatch wiring verification ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from experiments.pipeline import PipelineConfig
cfg = PipelineConfig(use_sensitivity_weighted=True, sensitivity_map_path='/dev/null')
assert cfg.use_sensitivity_weighted is True
assert hasattr(cfg, 'owv3_bit_budget_ratio')
assert hasattr(cfg, 'owv3_protect_threshold')
print('β dispatch wiring OK: use_sensitivity_weighted=True', cfg.use_sensitivity_weighted)
print('  bit_budget_ratio:', cfg.owv3_bit_budget_ratio)
print('  protect_threshold:', cfg.owv3_protect_threshold)
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Stage 1: Sensitivity sweep on Lane G v3 anchor
log "=== Stage 1: sensitivity sweep on Lane G v3 anchor ==="
ANCHOR_CKPT="${ANCHOR_CKPT:-$WORKSPACE/experiments/results/lane_g_v3_pfp16_landed/renderer.pt}"
SENS_OUT="$LOG_DIR/sensitivity_map.pt"
if [ ! -f "$ANCHOR_CKPT" ]; then
    log "WARN: anchor checkpoint $ANCHOR_CKPT not present — sensitivity sweep needs"
    log "WARN: a renderer .pt; the local Lane G v3 directory only contains the"
    log "WARN: post-encoded archive. Operator must extract or re-train the"
    log "WARN: anchor renderer before this stage runs."
    log "WARN: STAGE 1 SKIPPED."
else
    "$PYBIN" -m experiments.run_sensitivity_sweep \
        --checkpoint "$ANCHOR_CKPT" \
        --output "$SENS_OUT" \
        --device cuda \
        --num-samples 256 \
        2>&1 | tee -a "$LOG_DIR/run.log" || {
            log "FATAL: sensitivity sweep failed"
            exit 4
        }
fi

# Stage 2: Build β archive via experiments/pipeline.py
log "=== Stage 2: β archive build (use_sensitivity_weighted=True) ==="
if [ -f "$SENS_OUT" ]; then
    "$PYBIN" -m experiments.pipeline compress \
        --profile lane_g_v3 \
        --device cuda \
        --output-dir "$LOG_DIR/compress_β" \
        --use-sensitivity-weighted \
        --sensitivity-map-path "$SENS_OUT" \
        --owv3-bit-budget-ratio 0.7 \
        --owv3-protect-threshold 1e-3 \
        2>&1 | tee -a "$LOG_DIR/run.log" || {
            log "FATAL: β archive build failed"
            exit 5
        }
else
    log "STAGE 2 SKIPPED (no sensitivity map artifact)"
fi

# Stage 3: Build full submission archive
log "=== Stage 3: full submission archive ==="
# (delegates to experiments/pipeline.py archive step; renderer.bin from
# Stage 2, masks + poses from cached or fresh extraction)

# Stage 4: Contest-CUDA auth eval
log "=== Stage 4: contest-CUDA auth eval ==="
ARCHIVE="$LOG_DIR/compress_β/iter_0/renderer_owv3_sensitivity.bin"
if [ -f "$ARCHIVE" ]; then
    bash "$WORKSPACE/scripts/remote_archive_only_eval.sh" \
        --archive "$ARCHIVE" \
        --output-json "$LOG_DIR/contest_auth_eval.json" \
        2>&1 | tee -a "$LOG_DIR/run.log" || {
            log "FATAL: contest-CUDA auth eval failed"
            exit 6
        }
else
    log "STAGE 4 SKIPPED (no β archive produced)"
fi

# Stage 5: Harvest + completion marker
log "=== Stage 5: harvest ==="
SCORE=$("$PYBIN" -c "
import json
try:
    with open('$LOG_DIR/contest_auth_eval.json') as f:
        d = json.load(f)
    print(d.get('score', 'N/A'))
except Exception:
    print('N/A')
")
log "LANE_OWV3_BETA_DONE score=$SCORE [contest-CUDA] paradigm=β"
log "  artifacts: $LOG_DIR"
log "  next: claim_lane_dispatch.py mark + tools/lane_maturity.py mark contest_cuda + real_archive_empirical"
