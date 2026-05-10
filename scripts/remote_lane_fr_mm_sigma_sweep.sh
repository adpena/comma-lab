#!/bin/bash
# Lane FR-MM — Variable-σ Gaussian-LUT sigma sweep over the Lane MM
#              encoder + LUT-decoder pipeline.
#
# Lane MM hard-coded sigma=15 (Selfcomp default). Olah eureka: sigma is a
# soft-decision-boundary parameter; smaller σ -> more confident class
# assignments (closer to one-hot, less robust to AV1 quant noise); larger
# σ -> smoother class transitions (more robust but possibly fuzzier
# decision regions). Bake-off:
#   sigma ∈ {8, 12, 15, 18, 22, 25, 30}
#
# Same Lane A renderer.bin + same poses + same grayscale.mkv encoding
# (CRF=50). Only the inflate-time LUT sigma changes per run via the
# LANE_MM_SIGMA env var (read by inflate_renderer_grayscale.py at decode
# time). 7 runs × ~5 min each = ~35 min total wall clock.
#
# Cost cap: $0.50 (no training; encode once + 7 fast auth evals on shared
# archive). Predicted band [0.65, 0.85] [contest-CUDA] for the BEST sigma
# (default 15 lands at ~0.76; the sweep should find a sub-0.70 alternative).
#
# Anchor: experiments/results/lane_a_landed/archive_lane_a.zip (1.15 verified).
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: encoder-only sweep reuses Lane MM build path (already-tested) and only varies the inflate-time sigma; the inflate path tac.mask_grayscale_lut.create_gaussian_softmax_lut(sigma=...) is the documented soft-decision contract.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_fr_mm_sigma_sweep"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-fr-mm] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_fr_mm_sigma_sweep.sh',
    'output_dir': '$LOG_DIR',
    'sigma_sweep': [8, 12, 15, 18, 22, 25, 30],
    'predicted_band': [0.65, 0.85],
    'anchor_archive': 'experiments/results/lane_a_landed/archive_lane_a.zip',
    'anchor_score_baseline': 1.15,
    'controlled_baseline': 'lane_a_landed (1.15 contest-CUDA)',
    'paradigm': 'lane_mm_sigma_sweep',
    'inflate_path': 'PYTHON_INFLATE=renderer_grayscale + LANE_MM_SIGMA env var',
    'cost_estimate_usd': 0.30,
    'cost_cap_usd': 0.50,
    'wall_clock_estimate_hours': 0.6,
    'wall_clock_cap_hours': 1.5,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=FR-MM gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- destroy this Vast.ai instance."
    exit 2
}

# Pre-flight: required artifacts present. Anchor on Lane A.
ANCHOR_ARCHIVE="experiments/results/lane_a_landed/archive_lane_a.zip"
for f in "$ANCHOR_ARCHIVE" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         submissions/robust_current/inflate.sh \
         submissions/robust_current/inflate_renderer.py \
         submissions/robust_current/inflate_renderer_grayscale.py; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: build the SHARED Lane MM archive (sigma is decode-time only) ==="
ARCHIVE="$LOG_DIR/archive_lane_fr_mm.zip"
set +e
"$PYBIN" -u experiments/build_lane_mm_archive.py \
    --anchor-archive "$ANCHOR_ARCHIVE" \
    --output "$ARCHIVE" \
    --crf 50 \
    --sigma 15 2>&1 | tee "$LOG_DIR/build.log" | tail -10
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: build_lane_mm_archive exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
if [ -z "${ARCHIVE_BYTES:-}" ] || [ "$ARCHIVE_BYTES" -le 0 ]; then
    echo "FATAL: archive size empty or zero — refusing to call auth_eval"
    exit 2
fi
log "  archive bytes: $ARCHIVE_BYTES"

# Sweep loop: 7 sigma values, each with its own config.env + auth eval.
SIGMAS=(8 12 15 18 22 25 30)
SWEEP_RESULTS_JSON="$LOG_DIR/sigma_sweep_summary.json"
"$PYBIN" -c "import json; open('$SWEEP_RESULTS_JSON', 'w').write(json.dumps({'sigmas': []}, indent=2))"

for SIGMA in "${SIGMAS[@]}"; do
    log "===================================================================="
    log "=== sweep sigma=$SIGMA ==="
    log "===================================================================="

    # config.env: tell inflate.sh to dispatch via grayscale arm AND export
    # LANE_MM_SIGMA so inflate_renderer_grayscale.py overrides the LUT default.
    INFLATE_CONFIG="$LOG_DIR/lane_fr_mm_sigma${SIGMA}_config.env"
    cat > "$INFLATE_CONFIG" <<EOF
PYTHON_INFLATE=renderer_grayscale
LANE_MM_SIGMA=$SIGMA
export LANE_MM_SIGMA
EOF

    EVAL_DIR="$LOG_DIR/eval_sigma${SIGMA}"
    mkdir -p "$EVAL_DIR"
    rm -rf "$EVAL_DIR/eval_work"

    set +e
    CONFIG_ENV_PATH="$INFLATE_CONFIG" "$PYBIN" -u experiments/contest_auth_eval.py \
        --archive "$ARCHIVE" \
        --inflate-sh submissions/robust_current/inflate.sh \
        --upstream-dir upstream \
        --device cuda \
        --keep-work-dir \
        --work-dir "$EVAL_DIR/eval_work" 2>&1 | tee "$EVAL_DIR/auth_eval.log" | tail -15
    EVAL_RC=$?
    set -e

    "$PYBIN" -c "
import json, os
from pathlib import Path

d = json.load(open('$SWEEP_RESULTS_JSON'))
entry = {
    'sigma': $SIGMA,
    'archive_bytes': os.path.getsize('$ARCHIVE'),
    'eval_rc': $EVAL_RC,
}
result_path = Path('$EVAL_DIR/eval_work/contest_auth_eval.json')
if result_path.exists():
    payload = json.loads(result_path.read_text())
    entry['score'] = float(payload['score_recomputed_from_components'])
    entry['final_score_reported'] = float(payload['final_score'])
d['sigmas'].append(entry)
json.dump(d, open('$SWEEP_RESULTS_JSON', 'w'), indent=2)
print(f'  recorded sigma sweep entry: {entry}')
"
done

log "=== Stage 6: aggregate sigma sweep results ==="
"$PYBIN" -c "
import json
d = json.load(open('$SWEEP_RESULTS_JSON'))
sigmas = d.get('sigmas', [])
scored = [s for s in sigmas if 'score' in s]
scored.sort(key=lambda s: s['score'])
d['ranked'] = scored
d['best'] = scored[0] if scored else None
json.dump(d, open('$SWEEP_RESULTS_JSON', 'w'), indent=2)
print('=== sigma sweep summary ===')
for s in sigmas:
    print(f\"  sigma={s['sigma']:>3d}  score={s.get('score', 'FAIL')}\")
print('=== best sigma:', d.get('best'))
"

"$PYBIN" -c "
import json, os, time
prov = json.load(open('$PROVENANCE'))
prov['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
prov['archive_path'] = '$ARCHIVE'
prov['archive_bytes'] = os.path.getsize('$ARCHIVE')
prov['sigma_sweep_results_path'] = '$SWEEP_RESULTS_JSON'
prov['lane_status'] = 'COMPLETE'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_FR_MM_DONE — see $SWEEP_RESULTS_JSON for ranked sigmas [contest-CUDA] ==="
