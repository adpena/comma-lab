#!/bin/bash
# Lane QAT-Sweep: Bayesian sweep over Lane F-V3 FP4 QAT hyperparameters.
#
# Replaces the hand-tuned Lane F-V3 schedule (int8_warmup=50, fp4_epochs=500,
# lr=2.5e-6, cosine, baseline band [1.30, 1.80]) with a 30-trial TPE Bayesian
# search. Predicted Lane QAT-Sweep band: [1.10, 1.60].
#
# Trial template placeholders (every __PARAM_X__ is required, validated at
# sweep construction):
#   __PARAM_INT8_WARMUP_EPOCHS__ -> int, sampled
#   __PARAM_FP4_EPOCHS__         -> int, sampled
#   __PARAM_LR__                 -> float, log-sampled
#   __PARAM_LR_SCHEDULE__        -> categorical (cosine|linear|constant)
#                                   (recorded in provenance only — qat_finetune.py
#                                   currently bakes cosine in; flag wiring TBD)
#   __PARAM_DEVICE__             -> "cuda" (fixed per CLAUDE.md)
#   __SWEEP_NAME__               -> "lane_qat_fp4"
#   __SWEEP_TRIAL_NUMBER__       -> e.g. "0017" or "BEST"
#   __SWEEP_SEARCH_SPACE_HASH__  -> SHA256[:16] for reproducibility
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234

SWEEP_NAME="__SWEEP_NAME__"
TRIAL_NUMBER="__SWEEP_TRIAL_NUMBER__"
SEARCH_SPACE_HASH="__SWEEP_SEARCH_SPACE_HASH__"

# Operational modes — same dual-purpose pattern as remote_lane_a_sweep.sh.
if [[ "$TRIAL_NUMBER" == "__SWEEP_TRIAL_NUMBER__" ]]; then
    LOG_DIR="$WORKSPACE/lane_qat_sweep_results"
    mkdir -p "$LOG_DIR"

    PROVENANCE="$LOG_DIR/provenance.json"
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
    'lane_script': 'scripts/remote_lane_qat_sweep.sh',
    'lane_name': 'lane_qat_sweep',
    'is_sweep': True,
    'n_trials': 30,
    'sweep_name': 'lane_qat_fp4',
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"

    echo "=== Stage 0: NVDEC probe ==="
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        echo "FATAL: NVDEC probe failed — destroy this instance, pick a different host."
        exit 2
    }

    echo "=== Stage 1: bootstrap ==="
    [ -d "$WORKSPACE/upstream" ] || {
        echo "FATAL: workspace missing upstream/ — run remote_setup_full.sh first"
        exit 1
    }

    echo "=== Stage 2: install optuna ==="
    uv pip install --system optuna 2>&1 | tail -3

    echo "=== Stage 3: Bayesian sweep over QAT hyperparameters ==="
    "$PYBIN" -u experiments/sweep_lane_qat.py \
        --n-trials 30 \
        --objective auth_score \
        --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/sweep.log" | tail -50

    OPT_SCRIPT="$WORKSPACE/scripts/remote_lane_qat_optimized.sh"
    [ -f "$OPT_SCRIPT" ] || {
        echo "FATAL: sweep driver did not emit $OPT_SCRIPT"
        exit 2
    }
    echo "=== Stage 4: official Lane QAT-Sweep result ==="
    bash "$OPT_SCRIPT" 2>&1 | tee "$LOG_DIR/optimized.log" | tail -30

    grep -q '^RESULT_JSON' "$LOG_DIR/optimized.log" || {
        echo "FATAL: optimized run did not produce RESULT_JSON"
        exit 2
    }
    echo "=== LANE_QAT_SWEEP_DONE — see $LOG_DIR ==="
    exit 0
fi

# ----- TRIAL MODE -----

LOG_DIR="$WORKSPACE/lane_qat_sweep_results/trial_${TRIAL_NUMBER}"
mkdir -p "$LOG_DIR"

log() { echo "[lane-qat-sweep trial=${TRIAL_NUMBER}] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

log "=== trial provenance ==="
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'sweep_name': '$SWEEP_NAME',
    'trial_number': '$TRIAL_NUMBER',
    'search_space_hash': '$SEARCH_SPACE_HASH',
    'is_sweep': True,
    'int8_warmup_epochs': __PARAM_INT8_WARMUP_EPOCHS__,
    'fp4_epochs': __PARAM_FP4_EPOCHS__,
    'lr': __PARAM_LR__,
    'lr_schedule': '__PARAM_LR_SCHEDULE__',
    'device': '__PARAM_DEVICE__',
    'cuda_available': torch.cuda.is_available(),
}
with open('$LOG_DIR/trial_provenance.json', 'w') as f:
    json.dump(prov, f, indent=2)
print('trial_provenance:', json.dumps(prov))
"

# Anchor on Lane A (1.15 [contest-CUDA]), same as Lane F-V3 baseline.
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_a_landed/extracted/masks.mkv"
for f in "$ANCHOR_RENDERER" "$ANCHOR_POSES" "$ANCHOR_MASKS" \
         upstream/videos/0.mkv upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== stage Lane A masks ==="
mkdir -p "$LOG_DIR/extracted"
cp "$ANCHOR_MASKS" "$LOG_DIR/extracted/masks.mkv"

log "=== FP4 QAT fine-tune with sampled hyperparameters ==="
log "  int8_warmup=__PARAM_INT8_WARMUP_EPOCHS__ fp4_epochs=__PARAM_FP4_EPOCHS__"
log "  lr=__PARAM_LR__ lr_schedule=__PARAM_LR_SCHEDULE__ (provenance only — flag pending)"
"$PYBIN" -u experiments/qat_finetune.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --poses "$ANCHOR_POSES" \
    --upstream upstream \
    --output-dir "$LOG_DIR/qat" \
    --device __PARAM_DEVICE__ \
    --base-ch 36 --mid-ch 60 --pose-dim 6 --motion-hidden 32 --depth 1 --embed-dim 6 \
    --use-zoom-flow --padding-mode zeros \
    --int8-warmup-epochs __PARAM_INT8_WARMUP_EPOCHS__ \
    --fp4-epochs __PARAM_FP4_EPOCHS__ \
    --lr __PARAM_LR__ \
    --batch-size 4 2>&1 | tee "$LOG_DIR/qat.log" | tail -20

FP4_BIN=$(find "$LOG_DIR/qat" -name "renderer_fp4.bin" -o -name "*_fp4*.bin" 2>/dev/null | head -1)
[ -n "$FP4_BIN" ] && [ -f "$FP4_BIN" ] || { echo "FATAL: qat_finetune didn't produce FP4 binary"; ls -la "$LOG_DIR/qat/"; exit 2; }

log "=== build trial archive (Python zipfile, NOT shell zip) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$FP4_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_trial.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        z.write(p, arcname=n)
print(f'archive {dst}: {os.path.getsize(dst)} bytes')
"
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
if [ -z "${ARCHIVE_BYTES:-}" ] || [ "$ARCHIVE_BYTES" -le 0 ]; then
    echo "FATAL: archive size empty or zero — refusing to call auth_eval"
    exit 2
fi

log "=== contest_auth_eval ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device __PARAM_DEVICE__ \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -10

grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || {
    echo "FATAL: auth_eval did not produce RESULT_JSON"
    exit 2
}

SIDE_CAR="${BASH_SOURCE[0]%.sh}.result.json"
"$PYBIN" -c "
import json, re
log = open('$LOG_DIR/auth_eval.log').read()
m = re.search(r'^RESULT_JSON:\s*(\{.*\})\s*$', log, re.M)
assert m, 'RESULT_JSON missing after grep passed?!'
payload = json.loads(m.group(1))
out = {
    'final_score': payload.get('final_score'),
    'auth_score':  payload.get('final_score'),
    'avg_posenet_dist': payload.get('avg_posenet_dist'),
    'avg_segnet_dist':  payload.get('avg_segnet_dist'),
    'rate_unscaled':    payload.get('rate_unscaled'),
    'archive_size_bytes': payload.get('archive_size_bytes'),
    'trial_number': '$TRIAL_NUMBER',
    'sweep_name':   '$SWEEP_NAME',
}
with open('$SIDE_CAR', 'w') as f:
    json.dump(out, f, indent=2)
print('wrote sidecar:', '$SIDE_CAR')
"

log "=== TRIAL_DONE — sidecar at $SIDE_CAR ==="
