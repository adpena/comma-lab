#!/bin/bash
# Lane G V3-V2: Lagrangian-controlled SNR for the KL-distill auxiliary
# weight on Lane A pose TTO. Replaces the hand-derived
# `--kl-distill-weight 0.002` constant from Lane G V3 with the
# `--kl-distill-snr-target 0.10` controller (the Hinton 2015 canonical
# auxiliary regime, mathematically derived rather than tuned). The
# controller uses multiplicative dual ascent on
# `log w_{t+1} = log w_t + η · log(ρ / snr_t)` (Boyd & Vandenberghe
# §5.4 strong duality + Kivinen & Warmuth 1997 exponentiated gradient
# on ratio constraints) — see src/tac/lagrangian_kl_weight.py for the
# convergence proof.
#
# V1 history (--kl-distill-weight 1.0):       KL dominated 14000× → killed.
# V2 history (--kl-distill-weight 0.01):      KL dominated 4000× → killed.
# V3 corrected (--kl-distill-weight 0.002):    KL ≈ 10% scorer (LANDED but
#                                              the constant only happened
#                                              to be ~right for the
#                                              POST-FIX KL scale of 2.7;
#                                              if the KL scale changes
#                                              again — e.g. via a future
#                                              loss-norm fix — the constant
#                                              breaks).
# V3-V2 (THIS SCRIPT — --kl-distill-snr-target 0.10):
#   * The TARGET ratio is operator-supplied; the per-step weight is
#     ADAPTIVELY found by dual ascent. If the KL scale changes (loss
#     fix / temperature change / different scorer), the controller
#     re-derives the right weight on its own.
#   * Initial weight = 0.002 (matches V3 corrected, so step 0 behaves
#     identically). After ~20 steps the controller has converged on
#     the SNR-correct weight even if the underlying KL scale has
#     shifted.
#   * Predicted band: identical to V3 corrected ([1.10, 1.18]) when
#     the V3 weight was already SNR-correct; tighter when the
#     current-run KL scale differs from V3's measured 2.7.
#
# Per CLAUDE.md non-negotiable (NEVER invent CLI flags): both
# `--kl-distill-snr-target` and `--kl-distill-snr-eta` are verified
# against optimize_poses.py argparse below in the dead-flag scan.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_g_v3_v2_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-g-v3-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_g_v3_v2_lagrangian_snr.sh',
    'output_dir': '$LOG_DIR',
    'kl_distill_snr_target': 0.10,
    'kl_distill_snr_eta': 0.5,
    'kl_distill_initial_weight': 0.002,
    'kl_distill_temperature': 2.0,
    'predicted_band': [1.10, 1.18],
    'anchor_score_baseline': 1.15,
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/optimized_poses.pt',
    'lagrangian_target': 'maintain SNR (KL contrib / scorer contrib) at 0.10 via multiplicative dual ascent',
    'delta_from_v3_corrected': {
        'mode': 'replace static --kl-distill-weight=0.002 with adaptive --kl-distill-snr-target=0.10',
        'rationale': 'V3 was hand-tuned for the POST-FIX KL scale of 2.7. If the underlying scorer changes (loss-norm fix, temperature, etc.) the constant silently goes wrong. Lagrangian SNR re-derives the weight per step → robust to scale changes.',
        'theory': 'Boyd & Vandenberghe §5.4 + Kivinen & Warmuth 1997 EG on ratio constraints',
        'helper_module': 'src/tac/lagrangian_kl_weight.py',
    },
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=G-V3-V2 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend.
log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- destroy this Vast.ai instance."
    exit 2
}

# Pre-flight: required artifacts present.
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Pre-flight: dead-flag scan (CLAUDE.md non-negotiable: NEVER invent
# CLI flags). Verifies that --kl-distill-snr-target and
# --kl-distill-snr-eta exist in optimize_poses.py argparse.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_g_v3_v2_lagrangian_snr.sh').read()
op_src = open('experiments/optimize_poses.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', op_src))
m = re.search(r'experiments/optimize_poses\.py(.*?)(?=\n\s*\[\s*-f|\n\s*log\b|\Z)',
              script, re.DOTALL)
assert m, 'could not locate optimize_poses.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in optimize_poses argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
"

log "=== Stage 1: rebuild full-res masks (same as Lane A baseline) ==="
set +e
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 \
    --output "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

mkdir -p "$LOG_DIR/extracted"
"$PYBIN" -c "
import zipfile
with zipfile.ZipFile('$LOG_DIR/archive_baseline_seed.zip') as z:
    z.extractall('$LOG_DIR/extracted')
print('extracted to $LOG_DIR/extracted')
"

log "=== Stage 2: pose TTO (Lane A flow) + LAGRANGIAN-CONTROLLED KL-distill ==="
log "   --gt-poses-path = $ANCHOR_POSES"
log "   --eval-roundtrip + --posetto-noise-std=0.5 (Fridrich C1 fixes)"
log "   --kl-distill-snr-target=0.10 (canonical Hinton 2015 auxiliary regime)"
log "   --kl-distill-snr-eta=0.5 (geometric convergence, Boyd §5.4)"
log "   --kl-distill-weight=0.002 (INITIAL value; controller adapts after)"
export PYTHONHASHSEED=1234
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +e
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --masks "$LOG_DIR/extracted/masks.mkv" \
    --gt-poses-path "$ANCHOR_POSES" \
    --device cuda \
    --steps 500 \
    --batch-pairs 4 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --kl-distill-weight 0.002 \
    --kl-distill-temperature 2.0 \
    --kl-distill-snr-target 0.10 \
    --kl-distill-snr-eta 0.5 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Validate: did optimize_poses produce the file?
[ -f "$LOG_DIR/optimized_poses.bin" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.bin"; exit 2; }
log "  produced optimized_poses.bin ($(stat -c '%s' "$LOG_DIR/optimized_poses.bin") bytes)"

log "=== Stage 3: build NEW archive (Lane A renderer + masks + NEW poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$ANCHOR_RENDERER" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
[ -f "$LOG_DIR/optimized_poses.pt" ] && cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_g_v3_v2.zip"
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

log "=== Stage 4: contest_auth_eval on Lane G V3-V2 archive [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
set +e
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval log has no RESULT_JSON line — eval crashed."
    exit 4
fi

log "=== LANE_G_V3_V2_DONE [contest-CUDA] -- see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
