#!/bin/bash
# Lane Ω-V2: Lagrangian per-WEIGHT learnable bit-depth on Lane A.
#
# Council 2026-04-27: Lane Ω-V1 (water-fill heuristic, α=0.5 hard-coded)
# landed but the user noted everything could be LEARNABLE and mathematically
# optimal. Lane Ω-V2 replaces the post-hoc water-fill with Lagrangian dual
# ascent + per-element learnable bit-depth.
#
# Math: rate-distortion duality (KKT) says at the constrained optimum
#     ∂D/∂bits_ij = -λ · ∂R/∂bits_ij    ∀ (i,j)
# Learnable bits + λ annealed via dual ascent reaches the SAME optimum as
# closed-form water-fill, BUT adapts to the actual fake-quantization
# distortion landscape (not the Gaussian-noise approximation that
# water-fill assumes). So Ω-V2 is at-least-as-good as Ω-V1 in the limit
# and strictly better when the distortion is non-Gaussian.
#
# Predicted band: [0.65, 1.05] [contest-CUDA] — slightly tighter than
# Ω-V1's [0.70, 1.05] because Lagrangian convergence is more reliable than
# water-fill heuristic (Yousfi council).
#
# Pipeline:
#   Stage 0 — NVDEC probe (catches bad-host in 5s).
#   Stage 1 — OPTIONAL: profile per-weight Hessian (warm-start Ω-V2 bits).
#             Skipped when LANE_OMEGA_V2_HESSIAN_INIT=0 (default).
#   Stage 2 — Lagrangian QAT: load Lane A, swap Conv2d → LearnableBitConv2d,
#             fine-tune 200 epochs at lr=2.5e-6, bits-lr=2.5e-7.
#             Target 2.5 bits/weight (≈75KB renderer.bin).
#   Stage 3 — OMG1 export of best checkpoint (auto-extracts bits from
#             LearnableBitConv2d wrappers — same on-disk OMG1 format as Ω-V1).
#   Stage 4 — Build archive (Ω-V2 renderer + Lane A masks + Lane A poses).
#   Stage 5 — contest_auth_eval [contest-CUDA].
#
# Anchored artifacts:
#   * renderer.bin: experiments/results/lane_a_landed/iter_0/renderer.bin (290KB FP32)
#   * masks.mkv:    experiments/results/lane_a_landed/iter_0/masks.mkv
#   * poses.pt:     experiments/results/lane_a_landed/iter_0/optimized_poses.pt
#
# Council preflight (CLAUDE.md non-negotiable: NEVER invent CLI flags).
# Verified 2026-04-27 against argparse:
#   * qat_omega_lagrangian.py: --checkpoint --video --masks-mkv --poses
#     --upstream --output-dir --init-bits --target-bits --lambda-start
#     --lambda-end --lambda-ramp-start-frac --total-epochs --lr
#     --bits-lr-scale --noise-std --kl-weight --seg-weight --pose-weight
#     --hessian-init --device --seed --log-every
#   * profile_hessian_per_weight.py: --checkpoint --video --masks-mkv
#     --poses --upstream --output --top-k --all-pairs --device --pair-batch
#   * contest_auth_eval.py: --archive --inflate-sh --upstream-dir --device
#     --keep-work-dir --work-dir
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_omega_v2_results"
mkdir -p "$LOG_DIR"
TAG="lane_omega_v2_lagrangian"

log() { echo "[lane-omega-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline + memory
# feedback_canonical_remote_bootstraps).
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
    'lane_script': 'scripts/remote_lane_omega_v2_lagrangian.sh',
    'lane_name': 'lane_omega_v2_lagrangian_on_lane_a',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/iter_0/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/iter_0/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [0.65, 1.05],
    'rationale': 'Lagrangian dual ascent on per-WEIGHT learnable bit-depth (vs Ω-V1 closed-form water-fill). KKT optimum: ∂D/∂bits = -λ · ∂R/∂bits. Adapts to actual STE fake-quant distortion landscape rather than Gaussian-noise approximation. Predicted band tighter than Ω-V1 because Lagrangian convergence is more reliable than water-fill heuristic.',
    'target_total_bits': 600000,
    'target_bits_per_weight': 2.5,
    'init_bits': 8.0,
    'lambda_start': 0.0,
    'lambda_end': 1.0,
    'lambda_ramp_start_frac': 0.3,
    'total_epochs': 200,
    'lr': 2.5e-6,
    'bits_lr_scale': 0.1,
    'noise_std': 0.5,
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=Omega-V2 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 — NVDEC probe BEFORE any GPU spend (memory:
# feedback_vastai_nvdec_host_variation).
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: anchor on Lane A artifacts committed to the repo.
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/iter_0/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_a_landed/iter_0/masks.mkv"
for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         "$ANCHOR_MASKS" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer: $ANCHOR_RENDERER ($(stat -c '%s' "$ANCHOR_RENDERER") bytes, ASYM FP32)"
log "  anchor_poses:    $ANCHOR_POSES"
log "  anchor_masks:    $ANCHOR_MASKS ($(stat -c '%s' "$ANCHOR_MASKS") bytes)"

# Pre-flight: dead-flag scan (CLAUDE.md non-negotiable: NEVER invent CLI flags).
# Scans both qat_omega_lagrangian.py and (optionally)
# profile_hessian_per_weight.py invocations for invented flags.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_omega_v2_lagrangian.sh').read()
qat_src = open('experiments/qat_omega_lagrangian.py').read()
prof_src = open('experiments/profile_hessian_per_weight.py').read()
qat_real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', qat_src))
prof_real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', prof_src))
# QAT invocation block
mq = re.search(r'experiments/qat_omega_lagrangian\.py(.*?)(?=\n# Stage|\nlog \"===|\Z)',
               script, re.DOTALL)
assert mq, 'could not locate qat_omega_lagrangian.py invocation'
qat_used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', mq.group(0)))
qat_invented = qat_used - qat_real
if qat_invented:
    print(f'INVENTED FLAGS (qat): {sorted(qat_invented)}', file=sys.stderr); sys.exit(3)
# Profiler invocation block (optional, in Stage 1)
mp = re.search(r'experiments/profile_hessian_per_weight\.py(.*?)(?=\n# Stage|\nlog \"===|\Z)',
               script, re.DOTALL)
if mp:
    prof_used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', mp.group(0)))
    prof_invented = prof_used - prof_real
    if prof_invented:
        print(f'INVENTED FLAGS (profiler): {sorted(prof_invented)}',
              file=sys.stderr); sys.exit(3)
    print(f'OK: qat={len(qat_used)} prof={len(prof_used)} flags all real')
else:
    print(f'OK: qat={len(qat_used)} flags all real (profiler skipped)')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Stage 1 — OPTIONAL: profile per-weight Hessian importance for warm-start.
# Disabled by default (LANE_OMEGA_V2_HESSIAN_INIT=0) because Lagrangian dual
# ascent converges to the KKT optimum from any reasonable init; warm-start
# only speeds up convergence by an estimated 20-30% (council Yousfi).
log "=== Stage 1: per-weight Hessian profile (OPTIONAL warm-start) ==="
HESSIAN_PT=""
if [ "${LANE_OMEGA_V2_HESSIAN_INIT:-0}" = "1" ]; then
    HESSIAN_PT="$LOG_DIR/hessian_per_weight.pt"
    set +e
    "$PYBIN" -u experiments/profile_hessian_per_weight.py \
        --checkpoint "$ANCHOR_RENDERER" \
        --video upstream/videos/0.mkv \
        --masks-mkv "$ANCHOR_MASKS" \
        --poses "$ANCHOR_POSES" \
        --upstream upstream \
        --output "$HESSIAN_PT" \
        --top-k 30 \
        --all-pairs \
        --device cuda \
        --pair-batch 4 2>&1 | tee "$LOG_DIR/profile.log" | tail -20
        PIPE_RC=("${PIPESTATUS[@]}")
    set -e
        if [ "${PIPE_RC[0]}" -ne 0 ]; then
            echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
        fi
    [ -f "$HESSIAN_PT" ] || { echo "FATAL: profiler didn't produce $HESSIAN_PT"; exit 2; }
    log "  hessian profile: $HESSIAN_PT ($(stat -c '%s' "$HESSIAN_PT") bytes)"
else
    log "  Hessian warm-start DISABLED (set LANE_OMEGA_V2_HESSIAN_INIT=1 to enable)"
fi

# Stage 2 — Lagrangian QAT loop (the actual Lane Ω-V2 work).
log "=== Stage 2: Lagrangian QAT (per-weight learnable bit-depth) ==="
QAT_DIR="$LOG_DIR/qat"
mkdir -p "$QAT_DIR"
HESSIAN_FLAG=""
if [ -n "$HESSIAN_PT" ] && [ -f "$HESSIAN_PT" ]; then
    HESSIAN_FLAG="--hessian-init $HESSIAN_PT"
fi
set +e
"$PYBIN" -u experiments/qat_omega_lagrangian.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --video upstream/videos/0.mkv \
    --masks-mkv "$ANCHOR_MASKS" \
    --poses "$ANCHOR_POSES" \
    --upstream upstream \
    --output-dir "$QAT_DIR" \
    --init-bits 8.0 \
    --target-bits 2.5 \
    --lambda-start 0.0 \
    --lambda-end 1.0 \
    --lambda-ramp-start-frac 0.3 \
    --total-epochs 200 \
    --lr 2.5e-6 \
    --bits-lr-scale 0.1 \
    --noise-std 0.5 \
    --seg-weight 100.0 \
    --pose-weight 10.0 \
    --device cuda \
    --seed 1234 \
    --log-every 10 \
    $HESSIAN_FLAG 2>&1 | tee "$LOG_DIR/qat.log" | tail -40
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Stage 3 — OMG1 export of best checkpoint.
# qat_omega_lagrangian.py already exports renderer.bin to $QAT_DIR/renderer.bin
# at the end of Stage 2. We just verify it's there.
log "=== Stage 3: OMG1 export verification ==="
OMEGA_BIN="$QAT_DIR/renderer.bin"
[ -f "$OMEGA_BIN" ] || { echo "FATAL: QAT didn't produce $OMEGA_BIN"; exit 2; }
OMEGA_SIZE=$(stat -c '%s' "$OMEGA_BIN")
log "  Ω-V2 renderer: $OMEGA_BIN ($OMEGA_SIZE bytes)"

# Stage 4 — Build archive (Ω-V2 renderer + Lane A masks + Lane A poses).
# CLAUDE.md non-negotiable: Python zipfile, NOT shell `zip` binary
# (memory: feedback_zip_dep_bootstrap_trap).
log "=== Stage 4: build archive ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$OMEGA_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$ANCHOR_MASKS" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_omega_v2.zip"
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
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo "FATAL: archive empty" >&2; exit 2; }
log "  archive: $ARCHIVE ($ARCHIVE_BYTES bytes)"

# Stage 5 — contest_auth_eval on the EXACT archive that would be submitted.
log "=== Stage 5: contest_auth_eval on Lane Ω-V2 archive [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
set +e
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Sanity-check the auth eval emitted RESULT_JSON (catches silent crashes,
# LANE-B 2026-04-26 cascade pattern).
if ! grep "RESULT_JSON" "$LOG_DIR/auth_eval.log" >/dev/null 2>&1; then
    log "FATAL: auth_eval log has no RESULT_JSON line — eval crashed."
    exit 4
fi

log "=== LANE_OMEGA_V2_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
