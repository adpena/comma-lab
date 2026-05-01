#!/bin/bash
# Lane RM — Riemannian SE(3) geometric pose optimization, anchored on
# Lane A's 1.15 [contest-CUDA] floor.
#
# Hypothesis: poses live on the SE(3) Lie group, NOT in flat ℝ⁶.
# Standard Euclidean SGD/Adam on (axis-angle ω, translation t)
# accumulates orthogonality drift on the rotation factor across steps; a
# Riemannian optimiser uses the SE(3) exponential map as the retraction
# so rotations stay in SO(3) by construction. Per Bonnabel (2013) the
# convergence rate matches Euclidean SGD on smooth manifolds, but the
# constant factor improves on the SO(3) submanifold.
#
# Math references (in src/tac/se3.py docstrings):
#   * Absil-Mahony-Sepulchre, *Optimization Algorithms on Matrix
#     Manifolds*, Princeton UP 2008 — manifold + retraction theory.
#   * Boumal, *An Introduction to Optimization on Smooth Manifolds*,
#     Cambridge UP 2023 — Chapter 10 (Riemannian SGD), §10.5 (momentum).
#   * Bonnabel, *Stochastic gradient descent on Riemannian manifolds*,
#     IEEE TAC 58(9), 2013 — convergence rate matches Euclidean SGD.
#   * Sola-Deray-Atchuthan, *A micro Lie theory for state estimation in
#     robotics*, arXiv:1812.01537, 2018 — closed-form SE(3) primitives.
#
# Single variable vs Lane A: --optimizer riemannian-sgd (everything else
# identical to remote_lane_a_pose_tto.sh, including --eval-roundtrip,
# --posetto-noise-std=0.5, --batch-pairs=8, --steps=500, warm-start
# poses, renderer, and masks).
#
# Predicted band: [1.05, 1.15] [contest-CUDA].
# Anchor:         Lane A's 1.15 [contest-CUDA] artifacts at
#                 experiments/results/lane_a_landed/.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_rm_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-rm] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps): every remote run must
# emit provenance.json so a fresh agent can reconstruct the experiment.
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
    'lane_script': 'scripts/remote_lane_rm_riemannian_pose_tto.sh',
    'output_dir': '$LOG_DIR',
    'lane_internal_name': 'lane_rm',
    'optimizer': 'riemannian-sgd',
    'riemannian_momentum': 0.9,
    'manifold': 'SE(3) = SO(3) ⋉ R^3',
    'retraction': 'SE(3) exponential map (Rodrigues for SO(3))',
    'metric': 'left-invariant',
    'anchor_score_baseline': 1.15,
    'anchor_artifacts': 'experiments/results/lane_a_landed/',
    'predicted_band': [1.05, 1.15],
    'delta_from_lane_a': 'optimizer=riemannian-sgd (SE(3) geodesic step)',
    'math_references': [
        'Absil, Mahony, Sepulchre 2008 (Princeton UP) — manifold theory',
        'Boumal 2023 (Cambridge UP) — Riemannian SGD',
        'Bonnabel 2013 (IEEE TAC 58:9) — convergence rate',
        'Sola, Deray, Atchuthan 2018 (arXiv:1812.01537) — SE(3) primitives',
    ],
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=RM gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 NVDEC probe — refuse to spend GPU on a host that cannot run
# upstream/evaluate.py at the end (memory: feedback_vastai_nvdec_host_variation).
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py at the end. Destroy this"
    log "       Vast.ai instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present (Lane A's verified renderer +
# masks + warm-start poses). Lane RM uses Lane A's exact renderer + masks
# so the SE(3) optimiser is the SINGLE variable being tested.
for f in experiments/results/lane_a_landed/iter_0/renderer.bin \
         experiments/results/lane_a_landed/iter_0/masks.mkv \
         experiments/results/lane_a_landed/optimized_poses.pt \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: pose TTO with --optimizer riemannian-sgd, warm-start from Lane A ==="
log "   --gt-poses-path = experiments/results/lane_a_landed/optimized_poses.pt"
log "   --optimizer riemannian-sgd (Lane RM — SE(3) exponential-map retraction)"
log "   --riemannian-momentum 0.9 (Polyak heavy-ball in se(3) coords)"
log "   --eval-roundtrip + --posetto-noise-std=0.5 (Fridrich C1 fixes)"
# Determinism: pin seeds + python hash (CLAUDE.md non-negotiable).
export PYTHONHASHSEED=1234
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint experiments/results/lane_a_landed/iter_0/renderer.bin \
    --masks experiments/results/lane_a_landed/iter_0/masks.mkv \
    --gt-poses-path experiments/results/lane_a_landed/optimized_poses.pt \
    --device cuda \
    --steps 500 \
    --batch-pairs 8 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --optimizer riemannian-sgd \
    --riemannian-momentum 0.9 \
    --pose-mode full-6dof \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Validate: did optimize_poses produce the file?
[ -f "$LOG_DIR/optimized_poses.pt" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.pt"; exit 2; }
RM_BYTES=$(stat -c '%s' "$LOG_DIR/optimized_poses.pt" 2>/dev/null || stat -f '%z' "$LOG_DIR/optimized_poses.pt")
log "  produced optimized_poses.pt (${RM_BYTES} bytes — Lane RM SE(3))"

# Sanity check: the saved poses MUST materialise to a (600, 6) tensor
# (one SE(3) element per pair) AND every rotation factor MUST round-trip
# through exp_map_so3 → log_map_so3 with < 1e-3 axis-angle drift. This
# catches any silent corruption from the optimiser path before we burn
# $0.20 of contest_auth_eval discovering it.
"$PYBIN" -c "
import sys, torch
from tac.se3 import exp_map_so3, log_map_so3
poses = torch.load('$LOG_DIR/optimized_poses.pt', map_location='cpu', weights_only=False)
if not isinstance(poses, torch.Tensor):
    print('FATAL: optimized_poses.pt is not a Tensor; got', type(poses).__name__, file=sys.stderr)
    sys.exit(2)
if poses.shape != (600, 6):
    print(f'FATAL: poses shape {tuple(poses.shape)} != (600, 6)', file=sys.stderr)
    sys.exit(2)
omega = poses[:, 0:3].float()
R = exp_map_so3(omega)
omega_rt = log_map_so3(R)
drift = (omega_rt - omega).abs().max().item()
if drift > 1e-3:
    print(f'FATAL: SE(3) round-trip drift {drift:.2e} > 1e-3', file=sys.stderr)
    sys.exit(2)
print(f'[Lane RM] sanity OK — (600, 6) poses, SE(3) round-trip drift {drift:.2e}')
"

log "=== Stage 2: build NEW archive (Lane A renderer + Lane A masks + Lane RM poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp experiments/results/lane_a_landed/iter_0/renderer.bin "$LOG_DIR/iter_0/renderer.bin"
cp experiments/results/lane_a_landed/iter_0/masks.mkv "$LOG_DIR/iter_0/masks.mkv"
cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_rm.zip"
# Python zipfile, NOT shell `zip` — PyTorch container has no `zip`
# binary (memory: feedback_zip_dep_bootstrap_trap).
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

log "=== Stage 3: contest_auth_eval on Lane RM archive ==="
rm -rf "$LOG_DIR/eval_work"

# Codex F5 fix (2026-04-28): canonical pre-flight for inflate.sh's
# PYTHON_INFLATE env. Without config.env (which sets PYTHON_INFLATE=renderer),
# inflate.sh falls into the ffmpeg path and crashes opening extracted/0.mkv
# (which never exists in a renderer archive). The launcher tarball used to
# silently exclude .env files; that's fixed in scripts/launch_lane_on_vastai.py
# but we also defend in-script so any lane reusing this template is safe
# even if the deploy path regresses.
[ -f submissions/robust_current/config.env ] || {
    log "FATAL: submissions/robust_current/config.env missing -- inflate.sh"
    log "       will not know PYTHON_INFLATE=renderer and contest_auth_eval"
    log "       will crash opening extracted/0.mkv. Re-deploy with the fixed"
    log "       launcher (Codex F5 2026-04-28) which includes .env files."
    exit 3
}
grep -q '^PYTHON_INFLATE=renderer' submissions/robust_current/config.env || {
    log "FATAL: submissions/robust_current/config.env exists but does not set"
    log "       PYTHON_INFLATE=renderer. inflate.sh would call ffmpeg path."
    exit 3
}

"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

log "=== LANE_RM_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
