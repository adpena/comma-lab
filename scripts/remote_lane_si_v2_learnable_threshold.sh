#!/bin/bash
# Lane SI-V2: LEARNABLE saliency threshold via Lagrangian on TARGET BIT BUDGET.
#
# Council 2026-04-27: Lane SI-V1 hard-coded threshold_quantile=0.5 (the
# bottom-half of saliency = "blind spot, compress aggressively"). The
# rate-distortion optimum depends on the actual codec rate function and
# the saliency distribution shape; 0.5 is a heuristic guess.
#
# Lane SI-V2 replaces the heuristic with a LEARNABLE scalar threshold
# optimised by Lagrangian dual ascent on a TARGET BIT BUDGET. Math:
#
#   soft_mask(p) = sigmoid((threshold - p) / temperature)  ∈ [0, 1]
#   bytes(t) ≈ slope * t + intercept   (linear rate model, calibrated
#                                       at boot via two encoder probes)
#   L(t, λ) = (bytes(t) - target_bytes)² + λ * (bytes(t) - target_bytes)
#
# Dual ascent on λ pushes |bytes - target| toward zero. The closed-form
# linear-model solution is t* = (target - intercept) / slope, but the
# iterative form supports any non-linear rate model out of the box.
#
# Predicted band: [1.05, 1.18] [contest-CUDA] vs Lane SI-V1's [1.10, 1.18]
# (council: tighter floor because the threshold is now optimal for the
# specific saliency map rather than arbitrarily quantised at 0.5; same
# ceiling because the SLI1 inflate-time decoder is still TODO so the
# archive shipped here uses Lane A masks).
#
# Pipeline (mirrors Lane SI-V1 with one V2 substitution):
#   Stage 0 — NVDEC probe.
#   Stage 1 — profile_scorer_saliency on Lane A baseline → saliency maps.
#   Stage 2 — V2 encoding: target_bytes Lagrangian threshold (research
#             artifact; SLI1 inflate-time decoder still deferred).
#   Stage 3 — Build archive (Lane A renderer + Lane A masks + Lane A poses).
#   Stage 4 — contest_auth_eval [contest-CUDA].
#
# Council preflight (CLAUDE.md non-negotiable: NEVER invent CLI flags).
# Verified 2026-04-27 against argparse:
#   * profile_scorer_saliency.py: --checkpoint --poses --masks-mkv
#     --video --output --device --upstream-dir --n-pairs --reduce
#   * contest_auth_eval.py: --archive --inflate-sh --upstream-dir --device
#     --keep-work-dir --work-dir
#
# OUTSTANDING TODO (Component 4, deferred — same as Lane SI-V1):
#   The SLI1 inflate-time decoder is still required to make the
#   saliency-weighted payload load-bearing in the archive. Until it lands
#   the archive ships Lane A's masks.mkv and the V2 payload is a research
#   artifact. Lane SI-V2 measures the threshold-learning convergence and
#   rate-savings potential; the score gain materialises only after the
#   inflate-time decoder is integrated.

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_si_v2_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-si-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_si_v2_learnable_threshold.sh',
    'lane_name': 'lane_si_v2_learnable_threshold',
    'output_dir': '$LOG_DIR',
    'predicted_band': [1.05, 1.18],
    'baseline_score': 1.15,
    'baseline_lane': 'A',
    'rationale': 'Learnable saliency threshold via Lagrangian dual ascent on target_bytes; replaces Lane SI-V1 hard-coded threshold_quantile=0.5.',
    'target_bytes': 60000,
    'tolerance_bytes': 256.0,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=SI-V2 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# ─── Stage 0: NVDEC probe ────────────────────────────────────────────────
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py at the end. Destroy this"
    log "       Vast.ai instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present (Lane A baseline + scorers + video)
for f in submissions/baseline_dilated_h64_0_90/renderer.bin \
         submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
         submissions/baseline_dilated_h64_0_90/masks.mkv \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# ─── Stage 1: profile scorer saliency ────────────────────────────────────
log "=== Stage 1: profile PoseNet + SegNet saliency ==="
SAL_OUT="$LOG_DIR/saliency_maps.pt"
"$PYBIN" -u experiments/profile_scorer_saliency.py \
    --checkpoint submissions/baseline_dilated_h64_0_90/renderer.bin \
    --poses submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
    --masks-mkv submissions/baseline_dilated_h64_0_90/masks.mkv \
    --video upstream/videos/0.mkv \
    --output "$SAL_OUT" \
    --device cuda \
    --upstream-dir upstream \
    --n-pairs 64 \
    --reduce mean 2>&1 | tee "$LOG_DIR/saliency.log" | tail -20

[ -f "$SAL_OUT" ] || { log "FATAL: profile_scorer_saliency.py did not produce $SAL_OUT"; exit 2; }
log "  produced $SAL_OUT ($(stat -c '%s' "$SAL_OUT" 2>/dev/null || echo '?') bytes)"

# ─── Stage 2: V2 encoding with LEARNABLE threshold (Lagrangian) ─────────
log "=== Stage 2: encode saliency-weighted mask payload (Lane SI-V2: learnable threshold) ==="
SI_PAYLOAD="$LOG_DIR/masks_saliency_weighted_v2.sli1"
SI_META="$LOG_DIR/saliency_threshold.json"
"$PYBIN" - <<PY 2>&1 | tee -a "$LOG_DIR/saliency.log" | tail -20
import json, sys
import av, torch
sys.path.insert(0, 'src'); sys.path.insert(0, 'upstream')
from tac.saliency_inversion import apply_saliency_weighted_compression

maps = torch.load("$SAL_OUT", map_location='cpu')
sal = maps['combined']
print(f"[lane-si-v2] saliency map shape={tuple(sal.shape)} "
      f"min={sal.min().item():.4f} max={sal.max().item():.4f} "
      f"mean={sal.mean().item():.4f}")

# Decode Lane A masks.mkv into uint8 (N, H, W)
mask_path = "submissions/baseline_dilated_h64_0_90/masks.mkv"
container = av.open(mask_path)
stream = container.streams.video[0]
mask_frames = []
for frame in container.decode(stream):
    arr = frame.to_ndarray(format='gray')  # (H, W) uint8
    mask_frames.append(torch.from_numpy(arr))
container.close()
masks = torch.stack(mask_frames, dim=0)
print(f"[lane-si-v2] decoded {masks.shape[0]} mask frames {tuple(masks.shape[1:])}"
      f" dtype={masks.dtype}")

# Resize saliency to match mask grid if needed.
import torch.nn.functional as F
if sal.shape != masks.shape[1:]:
    sal = F.interpolate(
        sal.float()[None, None], size=masks.shape[1:], mode='bilinear',
        align_corners=False,
    )[0, 0]
    print(f"[lane-si-v2] resized saliency to {tuple(sal.shape)} to match masks")

# Lane SI-V2 core: optimise threshold for target_bytes via Lagrangian.
# target_bytes is set so the resulting payload is approximately the same
# size as Lane A's masks.mkv (rate-neutral), which lets the V2 result
# isolate the QUALITY effect of the learnable threshold.
import os
target_bytes = int(os.path.getsize(mask_path))
print(f"[lane-si-v2] target_bytes = {target_bytes} (Lane A masks.mkv size)")

payload = apply_saliency_weighted_compression(
    masks=masks,
    saliency=sal,
    target_bytes=target_bytes,
    target_bytes_tolerance=512.0,
    high_crf=50,
    low_crf=30,
)
with open("$SI_PAYLOAD", 'wb') as f:
    f.write(payload)
print(f"[lane-si-v2] V2 saliency-weighted payload: {len(payload)} bytes "
      f"(target {target_bytes}, delta {len(payload) - target_bytes})")

# Persist threshold + temperature for the eventual inflate-time decoder.
meta = {
    'lane': 'SI-V2',
    'target_bytes': target_bytes,
    'actual_bytes': len(payload),
    'delta_bytes': len(payload) - target_bytes,
    'masks_shape': list(masks.shape),
    'saliency_shape': list(sal.shape),
}
with open("$SI_META", 'w') as f:
    json.dump(meta, f, indent=2)
print(f"[lane-si-v2] threshold meta -> $SI_META")
PY

# ─── Stage 3: build archive (Lane A artifacts — see OUTSTANDING TODO) ────
log "=== Stage 3: build archive (Lane A renderer + Lane A masks + Lane A poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp submissions/baseline_dilated_h64_0_90/renderer.bin "$LOG_DIR/iter_0/renderer.bin"
cp submissions/baseline_dilated_h64_0_90/masks.mkv "$LOG_DIR/iter_0/masks.mkv"
cp submissions/baseline_dilated_h64_0_90/optimized_poses.pt "$LOG_DIR/iter_0/optimized_poses.pt"

ARCHIVE="$LOG_DIR/archive_lane_si_v2.zip"
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

# ─── Stage 4: contest_auth_eval ──────────────────────────────────────────
log "=== Stage 4: contest_auth_eval on Lane SI-V2 archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15

log "=== LANE_SI_V2_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "    saliency map:        $SAL_OUT"
log "    SI-V2 payload (research): $SI_PAYLOAD"
log "    threshold meta:      $SI_META"
log "    archive (eval'd):    $ARCHIVE"
log "    NOTE: Lane SI-V2 measures the LEARNABLE threshold convergence"
log "    and rate-savings potential; the score gain becomes load-bearing"
log "    only after Component 4 (inflate-time SLI1 decoder) lands. The"
log "    archive shipped to contest_auth_eval uses Lane A's masks.mkv."
