#!/bin/bash
# Lane SI: Saliency-Inversion compression weighting (Fridrich UNIWARD).
#
# Premise (project_council_shower_thoughts: "saliency inversion"):
#   "Errors hidden in textured regions are undetectable."
# Compress AGGRESSIVELY in scorer blind spots (low saliency); preserve
# quality where the scorer pays attention (high saliency).
#
# Predicted band: [1.10, 1.18] vs Lane A baseline 1.15 [contest-CUDA].
# Reasoning:
#   - Lane A masks.mkv is ~77 KB out of a 338 KB archive (≈23%).
#   - Saliency-weighted mask encoding can shave 30-50% of the mask
#     bytes by spending them only where SegNet/PoseNet is sensitive.
#   - 30-50% of 77 KB = 23-38 KB savings ⇒ rate term -0.015 to -0.025.
#   - Risk: too-aggressive blind-spot CRF could leak into a region the
#     scorer DID quietly care about (CNN blind spot != actually blind).
#   - Conservative band: small positive expectation, with downside if
#     the inverse-saliency mask threshold is mis-tuned (q=0.5 default).
#
# Stages:
#   0. NVDEC probe + provenance + heartbeat
#   1. profile PoseNet + SegNet pixel saliency on the source video
#   2. encode the saliency map into the experimental mask payload
#      (PRINTED ONLY: actual archive uses Lane A masks.mkv until the
#      inflate-time decoder lands — see "OUTSTANDING TODO" below)
#   3. build archive (Lane A renderer + Lane A masks.mkv + Lane A poses)
#   4. contest_auth_eval on the archive
#
# OUTSTANDING TODO (Component 4, deferred):
#   The saliency-weighted payload format ("SLI1") needs an inflate-time
#   decoder to recombine the two CRF streams. inflate_renderer.py
#   currently expects a single masks.mkv. Until that decoder lands the
#   archive ships with the Lane A masks.mkv; Stage 2 here only emits
#   the saliency map + payload as research artifacts so v2 can compare.
#   (Component 4 design note: ship the boolean region mask in the
#   archive — ~hundreds of bytes RLE+zlib — and patch
#   inflate_renderer.py to dispatch on the SLI1 magic header. NOT done
#   yet because it would require touching the strict-scorer-rule
#   decoder path; council review required first.)

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_si_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-si] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md feedback_canonical_remote_bootstraps)
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
    'lane_script': 'scripts/remote_lane_si_saliency_inversion.sh',
    'output_dir': '$LOG_DIR',
    'predicted_band': [1.10, 1.18],
    'baseline_score': 1.15,
    'baseline_lane': 'A',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=SI gpu=$GPU" >> "$HEARTBEAT"
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
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

[ -f "$SAL_OUT" ] || { log "FATAL: profile_scorer_saliency.py did not produce $SAL_OUT"; exit 2; }
log "  produced $SAL_OUT ($(stat -c '%s' "$SAL_OUT" 2>/dev/null || echo '?') bytes)"

# ─── Stage 2: experimental saliency-weighted mask encoding ───────────────
log "=== Stage 2: encode saliency-weighted mask payload (research artifact) ==="
SI_PAYLOAD="$LOG_DIR/masks_saliency_weighted.sli1"
"$PYBIN" - <<PY 2>&1 | tee -a "$LOG_DIR/saliency.log" | tail -10
import sys, av, torch
sys.path.insert(0, 'src'); sys.path.insert(0, 'upstream')
from tac.saliency_inversion import (
    compute_inverse_saliency_mask, apply_saliency_weighted_compression,
)

maps = torch.load("$SAL_OUT", map_location='cpu')
sal = maps['combined']
sal_inv = compute_inverse_saliency_mask(sal, threshold_quantile=0.5)
print(f"[lane-si] inverse-saliency mask: {sal_inv.sum().item()}/{sal_inv.numel()} pixels"
      f" ({100*sal_inv.float().mean().item():.1f}%) marked as blind-spot")

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
print(f"[lane-si] decoded {masks.shape[0]} mask frames {tuple(masks.shape[1:])}"
      f" dtype={masks.dtype}")

# If mask grid != saliency grid, downsample saliency mask to match.
if sal_inv.shape != masks.shape[1:]:
    import torch.nn.functional as F
    sal_inv = F.interpolate(
        sal_inv.float()[None, None], size=masks.shape[1:], mode='nearest',
    )[0, 0].bool()
    print(f"[lane-si] downsampled saliency mask to {tuple(sal_inv.shape)} to match masks")

payload = apply_saliency_weighted_compression(
    masks=masks, saliency_inv=sal_inv, high_crf=50, low_crf=30,
)
with open("$SI_PAYLOAD", 'wb') as f:
    f.write(payload)
print(f"[lane-si] saliency-weighted payload: {len(payload)} bytes "
      f"(vs Lane A masks.mkv = {torch.tensor(__import__('os').path.getsize(mask_path)).item()} bytes)")
PY

# ─── Stage 3: build archive (Lane A artifacts — see OUTSTANDING TODO) ────
# Until the inflate-time SLI1 decoder lands, the archive uses the Lane A
# masks.mkv. Stage 2's payload is preserved as a research artifact for
# Lane SI v2 (which adds the inflate-time decoder).
log "=== Stage 3: build archive (Lane A renderer + Lane A masks + Lane A poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp submissions/baseline_dilated_h64_0_90/renderer.bin "$LOG_DIR/iter_0/renderer.bin"
cp submissions/baseline_dilated_h64_0_90/masks.mkv "$LOG_DIR/iter_0/masks.mkv"
cp submissions/baseline_dilated_h64_0_90/optimized_poses.pt "$LOG_DIR/iter_0/optimized_poses.pt"

ARCHIVE="$LOG_DIR/archive_lane_si.zip"
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
log "=== Stage 4: contest_auth_eval on Lane SI archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

log "=== LANE_SI_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "    saliency map: $SAL_OUT"
log "    SI payload (research):    $SI_PAYLOAD"
log "    archive (eval'd):         $ARCHIVE"
log "    NOTE: SI v1 measures the saliency map quality; the SLI1"
log "    payload becomes load-bearing only after Component 4 (inflate-"
log "    time decoder) lands. See script header OUTSTANDING TODO."
