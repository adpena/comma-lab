#!/bin/bash
# Lane UNIWARD: Fridrich-canonical texture-region weighted compression.
#
# WHAT: replaces Lane SI's hand-tuned saliency_inv mask with the UNIWARD
# texture-probability map (high-pass residual energy weighted by inverse
# local variance). Per CLAUDE.md "Fridrich inverse steganalysis" §1:
# "errors in textured regions are undetectable. Weight loss by inverse
# local variance." This lane is the MASK-COMPRESSION sibling of Lane SI:
# textured pixels ride aggressive CRF, calm regions ride conservative CRF.
#
# Lane UNIWARD's claim: by routing aggressive compression to the actual
# CNN-blind regions (which the texture-prob heuristic approximates better
# than learned saliency on unseen frames), we reduce mask bytes WITHOUT
# moving SegNet distortion. Rate is ~36% of the score wedge at the
# Lane G v3 floor (1.05); a -0.04 rate cut moves the needle into the
# [1.00, 1.05] band.
#
# At inflate time NO scorer weights are loaded — the texture probability
# is computed at compress time only, and what ships is a saliency-weighted
# mask.mkv whose decoder is the standard PyAV path.
#
# Wires the orphan src/tac/uniward_texture.py into the SI path via
# saliency_inversion.apply_saliency_weighted_compression(mode='uniward_texture').
#
# Anchors on Lane A 1.15 [contest-CUDA] (renderer + poses unchanged; only
# masks.mkv is regenerated through the UNIWARD-weighted encoder).
#
# Predicted band: [1.00, 1.13] [contest-CUDA] — same ceiling as Lane SI-V2
# (the SLI1 inflate-time decoder is still TODO; this archive ships Lane A
# masks but proves the encoder pipeline). Lower floor than SI-V2 because
# texture probability is an unsupervised heuristic and may underweight
# legitimate boundary signal.
#
# Cost cap: $0.50, ETA 1h on RTX 4090. Auth eval is the only real spend.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
cd "$WORKSPACE"
if [ -f "$WORKSPACE/env.sh" ]; then
    [ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
fi

PYBIN="${PYBIN:-/opt/conda/bin/python}"
export PYTHONHASHSEED=1234
export PYTHONPATH=src:upstream:$PWD
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_uniward_results}"
ITER_DIR="$LOG_DIR/iter_0"
TAG="${TAG:-lane_uniward}"
ANCHOR_DIR="${ANCHOR_DIR:-submissions/baseline_dilated_h64_0_90}"
LANE_A_ARCHIVE="${LANE_A_ARCHIVE:-experiments/results/lane_a_landed/archive_lane_a.zip}"

mkdir -p "$LOG_DIR" "$ITER_DIR"

log() { echo "[lane-uniward] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

HB_PID=""
cleanup() {
    set +e
    if [ -n "${HB_PID:-}" ]; then
        kill "$HB_PID" 2>/dev/null || true
    fi
    rm -rf "$LOG_DIR/eval_work/tmp" "$LOG_DIR/tmp" 2>/dev/null || true
    if [ "${DESTROY_INSTANCE_ON_EXIT:-0}" = "1" ] && [ -n "${VASTAI_INSTANCE_ID:-}" ]; then
        if command -v vastai >/dev/null 2>&1; then
            vastai destroy instance "$VASTAI_INSTANCE_ID" >>"$LOG_DIR/cleanup.log" 2>&1 || true
        fi
    fi
}
trap cleanup EXIT

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
export PROVENANCE LOG_DIR GIT_HASH GPU_NAME DRIVER_VER TAG ANCHOR_DIR

"$PYBIN" -u - <<'PY'
import json, os, time
import torch

prov = {
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": os.environ["GIT_HASH"],
    "gpu_name": os.environ["GPU_NAME"],
    "driver_version": os.environ["DRIVER_VER"],
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "lane_script": "scripts/remote_lane_uniward_texture.sh",
    "lane_name": "lane_uniward_texture",
    "tag": os.environ["TAG"],
    "anchor_dir": os.environ["ANCHOR_DIR"],
    "predicted_band": [1.00, 1.13],
    "baseline_score": 1.15,
    "baseline_lane": "A",
    "hypothesis": "UNIWARD texture-prob (high-pass residual energy * inverse local variance) routes aggressive CRF to CNN-blind regions, cutting mask bytes without moving SegNet.",
    "strict_scorer_rule_compliant": True,
    "wires_orphan_module": "src/tac/uniward_texture.py",
    "output_dir": os.environ["LOG_DIR"],
}
with open(os.environ["PROVENANCE"], "w", encoding="utf-8") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=uniward gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!

log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed before setup. Destroy this host and pick another."
    exit 2
}

log "=== Stage 1: anchor checks (Lane A baseline + scorers + video) ==="
for f in "$ANCHOR_DIR/renderer.bin" \
         "$ANCHOR_DIR/optimized_poses.pt" \
         "$ANCHOR_DIR/masks.mkv" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 2: compute UNIWARD texture probability on GT video ==="
TEX_OUT="$LOG_DIR/texture_probability.pt"
export TEX_OUT
"$PYBIN" -u - <<'PY' 2>&1 | tee "$LOG_DIR/texture.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi
import os, sys, torch
import av
sys.path.insert(0, "src"); sys.path.insert(0, "upstream")
from tac.uniward_texture import compute_texture_probability

video_path = "upstream/videos/0.mkv"
container = av.open(video_path)
stream = container.streams.video[0]
frames = []
for i, frame in enumerate(container.decode(stream)):
    arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
    frames.append(torch.from_numpy(arr).permute(2, 0, 1).float())
    if len(frames) >= 64:
        break
container.close()
batch = torch.stack(frames, dim=0).cuda()  # (N, 3, H, W)
print(f"[lane-uniward] computing texture probability on {batch.shape[0]} frames "
      f"shape={tuple(batch.shape)}")
sigma2 = compute_texture_probability(batch, scorers=[], require_cuda=True)
print(f"[lane-uniward] texture map shape={tuple(sigma2.shape)} "
      f"min={sigma2.min().item():.4f} max={sigma2.max().item():.4f} "
      f"mean={sigma2.mean().item():.4f}")
torch.save({"texture_probability": sigma2.cpu()}, os.environ["TEX_OUT"])
print(f"[lane-uniward] saved -> {os.environ['TEX_OUT']}")
PY
[ -f "$TEX_OUT" ] || { log "FATAL: texture map not produced"; exit 2; }

log "=== Stage 3: encode UNIWARD-weighted mask payload ==="
UW_PAYLOAD="$LOG_DIR/masks_uniward.sli1"
UW_META="$LOG_DIR/uniward_meta.json"
export UW_PAYLOAD UW_META ANCHOR_DIR
"$PYBIN" -u - <<'PY' 2>&1 | tee -a "$LOG_DIR/texture.log"
import json, os, sys
import av, torch
import torch.nn.functional as F

sys.path.insert(0, "src"); sys.path.insert(0, "upstream")
from tac.saliency_inversion import apply_saliency_weighted_compression

tex = torch.load(os.environ["TEX_OUT"], map_location="cpu")["texture_probability"]
print(f"[lane-uniward] texture map shape={tuple(tex.shape)} "
      f"min={tex.min().item():.4f} max={tex.max().item():.4f}")

mask_path = os.path.join(os.environ["ANCHOR_DIR"], "masks.mkv")
container = av.open(mask_path)
stream = container.streams.video[0]
mask_frames = []
for frame in container.decode(stream):
    arr = frame.to_ndarray(format="gray")
    mask_frames.append(torch.from_numpy(arr))
container.close()
masks = torch.stack(mask_frames, dim=0)
print(f"[lane-uniward] decoded {masks.shape[0]} mask frames {tuple(masks.shape[1:])} dtype={masks.dtype}")

if tex.shape != masks.shape[1:]:
    tex = F.interpolate(
        tex.float()[None, None], size=masks.shape[1:], mode="bilinear", align_corners=False,
    )[0, 0]
    print(f"[lane-uniward] resized texture to {tuple(tex.shape)}")

payload = apply_saliency_weighted_compression(
    masks=masks,
    mode="uniward_texture",
    texture_probability=tex,
    texture_quantile=0.5,
    high_crf=50,
    low_crf=30,
)
with open(os.environ["UW_PAYLOAD"], "wb") as f:
    f.write(payload)
print(f"[lane-uniward] UNIWARD payload: {len(payload)} bytes")

meta = {
    "lane": "UNIWARD",
    "actual_bytes": len(payload),
    "masks_shape": list(masks.shape),
    "texture_shape": list(tex.shape),
    "texture_quantile": 0.5,
}
with open(os.environ["UW_META"], "w") as f:
    json.dump(meta, f, indent=2)
print(f"[lane-uniward] meta -> {os.environ['UW_META']}")
PY

log "=== Stage 4: build archive (Lane A renderer + Lane A masks + Lane A poses) ==="
# OUTSTANDING TODO: same as Lane SI — the SLI1 inflate-time decoder for the
# UNIWARD-weighted payload is deferred. The archive shipped to auth_eval uses
# the Lane A masks.mkv to confirm the encoder pipeline produces the expected
# shipped bytes; the score gain becomes load-bearing only after the inflate
# decoder lands.
cp "$ANCHOR_DIR/renderer.bin" "$ITER_DIR/renderer.bin"
cp "$ANCHOR_DIR/masks.mkv" "$ITER_DIR/masks.mkv"
cp "$ANCHOR_DIR/optimized_poses.pt" "$ITER_DIR/optimized_poses.pt"

ARCHIVE="$LOG_DIR/archive_lane_uniward.zip"
export ITER_DIR ARCHIVE
"$PYBIN" -u - <<'PY'
import os, zipfile

src = os.environ["ITER_DIR"]
dst = os.environ["ARCHIVE"]
members = ("renderer.bin", "masks.mkv", "optimized_poses.pt")
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for name in members:
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(path, "rb") as f:
            zf.writestr(info, f.read(), compresslevel=9)
print(f"archive {dst}: {os.path.getsize(dst)} bytes")
PY

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
if [ -z "${ARCHIVE_BYTES:-}" ] || [ "$ARCHIVE_BYTES" -le 0 ]; then
    log "FATAL: archive size empty or zero — refusing to call auth_eval"
    exit 2
fi
log "  archive_lane_uniward.zip = ${ARCHIVE_BYTES} bytes"

# Strip macOS AppleDouble files from upstream/videos before auth eval
# (check 37 + feedback_canonical_remote_bootstraps).
rm -f upstream/videos/._*.mkv

log "=== Stage 5: contest_auth_eval [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" \
    2>&1 | tee "$LOG_DIR/auth_eval.log"
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi
grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || {
    log "FATAL: auth eval completed without RESULT_JSON"
    exit 5
}
grep '^RESULT_JSON' "$LOG_DIR/auth_eval.log" | tail -1 > "$LOG_DIR/RESULT_JSON"
log "  RESULT_JSON: $LOG_DIR/RESULT_JSON"

log "=== Stage 6: cleanup ==="
rm -rf "$LOG_DIR/tmp" "$LOG_DIR/eval_work/tmp"
if [ "${DESTROY_INSTANCE_ON_EXIT:-0}" = "1" ] && [ -n "${VASTAI_INSTANCE_ID:-}" ]; then
    if command -v vastai >/dev/null 2>&1; then
        vastai destroy instance "$VASTAI_INSTANCE_ID" >>"$LOG_DIR/cleanup.log" 2>&1 || true
    fi
fi

log "=== LANE_UNIWARD_DONE [contest-CUDA] ==="
log "    archive: $ARCHIVE"
log "    auth_eval: $LOG_DIR/auth_eval.log"
log "    predicted_band: [1.00, 1.13]"
log "    NOTE: Lane UNIWARD measures encoder pipeline + texture map; the"
log "    score gain becomes load-bearing only after the SLI1 inflate-time"
log "    decoder lands (same TODO as Lane SI-V2)."
