#!/bin/bash
# Lane HM: analytical road-plane homography motion module.
#
# Replaces the renderer's learned-CNN motion submodule with the orphan
# src/tac/contrib/homography_motion.py:HomographyMotionModule — a
# zero-parameter FOE-centered perspective-zoom field. Saves ~80K
# motion-module params (the renderer head is unchanged from PROVEN_BASELINE).
#
# Wires the orphan into renderer.build_renderer via the new native dispatch
# `motion_type="homography_analytical"` (replaces the prior monkey-patch).
#
# Anchors on PROVEN_BASELINE (NOT Lane A — Lane HM is a from-scratch
# retrain because the motion module shape changes invalidate the Lane A
# checkpoint).
#
# Predicted band: [1.30, 2.20] [contest-CUDA]. Wide because:
#   - Floor case (1.30): homography captures most of road-scene motion;
#     the saved 80K motion params translate to FP4 byte savings.
#   - Ceiling case (2.20): homography misses per-class flow signal the
#     CNN motion module captures (vehicles, pedestrians, ego-motion noise).
#
# Cost cap: $3, ETA 8h on RTX 4090 (full PROVEN_BASELINE training).
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
cd "$WORKSPACE"
if [ -f "$WORKSPACE/env.sh" ]; then
    [ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
fi

export PYTHONHASHSEED=1234
export PYTHONPATH=src:upstream:$PWD
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_hm_results}"
TRAIN_DIR="$LOG_DIR/train"
EXTRACT_DIR="$LOG_DIR/extracted_lane_a"
ITER_DIR="$LOG_DIR/iter_0"
TAG="${TAG:-lane_hm}"
LANE_A_ARCHIVE="${LANE_A_ARCHIVE:-experiments/results/lane_a_landed/archive_lane_a.zip}"
if [ ! -f "$LANE_A_ARCHIVE" ] && [ -f submissions/robust_current/archive.zip ]; then
    LANE_A_ARCHIVE="submissions/robust_current/archive.zip"
fi

mkdir -p "$LOG_DIR" "$TRAIN_DIR" "$EXTRACT_DIR" "$ITER_DIR"

log() { echo "[lane-hm] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
export PROVENANCE LOG_DIR GIT_HASH GPU_NAME DRIVER_VER TAG LANE_A_ARCHIVE

python3 -u - <<'PY'
import json
import os
import time

import torch

prov = {
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": os.environ["GIT_HASH"],
    "gpu_name": os.environ["GPU_NAME"],
    "driver_version": os.environ["DRIVER_VER"],
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "lane_script": "scripts/remote_lane_hm.sh",
    "lane_name": "lane_hm",
    "tag": os.environ["TAG"],
    "profile": "hm_dilated_h64",
    "anchor_archive": os.environ["LANE_A_ARCHIVE"],
    "predicted_band": [1.30, 2.20],
    "hypothesis": "Analytical FOE-centered homography motion replaces 80K-param learned CNN motion; bytes saved + geometric prior aligned with Fridrich texture-first inverse-steganalysis.",
    "wires_orphan_module": "src/tac/contrib/homography_motion.py",
    "strict_scorer_rule_compliant": True,
    "output_dir": os.environ["LOG_DIR"],
}
with open(os.environ["PROVENANCE"], "w", encoding="utf-8") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=hm gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!

log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed before setup. Destroy this host and pick another."
    exit 2
}

log "=== Stage 1: extract Lane A archive anchor checkpoint ==="
for f in "$LANE_A_ARCHIVE" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
export EXTRACT_DIR
python3 -u - <<'PY'
import os
import sys
import zipfile

src = os.environ["LANE_A_ARCHIVE"]
dst = os.environ["EXTRACT_DIR"]
required = {"renderer.bin", "masks.mkv", "optimized_poses.pt"}
with zipfile.ZipFile(src) as zf:
    names = set(zf.namelist())
    missing = required - names
    if missing:
        print(f"FATAL: Lane A archive missing {sorted(missing)}", file=sys.stderr)
        sys.exit(2)
    for name in sorted(required):
        zf.extract(name, dst)
print(f"extracted {sorted(required)} from {src} to {dst}")
PY
log "  anchor renderer.bin: $(stat -c '%s' "$EXTRACT_DIR/renderer.bin") bytes"
log "  anchor masks.mkv: $(stat -c '%s' "$EXTRACT_DIR/masks.mkv") bytes"
log "  anchor optimized_poses.pt: $(stat -c '%s' "$EXTRACT_DIR/optimized_poses.pt") bytes"

log "=== Stage 2: train_renderer.py --profile hm_dilated_h64 (motion_type=homography_analytical) ==="
python3 -u src/tac/experiments/train_renderer.py \
    --profile hm_dilated_h64 \
    --tag "$TAG" \
    --device cuda \
    --precomputed experiments/precomputed_local \
    --output-dir "$TRAIN_DIR" \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log"

BEST_FP32="$TRAIN_DIR/renderer_${TAG}_best_fp32.pt"
[ -f "$BEST_FP32" ] || {
    echo "FATAL: train_renderer did not produce $BEST_FP32" >&2
    ls -la "$TRAIN_DIR" >&2
    exit 3
}

RENDERER_BIN="$ITER_DIR/renderer.bin"
export BEST_FP32 RENDERER_BIN
python3 -u - <<'PY'
import os
import sys

import torch

sys.path.insert(0, "src")
from tac.renderer import build_renderer
from tac.renderer_export import export_asymmetric_checkpoint_fp4

payload = torch.load(os.environ["BEST_FP32"], map_location="cpu", weights_only=False)
state = payload["model_state_dict"]
arch = payload["__meta__"]
model = build_renderer(
    num_classes=5,
    embed_dim=arch["embed_dim"],
    base_ch=arch["base_ch"],
    mid_ch=arch["mid_ch"],
    motion_hidden=arch["motion_hidden"],
    depth=arch["depth"],
    blend_mode=arch.get("blend_mode", "scalar"),
    noise_mode=arch.get("noise_mode", "deterministic"),
    motion_type=arch.get("motion_type", "learned_cnn"),
    use_zoom_flow=arch["use_zoom_flow"],
    use_dsconv=arch["use_dsconv"],
    use_ghost=arch.get("use_ghost", False),
    padding_mode=arch["padding_mode"],
    use_dilation=arch["use_dilation"],
    pose_dim=arch.get("pose_dim", 0) or 0,
)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
nbytes = export_asymmetric_checkpoint_fp4(
    model,
    os.environ["RENDERER_BIN"],
    codebook_name=arch.get("fp4_codebook", "default"),
    robust_scale=arch.get("fp4_robust_scale", False),
)
print(f"exported FP4A renderer.bin: {os.environ['RENDERER_BIN']} ({nbytes} bytes)")
PY

log "=== Stage 3: pose TTO on best checkpoint ==="
python3 -u experiments/optimize_poses.py \
    --checkpoint "$RENDERER_BIN" \
    --masks "$EXTRACT_DIR/masks.mkv" \
    --gt-poses-path "$EXTRACT_DIR/optimized_poses.pt" \
    --device cuda \
    --steps "${POSE_TTO_STEPS:-500}" \
    --batch-pairs "${POSE_TTO_BATCH_PAIRS:-8}" \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --output-dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/optimize_poses.log"
[ -f "$LOG_DIR/optimized_poses.pt" ] || { echo "FATAL: pose TTO did not produce optimized_poses.pt" >&2; exit 4; }

log "=== Stage 4: build archive ==="
cp "$EXTRACT_DIR/masks.mkv" "$ITER_DIR/masks.mkv"
cp "$LOG_DIR/optimized_poses.pt" "$ITER_DIR/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_hm.zip"
export ITER_DIR ARCHIVE
python3 -u - <<'PY'
import os
import zipfile

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

log "=== Stage 5: CUDA auth eval writing RESULT_JSON ==="
rm -rf "$LOG_DIR/eval_work"
python3 -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" \
    2>&1 | tee "$LOG_DIR/auth_eval.log"
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

log "=== LANE_HM_DONE [contest-CUDA] ==="
log "    archive: $ARCHIVE"
log "    auth_eval: $LOG_DIR/auth_eval.log"
log "    predicted_band: [1.30, 2.20]"
