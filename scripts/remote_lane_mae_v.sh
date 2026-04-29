#!/bin/bash
# Lane MAE-V: Masked Autoencoder Variant — random patch masking on input
# masks during training with a learnable Gumbel-softmax categorical mask
# token. Anchored on Lane G v3 1.05 [contest-CUDA].
#
# Mechanism: src/tac/mae_mask_aug.py drops MAEMaskAugmenter into the per-
# step pair construction. A fraction (mask_ratio=0.25) of input mask
# patches (patch_size=16) are replaced by samples from a learnable 5-class
# categorical token. Eval-mode is a strict passthrough so the inference
# distribution exactly matches what the contest scorer sees — only
# training-time input distribution shifts.
#
# Composes with Lane SAUG-V2 (orthogonal: SAUG perturbs input numerics,
# MAE-V perturbs patch occupancy) and Lane J-JBL (orthogonal: JBL is a
# loss family). Predicted band [0.85, 1.10] per Cosmos research synthesis
# (.omx/research/jack_skunkworks_segnet_rate_research_20260428.md).
#
# Cost cap: $4 (RTX 4090 @ $0.25/hr × ~16h = $4.00). Hard kill if exceeded
# via the per-instance verify watchdog (scripts/verify_vast_instances.py).
#
# CLAUDE.md compliance:
#   * eval_roundtrip remains True (inherited from Lane G v3 anchor profile).
#   * No scorer at inflate (augmenter is compress-time only).
#   * Stage 4 contest_auth_eval [contest-CUDA] writes RESULT_JSON.
#   * Provenance.json + heartbeat.log per feedback_canonical_remote_bootstraps.
#   * AppleDouble cleanup + ARCHIVE_BYTES guard before auth eval.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
cd "$WORKSPACE"
if [ -f "$WORKSPACE/env.sh" ]; then
    [ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
fi

export PYTHONHASHSEED=1234
export PYTHONPATH=src:upstream:$PWD
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_mae_v_results}"
TRAIN_DIR="$LOG_DIR/train"
EXTRACT_DIR="$LOG_DIR/extracted_lane_g_v3"
ITER_DIR="$LOG_DIR/iter_0"
TAG="${TAG:-lane_mae_v}"
# Anchor on Lane G v3's verified 1.05 [contest-CUDA] archive: borrow its
# masks + optimized poses, ship a NEW renderer trained with MAE-V augmenter.
LANE_G_V3_ARCHIVE="${LANE_G_V3_ARCHIVE:-experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip}"
if [ ! -f "$LANE_G_V3_ARCHIVE" ] && [ -f experiments/results/lane_a_landed/archive_lane_a.zip ]; then
    LANE_G_V3_ARCHIVE="experiments/results/lane_a_landed/archive_lane_a.zip"
fi

mkdir -p "$LOG_DIR" "$TRAIN_DIR" "$EXTRACT_DIR" "$ITER_DIR"

log() { echo "[lane-mae-v] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
export PROVENANCE LOG_DIR GIT_HASH GPU_NAME DRIVER_VER TAG LANE_G_V3_ARCHIVE

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
    "lane_script": "scripts/remote_lane_mae_v.sh",
    "lane_name": "lane_mae_v",
    "tag": os.environ["TAG"],
    "profile": "mae_v_dilated_h64",
    "anchor_archive": os.environ["LANE_G_V3_ARCHIVE"],
    "anchor_score_baseline": 1.05,
    "predicted_band": [0.85, 1.10],
    "hypothesis": "MAE-style patch masking with learnable Gumbel-softmax token forces sparser, more compressible representations; orthogonal to Lane SAUG-V2 and Lane J-JBL.",
    "strict_scorer_rule_compliant": True,
    "output_dir": os.environ["LOG_DIR"],
    "cost_cap_usd": 4.0,
}
with open(os.environ["PROVENANCE"], "w", encoding="utf-8") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=mae-v gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!

log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed before setup. Destroy this host and pick another."
    exit 2
}

log "=== Stage 1: extract Lane G v3 archive anchor (renderer/masks/poses) ==="
for f in "$LANE_G_V3_ARCHIVE" \
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

src = os.environ["LANE_G_V3_ARCHIVE"]
dst = os.environ["EXTRACT_DIR"]
required = {"renderer.bin", "masks.mkv", "optimized_poses.pt"}
with zipfile.ZipFile(src) as zf:
    names = set(zf.namelist())
    missing = required - names
    if missing:
        print(f"FATAL: Lane G v3 archive missing {sorted(missing)}", file=sys.stderr)
        sys.exit(2)
    for name in sorted(required):
        zf.extract(name, dst)
print(f"extracted {sorted(required)} from {src} to {dst}")
PY
log "  anchor renderer.bin: $(stat -c '%s' "$EXTRACT_DIR/renderer.bin") bytes"
log "  anchor masks.mkv: $(stat -c '%s' "$EXTRACT_DIR/masks.mkv") bytes"
log "  anchor optimized_poses.pt: $(stat -c '%s' "$EXTRACT_DIR/optimized_poses.pt") bytes"

log "=== Stage 2: train_renderer.py --profile mae_v_dilated_h64 --use-mae-mask-aug ==="
python3 -u src/tac/experiments/train_renderer.py \
    --profile mae_v_dilated_h64 \
    --use-mae-mask-aug \
    --mae-mask-ratio 0.25 \
    --mae-patch-size 16 \
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

log "=== Stage 3: build archive (NEW renderer + Lane G v3 anchor masks + poses) ==="
cp "$EXTRACT_DIR/masks.mkv" "$ITER_DIR/masks.mkv"
cp "$EXTRACT_DIR/optimized_poses.pt" "$ITER_DIR/optimized_poses.pt"

# AppleDouble cleanup (macOS dotfiles ._* leak through scp/tarball deploys
# and inflate archive size while contributing zero useful bytes). Mirror
# the cleanup pattern from setup_full ._* purge (commit 0380a869).
find "$ITER_DIR" -name '._*' -delete 2>/dev/null || true

ARCHIVE="$LOG_DIR/archive_lane_mae_v.zip"
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

# ARCHIVE_BYTES guard — fail loud if the archive landed empty or absurdly
# large (catches the LANE-B trap where empty ARCHIVE_BYTES crashed eval).
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")
if [ -z "$ARCHIVE_BYTES" ] || [ "$ARCHIVE_BYTES" -lt 50000 ] || [ "$ARCHIVE_BYTES" -gt 2000000 ]; then
    log "FATAL: ARCHIVE_BYTES=$ARCHIVE_BYTES out of plausible range [50k, 2M]"
    exit 4
fi
log "  archive bytes: $ARCHIVE_BYTES"

log "=== Stage 4: CUDA auth eval [contest-CUDA] writing RESULT_JSON ==="
rm -rf "$LOG_DIR/eval_work"
python3 -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" \
    2>&1 | tee "$LOG_DIR/auth_eval.log"
grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || {
    log "FATAL: auth eval completed without RESULT_JSON"
    exit 5
}
grep '^RESULT_JSON' "$LOG_DIR/auth_eval.log" | tail -1 > "$LOG_DIR/RESULT_JSON"
log "  RESULT_JSON: $LOG_DIR/RESULT_JSON"

log "=== Stage 5: cleanup ==="
rm -rf "$LOG_DIR/tmp" "$LOG_DIR/eval_work/tmp"
if [ "${DESTROY_INSTANCE_ON_EXIT:-0}" = "1" ] && [ -n "${VASTAI_INSTANCE_ID:-}" ]; then
    if command -v vastai >/dev/null 2>&1; then
        vastai destroy instance "$VASTAI_INSTANCE_ID" >>"$LOG_DIR/cleanup.log" 2>&1 || true
    fi
fi

log "=== LANE_MAE_V_DONE [contest-CUDA] ==="
log "    archive: $ARCHIVE"
log "    auth_eval: $LOG_DIR/auth_eval.log"
log "    predicted_band: [0.85, 1.10]"
