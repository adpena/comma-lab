#!/bin/bash
# Lane WC: Cosmos Curator soft-DTW outlier weighting.
#
# Replaces Lane W's circular loss-based hard-pair signal with independent
# pair-typicality from SegNet penultimate feature geometry:
#   SegNet features -> PCA-3 -> TS-KMeans -> soft-DTW outlier score -> weights.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/results/lane_wc}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-43200}"
START_TS="$(date +%s)"

cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export PYTHONPATH="src:upstream:${PYTHONPATH:-}"
mkdir -p "$LOG_DIR"

log() { echo "[lane-wc] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

cost_guard() {
    now="$(date +%s)"
    elapsed=$((now - START_TS))
    if [ "$elapsed" -gt "$MAX_RUNTIME_SECONDS" ]; then
        log "FATAL: hard runtime cap exceeded: ${elapsed}s > ${MAX_RUNTIME_SECONDS}s"
        exit 70
    fi
}

HEARTBEAT="$LOG_DIR/heartbeat.log"
( while true; do
    GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')"
    echo "[$(date -u +%FT%TZ)] lane=WC gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe.
cost_guard
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed. Destroy this host and choose another."
    exit 2
}

# Stage 0b: self-bootstrap uv (CLAUDE.md non-negotiable, PCC5).
# Fresh Vast.ai instances don't have uv on PATH. The canonical helper is
# scripts/ensure_remote_uv.sh — it idempotently installs uv and symlinks
# it into /usr/local/bin so subprocesses inherit it. Cost of skipping
# this: ~$0.30 + 5-10 min wasted per dispatch (memory: feedback_uv_not_on_path_vast_instance_20260501).
log "=== Stage 0b: uv bootstrap ==="
bash "$WORKSPACE/scripts/ensure_remote_uv.sh" --symlink-system >/dev/null || {
    log "FATAL: uv bootstrap failed."
    exit 3
}

# Stage 1: code parity.
cost_guard
# CODE PARITY: launcher tarball is authoritative — do NOT git reset --hard.
# Doing so wipes local-only anchor files (archive_lane_a.zip, baseline dirs,
# etc.) that the launcher just SCP'd. The tarball IS the parity mechanism.
# (memory: feedback_git_reset_nukes_anchors_20260429)
python3 -u -m pip install -e .

# Stage 2: extract Lane A archive as anchor.
cost_guard
log "=== Stage 2: extract Lane A archive as anchor ==="
ANCHOR_ARCHIVE="${ANCHOR_ARCHIVE:-experiments/results/lane_a_landed/archive_lane_a.zip}"
ANCHOR_DIR="$LOG_DIR/anchor"
rm -rf "$ANCHOR_DIR"
mkdir -p "$ANCHOR_DIR"
python3 -u - <<PY
from pathlib import Path
import zipfile

archive = Path("$ANCHOR_ARCHIVE")
out = Path("$ANCHOR_DIR")
if not archive.is_file():
    raise SystemExit(f"FATAL: missing Lane A archive: {archive}")
with zipfile.ZipFile(archive) as zf:
    zf.extractall(out)
for name in ("renderer.bin", "masks.mkv", "optimized_poses.pt"):
    p = out / name
    if not p.is_file():
        raise SystemExit(f"FATAL: Lane A archive missing {name}")
    print(f"{name}: {p.stat().st_size} bytes")
PY

ANCHOR_RENDERER="$ANCHOR_DIR/renderer.bin"
ANCHOR_MASKS="$ANCHOR_DIR/masks.mkv"
ANCHOR_POSES="$ANCHOR_DIR/optimized_poses.pt"
GT_VIDEO="${GT_VIDEO:-upstream/videos/0.mkv}"
SEGNET_WEIGHTS="upstream/models/segnet.safetensors"
POSENET_WEIGHTS="upstream/models/posenet.safetensors"
for f in "$ANCHOR_RENDERER" "$ANCHOR_MASKS" "$ANCHOR_POSES" \
         "$GT_VIDEO" "$SEGNET_WEIGHTS" "$POSENET_WEIGHTS"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Stage 3: extract SegNet penultimate features for all 600 pairs.
cost_guard
log "=== Stage 3: extract SegNet penultimate features ==="
FEATURES="$LOG_DIR/features.pt"
export LANE_WC_FEATURES="$FEATURES"
python3 -u - <<'PY'
from pathlib import Path
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, "src")
sys.path.insert(0, "upstream")

from modules import SegNet
from tac.data import decode_video, pair_from_frames, pair_start_indices

device = torch.device("cuda")
video = Path("upstream/videos/0.mkv")
segnet_path = Path("upstream/models/segnet.safetensors")
out_path = Path(os.environ["LANE_WC_FEATURES"])

segnet = SegNet().eval().to(device)
segnet.load_state_dict(load_file(str(segnet_path), device="cpu"))
for p in segnet.parameters():
    p.requires_grad = False

frames = decode_video(video)[:1200]
starts = pair_start_indices(len(frames))
if len(starts) != 600:
    raise SystemExit(f"FATAL: expected 600 pairs, got {len(starts)}")

target = getattr(segnet, "segmentation_head", None)
if target is None:
    raise SystemExit("FATAL: SegNet has no segmentation_head for penultimate hook")

features = []
for pair_idx, start in enumerate(starts):
    captured = []

    def hook(_module, inputs, _output):
        feat = inputs[0]
        pooled = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        captured.append(pooled.detach().cpu())

    handle = target.register_forward_hook(hook)
    pair_hwc = pair_from_frames(frames, start)
    pair_chw = pair_hwc.permute(0, 1, 4, 2, 3).contiguous().float().to(device)
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(pair_chw)
        _ = segnet(seg_in)
    handle.remove()
    if not captured:
        raise SystemExit(f"FATAL: hook captured no features for pair {pair_idx}")
    features.append(captured[0].squeeze(0))

features_t = torch.stack(features).contiguous()
if features_t.shape[0] != 600:
    raise SystemExit(f"FATAL: expected features (600,D), got {tuple(features_t.shape)}")
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(features_t, out_path)
print(f"saved {tuple(features_t.shape)} to {out_path}")
PY
[ -f "$FEATURES" ] || { echo "FATAL: missing $FEATURES" >&2; exit 2; }

# Stage 4: fit Curator outlier weights.
cost_guard
log "=== Stage 4: fit_curator_outlier_weights.py ==="
set +e
python3 -u experiments/fit_curator_outlier_weights.py \
    --segnet-features "$FEATURES" \
    --n-pairs 600 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/fit_curator_outlier_weights.log"
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi
PAIR_WEIGHTS="$LOG_DIR/pair_weights.pt"
SCORER_STATE="$LOG_DIR/curator_outlier_scorer.pt"
[ -f "$PAIR_WEIGHTS" ] || { echo "FATAL: missing $PAIR_WEIGHTS" >&2; exit 2; }
[ -f "$SCORER_STATE" ] || { echo "FATAL: missing $SCORER_STATE" >&2; exit 2; }

# Stage 5: train renderer with independent pair weights.
#
# Round 11 Finding 1 fix (2026-04-28): NEVER pass renderer.bin (ASYM/FP4A
# binary format) to --resume-from. train_renderer.py expects a torch.save
# pickle (training_state_*.pt OR renderer_*_best_fp32.pt) and would crash
# on torch.load of the raw quantised binary.
#
# Operator must supply LANE_A_FLOAT_CHECKPOINT pointing at a fp32 PyTorch
# checkpoint (e.g., a renderer_lane_a_*_best_fp32.pt produced by the Lane
# A training run). If unset, the lane fails loud here BEFORE Stage 2/3
# work is wasted — no silent fall-through to a from-scratch run.
cost_guard
log "=== Stage 5: train_renderer.py --pair-weights-path ==="
LANE_A_FLOAT_CHECKPOINT="${LANE_A_FLOAT_CHECKPOINT:-}"
if [ -z "$LANE_A_FLOAT_CHECKPOINT" ]; then
    echo "FATAL: LANE_A_FLOAT_CHECKPOINT env var unset." >&2
    echo "  Lane WC requires a fp32 PyTorch checkpoint to warm-start from." >&2
    echo "  The Lane A archive's renderer.bin is in ASYM/FP4A binary format" >&2
    echo "  and CANNOT be loaded by train_renderer.py --resume-from." >&2
    echo "  Set LANE_A_FLOAT_CHECKPOINT=/path/to/renderer_lane_a_*_best_fp32.pt" >&2
    echo "  (typically downloaded from the original Lane A training run) and" >&2
    echo "  re-launch." >&2
    exit 3
fi
if [ ! -f "$LANE_A_FLOAT_CHECKPOINT" ]; then
    echo "FATAL: LANE_A_FLOAT_CHECKPOINT does not exist: $LANE_A_FLOAT_CHECKPOINT" >&2
    exit 3
fi
# Magic-byte sanity: refuse a renderer .bin even if the env var was wired
# at the wrong path (defence-in-depth — train_renderer.py also checks).
_magic="$(head -c 4 "$LANE_A_FLOAT_CHECKPOINT" 2>/dev/null || true)"
case "$_magic" in
    FP4A|ASYM|DPSM|I4LZ|CCh1|C3R1|SCv1|OMG1)
        echo "FATAL: LANE_A_FLOAT_CHECKPOINT looks like a renderer .bin (magic=$_magic)." >&2
        echo "  Pass a float checkpoint (training_state_*.pt or renderer_*_best_fp32.pt)." >&2
        exit 3
        ;;
esac
log "Lane WC warm-start from fp32 checkpoint: $LANE_A_FLOAT_CHECKPOINT"
TRAIN_OUT="$LOG_DIR/train"
mkdir -p "$TRAIN_OUT"
set +e
python3 -u -m tac.experiments.train_renderer \
    --profile wc_dilated_h64 \
    --tag lane_wc_curator_outlier \
    --output-dir "$TRAIN_OUT" \
    --device cuda \
    --resume-from "$LANE_A_FLOAT_CHECKPOINT" \
    --use-self-compress-codec \
    --pair-weights-path "$PAIR_WEIGHTS" \
    --epochs 500 \
    --lr 5e-5 \
    --no-auth-eval-on-best 2>&1 | tee "$LOG_DIR/train.log" | tail -40
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

BEST_FP32="$(ls -t "$TRAIN_OUT"/renderer_*_best_fp32.pt 2>/dev/null | head -1)"
[ -n "$BEST_FP32" ] && [ -f "$BEST_FP32" ] || {
    echo "FATAL: training produced no renderer_*_best_fp32.pt" >&2
    exit 2
}
log "best checkpoint: $BEST_FP32"

# Stage 6: Pose TTO on best checkpoint.
cost_guard
log "=== Stage 6: Pose TTO on best checkpoint ==="
POSE_OUT="$LOG_DIR/pose_tto"
mkdir -p "$POSE_OUT"
set +e
python3 -u experiments/optimize_poses.py \
    --checkpoint "$BEST_FP32" \
    --device cuda \
    --n-frames 1200 \
    --steps 500 \
    --lr 0.01 \
    --batch-pairs 50 \
    --masks "$ANCHOR_MASKS" \
    --gt-poses-path "$ANCHOR_POSES" \
    --video "$GT_VIDEO" \
    --upstream upstream \
    --output-dir "$POSE_OUT" \
    --eval-roundtrip 2>&1 | tee "$LOG_DIR/pose_tto.log" | tail -40
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi
OPTIMIZED_POSES="$POSE_OUT/optimized_poses.pt"
[ -f "$OPTIMIZED_POSES" ] || { echo "FATAL: Pose TTO produced no optimized_poses.pt" >&2; exit 2; }

# Stage 7: build archive.
cost_guard
log "=== Stage 7: build archive ==="
ITER_DIR="$LOG_DIR/iter_0"
mkdir -p "$ITER_DIR"
SCV1_BIN="$ITER_DIR/renderer.bin"
python3 -u - <<PY
from pathlib import Path
import torch

from tac.renderer import AsymmetricPairGenerator
from tac.renderer_export import export_self_compressed_renderer
from tac.self_compress import swap_renderer_convs_with_self_compress

ckpt = "$BEST_FP32"
out = Path("$SCV1_BIN")
state = torch.load(ckpt, map_location="cpu", weights_only=False)
sd = state.get("model_state_dict") or state.get("state_dict") or state.get("model") or state
meta = state.get("__meta__", {}) or state.get("arch_meta", {}) or {}
model = AsymmetricPairGenerator(
    num_classes=meta.get("num_classes", 5),
    embed_dim=meta.get("embed_dim", 6),
    base_ch=meta.get("base_ch", 36),
    mid_ch=meta.get("mid_ch", 60),
    motion_hidden=meta.get("motion_hidden", 32),
    depth=meta.get("depth", 1),
    pose_dim=meta.get("pose_dim", 6),
    use_dsconv=meta.get("use_dsconv", False),
    padding_mode=meta.get("padding_mode", "zeros"),
    use_dilation=meta.get("use_dilation", False),
    use_zoom_flow=meta.get("use_zoom_flow", True),
)
swap_renderer_convs_with_self_compress(model, init_bits=8.0)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
n = export_self_compressed_renderer(model, out, use_lzma=True)
print(f"exported {n} bytes to {out}")
PY
cp "$ANCHOR_MASKS" "$ITER_DIR/masks.mkv"
cp "$OPTIMIZED_POSES" "$ITER_DIR/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_wc.zip"
python3 -u - <<PY
from pathlib import Path
import zipfile

src = Path("$ITER_DIR")
dst = Path("$ARCHIVE")
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name in ("renderer.bin", "masks.mkv", "optimized_poses.pt"):
        p = src / name
        if not p.is_file():
            raise SystemExit(f"FATAL: missing archive input {p}")
        z.write(p, arcname=name)
print(f"archive {dst}: {dst.stat().st_size} bytes")
PY
[ -f "$ARCHIVE" ] || { echo "FATAL: missing archive" >&2; exit 2; }

# Stage 8: CUDA auth eval.
cost_guard
log "=== Stage 8: CUDA auth eval ==="
EVAL_WORK="$LOG_DIR/eval_work"
RESULT_JSON="$LOG_DIR/RESULT_JSON"
rm -rf "$EVAL_WORK"
# Strip macOS AppleDouble files before contest_auth_eval — Lane F-V2 bug 2026-04-27.
rm -f upstream/videos/._*.mkv
set +e
python3 -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$EVAL_WORK" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -30
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi
if [ -f "$EVAL_WORK/contest_auth_eval.json" ]; then
    cp "$EVAL_WORK/contest_auth_eval.json" "$RESULT_JSON"
elif [ -f "$LOG_DIR/auth_eval/contest_auth_eval.json" ]; then
    cp "$LOG_DIR/auth_eval/contest_auth_eval.json" "$RESULT_JSON"
else
    echo "FATAL: auth eval did not write contest_auth_eval.json; refusing log JSON scrape" >&2
    exit 2
fi
[ -s "$RESULT_JSON" ] || { echo "FATAL: auth eval did not write RESULT_JSON" >&2; exit 2; }

# Stage 9: write provenance.
cost_guard
log "=== Stage 9: write provenance.json ==="
python3 -u - <<PY
from pathlib import Path
import json
import subprocess
import time

log_dir = Path("$LOG_DIR")
prov = {
    "lane_name": "lane_wc",
    "predicted_band": [0.78, 1.05],
    "hypothesis": "Cosmos Curator soft-DTW outlier weighting replaces Lane W circular loss-based signal with independent typicality",
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime($START_TS)),
    "finished_provenance_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "archive": "$ARCHIVE",
    "result_json": "$RESULT_JSON",
    "features": "$FEATURES",
    "pair_weights": "$PAIR_WEIGHTS",
    "scorer_state": "$SCORER_STATE",
    "best_fp32_checkpoint": "$BEST_FP32",
    "strict_scorer_rule": "SegNet is used only during compress-time feature extraction; inflate receives only archive artifacts.",
}
(log_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
print(json.dumps(prov, indent=2))
PY

# Stage 10: cleanup.
cost_guard
log "=== Stage 10: cleanup and final record ==="
python3 -u - <<PY
from pathlib import Path
import json
import time

record = {
    "lane": "WC",
    "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "archive": "$ARCHIVE",
    "result_json": "$RESULT_JSON",
    "provenance": "$LOG_DIR/provenance.json",
}
Path("$LOG_DIR/final_record.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

if [ "${AUTO_DESTROY_VAST:-0}" = "1" ] && [ -n "${VAST_INSTANCE_ID:-}" ]; then
    log "AUTO_DESTROY_VAST=1: destroying Vast instance $VAST_INSTANCE_ID"
    vastai destroy instance "$VAST_INSTANCE_ID" || true
fi

log "=== LANE_WC_DONE [contest-CUDA] ==="
