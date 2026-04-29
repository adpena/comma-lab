#!/bin/bash
# Lane J-NWC: Neural Weight Compression for renderer.bin.
#
# arXiv 2510.11234 — "Neural Weight Compression for Language Models" (late 2025).
# Trains a tiny VQ-VAE-style codec on a corpus of small renderer checkpoints,
# then applies the codec to compress the Lane G v3 frontier renderer (296KB)
# down to ~17KB. Predicted rate gain: −0.084 → score band [0.92, 1.02].
#
# Stages:
#   0. NVDEC probe (memory feedback_vastai_nvdec_host_variation)
#   1. canonical git sync (fetch + reset --hard origin/main) + pip install -e .
#   2. Train neural weight codec on a corpus of saved .pt checkpoints
#   3. Export Lane G v3 renderer via NWC1 → renderer.bin
#   4. Build archive (renderer.bin + masks.mkv + optimized_poses.pt)
#   5. CUDA contest_auth_eval [contest-CUDA]
#   6. Provenance + final record
#
# The codec is trained on CPU because the corpus is small (≤200 checkpoints,
# ≤4M blocks, ~256MB RAM) and AdamW for ~16K params @ 2000 steps converges
# well before GPU acceleration matters.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/results/lane_j_nwc}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-43200}"  # 12h cap
START_TS="$(date +%s)"

# ANCHOR_LANE_G_V3_ARCHIVE: discoverable by Check 43 tarball-anchor scanner.
ANCHOR_LANE_G_V3_ARCHIVE="${ANCHOR_LANE_G_V3_ARCHIVE:-experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip}"
ANCHOR_CORPUS_DIR="${ANCHOR_CORPUS_DIR:-experiments/results}"

cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export PYTHONPATH="src:upstream:${PYTHONPATH:-}"
mkdir -p "$LOG_DIR"

log() { echo "[lane-j-nwc] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

cost_guard() {
    now="$(date +%s)"
    elapsed=$((now - START_TS))
    if [ "$elapsed" -gt "$MAX_RUNTIME_SECONDS" ]; then
        log "FATAL: hard runtime cap exceeded: ${elapsed}s > ${MAX_RUNTIME_SECONDS}s"
        exit 70
    fi
}

# Heartbeat (memory feedback_remote_code_parity_required §3).
HEARTBEAT="$LOG_DIR/heartbeat.log"
( while true; do
    GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')"
    echo "[$(date -u +%FT%TZ)] lane=J-NWC gpu=$GPU" >> "$HEARTBEAT"
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

# Stage 1: code parity + install.
cost_guard
log "=== Stage 1: canonical git sync + pip install -e . ==="
# Nuke local junk from prior failed deploys, then sync to origin/main exactly.
git fetch origin main && git reset --hard origin/main
python3 -u -m pip install -e .

# Pre-flight: required artifacts.
[ -f "$ANCHOR_LANE_G_V3_ARCHIVE" ] || {
    log "FATAL: missing Lane G v3 anchor archive: $ANCHOR_LANE_G_V3_ARCHIVE"
    exit 1
}
[ -d "$ANCHOR_CORPUS_DIR" ] || {
    log "FATAL: missing corpus dir: $ANCHOR_CORPUS_DIR"
    exit 1
}
GT_VIDEO="${GT_VIDEO:-upstream/videos/0.mkv}"
SEGNET_WEIGHTS="upstream/models/segnet.safetensors"
POSENET_WEIGHTS="upstream/models/posenet.safetensors"
for f in "$GT_VIDEO" "$SEGNET_WEIGHTS" "$POSENET_WEIGHTS"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Stage 1b: extract Lane G v3 archive as anchor.
cost_guard
log "=== Stage 1b: extract Lane G v3 archive ==="
ANCHOR_DIR="$LOG_DIR/anchor"
rm -rf "$ANCHOR_DIR"
mkdir -p "$ANCHOR_DIR"
python3 -u - <<PY
from pathlib import Path
import zipfile

archive = Path("$ANCHOR_LANE_G_V3_ARCHIVE")
out = Path("$ANCHOR_DIR")
if not archive.is_file():
    raise SystemExit(f"FATAL: missing Lane G v3 archive: {archive}")
with zipfile.ZipFile(archive) as zf:
    zf.extractall(out)
for name in ("renderer.bin", "masks.mkv", "optimized_poses.pt"):
    p = out / name
    if not p.is_file():
        raise SystemExit(f"FATAL: Lane G v3 archive missing {name}")
    print(f"{name}: {p.stat().st_size} bytes")
PY

ANCHOR_RENDERER_BIN="$ANCHOR_DIR/renderer.bin"
ANCHOR_MASKS="$ANCHOR_DIR/masks.mkv"
ANCHOR_POSES="$ANCHOR_DIR/optimized_poses.pt"

# Stage 2: train the neural weight codec on a corpus of saved .pt checkpoints.
cost_guard
log "=== Stage 2: train NWC codec on corpus ==="
CODEC_PT="$LOG_DIR/codec.pt"
python3 -u experiments/train_neural_weight_codec.py \
    --corpus-dir "$ANCHOR_CORPUS_DIR" \
    --output "$CODEC_PT" \
    --num-steps 2000 \
    --batch-size 256 \
    --lr 1e-3 \
    --device cpu \
    --block-size 16 \
    --codebook-size 64 \
    --latent-dim 16 \
    --hidden 64 \
    --max-corpus-files 200 \
    --max-blocks-per-ckpt 50000 \
    --seed 1234 2>&1 | tee "$LOG_DIR/train_codec.log" | tail -40
[ -f "$CODEC_PT" ] || { log "FATAL: codec training did not produce $CODEC_PT"; exit 2; }

# Stage 3: export Lane G v3 renderer via NWC1.
#
# Lane G v3's renderer.bin is currently in ASYM/FP4A binary format. We must
# first load it back to a fp32 model state, then re-export through NWC1.
cost_guard
log "=== Stage 3: export Lane G v3 renderer via NWC1 ==="
NWC_RENDERER_BIN="$LOG_DIR/renderer_nwc.bin"
python3 -u - <<PY
from pathlib import Path
import torch

from tac.renderer_export import (
    export_neural_compressed_checkpoint,
    load_any_renderer_checkpoint,
)

src_bin = Path("$ANCHOR_RENDERER_BIN")
codec_pt = Path("$CODEC_PT")
out_bin = Path("$NWC_RENDERER_BIN")

# Step 3a: load Lane G v3 renderer back to a fp32 nn.Module.
print(f"loading {src_bin} ({src_bin.stat().st_size:,} bytes)")
model = load_any_renderer_checkpoint(src_bin, device="cpu")
print(f"loaded model: {type(model).__name__}, "
      f"{sum(p.numel() for p in model.parameters()):,} params")

# Step 3b: export through NWC1 with the trained codec.
nbytes = export_neural_compressed_checkpoint(
    model,
    codec_path=codec_pt,
    output_path=out_bin,
)
print(f"NWC1 renderer.bin: {nbytes:,} bytes (was {src_bin.stat().st_size:,} bytes)")
delta = src_bin.stat().st_size - nbytes
print(f"delta: {delta:+,} bytes ({delta/37545489:.6f} rate units)")
PY
[ -f "$NWC_RENDERER_BIN" ] || { log "FATAL: NWC1 export did not produce $NWC_RENDERER_BIN"; exit 2; }

# Stage 4: build the archive (renderer.bin + masks.mkv + optimized_poses.pt).
cost_guard
log "=== Stage 4: build archive ==="
ITER_DIR="$LOG_DIR/iter_0"
mkdir -p "$ITER_DIR"
cp "$NWC_RENDERER_BIN" "$ITER_DIR/renderer.bin"
cp "$ANCHOR_MASKS" "$ITER_DIR/masks.mkv"
cp "$ANCHOR_POSES" "$ITER_DIR/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_j_nwc.zip"
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
[ -f "$ARCHIVE" ] || { log "FATAL: missing archive"; exit 2; }
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ -n "${ARCHIVE_BYTES:-}" ] && [ "$ARCHIVE_BYTES" -gt 0 ] || {
    log "FATAL: archive size empty or zero — refusing to call auth_eval"
    exit 2
}

# Stage 5: CUDA auth eval [contest-CUDA].
cost_guard
log "=== Stage 5: CUDA auth eval [contest-CUDA] ==="
EVAL_WORK="$LOG_DIR/eval_work"
RESULT_JSON="$LOG_DIR/RESULT_JSON"
rm -rf "$EVAL_WORK"
# AppleDouble cleanup before contest_auth_eval (Lane F-V2 bug 2026-04-27).
rm -f upstream/videos/._*.mkv
python3 -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$EVAL_WORK" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -30

if [ -f "$EVAL_WORK/contest_auth_eval.json" ]; then
    cp "$EVAL_WORK/contest_auth_eval.json" "$RESULT_JSON"
elif [ -f "$LOG_DIR/auth_eval/contest_auth_eval.json" ]; then
    cp "$LOG_DIR/auth_eval/contest_auth_eval.json" "$RESULT_JSON"
else
    grep -Eo '\{.*\}' "$LOG_DIR/auth_eval.log" | tail -1 > "$RESULT_JSON" || true
fi
[ -s "$RESULT_JSON" ] || { log "FATAL: auth eval did not write RESULT_JSON"; exit 2; }

# Stage 6: provenance + final record.
cost_guard
log "=== Stage 6: write provenance.json ==="
python3 -u - <<PY
from pathlib import Path
import json
import subprocess
import time

log_dir = Path("$LOG_DIR")
prov = {
    "lane_name": "lane_j_nwc",
    "predicted_band": [0.92, 1.02],
    "hypothesis": (
        "Lane J-NWC: arXiv 2510.11234 neural weight compression. "
        "Train a tiny VQ-VAE codec on a corpus of saved renderers, then "
        "apply it to Lane G v3 renderer.bin. Target 4 bits/weight on 88K "
        "params → -126KB rate → -0.084 score."
    ),
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime($START_TS)),
    "finished_provenance_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "anchor_archive": "$ANCHOR_LANE_G_V3_ARCHIVE",
    "corpus_dir": "$ANCHOR_CORPUS_DIR",
    "codec_pt": "$CODEC_PT",
    "renderer_nwc_bin": "$NWC_RENDERER_BIN",
    "archive": "$ARCHIVE",
    "result_json": "$RESULT_JSON",
    "strict_scorer_rule": (
        "NWC codec is loaded ONLY at compress time to encode renderer.bin; "
        "the codec weights are bundled INTO renderer.bin so inflate can "
        "decode without external state. No SegNet/PoseNet at inflate."
    ),
}
(log_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
print(json.dumps(prov, indent=2))
PY

cost_guard
log "=== Stage 7: final record ==="
python3 -u - <<PY
from pathlib import Path
import json
import time

record = {
    "lane": "J-NWC",
    "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "archive": "$ARCHIVE",
    "result_json": "$RESULT_JSON",
    "provenance": "$LOG_DIR/provenance.json",
    "predicted_band": [0.92, 1.02],
}
Path("$LOG_DIR/final_record.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

if [ "${AUTO_DESTROY_VAST:-0}" = "1" ] && [ -n "${VAST_INSTANCE_ID:-}" ]; then
    log "AUTO_DESTROY_VAST=1: destroying Vast instance $VAST_INSTANCE_ID"
    vastai destroy instance "$VAST_INSTANCE_ID" || true
fi

log "=== LANE_J_NWC_DONE [contest-CUDA] ==="
