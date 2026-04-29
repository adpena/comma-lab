#!/bin/bash
# Lane EA: arithmetic-coded entropy archive research deploy.
#
# WHAT: exercises the orphan src/tac/entropy_archive.py
# (NeuralEntropyModel + ArithmeticCoder/Decoder + build_entropy_archive)
# on Lane A's actual artifacts. Distinct from Lane EBR — EBR adds an
# entropy-bottleneck regularization term during training; Lane EA replaces
# the deflate compressor with an arithmetic coder over learned per-symbol
# probability models.
#
# Lane EA's claim: with calibrated learned probabilities, arithmetic
# coding hits Shannon entropy. Target 30-50% smaller than the deflate
# baseline.
#
# OUTSTANDING TODO: this is a RESEARCH lane. The codec produces a payload
# but no inflate-time arithmetic decoder exists yet. The archive shipped
# to auth_eval uses Lane A's masks.mkv; this lane measures encoder-only
# rate potential. Promote when an inflate decoder lands.
#
# Cost cap: $0.30, ETA 30min on RTX 4090. Encoder-only.

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

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_ea_results}"
ITER_DIR="$LOG_DIR/iter_0"
TAG="${TAG:-lane_ea}"
ANCHOR_DIR="${ANCHOR_DIR:-submissions/baseline_dilated_h64_0_90}"

mkdir -p "$LOG_DIR" "$ITER_DIR"
log() { echo "[lane-ea] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

HB_PID=""
cleanup() {
    set +e
    [ -n "${HB_PID:-}" ] && kill "$HB_PID" 2>/dev/null || true
    rm -rf "$LOG_DIR/eval_work/tmp" 2>/dev/null || true
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
    "lane_script": "scripts/remote_lane_ea_entropy_archive.sh",
    "lane_name": "lane_ea_entropy_archive",
    "tag": os.environ["TAG"],
    "anchor_dir": os.environ["ANCHOR_DIR"],
    "predicted_band": [1.10, 1.20],
    "baseline_score": 1.15,
    "baseline_lane": "A",
    "hypothesis": "RESEARCH: arithmetic coding with learned probability model approaches Shannon-entropy bound on pose targets.",
    "research_only": True,
    "strict_scorer_rule_compliant": True,
    "wires_orphan_module": "src/tac/entropy_archive.py",
    "output_dir": os.environ["LOG_DIR"],
}
with open(os.environ["PROVENANCE"], "w") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=ea gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!

log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || { log "FATAL: NVDEC probe failed"; exit 2; }

log "=== Stage 1: anchor checks ==="
for f in "$ANCHOR_DIR/renderer.bin" \
         "$ANCHOR_DIR/optimized_poses.pt" \
         "$ANCHOR_DIR/masks.mkv" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 2: arithmetic-code Lane A pose targets ==="
EA_META="$LOG_DIR/ea_meta.json"
export EA_META ANCHOR_DIR
"$PYBIN" -u - <<'PY' 2>&1 | tee "$LOG_DIR/ea.log"
import json, os, sys
import torch
sys.path.insert(0, "src"); sys.path.insert(0, "upstream")
from tac.entropy_archive import compress_byte_stream
from tac.submission_archive import load_optimized_poses

poses_path = os.path.join(os.environ["ANCHOR_DIR"], "optimized_poses.pt")
poses = load_optimized_poses(poses_path, pose_dim=6)
print(f"[lane-ea] poses shape={tuple(poses.shape)}")

# Quantize poses to int8, then arithmetic-code the byte stream.
qmin, qmax = -128, 127
amax = poses.abs().max().item()
scale = amax / qmax
qposes = (poses / scale).round().clamp(qmin, qmax).to(torch.int8)
raw_bytes = qposes.numpy().tobytes()
print(f"[lane-ea] quantized poses: {len(raw_bytes)} raw int8 bytes (scale={scale:.6e})")

compressed = compress_byte_stream(raw_bytes)
print(f"[lane-ea] arithmetic-coded: {len(compressed)} bytes ({len(compressed)/max(len(raw_bytes),1)*100:.1f}% of raw)")

original_pt_size = os.path.getsize(poses_path)
print(f"[lane-ea] original optimized_poses.pt: {original_pt_size} bytes")
print(f"[lane-ea] potential savings vs original: {(1 - len(compressed)/max(original_pt_size,1))*100:.1f}%")

meta = {
    "lane": "EA",
    "raw_int8_bytes": len(raw_bytes),
    "arithmetic_coded_bytes": len(compressed),
    "original_pt_bytes": original_pt_size,
    "rate_savings_potential_vs_pt": 1.0 - (len(compressed) / max(original_pt_size, 1)),
    "research_only": True,
}
with open(os.environ["EA_META"], "w") as f:
    json.dump(meta, f, indent=2)
print(f"[lane-ea] meta -> {os.environ['EA_META']}")
PY

log "=== Stage 3: build archive (Lane A artifacts — research lane) ==="
cp "$ANCHOR_DIR/renderer.bin" "$ITER_DIR/renderer.bin"
cp "$ANCHOR_DIR/masks.mkv" "$ITER_DIR/masks.mkv"
cp "$ANCHOR_DIR/optimized_poses.pt" "$ITER_DIR/optimized_poses.pt"

ARCHIVE="$LOG_DIR/archive_lane_ea.zip"
export ITER_DIR ARCHIVE
"$PYBIN" -u - <<'PY'
import os, zipfile
src = os.environ["ITER_DIR"]; dst = os.environ["ARCHIVE"]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for name in ("renderer.bin", "masks.mkv", "optimized_poses.pt"):
        path = os.path.join(src, name)
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(path, "rb") as f:
            zf.writestr(info, f.read(), compresslevel=9)
print(f"archive {dst}: {os.path.getsize(dst)} bytes")
PY

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ -n "${ARCHIVE_BYTES:-}" ] && [ "$ARCHIVE_BYTES" -gt 0 ] || { log "FATAL: empty archive"; exit 2; }
log "  archive_lane_ea.zip = ${ARCHIVE_BYTES} bytes"

rm -f upstream/videos/._*.mkv

log "=== Stage 4: contest_auth_eval [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" \
    2>&1 | tee "$LOG_DIR/auth_eval.log"
grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || { log "FATAL: no RESULT_JSON"; exit 5; }
grep '^RESULT_JSON' "$LOG_DIR/auth_eval.log" | tail -1 > "$LOG_DIR/RESULT_JSON"
log "  RESULT_JSON: $LOG_DIR/RESULT_JSON"

log "=== LANE_EA_DONE [contest-CUDA] (research artifact) ==="
log "    archive: $ARCHIVE   ea_meta: $EA_META"
log "    predicted_band: [1.10, 1.20]  (research-only; rate gain pending inflate decoder)"
