#!/bin/bash
# Lane 12 OWv3 0120 NeRV stack: replace owv3_0120 (1.0024) masks.mkv (421KB)
# with jsonfix40 NeRV (23KB), regenerate poses against the new mask stream
# under the owv3_0120 pruned renderer, then run CUDA contest auth eval.
#
# This composes two prior-validated artifacts:
#   * Mask codec: experiments/results/lane_12_nerv_20260430_codex_jsonfix40/
#       archive_lane_12_nerv.zip (NeRV trained 60K steps on Lane G v3 baseline
#       masks; same masks.mkv SHA as owv3_0120 -> direct retarget OK)
#   * Renderer: experiments/results/lane_g_v3_owv3_wave3_LANDED_20260501/
#       archive_lane_g_v3_owv3_0120_LANDED.zip renderer.bin (211,903 bytes,
#       FP4-pruned via OWv3 wave3 R7)
#
# The pose stream MUST be regenerated -- the NeRV-decoded masks differ from
# AV1-decoded masks by ~1.2% per pixel, so the prior optimized_poses.pt
# (tuned for AV1 reconstruction under the owv3_0120 renderer) drifts.
# Predicted band [contest-CUDA] = [0.85, 1.05]: lower bound assumes 1.2%
# mask-pixel disagreement is fully absorbed by pose-regen (-0.05 vs 1.0024
# from rate savings); upper bound assumes residual 5% pose-drift unfixable.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-python3}"
cd "$WORKSPACE"
if [ -f "$WORKSPACE/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$WORKSPACE/env.sh"
fi

export PYTHONPATH="src:upstream:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-1234}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_12_owv3_0120_nerv_stack_results}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
HEARTBEAT="$LOG_DIR/heartbeat.log"
PROVENANCE="$LOG_DIR/provenance.json"

OWV3_0120_ARCHIVE="${OWV3_0120_ARCHIVE:-$WORKSPACE/wave3_archives/owv3_0120_bbr0p66_protect0p002_aggr1em05.zip}"
NERV_CANDIDATE_ARCHIVE="${NERV_CANDIDATE_ARCHIVE:-$WORKSPACE/experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip}"
WARM_POSES="${WARM_POSES:-$WORKSPACE/experiments/results/lane_g_v3_landed/optimized_poses.pt}"
GT_POSE_TARGETS="${GT_POSE_TARGETS:-$WORKSPACE/experiments/results/lane_a_landed/gt_pose_targets.pt}"

POSE_STEPS="${POSE_STEPS:-300}"
POSE_BATCH_PAIRS="${POSE_BATCH_PAIRS:-50}"
POSE_LR="${POSE_LR:-0.01}"
POSE_SEG_WEIGHT="${POSE_SEG_WEIGHT:-100.0}"
POSE_WEIGHT="${POSE_WEIGHT:-10.0}"

EXTRACT_DIR="$LOG_DIR/extracted"
POSE_DIR="$LOG_DIR/pose_regen"
ARCHIVE_SRC_DIR="$LOG_DIR/archive_src"
CANDIDATE_MASKS_PT="$LOG_DIR/candidate_masks.pt"
ARCHIVE="$LOG_DIR/archive_lane_12_owv3_0120_nerv_stack.zip"
EVAL_WORK_DIR="$LOG_DIR/eval_work"

log() { echo "[lane12-owv3-0120-nerv] $(date -u +%FT%TZ) $*" | tee -a "$RUN_LOG"; }

HB_PID=""
cleanup() {
    set +e
    if [ -n "${HB_PID:-}" ]; then
        kill "$HB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

log "=== Stage 0: NVDEC + CUDA preflight ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed"
    exit 2
}

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || true)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || true)
else
    GPU_NAME="nvidia-smi unavailable"
    DRIVER_VER="nvidia-smi unavailable"
fi
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
export GPU_NAME DRIVER_VER GIT_HASH PROVENANCE LOG_DIR
export OWV3_0120_ARCHIVE NERV_CANDIDATE_ARCHIVE WARM_POSES GT_POSE_TARGETS
export POSE_STEPS POSE_BATCH_PAIRS POSE_LR POSE_SEG_WEIGHT POSE_WEIGHT

"$PYBIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import time
import torch

def file_meta(path_s: str) -> dict:
    path = Path(path_s)
    if not path.is_file():
        raise SystemExit(f"FATAL: required input missing: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}

payload = {
    "schema_version": 1,
    "lane": "lane_12_owv3_0120_nerv_stack",
    "status": "started",
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "purpose": "compose owv3_0120 pruned renderer + jsonfix40 NeRV mask codec + regen poses",
    "promotion_policy": "score truth only after CUDA contest_auth_eval and adjudication",
    "git_hash": os.environ["GIT_HASH"],
    "gpu_name": os.environ["GPU_NAME"],
    "driver_version": os.environ["DRIVER_VER"],
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "inputs": {
        "owv3_0120_archive": file_meta(os.environ["OWV3_0120_ARCHIVE"]),
        "nerv_candidate_archive": file_meta(os.environ["NERV_CANDIDATE_ARCHIVE"]),
        "warm_poses": file_meta(os.environ["WARM_POSES"]),
        "gt_pose_targets": file_meta(os.environ["GT_POSE_TARGETS"]),
    },
    "pose_regen": {
        "steps": int(os.environ["POSE_STEPS"]),
        "batch_pairs": int(os.environ["POSE_BATCH_PAIRS"]),
        "lr": float(os.environ["POSE_LR"]),
        "seg_weight": float(os.environ["POSE_SEG_WEIGHT"]),
        "pose_weight": float(os.environ["POSE_WEIGHT"]),
        "eval_roundtrip": True,
        "posetto_noise_std": 0.5,
    },
    "predicted_band": [0.85, 1.05],
    "champion_to_beat": {"score": 1.0024, "archive_bytes": 617410, "label": "owv3_0120"},
}
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is required")
Path(os.environ["PROVENANCE"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("provenance:", json.dumps(payload, sort_keys=True))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=12-owv3-0120-nerv gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!

log "=== Stage 0b: verify tac importable (skip pip install in venv) ==="
"$PYBIN" -c "import tac; print('tac:', tac.__file__)"

for f in "$OWV3_0120_ARCHIVE" "$NERV_CANDIDATE_ARCHIVE" "$WARM_POSES" "$GT_POSE_TARGETS" "$TAC_UPSTREAM_DIR/videos/0.mkv"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: extract owv3_0120 renderer + jsonfix40 NeRV ==="
mkdir -p "$EXTRACT_DIR"
export EXTRACT_DIR
"$PYBIN" - <<'PY'
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

extract_dir = Path(os.environ["EXTRACT_DIR"])
extract_dir.mkdir(parents=True, exist_ok=True)

# Extract owv3_0120 renderer.bin
with zipfile.ZipFile(os.environ["OWV3_0120_ARCHIVE"], "r") as zin:
    found = {info.filename for info in zin.infolist()}
    if "renderer.bin" not in found:
        raise SystemExit(f"FATAL: owv3_0120 archive missing renderer.bin (has {found})")
    (extract_dir / "renderer.bin").write_bytes(zin.read("renderer.bin"))

# Extract jsonfix40 masks.nrv
with zipfile.ZipFile(os.environ["NERV_CANDIDATE_ARCHIVE"], "r") as zin:
    found = {info.filename for info in zin.infolist()}
    if "masks.nrv" not in found:
        raise SystemExit(f"FATAL: NeRV archive missing masks.nrv (has {found})")
    (extract_dir / "masks.nrv").write_bytes(zin.read("masks.nrv"))

print(json.dumps({
    "renderer_bin_bytes": (extract_dir / "renderer.bin").stat().st_size,
    "masks_nrv_bytes": (extract_dir / "masks.nrv").stat().st_size,
    "extract_dir": str(extract_dir),
}, indent=2))
PY

log "=== Stage 2: decode masks.nrv to candidate_masks.pt for pose-regen ==="
export CANDIDATE_MASKS_PT
"$PYBIN" - <<'PY'
import os
from pathlib import Path
import torch
from tac.nerv_mask_codec import decode_nerv_codec, render_mask_argmax

extract_dir = Path(os.environ["EXTRACT_DIR"])
nrv = (extract_dir / "masks.nrv").read_bytes()
codec = decode_nerv_codec(nrv)
print(f"NeRV codec: hidden={codec.hidden_dim} freqs={codec.num_freqs} depth={codec.depth} params={codec.num_params()}", flush=True)

# Decode at full resolution: 1200 frames x 384 x 512
device = "cuda" if torch.cuda.is_available() else "cpu"
masks = render_mask_argmax(codec, num_frames=1200, height=384, width=512, batch_size=131072, device=device)
masks = masks.long().cpu()  # (T, H, W)
torch.save(masks, os.environ["CANDIDATE_MASKS_PT"])
print(f"Saved decoded NeRV masks to {os.environ['CANDIDATE_MASKS_PT']}: shape={tuple(masks.shape)} dtype={masks.dtype}", flush=True)

# Sanity: per-class counts
import numpy as np
counts = np.bincount(masks.numpy().reshape(-1), minlength=5)
print(f"Class distribution: {counts.tolist()} (sum={counts.sum()})", flush=True)
PY

log "=== Stage 3: pose regeneration against decoded masks.nrv on owv3_0120 renderer ==="
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint "$EXTRACT_DIR/renderer.bin" \
    --masks "$CANDIDATE_MASKS_PT" \
    --gt-poses-path "$WARM_POSES" \
    --gt-pose-targets "$GT_POSE_TARGETS" \
    --device cuda \
    --n-frames 1200 \
    --steps "$POSE_STEPS" \
    --batch-pairs "$POSE_BATCH_PAIRS" \
    --lr "$POSE_LR" \
    --seg-weight "$POSE_SEG_WEIGHT" \
    --pose-weight "$POSE_WEIGHT" \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --output-dir "$POSE_DIR" \
    2>&1 | tee "$LOG_DIR/optimize_poses.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: optimize_poses failed rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

[ -f "$POSE_DIR/optimized_poses.bin" ] || { echo "FATAL: missing regenerated optimized_poses.bin" >&2; exit 3; }

log "=== Stage 4: deterministic archive rebuild ==="
mkdir -p "$ARCHIVE_SRC_DIR"
cp "$EXTRACT_DIR/renderer.bin" "$ARCHIVE_SRC_DIR/renderer.bin"
cp "$EXTRACT_DIR/masks.nrv" "$ARCHIVE_SRC_DIR/masks.nrv"
cp "$POSE_DIR/optimized_poses.bin" "$ARCHIVE_SRC_DIR/optimized_poses.bin"
export ARCHIVE ARCHIVE_SRC_DIR
"$PYBIN" - <<'PY'
import json
import os
from pathlib import Path
import zipfile
from tac.submission_archive import detect_pose_manifest, validate_archive

src = Path(os.environ["ARCHIVE_SRC_DIR"])
dst = Path(os.environ["ARCHIVE"])
members = ("renderer.bin", "masks.nrv", "optimized_poses.bin")
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
    for name in members:
        data = (src / name).read_bytes()
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (0o644 & 0xFFFF) << 16
        zout.writestr(info, data, compresslevel=9)
result = validate_archive(dst, manifest=detect_pose_manifest(dst), strict=True)
if not result.valid:
    raise SystemExit(f"FATAL: rebuilt archive failed validation:\n{result.summary()}")
print(json.dumps({"archive": str(dst), "bytes": dst.stat().st_size, "members": list(members)}, indent=2))
PY

log "=== Stage 5: CUDA contest auth eval ==="
rm -rf "$EVAL_WORK_DIR"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$EVAL_WORK_DIR" \
    2>&1 | tee "$LOG_DIR/auth_eval.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval failed rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

CONTEST_JSON="$EVAL_WORK_DIR/contest_auth_eval.json"
[ -f "$CONTEST_JSON" ] || { echo "FATAL: missing contest_auth_eval.json" >&2; exit 4; }

# Surface the score immediately.
SCORE_LINE=$("$PYBIN" -c "
import json
d = json.load(open('$CONTEST_JSON'))
print(f\"score={d.get('final_score')} archive_bytes={d.get('archive_size_bytes')} pose={d.get('avg_posenet_dist')} seg={d.get('avg_segnet_dist')} rate={d.get('rate_unscaled')}\")
")
log "RESULT: $SCORE_LINE"
log "=== LANE_12_OWV3_0120_NERV_STACK_DONE [contest-CUDA] -- see $CONTEST_JSON ==="

# Echo a single-line marker so tee/tail consumers can grep it.
echo "RESULT_LANE_12_OWV3_0120_NERV: $SCORE_LINE"
