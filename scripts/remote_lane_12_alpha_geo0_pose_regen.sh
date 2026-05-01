#!/bin/bash
# Lane 12 Alpha-Geo-0: stale-pose isolation for the measured jsonfix40 NeRV
# mask payload. This is not new NeRV retraining. It keeps the exact measured
# masks.nrv and renderer, regenerates optimized poses against the decoded
# candidate mask stream, rebuilds a deterministic archive, and runs CUDA
# contest auth eval.
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

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_12_alpha_geo0_pose_regen_results}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"
HEARTBEAT="$LOG_DIR/heartbeat.log"
PROVENANCE="$LOG_DIR/provenance.json"

ANCHOR_CANDIDATE_ARCHIVE="${ANCHOR_CANDIDATE_ARCHIVE:-experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip}"
ANCHOR_BASELINE_ARCHIVE="${ANCHOR_BASELINE_ARCHIVE:-experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip}"
ANCHOR_WARM_POSES="${ANCHOR_WARM_POSES:-experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt}"
ANCHOR_GT_POSE_TARGETS="${ANCHOR_GT_POSE_TARGETS:-experiments/results/lane_a_landed/gt_pose_targets.pt}"

CANDIDATE_ARCHIVE="${CANDIDATE_ARCHIVE:-$WORKSPACE/$ANCHOR_CANDIDATE_ARCHIVE}"
BASELINE_ARCHIVE="${BASELINE_ARCHIVE:-$WORKSPACE/$ANCHOR_BASELINE_ARCHIVE}"
WARM_POSES="${WARM_POSES:-$WORKSPACE/$ANCHOR_WARM_POSES}"
GT_POSE_TARGETS="${GT_POSE_TARGETS:-$WORKSPACE/$ANCHOR_GT_POSE_TARGETS}"
POSE_STEPS="${POSE_STEPS:-500}"
POSE_BATCH_PAIRS="${POSE_BATCH_PAIRS:-8}"
POSE_LR="${POSE_LR:-0.01}"
POSE_SEG_WEIGHT="${POSE_SEG_WEIGHT:-100.0}"
POSE_WEIGHT="${POSE_WEIGHT:-10.0}"

EXTRACT_DIR="$LOG_DIR/extracted"
MASK_CACHE_DIR="$LOG_DIR/mask_cache"
POSE_DIR="$LOG_DIR/pose_regen"
ARCHIVE_SRC_DIR="$LOG_DIR/archive_src"
CANDIDATE_MASKS_PT="$LOG_DIR/candidate_masks.pt"
ALPHA_GEO_JSON="$LOG_DIR/alpha_geo_0_geometry.json"
ALPHA_PRIMITIVE_CONTRACT="$LOG_DIR/alpha_geo_0_primitive_contract.json"
ARCHIVE="$LOG_DIR/archive_lane_12_alpha_geo0_pose_regen.zip"
EVAL_WORK_DIR="$LOG_DIR/eval_work"

log() { echo "[lane12-alpha-geo0] $(date -u +%FT%TZ) $*" | tee -a "$RUN_LOG"; }

HB_PID=""
cleanup() {
    set +e
    if [ -n "${HB_PID:-}" ]; then
        kill "$HB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

log "=== Stage 0: NVDEC and CUDA preflight ==="
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
export GPU_NAME DRIVER_VER GIT_HASH PROVENANCE LOG_DIR CANDIDATE_ARCHIVE BASELINE_ARCHIVE WARM_POSES GT_POSE_TARGETS
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
    "lane": "lane_12_alpha_geo0_pose_regen",
    "status": "started",
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "purpose": "stale-pose isolation for exact jsonfix40 masks.nrv; no NeRV retraining",
    "promotion_policy": "score truth only after CUDA contest_auth_eval and adjudication",
    "git_hash": os.environ["GIT_HASH"],
    "gpu_name": os.environ["GPU_NAME"],
    "driver_version": os.environ["DRIVER_VER"],
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "inputs": {
        "candidate_archive": file_meta(os.environ["CANDIDATE_ARCHIVE"]),
        "baseline_archive": file_meta(os.environ["BASELINE_ARCHIVE"]),
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
    "predicted_band": [0.95, 1.10],
}
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is required for Alpha-Geo-0 pose regeneration")
Path(os.environ["PROVENANCE"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("alpha_geo0_provenance:", json.dumps(payload, sort_keys=True))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=12-alpha-geo0 gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!

log "=== Stage 0b: install editable package ==="
"$PYBIN" -m pip install -e .

for f in "$CANDIDATE_ARCHIVE" "$BASELINE_ARCHIVE" "$WARM_POSES" "$GT_POSE_TARGETS" "$TAC_UPSTREAM_DIR/videos/0.mkv"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: deterministic Alpha-Geo decode and cache ==="
mkdir -p "$EXTRACT_DIR" "$MASK_CACHE_DIR"
"$PYBIN" -u experiments/diagnose_nerv_geometry.py \
    --baseline "$BASELINE_ARCHIVE" \
    --baseline-member masks.mkv \
    --candidate "$CANDIDATE_ARCHIVE" \
    --candidate-member masks.nrv \
    --output-json "$ALPHA_GEO_JSON" \
    --primitive-contract-json "$ALPHA_PRIMITIVE_CONTRACT" \
    --mask-cache-dir "$MASK_CACHE_DIR" \
    --threshold-preset exploratory \
    --num-frames 1200 \
    --height 384 \
    --width 512 \
    2>&1 | tee "$LOG_DIR/diagnose_nerv_geometry.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: diagnose_nerv_geometry failed rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

export MASK_CACHE_DIR CANDIDATE_ARCHIVE CANDIDATE_MASKS_PT
"$PYBIN" - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


cache_dir = Path(os.environ["MASK_CACHE_DIR"])
candidate_sha = sha256_file(Path(os.environ["CANDIDATE_ARCHIVE"]))
matches = []
for meta_path in sorted(cache_dir.glob("*.json")):
    payload = json.loads(meta_path.read_text())
    fp = payload.get("fingerprint") or {}
    if fp.get("source_sha256") == candidate_sha and fp.get("archive_member_resolved") == "masks.nrv":
        tensor_file = meta_path.with_name(payload["tensor_file"])
        if tensor_file.is_file():
            matches.append((meta_path, tensor_file, payload))
if len(matches) != 1:
    raise SystemExit(f"FATAL: expected exactly one candidate mask cache tensor, found {len(matches)}")
_, tensor_file, payload = matches[0]
dst = Path(os.environ["CANDIDATE_MASKS_PT"])
shutil.copy2(tensor_file, dst)
print(json.dumps({
    "candidate_masks_pt": str(dst),
    "decoded_mask_sha256": payload.get("decoded_mask_sha256"),
    "decoded_mask_shape": payload.get("decoded_mask_shape"),
}, sort_keys=True))
PY

log "=== Stage 2: extract candidate renderer and masks.nrv ==="
export EXTRACT_DIR
"$PYBIN" - <<'PY'
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

candidate = Path(os.environ["CANDIDATE_ARCHIVE"])
extract_dir = Path(os.environ["EXTRACT_DIR"])
extract_dir.mkdir(parents=True, exist_ok=True)
allowed = {"renderer.bin", "masks.nrv"}
with zipfile.ZipFile(candidate, "r") as zin:
    names = set()
    for info in zin.infolist():
        p = PurePosixPath(info.filename)
        if info.filename in names:
            raise SystemExit(f"FATAL: duplicate archive member {info.filename!r}")
        names.add(info.filename)
        if info.is_dir() or p.is_absolute() or ".." in p.parts:
            raise SystemExit(f"FATAL: unsafe archive member {info.filename!r}")
        if info.filename in allowed:
            (extract_dir / info.filename).write_bytes(zin.read(info.filename))
missing = allowed - {p.name for p in extract_dir.iterdir() if p.is_file()}
if missing:
    raise SystemExit(f"FATAL: candidate archive missing {sorted(missing)}")
print(json.dumps({"extracted": sorted(allowed), "dir": str(extract_dir)}, sort_keys=True))
PY

log "=== Stage 3: pose regeneration against decoded masks.nrv ==="
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
[ -f "$POSE_DIR/optimized_poses.meta" ] || { echo "FATAL: missing regenerated optimized_poses.meta" >&2; exit 3; }

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
    raise SystemExit(f"FATAL: Alpha-Geo-0 archive failed validation:\n{result.summary()}")
print(json.dumps({"archive": str(dst), "bytes": dst.stat().st_size, "members": list(members)}, sort_keys=True))
PY

log "=== Stage 5: CUDA auth eval ==="
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
RESULT_JSON="$LOG_DIR/contest_auth_eval.json"
ADJUDICATION_LOG="$LOG_DIR/adjudication.log"
"$PYBIN" -u scripts/adjudicate_contest_auth_eval.py \
    --contest-json "$CONTEST_JSON" \
    --provenance "$PROVENANCE" \
    --archive "$ARCHIVE" \
    --result-copy "$RESULT_JSON" \
    --baseline-score 1.043987524793892 \
    --baseline-archive-bytes 686635 \
    --predicted-band 0.70 1.30 \
    --regression-threshold 1.30 \
    --delta-key score_delta_vs_pfp16_a_plus_plus \
    --required-device cuda \
    --required-samples 600 \
    --max-sane-score 100.0 \
    --allow-component-gate-forensic-success \
    2>&1 | tee "$ADJUDICATION_LOG"

cp "$RESULT_JSON" "$LOG_DIR/RESULT_JSON"
log "contest_auth_eval JSON written to $RESULT_JSON"
log "=== LANE_12_ALPHA_GEO0_POSE_REGEN_DONE [contest-CUDA] -- see $RESULT_JSON ==="
