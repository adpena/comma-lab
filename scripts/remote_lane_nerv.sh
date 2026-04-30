#!/bin/bash
# Lane 12: NeRV mask codec (Phase 2 ACCELERATE).
# predicted_band=[0.95, 1.30] [contest-CUDA]
# Replaces Lane G v3 masks.mkv (AV1, 421 KB) with masks.nrv (NeRV, ~12-23 KB).
# Renderer + poses unchanged from Lane G v3 anchor.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-python3}"
cd "$WORKSPACE"
if [ -f "$WORKSPACE/env.sh" ]; then
    # shellcheck disable=SC1091
    [ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
fi

export PYTHONPATH="src:upstream:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_12_nerv_results}"
NERV_DIR="$LOG_DIR/nerv"
BASE_ARCHIVE="${BASE_ARCHIVE:-$WORKSPACE/submissions/robust_current/archive.zip}"
ARCHIVE="$LOG_DIR/archive_lane_12_nerv.zip"
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
PROFILE="${PROFILE:-nerv_mask_lane_g_v3}"
GT_MASKS_SOURCE="${GT_MASKS_SOURCE:-segnet}"
mkdir -p "$LOG_DIR" "$NERV_DIR"

log() { echo "[lane-12-nerv] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

HB_PID=""
cleanup() {
    set +e
    if [ -n "${HB_PID:-}" ]; then
        kill "$HB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
export PROVENANCE LOG_DIR GIT_HASH GPU_NAME DRIVER_VER BASE_ARCHIVE NERV_DIR PROFILE GT_MASKS_SOURCE
"$PYBIN" - <<'PY'
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
    "lane_script": "scripts/remote_lane_nerv.sh",
    "lane_name": "lane_12_nerv_mask_codec",
    "predicted_band": [0.95, 1.30],
    "strict_scorer_rule_compliant": True,
    "base_archive": os.environ["BASE_ARCHIVE"],
    "profile": os.environ["PROFILE"],
    "gt_masks_source": os.environ["GT_MASKS_SOURCE"],
    "output_dir": os.environ["LOG_DIR"],
}
with open(os.environ["PROVENANCE"], "w", encoding="utf-8") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=12-nerv gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!

log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed"
    exit 2
}
"$PYBIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is required for remote lane 12 (NeRV)")
print("cuda:", torch.cuda.get_device_name(0))
PY

log "=== Stage 0b: install editable package ==="
# CODE PARITY: launcher tarball is authoritative — do NOT git reset --hard.
# (memory: feedback_git_reset_nukes_anchors_20260429)
"$PYBIN" -m pip install -e .

for f in "$BASE_ARCHIVE" "$TAC_UPSTREAM_DIR/videos/0.mkv"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: train NeRV mask codec ==="
"$PYBIN" -u experiments/train_nerv_mask.py \
    --profile "$PROFILE" \
    --device cuda \
    --upstream "$TAC_UPSTREAM_DIR" \
    --gt-masks-source "$GT_MASKS_SOURCE" \
    --output-dir "$NERV_DIR" \
    2>&1 | tee "$LOG_DIR/train_nerv_mask.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

NERV_PAYLOAD="$NERV_DIR/masks.nrv"
[ -f "$NERV_PAYLOAD" ] || { echo "FATAL: missing $NERV_PAYLOAD" >&2; exit 3; }
NERV_BYTES=$(stat -f%z "$NERV_PAYLOAD" 2>/dev/null || stat -c%s "$NERV_PAYLOAD")
log "NeRV payload: $NERV_PAYLOAD ($NERV_BYTES bytes)"

# Kill criterion: > 100 KB → abandon (per Phase B council)
if [ "$NERV_BYTES" -gt 100000 ]; then
    log "FATAL: NeRV payload $NERV_BYTES B > 100 KB kill criterion"
    exit 4
fi

log "=== Stage 2: rebuild archive with masks.nrv replacing masks.mkv ==="
export BASE_ARCHIVE ARCHIVE NERV_PAYLOAD
"$PYBIN" - <<'PY'
import os
import zipfile

src = os.environ["BASE_ARCHIVE"]
dst = os.environ["ARCHIVE"]
nerv = os.environ["NERV_PAYLOAD"]
# Replace any existing masks.* with masks.nrv. Keep everything else
# (renderer.bin, poses.pt, inflate scripts) bit-identical to base.
mask_extensions = (".mkv", ".amrc", ".stcb", ".nrv")
with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(
    dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9
) as zout:
    for info in zin.infolist():
        # Skip any pre-existing mask payload — we ship masks.nrv instead.
        if info.filename.startswith("masks.") and info.filename.endswith(mask_extensions):
            continue
        data = zin.read(info.filename)
        out_info = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
        out_info.compress_type = zipfile.ZIP_DEFLATED
        zout.writestr(out_info, data, compresslevel=9)
    # Add masks.nrv
    out_info = zipfile.ZipInfo("masks.nrv", date_time=(1980, 1, 1, 0, 0, 0))
    out_info.compress_type = zipfile.ZIP_DEFLATED
    with open(nerv, "rb") as f:
        zout.writestr(out_info, f.read(), compresslevel=9)
print(f"archive {dst}: {os.path.getsize(dst)} bytes (with masks.nrv)")
PY

log "=== Stage 3: CUDA auth eval [contest-CUDA] ==="
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
    log "FATAL: auth eval did not produce RESULT_JSON"
    exit 5
}
grep '^RESULT_JSON' "$LOG_DIR/auth_eval.log" | tail -1 > "$LOG_DIR/RESULT_JSON"
log "RESULT_JSON written to $LOG_DIR/RESULT_JSON"
log "=== LANE_12_NERV_DONE [contest-CUDA] ==="
