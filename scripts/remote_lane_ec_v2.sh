#!/bin/bash
# Lane EC-V2: greedy water-fill SegNet correction deployment.
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

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_ec_v2_results}"
CORR_DIR="$LOG_DIR/corrections"
BASE_ARCHIVE="${BASE_ARCHIVE:-$WORKSPACE/submissions/robust_current/archive.zip}"
CHECKPOINT="${CHECKPOINT:-$WORKSPACE/submissions/robust_current/renderer.bin}"
GT_POSES="${GT_POSES:-$WORKSPACE/submissions/robust_current/optimized_poses.pt}"
ARCHIVE="$LOG_DIR/archive_lane_ec_v2.zip"
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
mkdir -p "$LOG_DIR" "$CORR_DIR"

log() { echo "[lane-ec-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
export PROVENANCE LOG_DIR GIT_HASH GPU_NAME DRIVER_VER BASE_ARCHIVE CHECKPOINT
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
    "lane_script": "scripts/remote_lane_ec_v2.sh",
    "lane_name": "lane_ec_v2_greedy_waterfill",
    "predicted_band": [0.80, 1.20],
    "strict_scorer_rule_compliant": True,
    "allocation_strategy": "greedy",
    "rate_cap_bytes": 50000,
    "base_archive": os.environ["BASE_ARCHIVE"],
    "checkpoint": os.environ["CHECKPOINT"],
    "output_dir": os.environ["LOG_DIR"],
}
with open(os.environ["PROVENANCE"], "w", encoding="utf-8") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=ec-v2 gpu=$GPU" >> "$HEARTBEAT"
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
    raise SystemExit("FATAL: CUDA is required for remote lane EC-V2")
print("cuda:", torch.cuda.get_device_name(0))
PY

log "=== Stage 0b: sync code and install editable package ==="
# CODE PARITY: launcher tarball is authoritative — do NOT git reset --hard.
# Doing so wipes local-only anchor files (archive_lane_a.zip, baseline dirs,
# etc.) that the launcher just SCP'd. The tarball IS the parity mechanism.
# (memory: feedback_git_reset_nukes_anchors_20260429)
"$PYBIN" -m pip install -e .

for f in "$BASE_ARCHIVE" "$CHECKPOINT" "$GT_POSES" "$TAC_UPSTREAM_DIR/videos/0.mkv"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 1: precompute greedy gradient corrections ==="
set +e
"$PYBIN" -u experiments/precompute_gradient_corrections.py \
    --checkpoint "$CHECKPOINT" \
    --device cuda \
    --n-frames 1200 \
    --batch-pairs "${BATCH_PAIRS:-10}" \
    --allocation-strategy greedy \
    --rate-cap-bytes "${RATE_CAP_BYTES:-50000}" \
    --eval-roundtrip \
    --gt-poses-path "$GT_POSES" \
    --upstream "$TAC_UPSTREAM_DIR" \
    --video "$TAC_UPSTREAM_DIR/videos/0.mkv" \
    --output-dir "$CORR_DIR" \
    2>&1 | tee "$LOG_DIR/precompute_gradient_corrections.log"
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

CORRECTIONS_BIN="$CORR_DIR/gradient_corrections.bin"
CORRECTION_MAP="$CORR_DIR/correction_map.npy"
[ -f "$CORRECTIONS_BIN" ] || { echo "FATAL: missing $CORRECTIONS_BIN" >&2; exit 3; }
[ -f "$CORRECTION_MAP" ] || { echo "FATAL: missing $CORRECTION_MAP" >&2; exit 3; }

log "=== Stage 2: build archive with Python zipfile and run CUDA auth eval ==="
export BASE_ARCHIVE ARCHIVE CORRECTIONS_BIN CORRECTION_MAP
"$PYBIN" - <<'PY'
import os
import zipfile

src = os.environ["BASE_ARCHIVE"]
dst = os.environ["ARCHIVE"]
extras = {
    "gradient_corrections.bin": os.environ["CORRECTIONS_BIN"],
    "correction_map.npy": os.environ["CORRECTION_MAP"],
}
with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
    skip = set(extras)
    for info in zin.infolist():
        if info.filename in skip:
            continue
        data = zin.read(info.filename)
        out_info = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
        out_info.compress_type = zipfile.ZIP_DEFLATED
        zout.writestr(out_info, data, compresslevel=9)
    for name, path in extras.items():
        out_info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        out_info.compress_type = zipfile.ZIP_DEFLATED
        with open(path, "rb") as f:
            zout.writestr(out_info, f.read(), compresslevel=9)
print(f"archive {dst}: {os.path.getsize(dst)} bytes")
PY

rm -rf "$LOG_DIR/eval_work"
set +e
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" \
    2>&1 | tee "$LOG_DIR/auth_eval.log"
    PIPE_RC=("${PIPESTATUS[@]}")
set -e
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || {
    log "FATAL: auth eval did not produce RESULT_JSON"
    exit 4
}
grep '^RESULT_JSON' "$LOG_DIR/auth_eval.log" | tail -1 > "$LOG_DIR/RESULT_JSON"
log "RESULT_JSON written to $LOG_DIR/RESULT_JSON"
log "=== LANE_EC_V2_DONE [contest-CUDA] ==="
