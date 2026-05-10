#!/bin/bash
# Lane SH — Shannon arithmetic coding of qint streams (EUREKA #4, 2026-04-29).
#
# Strategy: encoder-only. Take a SegMap-paradigm archive (Lane SA / FC / PA /
# any segmap_weights.tar.xz producer), unpack, re-encode the qint stream
# with a Shannon-optimal range coder, repack into a payload.bin sibling
# format, build a smaller archive, run contest_auth_eval.
#
# Reuses the upstream archive bytes for grayscale.mkv + optimized_poses.pt;
# only payload.tar.xz is replaced by payload.bin.
#
# Predicted band: -0.03 stacked vs upstream archive [contest-CUDA].
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: lane SH depends on tac.arithmetic_qint_codec.py + new
#   inflate_segmap_arithmetic.py; smoke fixture will be backfilled after the
#   first contest-CUDA score lands.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
SEGMAP_ARCH="${SEGMAP_ARCH:-segmap}"
SEGMAP_CLASS_TARGETS_FILENAME="${SEGMAP_CLASS_TARGETS_FILENAME:-class_targets.fp16}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_sh_shannon_arithmetic"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-sh] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# UPSTREAM_ARCHIVE: the SegMap-paradigm archive whose payload.tar.xz we will
# re-encode. Default to a Lane SA result; operators can override by exporting
# UPSTREAM_ARCHIVE before running this script. The archive MUST contain
# segmap_weights.tar.xz, grayscale.mkv, and (optionally) optimized_poses.pt.
UPSTREAM_ARCHIVE="${UPSTREAM_ARCHIVE:-$WORKSPACE/lane_sa_segmap_clone_results/archive_lane_sa_segmap_clone.zip}"

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || echo "no-gpu")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo "no-driver")
"$PYBIN" -c "
import json, time, torch
prov = {
    'lane_id': '$LANE_ID',
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_sh_shannon_arithmetic.sh',
    'output_dir': '$LOG_DIR',
    'variant': 'arithmetic_codec_repack',
    'predicted_band_delta': [-0.05, -0.02],
    'paradigm': 'segmap_arithmetic',
    'upstream_archive': '$UPSTREAM_ARCHIVE',
    'segmap_arch': '$SEGMAP_ARCH',
    'class_targets_filename': '$SEGMAP_CLASS_TARGETS_FILENAME',
    'controlled_baseline': 'UPSTREAM_ARCHIVE (encoder-only repack; only delta is xz -> arithmetic coder for qint streams)',
    'cost_estimate_usd': 1.00,
    'wall_clock_estimate_hours': 3.0,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=SH gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- destroy this Vast.ai instance."
    exit 2
}

[ -f "$UPSTREAM_ARCHIVE" ] || {
    echo "FATAL: missing upstream archive: $UPSTREAM_ARCHIVE" >&2
    echo "Hint: export UPSTREAM_ARCHIVE=path/to/archive_lane_*.zip before running." >&2
    exit 1
}

log "=== Stage 1: extract upstream archive ==="
EXTRACT_DIR="$LOG_DIR/upstream_extracted"
mkdir -p "$EXTRACT_DIR"
"$PYBIN" -c "
import zipfile, os
with zipfile.ZipFile('$UPSTREAM_ARCHIVE', 'r') as z:
    z.extractall('$EXTRACT_DIR')
print('extracted contents:', sorted(os.listdir('$EXTRACT_DIR')))
"

UPSTREAM_PAYLOAD="$EXTRACT_DIR/segmap_weights.tar.xz"
UPSTREAM_GRAYSCALE="$EXTRACT_DIR/grayscale.mkv"
UPSTREAM_POSES="$EXTRACT_DIR/optimized_poses.pt"
UPSTREAM_CLASS_TARGETS="$EXTRACT_DIR/$SEGMAP_CLASS_TARGETS_FILENAME"
[ -f "$UPSTREAM_PAYLOAD" ] || { echo "FATAL: missing $UPSTREAM_PAYLOAD" >&2; exit 1; }
[ -f "$UPSTREAM_GRAYSCALE" ] || { echo "FATAL: missing $UPSTREAM_GRAYSCALE" >&2; exit 1; }

log "=== Stage 2: re-encode payload.tar.xz -> payload.bin (arithmetic) ==="
ARITH_PAYLOAD="$LOG_DIR/payload.bin"
"$PYBIN" -c "
import json
from tac.arithmetic_qint_codec import (
    repack_payload_tar_xz_to_arithmetic,
    unpack_arithmetic_payload,
)
from tac.block_fp_codec import unpack_payload_tar_xz
import torch

stats = repack_payload_tar_xz_to_arithmetic('$UPSTREAM_PAYLOAD', '$ARITH_PAYLOAD')
print('repack stats:', json.dumps(stats, indent=2))

# Roundtrip: decode the arithmetic payload AND the original tar.xz, compare
# tensors. Tolerate exact match per-tensor since both go through the same
# decode_conv_weight path.
state_arith = unpack_arithmetic_payload('$ARITH_PAYLOAD')
state_orig = unpack_payload_tar_xz('$UPSTREAM_PAYLOAD')
for k, v in state_orig.items():
    if k not in state_arith:
        raise SystemExit(f'arithmetic payload missing key: {k}')
    if not torch.equal(v, state_arith[k]):
        diff = (v - state_arith[k]).abs().max().item()
        if diff > 1e-6:
            raise SystemExit(f'roundtrip mismatch on {k}: diff={diff}')
print('arithmetic-payload roundtrip: OK')
"

log "=== Stage 3: build new archive (payload.bin + grayscale.mkv + poses) ==="
mkdir -p "$LOG_DIR/archive_src"
cp "$ARITH_PAYLOAD" "$LOG_DIR/archive_src/payload.bin"
cp "$UPSTREAM_GRAYSCALE" "$LOG_DIR/archive_src/grayscale.mkv"
if [ -f "$UPSTREAM_POSES" ]; then
    cp "$UPSTREAM_POSES" "$LOG_DIR/archive_src/optimized_poses.pt"
fi
if [ -f "$UPSTREAM_CLASS_TARGETS" ]; then
    cp "$UPSTREAM_CLASS_TARGETS" "$LOG_DIR/archive_src/$SEGMAP_CLASS_TARGETS_FILENAME"
fi

ARCHIVE="$LOG_DIR/archive_${LANE_ID}.zip"
"$PYBIN" -c "
import os, zipfile
src = '$LOG_DIR/archive_src'
items = sorted(os.listdir(src))
with zipfile.ZipFile('$ARCHIVE', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in items:
        p = os.path.join(src, n)
        info = zipfile.ZipInfo(filename=n, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (0o644 & 0xFFFF) << 16
        with open(p, 'rb') as fh:
            z.writestr(info, fh.read(), compresslevel=9)
print('archive bytes:', os.path.getsize('$ARCHIVE'))
"

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo 'FATAL: archive empty'; exit 2; }
UPSTREAM_BYTES=$(stat -c '%s' "$UPSTREAM_ARCHIVE" 2>/dev/null || stat -f '%z' "$UPSTREAM_ARCHIVE")
log "archive_bytes=$ARCHIVE_BYTES (upstream was $UPSTREAM_BYTES)"

# Lane SH dispatch arm — segmap_arithmetic (NEW, added 2026-04-29).
INFLATE_CONFIG="$LOG_DIR/lane_sh_config.env"
cat > "$INFLATE_CONFIG" <<'EOF'
PYTHON_INFLATE=segmap_arithmetic
SEGMAP_GRAYSCALE_MODE=soft_lut
EOF
echo "SEGMAP_ARCH=$SEGMAP_ARCH" >> "$INFLATE_CONFIG"
if [ -f "$LOG_DIR/archive_src/$SEGMAP_CLASS_TARGETS_FILENAME" ]; then
    echo "SEGMAP_CLASS_TARGETS_FILENAME=$SEGMAP_CLASS_TARGETS_FILENAME" >> "$INFLATE_CONFIG"
fi

log "=== Stage 4: contest_auth_eval [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
set +e
CONFIG_ENV_PATH="$INFLATE_CONFIG" "$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

"$PYBIN" -c "
import json, os, time
prov = json.load(open('$PROVENANCE'))
prov['completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
prov['archive_path'] = '$ARCHIVE'
prov['archive_bytes'] = os.path.getsize('$ARCHIVE')
prov['upstream_archive_bytes'] = os.path.getsize('$UPSTREAM_ARCHIVE')
prov['arithmetic_payload_bytes'] = os.path.getsize('$ARITH_PAYLOAD')
prov['lane_status'] = 'COMPLETE'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_SH_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
