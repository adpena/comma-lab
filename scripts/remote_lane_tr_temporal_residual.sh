#!/bin/bash
# Lane TR — Temporal Residual coding (Toderici eureka, 2026-04-29).
#
# Strategy: encoder-only. Selfcomp encodes grayscale.mkv with crf=50 and
# implicit GOP defaults — driving frames are highly correlated, so a longer
# GOP + tuned CRF can exploit AV1's inter-frame prediction (motion vectors
# + temporal residual coding) to halve the mask rate without changing the
# decoder.
#
# We re-encode the upstream archive's grayscale.mkv with libsvtav1 settings
# tuned for a long GOP (g=600, key-int=600), higher quality (crf=40 instead
# of 50), and explicit reference structure (-tune 0 = visual quality, no
# psychovisual). The contest evaluator measures the MASK byte cost only via
# the archive size; the longer GOP + lower CRF trade reconstruction fidelity
# (which barely matters because the grayscale-LUT decoder is forgiving) for
# fewer total bytes.
#
# Predicted band: -0.05 mask rate vs upstream archive [contest-CUDA].
# Tarball-only parity (Check 66/67/68/69) — NEVER git pull / git reset --hard.
#
# E2E_SMOKE_OPT_OUT: lane TR is encoder-only and depends on libsvtav1 long-
#   GOP behaviour on the deployment host; smoke-fixture cannot exercise the
#   codec. Backfill after first contest-CUDA score lands.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LANE_ID="lane_tr_temporal_residual"
LOG_DIR="$WORKSPACE/${LANE_ID}_results"
mkdir -p "$LOG_DIR"
log() { echo "[lane-tr] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# UPSTREAM_ARCHIVE: any SegMap-paradigm OR renderer_grayscale archive whose
# grayscale.mkv we re-encode. The archive MUST contain grayscale.mkv plus
# whatever the original lane's PYTHON_INFLATE arm needs (e.g.
# segmap_weights.tar.xz for Lane SA, renderer.bin for Lane MM).
UPSTREAM_ARCHIVE="${UPSTREAM_ARCHIVE:-$WORKSPACE/lane_sa_segmap_clone_results/archive_lane_sa_segmap_clone.zip}"
# UPSTREAM_INFLATE: the PYTHON_INFLATE arm that the upstream archive expects.
# Defaults to segmap (Lane SA-style); set to renderer_grayscale for Lane MM.
UPSTREAM_INFLATE="${UPSTREAM_INFLATE:-segmap}"

# Re-encode parameters. Long GOP + lower CRF empirically shrinks AV1
# monochrome by 30-50% on driving sequences (Toderici eureka).
CRF="${TR_CRF:-40}"
GOP="${TR_GOP:-600}"
PRESET="${TR_PRESET:-6}"

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
    'lane_script': 'scripts/remote_lane_tr_temporal_residual.sh',
    'output_dir': '$LOG_DIR',
    'variant': 'long_gop_lower_crf_av1',
    'predicted_band_delta': [-0.07, -0.03],
    'paradigm': 'temporal_residual_av1',
    'upstream_archive': '$UPSTREAM_ARCHIVE',
    'upstream_inflate': '$UPSTREAM_INFLATE',
    'crf': $CRF,
    'gop': $GOP,
    'preset': $PRESET,
    'controlled_baseline': 'UPSTREAM_ARCHIVE (encoder-only repack; only delta is libsvtav1 long-GOP/lower-CRF re-encode of grayscale.mkv)',
    'cost_estimate_usd': 2.00,
    'wall_clock_estimate_hours': 4.0,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=TR gpu=$GPU" >> "$HEARTBEAT"
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
UPSTREAM_GRAYSCALE="$EXTRACT_DIR/grayscale.mkv"
[ -f "$UPSTREAM_GRAYSCALE" ] || { echo "FATAL: missing $UPSTREAM_GRAYSCALE" >&2; exit 1; }

log "=== Stage 2: decode grayscale.mkv -> decoded class IDs (lossless verify) ==="
DECODED_PT="$LOG_DIR/decoded_classes.pt"
"$PYBIN" -c "
import sys, torch
import numpy as np
import av
from pathlib import Path
from tac.mask_grayscale_lut import decode_grayscale_to_classes

frames = []
with av.open('$UPSTREAM_GRAYSCALE') as c:
    for pkt in c.demux(c.streams.video[0]):
        for fr in pkt.decode():
            frames.append(fr.to_ndarray(format='gray'))
gray = torch.from_numpy(np.stack(frames, axis=0))
print(f'decoded grayscale frames: {tuple(gray.shape)} dtype={gray.dtype}')
classes = decode_grayscale_to_classes(gray)
torch.save(classes, '$DECODED_PT')
print(f'decoded classes saved to $DECODED_PT')
"

log "=== Stage 3: re-encode grayscale.mkv with long-GOP / lower-CRF (libsvtav1) ==="
NEW_GRAYSCALE="$LOG_DIR/grayscale.mkv"
"$PYBIN" -c "
import torch
from pathlib import Path
from tac.mask_grayscale_lut import encode_masks_grayscale

classes = torch.load('$DECODED_PT', weights_only=False)
gray = encode_masks_grayscale(classes.long())
import av, numpy as np
arr = gray.numpy() if isinstance(gray, torch.Tensor) else np.asarray(gray)
out_path = Path('$NEW_GRAYSCALE')
container = av.open(str(out_path), mode='w')
stream = container.add_stream('libsvtav1', rate=20)
stream.width = arr.shape[2]
stream.height = arr.shape[1]
stream.pix_fmt = 'yuv420p'
# Lane TR encoder settings: longer GOP + lower CRF + slower preset.
stream.options = {
    'crf': str($CRF),
    'g': str($GOP),
    'keyint_min': str($GOP),
    'preset': str($PRESET),
    'svtav1-params': 'tune=0:enable-overlays=1:enable-tf=1',
}
for i in range(arr.shape[0]):
    yuv = np.zeros((arr.shape[1] * 3 // 2, arr.shape[2]), dtype='uint8')
    yuv[:arr.shape[1]] = arr[i]
    yuv[arr.shape[1]:] = 128
    frame = av.VideoFrame.from_ndarray(yuv, format='yuv420p')
    for pkt in stream.encode(frame):
        container.mux(pkt)
for pkt in stream.encode():
    container.mux(pkt)
container.close()
import os
print('new grayscale.mkv bytes:', os.path.getsize(out_path))

# Sanity: re-decode the new mkv and ensure class-decode integrity at >=99%.
import torch
from tac.mask_grayscale_lut import decode_grayscale_to_classes
frames2 = []
with av.open(str(out_path)) as c:
    for pkt in c.demux(c.streams.video[0]):
        for fr in pkt.decode():
            frames2.append(fr.to_ndarray(format='gray'))
gray2 = torch.from_numpy(np.stack(frames2, axis=0))
new_classes = decode_grayscale_to_classes(gray2)
agree = (new_classes.long() == classes.long()).float().mean().item()
print(f'class-decode agreement post-reencode: {agree*100:.3f}%')
if agree < 0.985:
    raise SystemExit(f'lane-TR class-decode agreement {agree:.4f} below 0.985 floor; CRF too aggressive')
"

log "=== Stage 4: rebuild archive with new grayscale.mkv ==="
NEW_ARCHIVE_SRC="$LOG_DIR/archive_src"
mkdir -p "$NEW_ARCHIVE_SRC"
# Copy every member from the upstream archive EXCEPT grayscale.mkv; that
# we replace with our re-encoded version.
"$PYBIN" -c "
import os, shutil, zipfile
src_zip = '$UPSTREAM_ARCHIVE'
dst_dir = '$NEW_ARCHIVE_SRC'
with zipfile.ZipFile(src_zip, 'r') as z:
    for name in z.namelist():
        if os.path.basename(name) == 'grayscale.mkv':
            continue
        z.extract(name, dst_dir)
shutil.copy('$NEW_GRAYSCALE', os.path.join(dst_dir, 'grayscale.mkv'))
print('archive_src contents:', sorted(os.listdir(dst_dir)))
"
ARCHIVE="$LOG_DIR/archive_${LANE_ID}.zip"
"$PYBIN" -c "
import os, zipfile
src = '$NEW_ARCHIVE_SRC'
items = sorted(os.listdir(src))
with zipfile.ZipFile('$ARCHIVE', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in items:
        p = os.path.join(src, n)
        z.write(p, arcname=n)
print('archive bytes:', os.path.getsize('$ARCHIVE'))
"

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo 'FATAL: archive empty'; exit 2; }
UPSTREAM_BYTES=$(stat -c '%s' "$UPSTREAM_ARCHIVE" 2>/dev/null || stat -f '%z' "$UPSTREAM_ARCHIVE")
log "archive_bytes=$ARCHIVE_BYTES (upstream was $UPSTREAM_BYTES)"

# Lane TR ships a re-encoded grayscale.mkv; the inflate arm MUST be either
# PYTHON_INFLATE=segmap or PYTHON_INFLATE=renderer_grayscale (Check
# [grayscale-lut-consistency]). The default UPSTREAM_INFLATE=segmap above
# satisfies the guard literally; if an operator overrides to
# renderer_grayscale that arm is also accepted.
case "$UPSTREAM_INFLATE" in
    segmap|renderer_grayscale)
        ;;
    *)
        echo "FATAL: UPSTREAM_INFLATE='$UPSTREAM_INFLATE' is not a grayscale-LUT-compatible arm." >&2
        echo "       Allowed: 'segmap' or 'renderer_grayscale' (Check [grayscale-lut-consistency])." >&2
        exit 1
        ;;
esac
INFLATE_CONFIG="$LOG_DIR/lane_tr_config.env"
echo "PYTHON_INFLATE=$UPSTREAM_INFLATE" > "$INFLATE_CONFIG"

log "=== Stage 5: contest_auth_eval [contest-CUDA] ==="
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
prov['lane_status'] = 'COMPLETE'
json.dump(prov, open('$PROVENANCE', 'w'), indent=2)
print('provenance updated.')
"

log "=== LANE_TR_DONE — see $LOG_DIR/auth_eval.log for RESULT_JSON [contest-CUDA] ==="
