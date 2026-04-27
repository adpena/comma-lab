#!/bin/bash
# probe_nvdec.sh — single-purpose NVDEC exposure probe for Vast.ai instances.
#
# Per memory feedback_vastai_nvdec_host_variation: Vast.ai 4090 instances on
# the same image / driver can have different NVDEC exposure. Compute CUDA may
# work fine while DALI's `fn.experimental.inputs.video` MIXED operator fails
# with `CUDA_ERROR_NO_DEVICE (100)`. The probe runs DALI's video pipeline
# build at minimum batch — if NVDEC is missing, build() raises and we exit
# non-zero so the calling lane script can refuse to proceed.
#
# Cost: ~5 seconds wall on a healthy host. ~$3-4 of GPU dollars saved every
# time it catches a bad host before pose TTO / training begins.
#
# Usage from a lane script:
#   bash scripts/probe_nvdec.sh || {
#       echo "FATAL: NVDEC probe failed — host cannot run upstream/evaluate.py"
#       exit 2
#   }
#
# Idempotent. Safe to call multiple times.
#
# Exit codes:
#   0 — NVDEC available, downstream auth eval will work
#   1 — DALI not installed (dependency error, not a host problem)
#   2 — NVDEC missing on this host (kill instance, pick a different one)
#
# Flags:
#   --ensure-dali  — if DALI is missing, install it and re-check (instead
#                    of exit 1). Use this from bootstraps that run on a
#                    fresh container that hasn't run remote_setup_full.sh.
#                    Codex R5-3-round-4 fix: closes the probe-before-DALI
#                    gap that bricked remote_train_bootstrap.sh and
#                    remote_pose_tto_bootstrap.sh on fresh hosts.
set -euo pipefail

ENSURE_DALI=0
for arg in "$@"; do
    case "$arg" in
        --ensure-dali) ENSURE_DALI=1 ;;
        *) echo "[probe_nvdec] unknown arg: $arg" >&2; exit 1 ;;
    esac
done

PYBIN="${PYBIN:-/opt/conda/bin/python}"

if ! "$PYBIN" -c "import nvidia.dali" 2>/dev/null; then
    if [ "$ENSURE_DALI" = "1" ]; then
        echo "[probe_nvdec] DALI missing; --ensure-dali set, installing..." >&2
        "$PYBIN" -m pip install -q --extra-index-url \
            https://developer.download.nvidia.com/compute/redist \
            nvidia-dali-cuda120 2>&1 | tail -3 >&2
        if ! "$PYBIN" -c "import nvidia.dali" 2>/dev/null; then
            echo "[probe_nvdec] FATAL: install completed but import still fails" >&2
            exit 1
        fi
        echo "[probe_nvdec] DALI installed; continuing to probe..." >&2
    else
        echo "[probe_nvdec] FATAL: nvidia.dali not installed on $PYBIN" >&2
        echo "[probe_nvdec]   Run: $PYBIN -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120" >&2
        echo "[probe_nvdec]   OR call this script with --ensure-dali to auto-install" >&2
        exit 1
    fi
fi

# The actual probe: build a minimal DALI pipeline that exercises the video
# MIXED operator AND actually decodes a frame (share_outputs).
#
# 2026-04-27 STRENGTHENED probe (Lane A relaunch finding):
# The previous probe only called pipe.build(), which validates handle
# allocation but does NOT exercise the NVDEC hardware decode path. A
# Hungary host PASSED build() but FAILED at share_outputs() during the
# Lane A auth-eval relaunch — wasting 22 min and a CPU-fallback inflate.
# The strengthened probe synthesizes a tiny in-memory MP4 (1 frame, 2x2
# pixel) using PyAV, feeds it to DALI via fn.experimental.inputs.video,
# runs a full schedule_run + share_outputs cycle, and verifies the
# decoded output tensor has the expected shape. If NVDEC is missing,
# share_outputs() raises CUDA_ERROR_NO_DEVICE here, NOT at the Lane A
# auth-eval invocation 22 min later.
PROBE_OUT=$("$PYBIN" -c "
import io, av, numpy as np
import torch  # noqa: F401  # ensure CUDA context inits before DALI

# Build a tiny in-memory MP4 (1 frame, 2x2 pixel) — forces NVDEC to
# actually decode something rather than just allocate handles.
buf = io.BytesIO()
container = av.open(buf, mode='w', format='mp4')
stream = container.add_stream('h264', rate=1)
stream.width, stream.height = 16, 16
stream.pix_fmt = 'yuv420p'
for _ in range(2):
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
    for packet in stream.encode(frame): container.mux(packet)
for packet in stream.encode(): container.mux(packet)
container.close()
buf.seek(0)
video_bytes = buf.read()

from nvidia.dali import pipeline_def, fn
import nvidia.dali.types as dali_types
@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def p():
    return fn.experimental.inputs.video(name='x', sequence_length=2, device='mixed')
pipe = p()
pipe.build()
# RUNTIME verification — this is what the previous probe missed.
pipe.feed_input('x', [np.frombuffer(video_bytes, dtype=np.uint8)])
pipe.schedule_run()
out = pipe.share_outputs()
shape = out[0].shape()
pipe.release_outputs()
assert shape == (1, 2, 16, 16, 3), f'unexpected NVDEC output shape: {shape}'
print('NVDEC_OK')
" 2>&1) || {
    echo "[probe_nvdec] FATAL: NVDEC probe failed on this host." >&2
    echo "[probe_nvdec]   Output: $PROBE_OUT" >&2
    echo "[probe_nvdec]   This Vast.ai host has compute CUDA but NOT NVDEC." >&2
    echo "[probe_nvdec]   DALI's video MIXED operator will fail at upstream/evaluate.py." >&2
    echo "[probe_nvdec]   Action: vastai destroy instance \$INSTANCE_ID && pick a different host." >&2
    echo "[probe_nvdec]   Reference: feedback_vastai_nvdec_host_variation memory entry." >&2
    exit 2
}

if [[ "$PROBE_OUT" != *"NVDEC_OK"* ]]; then
    echo "[probe_nvdec] FATAL: probe exit OK but no NVDEC_OK marker — output: $PROBE_OUT" >&2
    exit 2
fi

echo "[probe_nvdec] OK (NVDEC exposed, DALI video pipeline buildable)"
