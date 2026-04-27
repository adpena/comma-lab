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
#
# 2026-04-27 codex R5-4 #1 fix: the previous strengthened probe imported
# PyAV (`av`) to synthesize the test MP4 in-memory. On a fresh container
# without PyAV, the import failed and the probe exit-2'd, falsely
# reporting "host has no NVDEC" when the real failure was a missing
# Python dep. Fix: embed a tiny pre-built MP4 (16x16, 2 frames, h264/
# yuv420p) as base64 — no external Python deps needed beyond DALI's
# own torch+numpy. The fixture was generated once via:
#   ffmpeg -y -f lavfi -i color=black:s=16x16:d=2:r=1 \
#     -pix_fmt yuv420p -c:v libx264 -t 2 /tmp/tiny.mp4
#   base64 /tmp/tiny.mp4
# Total embedded payload: ~2.1 KiB. Decoded at runtime by stdlib base64.
PROBE_OUT=$("$PYBIN" -c "
import base64, numpy as np
import torch  # noqa: F401  # ensure CUDA context inits before DALI

# Pre-built tiny MP4 (16x16, 2 frames @ 1 fps, yuv420p, h264). Generated
# locally with ffmpeg; never regenerated at runtime. Forces NVDEC to
# actually decode something rather than just allocate handles. PyAV is
# NOT imported here — that would make a missing PyAV install masquerade
# as a missing-NVDEC host failure (codex R5-4 #1).
TINY_MP4_B64 = (
    'AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAtltZGF0AAACrQYF'
    '//+p3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NSByMzIyMiBiMzU2MDVhIC0gSC4y'
    'NjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAyNSAtIGh0dHA6Ly93d3cu'
    'dmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9j'
    'az0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9'
    'MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9'
    'MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3Fw'
    'X29mZnNldD0tMiB0aHJlYWRzPTEgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFk'
    'cz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0'
    'cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9'
    'MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBr'
    'ZXlpbnRfbWluPTEgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00'
    'MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5'
    'IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAQZYiEABb//vfTP8yy7Jol'
    'gQAAAAhBmiFsQV/+8AAAAzFtb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPoAAAH0AABAAAB'
    'AAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAA'
    'AAAAAAAAAAAAAAAAAAAAAAAAAAACAAACW3RyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAAAAEA'
    'AAAAAAAH0AAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAA'
    'AEAAAAAAEAAAABAAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAB9AAAAAAAAEAAAAAAdNt'
    'ZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAAEAAAACAAFXEAAAAAAAtaGRscgAAAAAAAAAAdmlk'
    'ZQAAAAAAAAAAAAAAAFZpZGVvSGFuZGxlcgAAAAF+bWluZgAAABR2bWhkAAAAAQAAAAAAAAAA'
    'AAAAJGRpbmYAAAAcZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAABPnN0YmwAAAC+c3RzZAAA'
    'AAAAAAABAAAArmF2YzEAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAEAAQAEgAAABIAAAAAAAA'
    'AAEVTGF2YzYyLjI4LjEwMCBsaWJ4MjY0AAAAAAAAAAAAAAAY//8AAAA0YXZjQwFkAAr/4QAX'
    'Z2QACqzZXsBEAAADAAQAAAMACDxIllgBAAZo6+PLIsD9+PgAAAAAEHBhc3AAAAABAAAAAQAA'
    'ABRidHJ0AAAAAAAAC0QAAAAAAAAAGHN0dHMAAAAAAAAAAQAAAAIAAEAAAAAAFHN0c3MAAAAA'
    'AAAAAQAAAAEAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAIAAAABAAAAHHN0c3oAAAAAAAAAAAAA'
    'AAIAAALFAAAADAAAABRzdGNvAAAAAAAAAAEAAAAwAAAAYnVkdGEAAABabWV0YQAAAAAAAAAh'
    'aGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEA'
    'AAABAAAAAExhdmY2Mi4xMi4xMDA='
)
video_bytes = base64.b64decode(TINY_MP4_B64)

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
