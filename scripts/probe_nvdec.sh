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
set -euo pipefail

PYBIN="${PYBIN:-/opt/conda/bin/python}"

if ! "$PYBIN" -c "import nvidia.dali" 2>/dev/null; then
    echo "[probe_nvdec] FATAL: nvidia.dali not installed on $PYBIN" >&2
    echo "[probe_nvdec]   Run: $PYBIN -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120" >&2
    exit 1
fi

# The actual probe: build a minimal DALI pipeline that exercises the video
# MIXED operator. If NVDEC is exposed, build() succeeds. If not, it raises
# with CUDA_ERROR_NO_DEVICE.
PROBE_OUT=$("$PYBIN" -c "
from nvidia.dali import pipeline_def, fn
@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def p():
    return fn.experimental.inputs.video(name='x', sequence_length=2, device='mixed')
pipe = p()
pipe.build()
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
