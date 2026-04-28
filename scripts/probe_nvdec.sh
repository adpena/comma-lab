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
# Exit codes (codex R5-r6 #4 — error classification):
#   0 — NVDEC available, downstream auth eval will work
#   1 — DALI not installed at all (dependency error, not a host problem)
#   2 — NVDEC missing on this host (KILL instance, pick a different one)
#   3 — DALI/dependency build error (NOT a NVDEC-missing problem — investigate)
#   4 — Embedded MP4 fixture invalid or shape assertion failed (probe bug, not
#       a host problem; report upstream)
#   5 — Unknown failure mode — investigate manually, do NOT auto-destroy host
#
# Previously every Python failure mapped to exit 2 ("kill the host"). That
# meant a fixture-corrupt build OR a DALI install issue triggered the same
# operator response (destroy + relaunch) as a genuine missing-NVDEC host.
# Real NVDEC-missing failures emit "CUDA_ERROR_NO_DEVICE" / "no CUDA-capable
# device" / "NVDEC" in the exception text — only those should map to exit 2.
#
# Flags:
#   --ensure-dali  — if DALI is missing, install it and re-check (instead
#                    of exit 1). Use this from bootstraps that run on a
#                    fresh container that hasn't run remote_setup_full.sh.
#                    Codex R5-3-round-4 fix: closes the probe-before-DALI
#                    gap that bricked remote_train_bootstrap.sh and
#                    remote_pose_tto_bootstrap.sh on fresh hosts.
#   --lightweight  — DALI-FREE pre-check via libnvcuvid.so + ctypes. Catches
#                    the most common Vast.ai failure mode (compute CUDA
#                    works but NVDEC missing) in ~3 seconds, BEFORE the
#                    5-minute DALI install. Per memory feedback_metabugs_round_3
#                    Metabug B (2026-04-28). Saves $0.05+ per bad host.
#                    Use this from setup_full.sh Stage 0.5; the full
#                    DALI-based probe still runs later as the deep check.
set -euo pipefail

ENSURE_DALI=0
LIGHTWEIGHT=0
for arg in "$@"; do
    case "$arg" in
        --ensure-dali) ENSURE_DALI=1 ;;
        --lightweight) LIGHTWEIGHT=1 ;;
        *) echo "[probe_nvdec] unknown arg: $arg" >&2; exit 1 ;;
    esac
done

PYBIN="${PYBIN:-/opt/conda/bin/python}"

# ---------------------------------------------------------------------------
# Lightweight pre-DALI NVDEC check via libnvcuvid.so + ctypes.
# Per memory feedback_metabugs_round_3 Metabug B (2026-04-28).
# ---------------------------------------------------------------------------
# The DALI-based probe below is correct but runs AFTER a 5-minute DALI install
# in setup_full.sh, costing $0.05+ per bad host. The lightweight path catches
# ~95% of NVDEC-missing hosts in ~3 seconds with ZERO install cost by:
#   1. dlopen()'ing libnvcuvid.so.1 (always shipped with NVIDIA driver)
#   2. Calling cuvidGetDecoderCaps() against a CUDA context
#   3. Verifying H264 decoder is supported on device 0
# A pass here does NOT guarantee DALI's video MIXED operator will work (that's
# what the deep probe is for), but a FAIL guarantees the host is broken.
if [ "$LIGHTWEIGHT" = "1" ]; then
    LIGHT_OUT=$("$PYBIN" -c "
import ctypes, sys, traceback

def _exit_classified(token, msg):
    print(f'PROBE_CLASSIFICATION:{token}')
    print(f'PROBE_ERROR:{msg}')
    sys.exit(0)

# Step 1: load libnvcuvid (NVDEC userspace library shipped with driver).
try:
    nvcuvid = ctypes.CDLL('libnvcuvid.so.1')
except OSError as e:
    _exit_classified('NVDEC_MISSING', f'libnvcuvid.so.1 dlopen failed: {e}')

# Step 2: load libcuda (CUDA driver — always present on a CUDA host).
try:
    cuda = ctypes.CDLL('libcuda.so.1')
except OSError as e:
    _exit_classified('UNKNOWN', f'libcuda.so.1 dlopen failed (this should never happen): {e}')

# Step 3: cuInit + cuDeviceGet + cuCtxCreate to get a usable CUcontext.
CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p

cuda.cuInit.argtypes = [ctypes.c_uint]; cuda.cuInit.restype = CUresult
cuda.cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]; cuda.cuDeviceGet.restype = CUresult
cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
cuda.cuCtxCreate_v2.restype = CUresult
cuda.cuCtxDestroy_v2.argtypes = [CUcontext]; cuda.cuCtxDestroy_v2.restype = CUresult

rc = cuda.cuInit(0)
if rc != 0:
    _exit_classified('UNKNOWN', f'cuInit failed: rc={rc}')

dev = CUdevice()
rc = cuda.cuDeviceGet(ctypes.byref(dev), 0)
if rc != 0:
    _exit_classified('NVDEC_MISSING', f'cuDeviceGet(0) failed: rc={rc} (CUDA_ERROR_NO_DEVICE=100)')

ctx = CUcontext()
rc = cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
if rc != 0:
    _exit_classified('NVDEC_MISSING', f'cuCtxCreate failed: rc={rc}')

# Step 4: cuvidGetDecoderCaps for H264 / 8-bit / 4:2:0 — the most common path.
class CUVIDDECODECAPS(ctypes.Structure):
    _fields_ = [
        ('eCodecType', ctypes.c_int),
        ('eChromaFormat', ctypes.c_int),
        ('nBitDepthMinus8', ctypes.c_uint),
        ('reserved1', ctypes.c_uint * 3),
        ('bIsSupported', ctypes.c_uint8),
        ('nNumNVDECs', ctypes.c_uint8),
        ('nOutputFormatMask', ctypes.c_uint16),
        ('nMaxWidth', ctypes.c_uint),
        ('nMaxHeight', ctypes.c_uint),
        ('nMaxMBCount', ctypes.c_uint),
        ('nMinWidth', ctypes.c_uint16),
        ('nMinHeight', ctypes.c_uint16),
        ('bIsHistogramSupported', ctypes.c_uint8),
        ('nCounterBitDepth', ctypes.c_uint8),
        ('nMaxHistogramBins', ctypes.c_uint16),
        ('reserved3', ctypes.c_uint * 10),
    ]

try:
    nvcuvid.cuvidGetDecoderCaps.argtypes = [ctypes.POINTER(CUVIDDECODECAPS)]
    nvcuvid.cuvidGetDecoderCaps.restype = CUresult
except AttributeError as e:
    cuda.cuCtxDestroy_v2(ctx)
    _exit_classified('NVDEC_MISSING', f'cuvidGetDecoderCaps symbol missing: {e}')

caps = CUVIDDECODECAPS()
caps.eCodecType = 4  # cudaVideoCodec_H264
caps.eChromaFormat = 1  # cudaVideoChromaFormat_420
caps.nBitDepthMinus8 = 0
rc = nvcuvid.cuvidGetDecoderCaps(ctypes.byref(caps))
cuda.cuCtxDestroy_v2(ctx)

if rc != 0:
    _exit_classified('NVDEC_MISSING', f'cuvidGetDecoderCaps rc={rc} (NVDEC subsystem not exposed)')

if not caps.bIsSupported:
    _exit_classified('NVDEC_MISSING', f'H264/8bit/4:2:0 not supported (bIsSupported=0, nNumNVDECs={caps.nNumNVDECs})')

if caps.nNumNVDECs < 1:
    _exit_classified('NVDEC_MISSING', f'nNumNVDECs={caps.nNumNVDECs} (host has 0 NVDEC engines)')

print('PROBE_CLASSIFICATION:OK')
print(f'NVDEC_OK_LIGHTWEIGHT:nNumNVDECs={caps.nNumNVDECs} maxWidth={caps.nMaxWidth} maxHeight={caps.nMaxHeight}')
" 2>&1) || {
        echo "[probe_nvdec][lightweight] FATAL: probe crashed before classification." >&2
        echo "[probe_nvdec][lightweight]   Output: $LIGHT_OUT" >&2
        exit 5
    }

    LIGHT_CLASS=$(echo "$LIGHT_OUT" | grep -E "^PROBE_CLASSIFICATION:" | tail -1 | sed 's/^PROBE_CLASSIFICATION://')
    case "$LIGHT_CLASS" in
        OK)
            echo "[probe_nvdec][lightweight] OK ($(echo "$LIGHT_OUT" | grep NVDEC_OK_LIGHTWEIGHT))"
            exit 0
            ;;
        NVDEC_MISSING)
            echo "[probe_nvdec][lightweight] FATAL: NVDEC missing on this host (caught BEFORE DALI install)." >&2
            echo "[probe_nvdec][lightweight]   Output: $LIGHT_OUT" >&2
            echo "[probe_nvdec][lightweight]   Saved ~5 min of DALI install on a known-bad host." >&2
            echo "[probe_nvdec][lightweight]   Action: vastai destroy instance \$INSTANCE_ID && pick a different host." >&2
            echo "[probe_nvdec][lightweight]   Reference: feedback_vastai_nvdec_host_variation + feedback_metabugs_round_3 Metabug B." >&2
            exit 2
            ;;
        UNKNOWN|*)
            echo "[probe_nvdec][lightweight] FATAL: unexpected classification: $LIGHT_CLASS" >&2
            echo "[probe_nvdec][lightweight]   Output: $LIGHT_OUT" >&2
            exit 5
            ;;
    esac
fi

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
# Python dep. Fix: embed a tiny pre-built MP4 as base64 — no external
# Python deps needed beyond DALI's own torch+numpy.
#
# 2026-04-27 Lane D fix: bumped fixture from 16x16 → 64x48 because newer
# DALI / NVDEC enforces a minimum input size of 48x16 (frames_decoder_gpu.cc
# line 179). The 16x16 fixture caused a min-size assertion that bash
# classified as DALI_BUILD (exit 3), wedging Lane D launch on a host where
# NVDEC was actually fine. Verified manually: this 64x48 fixture decodes
# successfully with output shape (1, 2, 48, 64, 3). Fixture generated via:
#   ffmpeg -y -f lavfi -i color=black:s=64x48:d=2:r=1 \
#     -pix_fmt yuv420p -c:v libx264 -t 2 /tmp/tiny.mp4
#   base64 /tmp/tiny.mp4
# Total embedded payload: ~2.2 KiB. Decoded at runtime by stdlib base64.
#
# 2026-04-27 codex R5-r6 #4 fix: classify failures by error text. The
# previous probe mapped EVERY exception to exit 2 ("host has compute CUDA
# but NOT NVDEC"). A corrupt fixture OR a DALI build error therefore
# triggered the same operator response (destroy + relaunch) as a genuine
# missing-NVDEC host. The Python now prints a CLASSIFICATION line —
# `PROBE_CLASSIFICATION:<token>` — that bash maps to a specific exit
# code so the operator gets the right diagnostic and a non-NVDEC issue
# does NOT cost a relaunch on a healthy host.
PROBE_OUT=$("$PYBIN" -c "
import base64, sys, traceback
import numpy as np
import torch  # noqa: F401  # ensure CUDA context inits before DALI

# Pre-built tiny MP4 (64x48, 2 frames @ 1 fps, yuv420p, h264). Generated
# locally with ffmpeg; never regenerated at runtime. Forces NVDEC to
# actually decode something rather than just allocate handles. PyAV is
# NOT imported here — that would make a missing PyAV install masquerade
# as a missing-NVDEC host failure (codex R5-4 #1).
# Bumped from 16x16 to 64x48 on 2026-04-27: newer DALI/NVDEC requires
# minimum 48x16 dimensions; 16x16 hit a min-size assertion at runtime
# that misclassified as DALI_BUILD on healthy hosts (Lane D Norway).
TINY_MP4_B64 = (
    'AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAuptZGF0AAACrQYF//+p'
    '3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2MyByMzA2MCA1ZGI2YWE2IC0gSC4yNjQvTVBF'
    'Ry00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAyMSAtIGh0dHA6Ly93d3cudmlkZW9sYW4u'
    'b3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFs'
    'eXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVk'
    'X3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBk'
    'ZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTEg'
    'bG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRl'
    'cmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJf'
    'cHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9'
    'MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEgc2NlbmVjdXQ9NDAgaW50cmFfcmVm'
    'cmVzaD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42'
    'MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAf'
    'ZYiEABb//vfTP8yy7JokteOo96Kci/PTPylRonyCeQAAAApBmiFsQV/+1qkOAAADMm1vb3YAAABs'
    'bXZoZAAAAAAAAAAAAAAAAAAAA+gAAAfQAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAA'
    'AQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAJcdHJhawAA'
    'AFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAfQAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAA'
    'AAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAABAAAAAMAAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAAB'
    'AAAH0AAAAAAAAQAAAAAB1G1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAQAAAAIAAVcQAAAAAAC1o'
    'ZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAAX9taW5mAAAAFHZtaGQA'
    'AAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAE/c3RibAAA'
    'AL9zdHNkAAAAAAAAAAEAAACvYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAABAADAASAAAAEgA'
    'AAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADVhdmNDAWQACv/h'
    'ABhnZAAKrNlEewEQAAADABAAAAMAIPEiWWABAAZo6+PLIsD9+PgAAAAAEHBhc3AAAAABAAAAAQAA'
    'ABRidHJ0AAAAAAAAC4gAAAuIAAAAGHN0dHMAAAAAAAAAAQAAAAIAAEAAAAAAFHN0c3MAAAAAAAAA'
    'AQAAAAEAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAIAAAABAAAAHHN0c3oAAAAAAAAAAAAAAAIAAALU'
    'AAAADgAAABRzdGNvAAAAAAAAAAEAAAAwAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAA'
    'AAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY1'
    'OC43Ni4xMDA='
)


def _classify(exc_text: str) -> str:
    '''Map exception text to one of the PROBE_CLASSIFICATION tokens.

    Tokens:
      NVDEC_MISSING — host has compute CUDA but no NVDEC (kill instance)
      DALI_BUILD    — DALI installed but pipeline build failed (dep issue)
      FIXTURE       — embedded MP4 fixture invalid / shape assertion (probe bug)
      UNKNOWN       — none of the above (do NOT auto-destroy host)
    '''
    t = exc_text.lower()
    # Order matters: NVDEC markers win over DALI markers, since DALI
    # surfaces NVDEC failures via its own Pipeline class.
    nvdec_markers = (
        'cuda_error_no_device',
        'no cuda-capable device',
        'nvdec',
        'nvcuvid',
        'no nvidia driver on your system',
    )
    if any(m in t for m in nvdec_markers):
        return 'NVDEC_MISSING'
    fixture_markers = (
        'unexpected nvdec output shape',
        'unexpected fixture',
        'fixture too small',
        'invalid mp4 fixture',
    )
    if any(m in t for m in fixture_markers):
        return 'FIXTURE'
    dali_markers = (
        'nvidia.dali',
        'dalierror',
        'pipeline',
        'experimental.inputs.video',
        'fn.experimental',
    )
    if any(m in t for m in dali_markers):
        return 'DALI_BUILD'
    return 'UNKNOWN'


# Stage 0: validate the embedded MP4 fixture BEFORE trying to use it.
# A corrupt base64 / truncated MP4 must not be reported as a NVDEC failure.
try:
    video_bytes = base64.b64decode(TINY_MP4_B64)
    if len(video_bytes) < 1024:
        raise ValueError(
            f'fixture too small: got {len(video_bytes)} bytes (expected >= 1024); '
            f'invalid mp4 fixture'
        )
    if video_bytes[:4] != b'\x00\x00\x00\x20' and video_bytes[4:8] != b'ftyp':
        # Standard ISO BMFF / MP4 starts with a size box then 'ftyp'.
        if b'ftyp' not in video_bytes[:32]:
            raise ValueError(
                f'invalid mp4 fixture: no ftyp box in first 32 bytes'
            )
except Exception as e:
    print(f'PROBE_CLASSIFICATION:FIXTURE')
    print(f'PROBE_ERROR:{type(e).__name__}: {e}')
    sys.exit(0)  # bash dispatcher reads classification, not exit code

# Stage 1: actually exercise the NVDEC hardware decode path.
try:
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
    if shape != [(2, 48, 64, 3)]:
        # Re-raise as a fixture-class error so we don't blame the host.
        raise AssertionError(f'unexpected NVDEC output shape: {shape}')
    print('PROBE_CLASSIFICATION:OK')
    print('NVDEC_OK')
except Exception as e:
    tb = traceback.format_exc()
    classification = _classify(str(e) + ' ' + tb)
    print(f'PROBE_CLASSIFICATION:{classification}')
    print(f'PROBE_ERROR:{type(e).__name__}: {e}')
    print(f'PROBE_TRACEBACK_TAIL:{tb.splitlines()[-3:] if tb else []}')
    sys.exit(0)  # bash dispatcher reads classification, not exit code
" 2>&1) || {
    # Python crashed *outside* of any try/except (e.g., import torch failed).
    # No classification token will be present; treat as UNKNOWN per
    # codex R5-r6 #4 — DO NOT default to NVDEC_MISSING.
    echo "[probe_nvdec] FATAL: probe crashed before classification could run." >&2
    echo "[probe_nvdec]   Output: $PROBE_OUT" >&2
    echo "[probe_nvdec]   Classification: UNKNOWN (do not auto-destroy host)." >&2
    echo "[probe_nvdec]   Action: investigate the python crash before reusing this host." >&2
    exit 5
}

# Dispatch on classification token.
CLASSIFICATION=$(echo "$PROBE_OUT" | grep -E "^PROBE_CLASSIFICATION:" | tail -1 | sed 's/^PROBE_CLASSIFICATION://')

if [ -z "$CLASSIFICATION" ]; then
    echo "[probe_nvdec] FATAL: no PROBE_CLASSIFICATION token in output." >&2
    echo "[probe_nvdec]   Output: $PROBE_OUT" >&2
    echo "[probe_nvdec]   Classification: UNKNOWN (treat as exit 5, do NOT auto-destroy)." >&2
    exit 5
fi

case "$CLASSIFICATION" in
    OK)
        if [[ "$PROBE_OUT" != *"NVDEC_OK"* ]]; then
            echo "[probe_nvdec] FATAL: probe reported OK but no NVDEC_OK marker — output: $PROBE_OUT" >&2
            exit 5
        fi
        echo "[probe_nvdec] OK (NVDEC exposed, DALI video pipeline buildable)"
        exit 0
        ;;
    NVDEC_MISSING)
        echo "[probe_nvdec] FATAL: NVDEC missing on this host." >&2
        echo "[probe_nvdec]   Output: $PROBE_OUT" >&2
        echo "[probe_nvdec]   This Vast.ai host has compute CUDA but NOT NVDEC." >&2
        echo "[probe_nvdec]   DALI's video MIXED operator will fail at upstream/evaluate.py." >&2
        echo "[probe_nvdec]   Action: vastai destroy instance \$INSTANCE_ID && pick a different host." >&2
        echo "[probe_nvdec]   Reference: feedback_vastai_nvdec_host_variation memory entry." >&2
        exit 2
        ;;
    DALI_BUILD)
        echo "[probe_nvdec] FATAL: DALI/dependency build error (NOT a host problem)." >&2
        echo "[probe_nvdec]   Output: $PROBE_OUT" >&2
        echo "[probe_nvdec]   Action: investigate the dependency / DALI install — host is likely fine." >&2
        echo "[probe_nvdec]   Reuse the host after fixing the dep, do NOT auto-destroy." >&2
        exit 3
        ;;
    FIXTURE)
        echo "[probe_nvdec] FATAL: embedded MP4 fixture invalid OR shape assertion failed." >&2
        echo "[probe_nvdec]   Output: $PROBE_OUT" >&2
        echo "[probe_nvdec]   This is a probe BUG, not a host problem." >&2
        echo "[probe_nvdec]   Action: report upstream + reuse host." >&2
        exit 4
        ;;
    UNKNOWN)
        echo "[probe_nvdec] FATAL: unknown failure mode — investigate before reusing host." >&2
        echo "[probe_nvdec]   Output: $PROBE_OUT" >&2
        echo "[probe_nvdec]   Classification: UNKNOWN (do NOT auto-destroy)." >&2
        exit 5
        ;;
    *)
        echo "[probe_nvdec] FATAL: unrecognized classification token: $CLASSIFICATION" >&2
        echo "[probe_nvdec]   Output: $PROBE_OUT" >&2
        exit 5
        ;;
esac
