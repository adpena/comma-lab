#!/usr/bin/env bash
# SJ-KL C067 bounded remote driver.
#
# This script is intended to run on an already-claimed remote CUDA host. It
# performs tensor prep, CUDA SJ-KL residual build, deterministic archive packing,
# and canonical CUDA auth eval. It does not create a dispatch claim; operators
# must claim `lane_sjkl_c067` before launching the remote process.
#
# NO_NVDEC_NEEDED — operates on already-decoded tensor inputs and the canonical
# CUDA auth eval (which delegates video decode to its own probe-gated flow).
# This driver does no DALI/NVDEC video work itself.

set -euo pipefail

if [ -z "${WORKSPACE:-}" ]; then
    if [ -d /teamspace/studios/this_studio/pact ]; then
        WORKSPACE="/teamspace/studios/this_studio/pact"
    else
        WORKSPACE="/workspace/pact"
    fi
fi
cd "$WORKSPACE"

if [ -f "$WORKSPACE/env.sh" ]; then
    # shellcheck source=/dev/null
    source "$WORKSPACE/env.sh"
fi

PYBIN="${PYBIN:-.venv/bin/python}"
if [ ! -x "$PYBIN" ]; then
    PYBIN="$(command -v python3 || true)"
fi
if [ -z "$PYBIN" ]; then
    echo "FATAL: no Python runtime found; set PYBIN" >&2
    exit 2
fi

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

RUN_ID="${SJKL_RUN_ID:-sjkl_c067_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_REL="${SJKL_OUTPUT_DIR:-experiments/results/${RUN_ID}}"
OUT_DIR="$WORKSPACE/$OUT_REL"
mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/remote_lane_sjkl_c067.log"
HEARTBEAT="$OUT_DIR/heartbeat.log"
log() { echo "[sjkl-c067] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

SOURCE_ARCHIVE="${SJKL_SOURCE_ARCHIVE:-experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/line_search_source_c067_fixedslice/archive.zip}"
RENDERER_FRAMES="${SJKL_RENDERER_FRAMES:-experiments/results/renderer_output_for_postfilter.pt}"
PREPARED_TENSOR_DIR="${SJKL_PREPARED_TENSOR_DIR:-}"
GT_VIDEO="${SJKL_GT_VIDEO:-upstream/videos/0.mkv}"
INFLATE_SH="${INFLATE_SH:-submissions/robust_current/inflate.sh}"
EXPECTED_SOURCE_ARCHIVE_SHA256="${SJKL_EXPECTED_SOURCE_ARCHIVE_SHA256:-226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a}"
EXPECTED_SOURCE_ARCHIVE_BYTES="${SJKL_EXPECTED_SOURCE_ARCHIVE_BYTES:-276214}"
FAST_CHIP_REGEX="${SJKL_FAST_CHIP_REGEX:-H100|L40S|A100|RTX 4090|RTX 6000|RTX PRO|A10G}"
MAX_SJKL_BYTES="${SJKL_MAX_BYTES:-32768}"

TENSOR_DIR="$OUT_DIR/tensors"
BUILD_DIR="$OUT_DIR/build"
PACK_DIR="$OUT_DIR/pack"
mkdir -p "$TENSOR_DIR" "$BUILD_DIR" "$PACK_DIR"

require_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        log "FATAL: missing required file: $path"
        exit 3
    fi
}

sha256_file() {
    sha256sum "$1" 2>/dev/null | cut -d ' ' -f 1 || shasum -a 256 "$1" | cut -d ' ' -f 1
}

size_file() {
    stat -c '%s' "$1" 2>/dev/null || stat -f '%z' "$1"
}

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$WORKSPACE/$1" ;;
    esac
}

SOURCE_ARCHIVE_ABS="$(abs_path "$SOURCE_ARCHIVE")"
RENDERER_FRAMES_ABS="$(abs_path "$RENDERER_FRAMES")"
PREPARED_TENSOR_DIR_ABS=""
if [ -n "$PREPARED_TENSOR_DIR" ]; then
    PREPARED_TENSOR_DIR_ABS="$(abs_path "$PREPARED_TENSOR_DIR")"
fi
GT_VIDEO_ABS="$(abs_path "$GT_VIDEO")"
INFLATE_SH_ABS="$(abs_path "$INFLATE_SH")"

require_file "$SOURCE_ARCHIVE_ABS"
if [ -z "$PREPARED_TENSOR_DIR_ABS" ]; then
    require_file "$RENDERER_FRAMES_ABS"
else
    require_file "$PREPARED_TENSOR_DIR_ABS/renderer_target_slot_chw.pt"
    require_file "$PREPARED_TENSOR_DIR_ABS/gt_pairs_btchw.pt"
fi
require_file "$GT_VIDEO_ABS"
require_file "$INFLATE_SH_ABS"
require_file "$WORKSPACE/experiments/prepare_sjkl_pair_tensors.py"
require_file "$WORKSPACE/experiments/build_sjkl_residual.py"
require_file "$WORKSPACE/experiments/build_sjkl_c067_archive.py"
require_file "$WORKSPACE/experiments/contest_auth_eval.py"
require_file "$WORKSPACE/scripts/remote_archive_only_eval.sh"

if [ -n "$EXPECTED_SOURCE_ARCHIVE_SHA256" ]; then
    ACTUAL_SOURCE_SHA="$(sha256_file "$SOURCE_ARCHIVE_ABS")"
    if [ "$ACTUAL_SOURCE_SHA" != "$EXPECTED_SOURCE_ARCHIVE_SHA256" ]; then
        log "FATAL: C067 source archive SHA mismatch expected=$EXPECTED_SOURCE_ARCHIVE_SHA256 actual=$ACTUAL_SOURCE_SHA"
        exit 4
    fi
fi
if [ -n "$EXPECTED_SOURCE_ARCHIVE_BYTES" ]; then
    ACTUAL_SOURCE_BYTES="$(size_file "$SOURCE_ARCHIVE_ABS")"
    if [ "$ACTUAL_SOURCE_BYTES" != "$EXPECTED_SOURCE_ARCHIVE_BYTES" ]; then
        log "FATAL: C067 source archive size mismatch expected=$EXPECTED_SOURCE_ARCHIVE_BYTES actual=$ACTUAL_SOURCE_BYTES"
        exit 4
    fi
fi
case "$MAX_SJKL_BYTES" in
    ''|*[!0-9]*)
        log "FATAL: SJKL_MAX_BYTES must be a positive integer; got '$MAX_SJKL_BYTES'"
        exit 4
        ;;
esac
if [ "$MAX_SJKL_BYTES" -le 0 ]; then
    log "FATAL: SJKL_MAX_BYTES must be positive; got '$MAX_SJKL_BYTES'"
    exit 4
fi

log "=== Stage 0: fail-closed CUDA and fast-chip preflight ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "FATAL: nvidia-smi missing; refusing CUDA/SJ-KL spend"
    exit 5
fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
if ! printf '%s\n' "$GPU_NAME" | grep -Eiq "$FAST_CHIP_REGEX"; then
    log "FATAL: GPU '$GPU_NAME' does not match SJKL_FAST_CHIP_REGEX='$FAST_CHIP_REGEX'"
    exit 5
fi

SJKL_FAST_CHIP_REGEX="$FAST_CHIP_REGEX" "$PYBIN" - <<'PY'
import json
import os
import re
import sys
import torch

payload = {
    "torch_version": torch.__version__,
    "torch_cuda": getattr(torch.version, "cuda", None),
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(payload, sort_keys=True))
if not payload["cuda_available"]:
    raise SystemExit("FATAL: torch.cuda.is_available() is false")
regex = os.environ["SJKL_FAST_CHIP_REGEX"]
if not re.search(regex, payload["device_name"] or "", re.I):
    raise SystemExit(f"FATAL: CUDA device does not match fast-chip regex: {payload['device_name']!r}")
PY

# ── provenance.json (per feedback_canonical_remote_bootstraps + preflight Check L) ──
GIT_HASH="$(git -C "$WORKSPACE" rev-parse HEAD 2>/dev/null || echo unknown)"
PROVENANCE="$OUT_DIR/provenance.json"
SJKL_OUT_PROV="$PROVENANCE" SJKL_RUN_ID="$RUN_ID" SJKL_GIT_HASH="$GIT_HASH" \
SJKL_GPU_NAME="$GPU_NAME" SJKL_DRIVER_VERSION="$DRIVER_VERSION" \
SJKL_SOURCE_ARCHIVE="$SOURCE_ARCHIVE_ABS" SJKL_PREPARED_TENSOR_DIR="${PREPARED_TENSOR_DIR_ABS:-}" \
SJKL_K="${SJKL_K:-8}" SJKL_ALPHA_BITS="${SJKL_ALPHA_BITS:-6}" \
SJKL_BASIS_QUANT_BITS="${SJKL_BASIS_QUANT_BITS:-6}" \
SJKL_N_ANCHOR_PAIRS="${SJKL_N_ANCHOR_PAIRS:-30}" \
SJKL_MAX_BYTES="$MAX_SJKL_BYTES" \
"$PYBIN" - <<'PY'
import json
import os
import time
import torch

prov = {
    "lane_id": "lane_sjkl_c067",
    "run_id": os.environ["SJKL_RUN_ID"],
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": os.environ["SJKL_GIT_HASH"],
    "gpu_name": os.environ["SJKL_GPU_NAME"],
    "driver_version": os.environ["SJKL_DRIVER_VERSION"],
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "source_archive": os.environ["SJKL_SOURCE_ARCHIVE"],
    "prepared_tensor_dir": os.environ["SJKL_PREPARED_TENSOR_DIR"],
    "config": {
        "k": int(os.environ["SJKL_K"]),
        "alpha_bits": int(os.environ["SJKL_ALPHA_BITS"]),
        "basis_quant_bits": int(os.environ["SJKL_BASIS_QUANT_BITS"]),
        "n_anchor_pairs": int(os.environ["SJKL_N_ANCHOR_PAIRS"]),
        "max_sjkl_bytes": int(os.environ["SJKL_MAX_BYTES"]),
    },
    # SJ-KL is a residual side-channel; predicted band is for the STACKED archive
    # (PR106 + sjkl.bin) NOT a standalone score. Tagged [predicted-band only].
    "predicted_band_stacked": [0.18, 0.22],
    "predicted_band_tag": "[predicted-band only — NEVER-VERIFIED at lane level]",
}
with open(os.environ["SJKL_OUT_PROV"], "w") as f:
    json.dump(prov, f, indent=2, sort_keys=True)
print(f"[stage-0c] provenance written -> {os.environ['SJKL_OUT_PROV']}")
PY

(
    while true; do
        echo "$(date -u +%FT%TZ) run_id=$RUN_ID gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')" >> "$HEARTBEAT"
        sleep 60
    done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

log "=== Stage 0b: source/artifact manifest ==="
SJKL_RUN_ID="$RUN_ID" \
SJKL_OUT_DIR="$OUT_DIR" \
SJKL_SOURCE_ARCHIVE="$SOURCE_ARCHIVE_ABS" \
SJKL_RENDERER_FRAMES="$RENDERER_FRAMES_ABS" \
SJKL_PREPARED_TENSOR_DIR="$PREPARED_TENSOR_DIR_ABS" \
SJKL_GT_VIDEO="$GT_VIDEO_ABS" \
SJKL_INFLATE_SH="$INFLATE_SH_ABS" \
SJKL_GPU_NAME="$GPU_NAME" \
SJKL_DRIVER_VERSION="$DRIVER_VERSION" \
"$PYBIN" - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

def meta(path: str) -> dict[str, object]:
    p = Path(path)
    return {
        "path": str(p),
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }

repo = Path.cwd()
tracked = [
    "experiments/prepare_sjkl_pair_tensors.py",
    "experiments/build_sjkl_residual.py",
    "experiments/build_sjkl_c067_archive.py",
    "experiments/contest_auth_eval.py",
    "scripts/remote_archive_only_eval.sh",
    "scripts/remote_lane_sjkl_c067.sh",
    "submissions/robust_current/inflate.sh",
    "submissions/robust_current/inflate_renderer.py",
    "submissions/robust_current/unpack_renderer_payload.py",
    "src/tac/sjkl_basis.py",
]
payload = {
    "schema_version": 1,
    "tool": "scripts/remote_lane_sjkl_c067.sh",
    "run_id": os.environ["SJKL_RUN_ID"],
    "score_claim": False,
    "promotion_eligible": False,
    "required_score_truth": "contest_auth_eval.json from archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip() or None,
    "gpu_name": os.environ["SJKL_GPU_NAME"],
    "driver_version": os.environ["SJKL_DRIVER_VERSION"],
    "inputs": {
        "source_archive": meta(os.environ["SJKL_SOURCE_ARCHIVE"]),
        "gt_video": meta(os.environ["SJKL_GT_VIDEO"]),
        "inflate_sh": meta(os.environ["SJKL_INFLATE_SH"]),
    },
    "source_files": {rel: meta(str(repo / rel)) for rel in tracked if (repo / rel).is_file()},
}
prepared_tensor_dir = os.environ.get("SJKL_PREPARED_TENSOR_DIR") or ""
if prepared_tensor_dir:
    prepared = Path(prepared_tensor_dir)
    payload["inputs"]["prepared_tensor_dir"] = str(prepared)
    payload["inputs"]["prepared_tensors"] = {
        "renderer_target_slot_chw": meta(str(prepared / "renderer_target_slot_chw.pt")),
        "gt_pairs_btchw": meta(str(prepared / "gt_pairs_btchw.pt")),
    }
    manifest = prepared / "sjkl_pair_tensor_prep_manifest.json"
    if manifest.is_file():
        payload["inputs"]["prepared_tensor_manifest"] = meta(str(manifest))
else:
    payload["inputs"]["renderer_frames"] = meta(os.environ["SJKL_RENDERER_FRAMES"])
out = Path(os.environ["SJKL_OUT_DIR"]) / "source_artifact_manifest.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"source_artifact_manifest": str(out), "score_claim": False}, sort_keys=True))
PY

if [ -z "$PREPARED_TENSOR_DIR_ABS" ]; then
    log "FATAL: SJ-KL C-067 lane requires SJKL_PREPARED_TENSOR_DIR (pre-extracted .pt tensors)."
    log "       The recovered prepare_sjkl_pair_tensors.py expects pre-extracted tensors,"
    log "       not raw renderer/GT videos. Use SJKL_PREPARED_TENSOR_DIR=<dir> with"
    log "       renderer_target_slot_chw.pt + gt_pairs_btchw.pt OR adapt your dispatch"
    log "       to extract tensors upstream (see docs/sjkl_pair_tensor_extraction.md)."
    exit 5
fi
log "=== Stage 1a: reuse prepared SJ-KL tensors (no score claim) ==="
RENDERER_TENSOR="$PREPARED_TENSOR_DIR_ABS/renderer_target_slot_chw.pt"
GT_PAIRS="$PREPARED_TENSOR_DIR_ABS/gt_pairs_btchw.pt"
require_file "$RENDERER_TENSOR"
require_file "$GT_PAIRS"
if [ -f "$PREPARED_TENSOR_DIR_ABS/sjkl_pair_tensor_prep_manifest.json" ]; then
    cp "$PREPARED_TENSOR_DIR_ABS/sjkl_pair_tensor_prep_manifest.json" "$OUT_DIR/sjkl_pair_tensor_prep_manifest.source.json"
fi
log "  renderer tensor: $RENDERER_TENSOR ($(size_file "$RENDERER_TENSOR") bytes)"
log "  gt pairs tensor: $GT_PAIRS ($(size_file "$GT_PAIRS") bytes)"

log "=== Stage 1b: build SJ-KL prep manifest from prepared tensors (no score claim) ==="
PREP_OUT_DIR="$TENSOR_DIR/manifest_workdir"
mkdir -p "$PREP_OUT_DIR"
PREP_NPAIRS_ARGS=""
if [ -n "${SJKL_N_ANCHOR_PAIRS:-}" ]; then
    PREP_NPAIRS_ARGS="--n-pairs $SJKL_N_ANCHOR_PAIRS"
fi
"$PYBIN" -u experiments/prepare_sjkl_pair_tensors.py \
    --renderer-output "$RENDERER_TENSOR" \
    --target-frames "$GT_PAIRS" \
    --output-dir "$PREP_OUT_DIR" \
    --anchor-pair-idx 0 \
    --seed "${SJKL_SEED:-0}" \
    $PREP_NPAIRS_ARGS 2>&1 | tee "$OUT_DIR/tensor_prep.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    log "FATAL: prepare_sjkl_pair_tensors.py failed rc=${PIPE_RC[0]}"
    exit "${PIPE_RC[0]}"
fi
SJKL_PREP_MANIFEST="$PREP_OUT_DIR/sjkl_pair_tensor_prep_manifest.json"
require_file "$SJKL_PREP_MANIFEST"

log "=== Stage 2: CUDA build_sjkl_residual.py ==="
SJKL_BUILD_OUT="$BUILD_DIR/sjkl_build_workdir"
mkdir -p "$SJKL_BUILD_OUT"
"$PYBIN" -u experiments/build_sjkl_residual.py \
    --pair-tensor-manifest "$SJKL_PREP_MANIFEST" \
    --output-dir "$SJKL_BUILD_OUT" \
    --rank "${SJKL_K:-8}" \
    --n-pairs "${SJKL_N_ANCHOR_PAIRS:-30}" \
    --alpha-bits "${SJKL_ALPHA_BITS:-6}" \
    --basis-quant-bits "${SJKL_BASIS_QUANT_BITS:-6}" \
    --max-bytes "$MAX_SJKL_BYTES" \
    --device cuda \
    --seed "${SJKL_SEED:-0}" 2>&1 | tee "$OUT_DIR/build_sjkl_residual.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    log "FATAL: build_sjkl_residual.py failed rc=${PIPE_RC[0]}"
    exit "${PIPE_RC[0]}"
fi
SJKL_BIN="$SJKL_BUILD_OUT/sjkl.bin"
SJKL_MANIFEST="$SJKL_BUILD_OUT/sjkl_manifest.json"
require_file "$SJKL_BIN"
require_file "$SJKL_MANIFEST"
SJKL_BIN_BYTES="$(size_file "$SJKL_BIN")"
if [ "$SJKL_BIN_BYTES" -gt "$MAX_SJKL_BYTES" ]; then
    log "FATAL: sjkl.bin exceeds SJKL_MAX_BYTES cap bytes=$SJKL_BIN_BYTES cap=$MAX_SJKL_BYTES"
    exit 6
fi
if grep -Eqi 'falling back to CPU|ADVISORY only|device=cpu|device=mps' "$OUT_DIR/build_sjkl_residual.log"; then
    log "FATAL: SJ-KL build log contains CPU/MPS/advisory fallback text"
    exit 6
fi
"$PYBIN" - "$SJKL_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("device") != "cuda":
    raise SystemExit(f"FATAL: sjkl manifest device is not cuda: {payload.get('device')!r}")
if payload.get("score_claim") is not False:
    raise SystemExit("FATAL: sjkl manifest must keep score_claim=false")
PY

log "=== Stage 3: deterministic archive packer (adds charged sjkl.bin) ==="
OUTPUT_ARCHIVE="$PACK_DIR/archive.zip"
PACK_MANIFEST="$PACK_DIR/sjkl_c067_archive_manifest.json"
"$PYBIN" -u experiments/build_sjkl_c067_archive.py \
    --source-archive "$SOURCE_ARCHIVE_ABS" \
    --sjkl-bin "$SJKL_BIN" \
    --output-dir "$PACK_DIR" \
    --sjkl-member-name p 2>&1 | tee "$OUT_DIR/build_sjkl_c067_archive.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    log "FATAL: build_sjkl_c067_archive.py failed rc=${PIPE_RC[0]}"
    exit "${PIPE_RC[0]}"
fi
require_file "$OUTPUT_ARCHIVE"
require_file "$PACK_MANIFEST"
"$PYBIN" - "$PACK_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("score_claim") is not False:
    raise SystemExit("FATAL: pack manifest must keep score_claim=false")
logical = payload.get("payload_member_names", {}).get("output_logical_runtime_members", [])
if "sjkl.bin" not in logical:
    raise SystemExit(f"FATAL: sjkl.bin missing from packed logical members: {logical}")
if payload.get("runtime_contract", {}).get("score_affecting_payload_charged_in_archive") is not True:
    raise SystemExit("FATAL: sjkl.bin must be charged inside the archive payload")
proof = payload.get("runtime_contract", {}).get("runtime_apply_proof") or {}
if proof.get("verified") is not True:
    raise SystemExit(f"FATAL: SJ-KL runtime-apply proof is absent: {proof}")
sjkl = payload.get("sjkl_payload", {})
if int(sjkl.get("bytes", -1)) > int(sjkl.get("max_bytes", -1)):
    raise SystemExit(f"FATAL: sjkl.bin exceeds manifest byte cap: {sjkl}")
PY

log "=== Stage 4: exact CUDA auth eval via remote_archive_only_eval.sh ==="
SOURCE_SHA_LINES="$(
    {
        printf 'scripts/remote_lane_sjkl_c067.sh=%s\n' "$(sha256_file "$WORKSPACE/scripts/remote_lane_sjkl_c067.sh")"
        printf 'experiments/build_sjkl_residual.py=%s\n' "$(sha256_file "$WORKSPACE/experiments/build_sjkl_residual.py")"
        printf 'experiments/build_sjkl_c067_archive.py=%s\n' "$(sha256_file "$WORKSPACE/experiments/build_sjkl_c067_archive.py")"
        printf 'experiments/prepare_sjkl_pair_tensors.py=%s\n' "$(sha256_file "$WORKSPACE/experiments/prepare_sjkl_pair_tensors.py")"
        printf 'submissions/robust_current/inflate_renderer.py=%s\n' "$(sha256_file "$WORKSPACE/submissions/robust_current/inflate_renderer.py")"
        printf 'submissions/robust_current/unpack_renderer_payload.py=%s\n' "$(sha256_file "$WORKSPACE/submissions/robust_current/unpack_renderer_payload.py")"
        printf 'src/tac/sjkl_basis.py=%s\n' "$(sha256_file "$WORKSPACE/src/tac/sjkl_basis.py")"
    }
)"
export ARCHIVE_PATH="$OUTPUT_ARCHIVE"
export ARCHIVE_LABEL="$RUN_ID"
export LOG_DIR="$OUT_DIR/exact_eval"
export INFLATE_SH="$INFLATE_SH_ABS"
export PREDICTED_LOW="${SJKL_PREDICTED_LOW:-0.20}"
export PREDICTED_HIGH="${SJKL_PREDICTED_HIGH:-0.60}"
export CONTROLLED_BASELINE="${SJKL_CONTROLLED_BASELINE:-C067 fixedslice source archive plus charged SJ-KL residual payload; no score claim until contest_auth_eval.json}"
export REQUIRED_SOURCE_SHA256S="$SOURCE_SHA_LINES"
export CONTEST_AUTH_EVAL_CUSTODY_FLAGS="--keep-work-dir --work-dir"
# Custody guard: remote_archive_only_eval.sh invokes experiments/contest_auth_eval.py
# with --keep-work-dir --work-dir "$LOG_DIR/eval_work" and copies
# eval_work/contest_auth_eval.json back to LOG_DIR. Keep the work dir by default
# for post-run adjudication and forensic replay.
export KEEP_EVAL_WORK="${KEEP_EVAL_WORK:-1}"
log "contest_auth_eval custody flags delegated via remote_archive_only_eval.sh: $CONTEST_AUTH_EVAL_CUSTODY_FLAGS"
export SJKL_REQUIRE_APPLIED="${SJKL_REQUIRE_APPLIED:-1}"
if [ "$SJKL_REQUIRE_APPLIED" != "1" ]; then
    log "FATAL: SJKL_REQUIRE_APPLIED must remain 1 for SJ-KL eval dispatch"
    exit 6
fi
bash scripts/remote_archive_only_eval.sh

CONTEST_JSON="$LOG_DIR/contest_auth_eval.json"
require_file "$CONTEST_JSON"
cp "$CONTEST_JSON" "$OUT_DIR/contest_auth_eval.json"
log "=== SJKL_C067_DONE [contest-CUDA] contest_auth_eval_json=$CONTEST_JSON archive=$OUTPUT_ARCHIVE ==="
