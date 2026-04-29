#!/bin/bash
# Lane SQ: per-class adaptive quantization research deploy.
#
# WHAT: exercises the orphan src/tac/semantic_quantization.py
# (semantic_adaptive_quantize: per-class bit allocation 8/6/6/4/4 for
# road/vehicles/pedestrians/sky/background) on Lane A's renderer
# state_dict. Distinct from Lane Ω (per-WEIGHT Hessian-aware quantization)
# — Lane SQ is per-CLASS, only meaningful for SPADE/CLADE renderers
# whose normalization layers carry per-class parameters.
#
# Lane SQ's claim: SegNet's 100x weight + scoring formula structure mean
# road pixels (high PoseNet sensitivity) deserve 8-bit, sky pixels
# tolerate 4-bit. Saves ~20% rate vs uniform 8-bit on the SPADE/CLADE
# normalization params.
#
# OUTSTANDING TODO: this is a RESEARCH lane. The current Lane A renderer
# is dilated-h64 (NOT SPADE/CLADE), so the per-class quantization will
# fall back to uniform 8-bit on backbone params. The encoder runs to
# prove the pipeline works; rate gain becomes meaningful once a
# CLADE/SPADE renderer profile lands.
#
# Cost cap: $0.30, ETA 30min on RTX 4090. Encoder-only.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
cd "$WORKSPACE"
if [ -f "$WORKSPACE/env.sh" ]; then
    [ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
fi

PYBIN="${PYBIN:-/opt/conda/bin/python}"
export PYTHONHASHSEED=1234
export PYTHONPATH=src:upstream:$PWD
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_sq_results}"
ITER_DIR="$LOG_DIR/iter_0"
TAG="${TAG:-lane_sq}"
ANCHOR_DIR="${ANCHOR_DIR:-submissions/baseline_dilated_h64_0_90}"

mkdir -p "$LOG_DIR" "$ITER_DIR"
log() { echo "[lane-sq] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

HB_PID=""
cleanup() {
    set +e
    [ -n "${HB_PID:-}" ] && kill "$HB_PID" 2>/dev/null || true
    rm -rf "$LOG_DIR/eval_work/tmp" 2>/dev/null || true
    if [ "${DESTROY_INSTANCE_ON_EXIT:-0}" = "1" ] && [ -n "${VASTAI_INSTANCE_ID:-}" ]; then
        if command -v vastai >/dev/null 2>&1; then
            vastai destroy instance "$VASTAI_INSTANCE_ID" >>"$LOG_DIR/cleanup.log" 2>&1 || true
        fi
    fi
}
trap cleanup EXIT

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
export PROVENANCE LOG_DIR GIT_HASH GPU_NAME DRIVER_VER TAG ANCHOR_DIR

"$PYBIN" -u - <<'PY'
import json, os, time
import torch
prov = {
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": os.environ["GIT_HASH"],
    "gpu_name": os.environ["GPU_NAME"],
    "driver_version": os.environ["DRIVER_VER"],
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "lane_script": "scripts/remote_lane_sq_semantic_quantization.sh",
    "lane_name": "lane_sq_semantic_quantization",
    "tag": os.environ["TAG"],
    "anchor_dir": os.environ["ANCHOR_DIR"],
    "predicted_band": [1.10, 1.20],
    "baseline_score": 1.15,
    "baseline_lane": "A",
    "hypothesis": "RESEARCH: per-class bit allocation 8/6/6/4/4 saves ~20pct rate on SPADE/CLADE renderer normalisation params.",
    "research_only": True,
    "strict_scorer_rule_compliant": True,
    "wires_orphan_module": "src/tac/semantic_quantization.py",
    "output_dir": os.environ["LOG_DIR"],
}
with open(os.environ["PROVENANCE"], "w") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=sq gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!

log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || { log "FATAL: NVDEC probe failed"; exit 2; }

log "=== Stage 1: anchor checks ==="
for f in "$ANCHOR_DIR/renderer.bin" \
         "$ANCHOR_DIR/optimized_poses.pt" \
         "$ANCHOR_DIR/masks.mkv" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 2: semantic-quantize Lane A renderer state_dict ==="
SQ_META="$LOG_DIR/sq_meta.json"
export SQ_META ANCHOR_DIR
"$PYBIN" -u - <<'PY' 2>&1 | tee "$LOG_DIR/sq.log"
import json, os, sys
import torch
sys.path.insert(0, "src"); sys.path.insert(0, "upstream")
from tac.semantic_quantization import semantic_adaptive_quantize, DEFAULT_CLASS_BITS
from tac.renderer_export import load_asymmetric_checkpoint_fp4

renderer_bin = os.path.join(os.environ["ANCHOR_DIR"], "renderer.bin")
print(f"[lane-sq] loading {renderer_bin}")
# Load the renderer state_dict via FP4 inflater
state, meta = load_asymmetric_checkpoint_fp4(renderer_bin)
print(f"[lane-sq] state_dict: {len(state)} tensors")

result = semantic_adaptive_quantize(state, class_bits=DEFAULT_CLASS_BITS)
print(f"[lane-sq] semantic-quantized {len(result['quantized_state_dict'])} tensors")
print(f"[lane-sq] estimated rate savings: {result['savings_estimate']:.4f} ({result['savings_estimate']*100:.1f}pct)")

class_specific_keys = [k for k, v in result["bits_used"].items() if isinstance(v, list)]
print(f"[lane-sq] class-specific keys (per-class bits): {len(class_specific_keys)}")
for k in class_specific_keys[:5]:
    print(f"  {k}: bits {result['bits_used'][k]}")

meta_obj = {
    "lane": "SQ",
    "n_tensors": len(state),
    "n_class_specific": len(class_specific_keys),
    "class_bits": DEFAULT_CLASS_BITS,
    "savings_estimate": result["savings_estimate"],
    "research_only": True,
    "applicable_arch": "SPADE/CLADE renderers (Lane A dilated-h64 has none, so only backbone uniform 8-bit applies)",
}
with open(os.environ["SQ_META"], "w") as f:
    json.dump(meta_obj, f, indent=2, default=str)
print(f"[lane-sq] meta -> {os.environ['SQ_META']}")
PY

log "=== Stage 3: build archive (Lane A artifacts — research lane) ==="
cp "$ANCHOR_DIR/renderer.bin" "$ITER_DIR/renderer.bin"
cp "$ANCHOR_DIR/masks.mkv" "$ITER_DIR/masks.mkv"
cp "$ANCHOR_DIR/optimized_poses.pt" "$ITER_DIR/optimized_poses.pt"

ARCHIVE="$LOG_DIR/archive_lane_sq.zip"
export ITER_DIR ARCHIVE
"$PYBIN" -u - <<'PY'
import os, zipfile
src = os.environ["ITER_DIR"]; dst = os.environ["ARCHIVE"]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for name in ("renderer.bin", "masks.mkv", "optimized_poses.pt"):
        path = os.path.join(src, name)
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(path, "rb") as f:
            zf.writestr(info, f.read(), compresslevel=9)
print(f"archive {dst}: {os.path.getsize(dst)} bytes")
PY

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ -n "${ARCHIVE_BYTES:-}" ] && [ "$ARCHIVE_BYTES" -gt 0 ] || { log "FATAL: empty archive"; exit 2; }
log "  archive_lane_sq.zip = ${ARCHIVE_BYTES} bytes"

rm -f upstream/videos/._*.mkv

log "=== Stage 4: contest_auth_eval [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" \
    2>&1 | tee "$LOG_DIR/auth_eval.log"
grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || { log "FATAL: no RESULT_JSON"; exit 5; }
grep '^RESULT_JSON' "$LOG_DIR/auth_eval.log" | tail -1 > "$LOG_DIR/RESULT_JSON"
log "  RESULT_JSON: $LOG_DIR/RESULT_JSON"

log "=== LANE_SQ_DONE [contest-CUDA] (research artifact) ==="
log "    archive: $ARCHIVE   sq_meta: $SQ_META"
log "    predicted_band: [1.10, 1.20]  (research-only; SPADE/CLADE renderer required for full gain)"
