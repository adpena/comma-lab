#!/usr/bin/env bash
# Lane G v3 + OWV3 Fisher sensitivity stack.
#
# Runs the full promotion path on a CUDA/NVDEC host:
#   1. profile per-weight Fisher importance for Lane G v3 on CUDA
#   2. convert Fisher -> OWV3 per-output-channel sensitivity map
#   3. build a Lane G v3 archive with only renderer.bin swapped to OWV3
#   4. run contest_auth_eval.py on the exact archive bytes with --device cuda
#
# Modal note: Modal is acceptable for RUN_CONTEST_EVAL=0 Fisher/build-only
# dispatch. Do not treat Modal CPU eval as promotion-grade evidence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="${WORKSPACE:-$DEFAULT_WORKSPACE}"
cd "$WORKSPACE"

if [ -f "$WORKSPACE/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$WORKSPACE/env.sh"
    cd "$WORKSPACE"
fi

if [ -z "${PYBIN:-}" ]; then
    if [ -x "$WORKSPACE/.venv/bin/python" ]; then
        PYBIN="$WORKSPACE/.venv/bin/python"
    else
        PYBIN="$(command -v python3 || command -v python)"
    fi
fi

export PYTHONHASHSEED="${PYTHONHASHSEED:-1234}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LANE_ID="${LANE_ID:-lane_g_v3_owv3_fisher_stack}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/${LANE_ID}_results}"
mkdir -p "$LOG_DIR"

log() { echo "[lane-owv3-fisher] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }
fail() { echo "FATAL: $*" >&2; exit 1; }

ANCHOR_DIR="${ANCHOR_DIR:-experiments/results/lane_g_v3_landed}"
ANCHOR_RENDERER="${ANCHOR_RENDERER:-$ANCHOR_DIR/iter_0/renderer.bin}"
ANCHOR_MASKS="${ANCHOR_MASKS:-$ANCHOR_DIR/iter_0/masks.mkv}"
ANCHOR_POSES="${ANCHOR_POSES:-$ANCHOR_DIR/iter_0/optimized_poses.pt}"
ANCHOR_ARCHIVE="${ANCHOR_ARCHIVE:-$ANCHOR_DIR/archive_lane_g_v3.zip}"
export ANCHOR_RENDERER ANCHOR_MASKS ANCHOR_POSES ANCHOR_ARCHIVE

FISHER_TOP_K="${FISHER_TOP_K:-30}"
PAIR_BATCH="${PAIR_BATCH:-4}"
BIT_BUDGET_RATIO="${BIT_BUDGET_RATIO:-0.7}"
PROTECT_THRESHOLD="${PROTECT_THRESHOLD:-1e-3}"
AGGRESSIVE_THRESHOLD="${AGGRESSIVE_THRESHOLD:-1e-5}"
MISSING_VALUE="${MISSING_VALUE:-1e-2}"
RUN_CONTEST_EVAL="${RUN_CONTEST_EVAL:-1}"
AUTH_DEVICE="${AUTH_EVAL_DEVICE:-cuda}"

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"

log "=== Stage 0: CUDA/NVDEC/anchor preflight ==="
"$PYBIN" - <<'PY'
import sys
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; Fisher artifacts from CPU/MPS are smoke-only and are rejected for OWV3 promotion")
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

# Stage 0: NVDEC probe (BEFORE any nvidia-smi info reads or GPU work).
# Per preflight Check 11 (feedback_vastai_nvdec_host_variation, eef64293):
# probe MUST come before any GPU-work marker including bare `nvidia-smi`.
if [ "${SKIP_NVDEC_PROBE:-0}" != "1" ] && [ -f "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
        log "FATAL: NVDEC/DALI probe failed; exact CUDA eval is not trustworthy on this host."
        exit 2
    }
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"
    DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)"
else
    GPU_NAME="unknown-no-nvidia-smi"
    DRIVER_VER="unknown-no-nvidia-smi"
fi

for f in \
    "$ANCHOR_RENDERER" \
    "$ANCHOR_MASKS" \
    "$ANCHOR_POSES" \
    "$ANCHOR_ARCHIVE" \
    upstream/videos/0.mkv \
    upstream/models/segnet.safetensors \
    upstream/models/posenet.safetensors \
    submissions/robust_current/inflate.sh \
    submissions/robust_current/inflate_renderer.py \
    experiments/profile_hessian_per_weight.py \
    experiments/convert_fisher_to_owv3_sensitivity_map.py \
    experiments/build_lane_g_v3_owv3_stack.py \
    experiments/contest_auth_eval.py; do
    [ -f "$f" ] || fail "missing required file: $f"
done

if ! grep -q 'magic == b"OWV3"' submissions/robust_current/inflate_renderer.py; then
    fail "inflate_renderer.py lacks OWV3 magic dispatch; rebuild/deploy current tree"
fi

ANCHOR_MAGIC="$("$PYBIN" -c "print(open('$ANCHOR_RENDERER','rb').read(4).decode('utf-8','replace'))")"
[ "$ANCHOR_MAGIC" = "ASYM" ] || fail "$ANCHOR_RENDERER magic is $ANCHOR_MAGIC; expected ASYM"

GIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo no-git)"
"$PYBIN" - <<PY
import json, time, torch
prov = {
    "lane_id": "$LANE_ID",
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": "$GIT_HASH",
    "gpu_name": "$GPU_NAME",
    "driver_version": "$DRIVER_VER",
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "lane_script": "scripts/remote_lane_g_v3_owv3_fisher_stack.sh",
    "anchor_renderer": "$ANCHOR_RENDERER",
    "anchor_masks": "$ANCHOR_MASKS",
    "anchor_poses": "$ANCHOR_POSES",
    "anchor_archive": "$ANCHOR_ARCHIVE",
    "baseline_lane": "lane_g_v3_landed",
    "baseline_score_contest_cuda": 1.05,
    "fisher_top_k": int("$FISHER_TOP_K"),
    "pair_batch": int("$PAIR_BATCH"),
    "bit_budget_ratio": float("$BIT_BUDGET_RATIO"),
    "protect_threshold": float("$PROTECT_THRESHOLD"),
    "aggressive_threshold": float("$AGGRESSIVE_THRESHOLD"),
    "run_contest_eval": "$RUN_CONTEST_EVAL" == "1",
    "auth_eval_device": "$AUTH_DEVICE",
    "output_dir": "$LOG_DIR",
    "predicted_band": [0.95, 1.05],
}
with open("$PROVENANCE", "w") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')"
    else
        GPU="nvidia-smi-unavailable"
    fi
    echo "[$(date -u +%FT%TZ)] lane=OWV3-FISHER gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT

log "=== Stage 1: Lane G v3 renderer/mask/pose smoke ==="
"$PYBIN" - <<'PY'
from pathlib import Path
import os
import av
import torch
from experiments.profile_hessian_per_weight import _gray_mask_to_class_ids
from tac.renderer_export import load_any_renderer_checkpoint
from tac.submission_archive import load_optimized_poses

ckpt = Path(os.environ["ANCHOR_RENDERER"])
mask_path = Path(os.environ["ANCHOR_MASKS"])
pose_path = Path(os.environ["ANCHOR_POSES"])
model = load_any_renderer_checkpoint(str(ckpt), device="cuda").eval()
container = av.open(str(mask_path))
stream = container.streams.video[0]
frames = []
for i, frame in enumerate(container.decode(stream)):
    frames.append(_gray_mask_to_class_ids(torch.from_numpy(frame.to_ndarray(format="gray"))))
    if i >= 1:
        break
container.close()
if len(frames) != 2:
    raise SystemExit("could not decode two mask frames")
m0 = frames[0].unsqueeze(0).to("cuda", dtype=torch.long)
m1 = frames[1].unsqueeze(0).to("cuda", dtype=torch.long)
if int(m0.min()) < 0 or int(m0.max()) > 4 or int(m1.min()) < 0 or int(m1.max()) > 4:
    raise SystemExit("mask class range outside [0, 4]")
kwargs = {}
pose_dim = int(getattr(model, "pose_dim", 0) or 0)
if pose_dim:
    poses = load_optimized_poses(str(pose_path), pose_dim=pose_dim, expected_n_pairs=600)
    kwargs["pose"] = poses[0:1].to("cuda")
with torch.no_grad():
    out = model(m0, m1, **kwargs)
print(f"smoke ok: out_shape={tuple(out.shape)} range=[{float(out.min()):.4f},{float(out.max()):.4f}]")
PY

PAIR_WEIGHT_ARGS=(--all-pairs)
PAIR_WEIGHT_LABEL="uniform_all_pairs"
if [ -n "${PAIR_WEIGHTS:-}" ]; then
    [ -f "$PAIR_WEIGHTS" ] || fail "PAIR_WEIGHTS set but file missing: $PAIR_WEIGHTS"
    PAIR_WEIGHT_ARGS=(--pair-weights "$PAIR_WEIGHTS")
    PAIR_WEIGHT_LABEL="$PAIR_WEIGHTS"
else
    for cand in \
        experiments/results/lane_lane_w_v2_modal/harvested_artifacts/lane_w_results/pair_weights.pt \
        experiments/results/lane_lane_w_modal/harvested_artifacts/lane_w_results/pair_weights.pt; do
        if [ -f "$cand" ]; then
            PAIR_WEIGHT_ARGS=(--pair-weights "$cand")
            PAIR_WEIGHT_LABEL="$cand"
            break
        fi
    done
fi
log "pair weights: $PAIR_WEIGHT_LABEL"

log "=== Stage 2: CUDA Fisher profile ==="
FISHER_PT="$LOG_DIR/hessian_per_weight.pt"
set +e
"$PYBIN" -u experiments/profile_hessian_per_weight.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --video upstream/videos/0.mkv \
    --masks-mkv "$ANCHOR_MASKS" \
    --poses "$ANCHOR_POSES" \
    --upstream upstream \
    --output "$FISHER_PT" \
    --top-k "$FISHER_TOP_K" \
    "${PAIR_WEIGHT_ARGS[@]}" \
    --device cuda \
    --pair-batch "$PAIR_BATCH" \
    --include-protected-conv2d 2>&1 | tee "$LOG_DIR/fisher_profile.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || exit "$RC"
[ -f "$FISHER_PT" ] || fail "Fisher profiler did not produce $FISHER_PT"
log "Fisher artifact: $FISHER_PT ($(stat -c '%s' "$FISHER_PT" 2>/dev/null || stat -f '%z' "$FISHER_PT") bytes)"

log "=== Stage 3: Fisher -> OWV3 sensitivity map ==="
SENSITIVITY_PT="$LOG_DIR/owv3_sensitivity_map.pt"
	"$PYBIN" -u experiments/convert_fisher_to_owv3_sensitivity_map.py \
	    --checkpoint "$ANCHOR_RENDERER" \
	    --fisher "$FISHER_PT" \
	    --output "$SENSITIVITY_PT" \
	    --aggregate sum \
	    --missing-policy error \
	    --missing-value "$MISSING_VALUE" \
	    --metadata-json "$LOG_DIR/owv3_sensitivity_map.metadata.json" 2>&1 | tee "$LOG_DIR/convert_sensitivity.log"
[ -f "$SENSITIVITY_PT" ] || fail "sensitivity conversion did not produce $SENSITIVITY_PT"

log "=== Stage 4: build Lane G v3 + OWV3 archive ==="
STACKED_ARCHIVE="$LOG_DIR/archive_lane_g_v3_owv3.zip"
BUILD_PROVENANCE="$LOG_DIR/build_provenance.json"
"$PYBIN" -u experiments/build_lane_g_v3_owv3_stack.py \
    --sensitivity-map "$SENSITIVITY_PT" \
    --output "$STACKED_ARCHIVE" \
    --provenance-json "$BUILD_PROVENANCE" \
    --bit-budget-ratio "$BIT_BUDGET_RATIO" \
    --protect-threshold "$PROTECT_THRESHOLD" \
    --aggressive-threshold "$AGGRESSIVE_THRESHOLD" 2>&1 | tee "$LOG_DIR/build_owv3.log"
[ -f "$STACKED_ARCHIVE" ] || fail "OWV3 archive not written"
ARCHIVE_BYTES="$(stat -c '%s' "$STACKED_ARCHIVE" 2>/dev/null || stat -f '%z' "$STACKED_ARCHIVE")"
ARCHIVE_SHA="$("$PYBIN" -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())" "$STACKED_ARCHIVE")"
log "OWV3 archive: $STACKED_ARCHIVE = $ARCHIVE_BYTES bytes sha256=$ARCHIVE_SHA"

if [ "$RUN_CONTEST_EVAL" != "1" ]; then
    log "RUN_CONTEST_EVAL=$RUN_CONTEST_EVAL; skipping exact eval. Archive requires CUDA eval before promotion."
    exit 0
fi

if [ "$AUTH_DEVICE" != "cuda" ] && [ "${ALLOW_NON_CUDA_EVAL:-0}" != "1" ]; then
    fail "AUTH_EVAL_DEVICE=$AUTH_DEVICE would be advisory-only. Set AUTH_EVAL_DEVICE=cuda for promotion or ALLOW_NON_CUDA_EVAL=1 for explicit smoke."
fi

log "=== Stage 5: contest_auth_eval on exact OWV3 archive bytes ==="
EVAL_WORK_DIR="$LOG_DIR/eval_work"
rm -rf "$EVAL_WORK_DIR"
set +e
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$STACKED_ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "$AUTH_DEVICE" \
    --keep-work-dir \
    --work-dir "$EVAL_WORK_DIR" 2>&1 | tee "$LOG_DIR/auth_eval.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || exit "$RC"

RESULT_JSON="$EVAL_WORK_DIR/contest_auth_eval.json"
[ -f "$RESULT_JSON" ] || fail "contest_auth_eval did not write $RESULT_JSON"
cp "$RESULT_JSON" "$LOG_DIR/contest_auth_eval.json"

"$PYBIN" - <<PY
import json, time
from pathlib import Path
result = json.loads(Path("$RESULT_JSON").read_text())
prov_path = Path("$PROVENANCE")
prov = json.loads(prov_path.read_text())
prov.update({
    "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "archive_path": "$STACKED_ARCHIVE",
    "archive_sha256": "$ARCHIVE_SHA",
    "archive_size_bytes": int("$ARCHIVE_BYTES"),
    "contest_auth_eval_json": "$LOG_DIR/contest_auth_eval.json",
    "final_score": result.get("final_score"),
    "score_recomputed_from_components": result.get("score_recomputed_from_components"),
    "avg_posenet_dist": result.get("avg_posenet_dist"),
    "avg_segnet_dist": result.get("avg_segnet_dist"),
    "rate_unscaled": result.get("rate_unscaled"),
    "lane_status": "COMPLETE_CONTEST_CUDA" if result.get("provenance", {}).get("device") == "cuda" else "COMPLETE_NON_CUDA_ADVISORY",
})
prov_path.write_text(json.dumps(prov, indent=2))
print("RESULT_JSON", json.dumps({
    "score_recomputed_from_components": result.get("score_recomputed_from_components"),
    "final_score": result.get("final_score"),
    "avg_posenet_dist": result.get("avg_posenet_dist"),
    "avg_segnet_dist": result.get("avg_segnet_dist"),
    "rate_unscaled": result.get("rate_unscaled"),
    "archive_size_bytes": result.get("archive_size_bytes"),
    "device": result.get("provenance", {}).get("device"),
}))
PY

log "=== LANE_G_V3_OWV3_FISHER_DONE [contest-CUDA] — see $LOG_DIR/contest_auth_eval.json ==="
