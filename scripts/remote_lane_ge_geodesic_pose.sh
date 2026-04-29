#!/bin/bash
# Lane GE: Chebyshev geodesic pose compression (rank-1 measurement).
#
# WHAT: Lane M+N's empirical finding was that PoseNet's useful signal is
# effectively rank-1 (dim 0 dominates). Lane GE asks the related but
# distinct question: can we replace 1200 6-DoF pose vectors (~14 KB
# .pt) with a 13-coefficient Chebyshev series along the road manifold
# (~52 bytes) and still hit competitive PoseNet?
#
# Wires the orphan src/tac/geodesic_pose.py:
#   1. Fit GeodesicPoseModel on Lane A's optimized_poses.pt dim 0.
#   2. Materialise the model back to (N, 6) — dim 1..5 = 0 by design.
#   3. Save as optimized_poses.pt and run contest_auth_eval.
#
# This is NOT the rate-saving deploy (which would require an inflate-time
# Chebyshev decoder). This is the SCORE-COST measurement of the rank-1
# pose hypothesis on Lane A's verified 1.15 anchor. Once we know the cost,
# Lane GE-V2 can choose to ship the compact 52-byte artifact and rebuild
# inflate to decode it.
#
# Per Lane M-V2 audit (2026-04-28): rank-1 PoseNet OUTPUT sensitivity
# != rank-1 renderer INPUT subspace. Lane GE feeds (zoom, 0, 0, 0, 0, 0)
# poses to the renderer; the BUG-1 mismatch is the same risk class.
# Lane GE accepts that risk to MEASURE the cost; future Lane GE-V3-clean
# would route through _project_to_renderer_pose (analogous to Lane M-V3).
#
# Anchors on Lane A 1.15 [contest-CUDA].
#
# Predicted band: [1.20, 2.00] [contest-CUDA]. Wide because:
#   - if rank-1 is RIGHT: floor near 1.20 (PoseNet barely degraded)
#   - if rank-1 is WRONG: ceiling near 2.00 (Lane M-V2 territory)
# Either way the experiment is informative.
#
# Cost cap: $0.50, ETA 1h on RTX 4090.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
cd "$WORKSPACE"
if [ -f "$WORKSPACE/env.sh" ]; then
    source "$WORKSPACE/env.sh"
fi

PYBIN="${PYBIN:-/opt/conda/bin/python}"
export PYTHONHASHSEED=1234
export PYTHONPATH=src:upstream:$PWD
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_ge_results}"
ITER_DIR="$LOG_DIR/iter_0"
TAG="${TAG:-lane_ge}"
ANCHOR_DIR="${ANCHOR_DIR:-submissions/baseline_dilated_h64_0_90}"

mkdir -p "$LOG_DIR" "$ITER_DIR"

log() { echo "[lane-ge] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

HB_PID=""
cleanup() {
    set +e
    if [ -n "${HB_PID:-}" ]; then
        kill "$HB_PID" 2>/dev/null || true
    fi
    rm -rf "$LOG_DIR/eval_work/tmp" "$LOG_DIR/tmp" 2>/dev/null || true
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
    "lane_script": "scripts/remote_lane_ge_geodesic_pose.sh",
    "lane_name": "lane_ge_geodesic_pose",
    "tag": os.environ["TAG"],
    "anchor_dir": os.environ["ANCHOR_DIR"],
    "predicted_band": [1.20, 2.00],
    "baseline_score": 1.15,
    "baseline_lane": "A",
    "hypothesis": "Rank-1 Chebyshev geodesic pose compression: 13 floats vs 1200x6 floats.",
    "strict_scorer_rule_compliant": True,
    "wires_orphan_module": "src/tac/geodesic_pose.py",
    "output_dir": os.environ["LOG_DIR"],
}
with open(os.environ["PROVENANCE"], "w", encoding="utf-8") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=ge gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!

log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed before setup. Destroy this host and pick another."
    exit 2
}

log "=== Stage 1: anchor checks ==="
for f in "$ANCHOR_DIR/renderer.bin" \
         "$ANCHOR_DIR/optimized_poses.pt" \
         "$ANCHOR_DIR/masks.mkv" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

log "=== Stage 2: fit GeodesicPoseModel on Lane A poses dim 0 ==="
GEODESIC_BIN="$LOG_DIR/geodesic_pose.bin"
GEODESIC_META="$LOG_DIR/geodesic_meta.json"
RECONSTRUCTED_POSES="$ITER_DIR/optimized_poses.pt"
export GEODESIC_BIN GEODESIC_META RECONSTRUCTED_POSES ANCHOR_DIR
"$PYBIN" -u - <<'PY' 2>&1 | tee "$LOG_DIR/geodesic.log"
import json, os, sys
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "upstream")
from tac.geodesic_pose import (
    GEODESIC_POSE_DEGREE,
    GeodesicPoseModel,
    fit_geodesic_pose,
    save_geodesic_pose,
    load_geodesic_pose,
)
from tac.submission_archive import load_optimized_poses

anchor_poses_path = os.path.join(os.environ["ANCHOR_DIR"], "optimized_poses.pt")
poses = load_optimized_poses(anchor_poses_path, pose_dim=6)
print(f"[lane-ge] anchor poses shape={tuple(poses.shape)} dtype={poses.dtype}")
print(f"[lane-ge] dim 0 stats: mean={poses[:, 0].mean().item():.6f} "
      f"std={poses[:, 0].std().item():.6f} min={poses[:, 0].min().item():.6f} "
      f"max={poses[:, 0].max().item():.6f}")
print(f"[lane-ge] dim 1..5 stats: mean={poses[:, 1:].mean().item():.6f} "
      f"std={poses[:, 1:].std().item():.6f}")

model = fit_geodesic_pose(poses)
n = poses.shape[0]
reconstructed = model.forward(n).detach()
fit_err = (reconstructed[:, 0] - poses[:, 0]).abs().mean().item()
discarded_norm = poses[:, 1:].norm().item()
print(f"[lane-ge] fit residual on dim 0: {fit_err:.6f}")
print(f"[lane-ge] discarded dim 1..5 L2 norm: {discarded_norm:.6f}")

save_geodesic_pose(model, os.environ["GEODESIC_BIN"])
print(f"[lane-ge] saved geodesic_pose.bin: {os.path.getsize(os.environ['GEODESIC_BIN'])} bytes "
      f"(vs {os.path.getsize(anchor_poses_path)} bytes for original)")

# Round-trip verify the load.
loaded = load_geodesic_pose(os.environ["GEODESIC_BIN"])
roundtrip = loaded.forward(n).detach()
roundtrip_err = (roundtrip - reconstructed).abs().max().item()
assert roundtrip_err < 1e-6, f"roundtrip mismatch {roundtrip_err}"
print(f"[lane-ge] roundtrip verified (max err {roundtrip_err:.2e})")

# Save reconstructed (N, 6) tensor as optimized_poses.pt for the archive.
torch.save(reconstructed, os.environ["RECONSTRUCTED_POSES"])
print(f"[lane-ge] saved reconstructed optimized_poses.pt: "
      f"{os.path.getsize(os.environ['RECONSTRUCTED_POSES'])} bytes")

meta = {
    "lane": "GE",
    "geodesic_degree": GEODESIC_POSE_DEGREE,
    "geodesic_bytes": os.path.getsize(os.environ["GEODESIC_BIN"]),
    "original_bytes": os.path.getsize(anchor_poses_path),
    "reconstructed_bytes": os.path.getsize(os.environ["RECONSTRUCTED_POSES"]),
    "fit_residual_dim0": fit_err,
    "discarded_norm_dim1_5": discarded_norm,
    "n_frames": int(n),
    "rate_savings_potential": 1.0 - (
        os.path.getsize(os.environ["GEODESIC_BIN"]) /
        max(os.path.getsize(anchor_poses_path), 1)
    ),
}
with open(os.environ["GEODESIC_META"], "w") as f:
    json.dump(meta, f, indent=2)
print(f"[lane-ge] meta -> {os.environ['GEODESIC_META']}")
print(f"[lane-ge] rate_savings_potential: {meta['rate_savings_potential']:.4f} (~{meta['rate_savings_potential']*100:.1f}%)")
PY

[ -f "$RECONSTRUCTED_POSES" ] || { log "FATAL: reconstructed poses not produced"; exit 2; }

log "=== Stage 3: build archive (Lane A renderer + Lane A masks + GE poses) ==="
cp "$ANCHOR_DIR/renderer.bin" "$ITER_DIR/renderer.bin"
cp "$ANCHOR_DIR/masks.mkv" "$ITER_DIR/masks.mkv"

ARCHIVE="$LOG_DIR/archive_lane_ge.zip"
export ITER_DIR ARCHIVE
"$PYBIN" -u - <<'PY'
import os, zipfile

src = os.environ["ITER_DIR"]
dst = os.environ["ARCHIVE"]
members = ("renderer.bin", "masks.mkv", "optimized_poses.pt")
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for name in members:
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(path, "rb") as f:
            zf.writestr(info, f.read(), compresslevel=9)
print(f"archive {dst}: {os.path.getsize(dst)} bytes")
PY

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
if [ -z "${ARCHIVE_BYTES:-}" ] || [ "$ARCHIVE_BYTES" -le 0 ]; then
    log "FATAL: archive size empty or zero"
    exit 2
fi
log "  archive_lane_ge.zip = ${ARCHIVE_BYTES} bytes"

rm -f upstream/videos/._*.mkv

log "=== Stage 4: contest_auth_eval [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" \
    2>&1 | tee "$LOG_DIR/auth_eval.log"
grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || {
    log "FATAL: auth eval completed without RESULT_JSON"
    exit 5
}
grep '^RESULT_JSON' "$LOG_DIR/auth_eval.log" | tail -1 > "$LOG_DIR/RESULT_JSON"
log "  RESULT_JSON: $LOG_DIR/RESULT_JSON"

log "=== Stage 5: cleanup ==="
rm -rf "$LOG_DIR/tmp" "$LOG_DIR/eval_work/tmp"
if [ "${DESTROY_INSTANCE_ON_EXIT:-0}" = "1" ] && [ -n "${VASTAI_INSTANCE_ID:-}" ]; then
    if command -v vastai >/dev/null 2>&1; then
        vastai destroy instance "$VASTAI_INSTANCE_ID" >>"$LOG_DIR/cleanup.log" 2>&1 || true
    fi
fi

log "=== LANE_GE_DONE [contest-CUDA] ==="
log "    archive: $ARCHIVE"
log "    auth_eval: $LOG_DIR/auth_eval.log"
log "    predicted_band: [1.20, 2.00]"
log "    NOTE: rate gain materializes only after inflate adds Chebyshev decoder."
