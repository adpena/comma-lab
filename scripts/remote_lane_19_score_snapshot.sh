#!/bin/bash
# Lane 19 score-snapshot dispatch — score the partial-training snapshot from
# instance 35899850 (destroyed 2026-04-30 at epoch 1340/1980) AS-IS, without
# resuming training. Path B from
# `project_lane_19_dispatch_launch_ready_20260501.md`.
#
# This is a 50-LOC subset of `scripts/remote_lane_19_logit_margin.sh` that:
#   * SKIPS Stage 1 (training) entirely — the snapshot is the input
#   * KEEPS Stage 1b (FP4A export) — runs on the snapshot's _best_fp4.pt
#   * KEEPS Stage 2 (build half-frame archive for masks)
#   * KEEPS Stage 3 (pose TTO with the snapshot renderer)
#   * KEEPS Stage 4 (archive bundling)
#   * KEEPS Stage 5 (contest_auth_eval + JSON adjudication)
#
# Cost: ~$0.20, ~30min on Vast.ai 4090. Path A (resume training) is the more
# expensive path that closes the same L3 gate at higher score quality.
#
# Snapshot SHA-pinned in
# `experiments/results/live_snapshot_35899850_lane_19_logit_margin_20260430/SHA256SUMS`.
#
# Per CLAUDE.md NEVER-INVENT-CLI-FLAGS: every flag passed to
# build_baseline_archive / optimize_poses / contest_auth_eval was verified by
# argparse-grep against the original `remote_lane_19_logit_margin.sh`.
#
# Council 6/0 APPROVE Path B FIRST as $0.20 pre-screen — see
# `project_lane_19_dispatch_launch_ready_20260501.md`.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=89
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"

LOG_DIR="$WORKSPACE/lane_19_score_snapshot_results"
mkdir -p "$LOG_DIR" "$LOG_DIR/iter_0" "$LOG_DIR/train"
TAG="lane_19_logit_margin"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$WORKSPACE/experiments/results/live_snapshot_35899850_lane_19_logit_margin_20260430}"
SNAPSHOT_FP4="$SNAPSHOT_DIR/train/renderer_${TAG}_best_fp4.pt"
SNAPSHOT_ZOOM="$SNAPSHOT_DIR/train/zoom_scalars.pt"

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
log() { echo "[lane-19-snap] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Heartbeat (every 60s) — preflight Check 41 requires this for remote_lane scripts.
( while true; do echo "$(date -u +%FT%TZ) heartbeat pid=$$ stage=running" >> "$HEARTBEAT"; sleep 60; done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

cat > "$PROVENANCE" <<JSON
{
  "schema_version": 1,
  "started_at_utc": "$(date -u +%FT%TZ)",
  "git_hash": "$GIT_HASH",
  "gpu_name": "$GPU_NAME",
  "driver_version": "$DRIVER_VER",
  "lane_script": "scripts/remote_lane_19_score_snapshot.sh",
  "tag": "$TAG",
  "profile": "$TAG",
  "predicted_band": [0.97, 1.04],
  "anchor_score_baseline": 1.043987524793892,
  "anchor_lane": "lane_g_v3_pfp16_a_plus_plus_t4",
  "snapshot_dir": "$SNAPSHOT_DIR",
  "snapshot_fp4": "$SNAPSHOT_FP4",
  "snapshot_epoch_est": 1340,
  "snapshot_total_epochs": 1980,
  "dispatch_path": "B (score-as-is, $0.20 pre-screen for Path A resume)",
  "council_review_memo": "project_lane_19_dispatch_launch_ready_20260501.md",
  "controlled_baseline": "lane_g_v3_pfp16_a_plus_plus_t4 (same anchor masks/poses; only renderer differs — Lane 19 partial-training renderer vs PFP16 frontier renderer)",
  "baseline_lane": "lane_g_v3_pfp16_a_plus_plus_t4_20260430",
  "baseline_score_contest_cuda": 1.043987524793892,
  "no_training": true,
  "strict_scorer_rule_compliant": true,
  "design_memo": "project_lane_19_dispatch_launch_ready_20260501.md"
}
JSON

log "=== Stage 0: CUDA/NVDEC preflight ==="
"$PYBIN" - <<'PY'
import sys, torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; Lane 19 snapshot scoring requires CUDA per CLAUDE.md")
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

if [ "${SKIP_NVDEC_PROBE:-0}" != "1" ] && [ -f "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
        log "FATAL: NVDEC/DALI probe failed; pick a different host."
        exit 2
    }
fi

# Pre-flight: required artifacts present
for f in upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
         "$SNAPSHOT_FP4"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  snapshot fp4 checkpoint: $SNAPSHOT_FP4 ($(stat -c '%s' "$SNAPSHOT_FP4" 2>/dev/null || stat -f '%z' "$SNAPSHOT_FP4") bytes)"

log "=== Stage 1b: FP4A export from snapshot fp4 -> iter_0/renderer.bin ==="
"$PYBIN" -c "
import sys, torch
sys.path.insert(0, 'src')
from tac.renderer import build_renderer
from tac.renderer_export import export_asymmetric_checkpoint_fp4
ckpt_path = '$SNAPSHOT_FP4'
out_bin = '$LOG_DIR/iter_0/renderer.bin'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt.get('model_state_dict', ckpt)
meta = ckpt.get('__meta__', {}) or {}
fp4_codebook = meta.get('fp4_codebook', 'residual')
fp4_robust_scale = bool(meta.get('fp4_robust_scale', True))
def m(key, default):
    return meta.get(key, default)
model = build_renderer(
    embed_dim=m('embed_dim', 6),
    base_ch=m('base_ch', 36),
    mid_ch=m('mid_ch', 60),
    motion_hidden=m('motion_hidden', 32),
    depth=m('depth', 1),
    pose_dim=m('pose_dim', 6),
    use_dsconv=m('use_dsconv', False),
    padding_mode=m('padding_mode', 'zeros'),
    use_dilation=m('use_dilation', False),
    use_zoom_flow=m('use_zoom_flow', True),
)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise RuntimeError(
        f'shape mismatch refusing to ship: missing={list(missing)[:3]} '
        f'unexpected={list(unexpected)[:3]}'
    )
nbytes = export_asymmetric_checkpoint_fp4(
    model, out_bin, codebook_name=fp4_codebook, robust_scale=fp4_robust_scale,
)
print(f'WROTE {out_bin}: {nbytes} bytes (codebook={fp4_codebook}, robust={fp4_robust_scale})')
" 2>&1 | tee -a "$LOG_DIR/run.log"

[ -f "$LOG_DIR/iter_0/renderer.bin" ] || { echo "FATAL: FP4A export failed" >&2; exit 2; }

log "=== Stage 2: build half-frame archive (renderer + 600 odd masks + poses) ==="
"$PYBIN" -c "
from tac.profiles import PROFILES
prof_name = 'lane_19_logit_margin'
p = PROFILES.get(prof_name)
assert p is not None, f'PROFILES is missing {prof_name}'
assert p.get('mask_half_sim_prob', 0) > 0 or p.get('use_zoom_flow', False), \
    f'profile {prof_name} must have mask_half_sim_prob>0 OR use_zoom_flow=True'
print(f'halfframe-profile-assertion OK: {prof_name}')
"
# Profile: --profile lane_19_logit_margin (Check F: profile name within 30 lines of --half-frame).
set +e
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 --half-frame \
    --output "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_halfframe_seed.zip" 2>&1 | tail -5
cd "$WORKSPACE"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"

log "=== Stage 3: pose TTO with the snapshot logit-margin renderer ==="
set +e
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint "$LOG_DIR/iter_0/renderer.bin" \
    --masks "$LOG_DIR/iter_0/masks.mkv" \
    --gt-poses-path submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
    --device cuda \
    --steps 500 \
    --batch-pairs 8 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

[ -f "$LOG_DIR/optimized_poses.pt" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.pt"; exit 3; }
cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"

# zoom_scalars from the SNAPSHOT (training already computed them at epoch 1340).
if [ -f "$SNAPSHOT_ZOOM" ]; then
    cp "$SNAPSHOT_ZOOM" "$LOG_DIR/iter_0/zoom_scalars.pt"
    log "  bundling snapshot zoom_scalars.pt ($(stat -c '%s' "$SNAPSHOT_ZOOM" 2>/dev/null || stat -f '%z' "$SNAPSHOT_ZOOM") bytes)"
else
    log "  WARN: no zoom_scalars.pt in snapshot — inflate will use identity zoom (degraded)"
fi

ARCHIVE="$LOG_DIR/archive_lane_19_snapshot.zip"
"$PYBIN" -c "
import hashlib, json, os, zipfile
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
files = ['renderer.bin', 'masks.mkv', 'optimized_poses.pt']
if os.path.isfile(os.path.join(src, 'zoom_scalars.pt')):
    files.append('zoom_scalars.pt')
manifest = []
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in files:
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        data = open(p, 'rb').read()
        info = zipfile.ZipInfo(filename=n, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (0o644 & 0xFFFF) << 16
        z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        manifest.append({
            'name': n,
            'size_bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
        })
archive_sha256 = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
archive_bytes = os.path.getsize(dst)
print(f'archive {dst}: {archive_bytes} bytes ({len(files)} files) sha256={archive_sha256}')
with open('$LOG_DIR/archive_manifest.json', 'w') as f:
    json.dump({
        'schema_version': 1,
        'lane': 'lane_19_score_snapshot',
        'archive_bytes': archive_bytes,
        'archive_sha256': archive_sha256,
        'fixed_member_order': files,
        'members': manifest,
    }, f, indent=2, sort_keys=True)
"

log "=== Stage 4: contest_auth_eval on Lane 19 snapshot archive ==="
rm -rf "$LOG_DIR/eval_work"
set +e
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

# RESULT_JSON guard (LANE-B silent-crash prevention).
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash."
    exit 4
fi
[ -f "$LOG_DIR/eval_work/contest_auth_eval.json" ] || {
    log "FATAL: missing eval_work/contest_auth_eval.json — JSON adjudication cannot proceed."
    exit 4
}

log "=== Stage 5: JSON adjudication against current PFP16 frontier gates ==="
set +e
"$PYBIN" scripts/adjudicate_contest_auth_eval.py \
    --contest-json "$LOG_DIR/eval_work/contest_auth_eval.json" \
    --provenance "$PROVENANCE" \
    --archive "$ARCHIVE" \
    --result-copy "$LOG_DIR/contest_auth_eval.json" \
    --baseline-score 1.043987524793892 \
    --baseline-archive-bytes 686635 \
    --predicted-band 0.97 1.04 \
    --regression-threshold 1.043987524793892 \
    --baseline-posenet-dist 0.00346442 \
    --baseline-segnet-dist 0.00400656 \
    --max-posenet-relative 1.002 \
    --max-segnet-relative 1.002 \
    --component-reference-label lane_g_v3_pfp16_a_plus_plus_t4_20260430 \
    --required-device cuda \
    --required-samples 600 2>&1 | tee "$LOG_DIR/adjudication.log"
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: JSON adjudication failed rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi
if ! grep -q "SCORE_RECOMPUTED" "$LOG_DIR/adjudication.log"; then
    log "FATAL: adjudication.log missing SCORE_RECOMPUTED — no machine JSON result."
    exit 4
fi

log "=== LANE_19_SNAPSHOT_DONE [contest-CUDA] — see $LOG_DIR/contest_auth_eval.json and adjudication.log ==="
log "  predicted band (Path B snapshot): [0.97, 1.04]"
log "  current frontier gate: PFP16 A++ score=1.043987524793892 bytes=686635"
log "  L3-gate decision: if score < 1.044 promote Lane 19 to L3; else trigger Path A (resume training, ~$1.25)"
