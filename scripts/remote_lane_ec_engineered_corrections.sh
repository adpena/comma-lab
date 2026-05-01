#!/bin/bash
# Lane EC: Engineered Corrections deployment.
#
# WHAT: per-pixel SegNet corrections precomputed at COMPRESS TIME by
# engineered_quant_noise.py (Eureka 6). For each frame, the script finds
# small (±max-delta) RGB perturbations that flip WRONG SegNet argmax
# predictions back to the GT class, packs them as sparse zlib-compressed
# int8 deltas in `gradient_corrections.bin`, and bundles that file into
# the submission archive.
#
# At inflate time, submissions/robust_current/inflate_renderer.py loads
# gradient_corrections.bin (already wired — see _unpack_sparse_corrections
# + _apply_gradient_corrections, dispatched at archive_dir/"gradient_corrections.bin")
# and additively applies the deltas BEFORE the upscale to camera resolution.
# NO scorer is loaded at inflate time — strict-scorer-rule preserved
# (CLAUDE.md non-negotiable: corrections.bin is data, not model).
#
# BASELINE: Lane A (`submissions/baseline_dilated_h64_0_90/`) at 1.15
# [contest-CUDA] (renderer.bin + masks.mkv + optimized_poses.pt). SegNet
# distortion at this baseline is ~0.0046; engineered corrections
# typically shave 30-70% of mis-classified pixels (council Yousfi: this
# is exactly the SegNet-as-steganalysis-detector inverse problem).
#
# PREDICTED BAND: [0.85, 1.15] [contest-CUDA].
#   Floor 0.85: best-case SegNet reduction 0.0046 → 0.001 (-0.34 score)
#               offset by +0.001 rate cost from a 50KB corrections.bin.
#   Ceiling 1.15: zero net change (corrections.bin too large to fit budget,
#                 or the renderer is already at the SegNet floor for the
#                 deltas the gradient-search can find at max-delta=2).
#   Council Quantizr: Lane EC is rate-cost-limited; max-artifact-bytes=51200
#   caps the rate hit at 0.0014 score, so the downside is bounded.
#
# CLAUDE.md compliance:
#   * set -euo pipefail (zip_dep_bootstrap_trap memory)
#   * Python `zipfile.ZipFile` (NOT shell `zip` — PyTorch container has none)
#   * --device cuda everywhere (no MPS/CPU fallback)
#   * Stage 0 NVDEC probe (check 33 + feedback_vastai_nvdec_host_variation)
#   * Verified CLI flags via grep add_argument:
#       experiments/engineered_quant_noise.py:
#         --checkpoint --device --n-frames --batch-size --max-delta
#         --output-dir --video --smoke --gt-poses-path --quantize-bits
#         --max-artifact-bytes
#       experiments/contest_auth_eval.py:
#         --archive --inflate-sh --upstream-dir --device --keep-work-dir
#         --work-dir
#   * predicted_band metadata + [contest-CUDA] tag in completion line
#   * provenance.json + heartbeat.log (canonical_remote_bootstraps memory)
#   * macOS AppleDouble cleanup via setup_full + env.sh source path
#   * Container Python /opt/conda/bin/python (NOT venv)

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_ec_engineered_corrections_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-ec] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps).
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_ec_engineered_corrections.sh',
    'lane_name': 'lane_ec_engineered_corrections',
    'predicted_band': [0.85, 1.15],
    'score_tag': '[contest-CUDA]',
    'baseline_score': 1.15,
    'baseline_artifact_dir': 'submissions/baseline_dilated_h64_0_90',
    'output_dir': '$LOG_DIR',
    'max_delta': 2,
    'max_artifact_bytes': 51200,
    'note': 'Engineered SegNet corrections: per-pixel deltas that flip '
            'wrong argmax predictions back to GT. Compress-time SegNet '
            'access only; inflate sees data not model (strict-scorer-rule).',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=EC gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend (check 33 + memory
# feedback_vastai_nvdec_host_variation: 7/12 4090 hosts had compute-CUDA but
# missing NVDEC, ~$0.20-3 wasted per bad host). Cost: 5s on healthy host.
log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed (exit $?). Refusing to spend GPU on a"
    log "       host that cannot run upstream/evaluate.py at the end."
    log "       Destroy this Vast.ai instance and pick a different host."
    exit 2
}

# Stage 1: stage Lane A baseline artifacts as the seed (verified 1.15
# [contest-CUDA] on 2026-04-27). All three files MUST exist; refusing to
# proceed without them rather than silent-skip per CLAUDE.md.
log "=== Stage 1: stage Lane A baseline artifacts (seed for engineered corrections) ==="
BASELINE_DIR="$WORKSPACE/submissions/baseline_dilated_h64_0_90"
RENDERER_BIN="$BASELINE_DIR/renderer.bin"
MASKS_MKV="$BASELINE_DIR/masks.mkv"
OPTIMIZED_POSES="$BASELINE_DIR/optimized_poses.pt"
GT_VIDEO="$WORKSPACE/upstream/videos/0.mkv"
SEGNET_WEIGHTS="$WORKSPACE/upstream/models/segnet.safetensors"
POSENET_WEIGHTS="$WORKSPACE/upstream/models/posenet.safetensors"

for f in "$RENDERER_BIN" "$MASKS_MKV" "$OPTIMIZED_POSES" \
         "$GT_VIDEO" "$SEGNET_WEIGHTS" "$POSENET_WEIGHTS"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

STAGED_DIR="$LOG_DIR/staged"
mkdir -p "$STAGED_DIR"
cp "$RENDERER_BIN" "$STAGED_DIR/renderer.bin"
cp "$MASKS_MKV" "$STAGED_DIR/masks.mkv"
cp "$OPTIMIZED_POSES" "$STAGED_DIR/optimized_poses.pt"
log "  staged renderer.bin ($(stat -c '%s' "$STAGED_DIR/renderer.bin" 2>/dev/null || stat -f '%z' "$STAGED_DIR/renderer.bin") bytes)"
log "  staged masks.mkv ($(stat -c '%s' "$STAGED_DIR/masks.mkv" 2>/dev/null || stat -f '%z' "$STAGED_DIR/masks.mkv") bytes)"
log "  staged optimized_poses.pt ($(stat -c '%s' "$STAGED_DIR/optimized_poses.pt" 2>/dev/null || stat -f '%z' "$STAGED_DIR/optimized_poses.pt") bytes)"

# Stage 2: engineered_quant_noise — find per-pixel SegNet-flipping deltas.
# All flags verified against experiments/engineered_quant_noise.py argparse
# (see header docstring; CLAUDE.md NEVER invent CLI flags).
log "=== Stage 2: engineered_quant_noise (compress-time SegNet correction search) ==="
log "  --max-delta=2 (perturbation magnitude per channel, int8 packed)"
log "  --max-artifact-bytes=51200 (50KB cap → bounded rate cost ~0.0014)"
log "  --gt-poses-path=baseline optimized_poses (so renderer FiLM matches eval)"
"$PYBIN" -u experiments/engineered_quant_noise.py \
    --checkpoint "$STAGED_DIR/renderer.bin" \
    --video "$GT_VIDEO" \
    --device cuda \
    --n-frames 1200 \
    --batch-size 32 \
    --max-delta 2 \
    --quantize-bits 8 \
    --gt-poses-path "$STAGED_DIR/optimized_poses.pt" \
    --max-artifact-bytes 51200 \
    --output-dir "$LOG_DIR/corrections" 2>&1 | tee "$LOG_DIR/engineered_quant_noise.log" | tail -30
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

CORRECTIONS_BIN="$LOG_DIR/corrections/gradient_corrections.bin"
if [ ! -f "$CORRECTIONS_BIN" ]; then
    log "FATAL: engineered_quant_noise did NOT produce gradient_corrections.bin."
    log "       Either subprocess failed (see engineered_quant_noise.log) or"
    log "       the rate-budget guardrail aborted (exit 2 on max-artifact-bytes)."
    log "       Without corrections.bin Lane EC has no signal to ship — abort."
    exit 2
fi
CORRECTIONS_BYTES=$(stat -c '%s' "$CORRECTIONS_BIN" 2>/dev/null || stat -f '%z' "$CORRECTIONS_BIN")
log "  gradient_corrections.bin = ${CORRECTIONS_BYTES} bytes"
if [ "$CORRECTIONS_BYTES" -le 0 ] || [ "$CORRECTIONS_BYTES" -gt 51200 ]; then
    log "FATAL: corrections.bin size ${CORRECTIONS_BYTES} outside [1, 51200]."
    log "       Guardrail must have been bypassed — refusing to bundle."
    exit 2
fi

# Stage 3: build new archive with Lane A artifacts + corrections.bin.
# Python zipfile.ZipFile (NOT shell zip — PyTorch container has none, per
# memory feedback_zip_dep_bootstrap_trap). Deterministic timestamps via
# ZipInfo.date_time = (1980, 1, 1, 0, 0, 0) so the archive bytes hash is
# stable across rebuilds (codex R5-r6 #5: deterministic-zip rule).
log "=== Stage 3: build Lane EC archive (renderer + masks + poses + corrections.bin) ==="
ARCHIVE="$LOG_DIR/archive_lane_ec.zip"
"$PYBIN" -c "
import os, zipfile
src = '$STAGED_DIR'
corr = '$CORRECTIONS_BIN'
dst = '$ARCHIVE'
files = [
    ('renderer.bin', os.path.join(src, 'renderer.bin')),
    ('masks.mkv', os.path.join(src, 'masks.mkv')),
    ('optimized_poses.pt', os.path.join(src, 'optimized_poses.pt')),
    ('gradient_corrections.bin', corr),
]
for name, path in files:
    assert os.path.isfile(path), f'missing {path}'
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name, path in files:
        # Deterministic ZipInfo (fixed timestamp) so archive bytes are
        # reproducible across reruns. Codex R5-r6 #5 archive-builder rule.
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(path, 'rb') as fh:
            z.writestr(info, fh.read(), compresslevel=9)
size = os.path.getsize(dst)
print(f'archive {dst}: {size} bytes')
"

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
if [ -z "${ARCHIVE_BYTES:-}" ] || [ "$ARCHIVE_BYTES" -le 0 ]; then
    log "FATAL: archive size empty or zero — refusing to call auth_eval"
    exit 2
fi
log "  archive_lane_ec.zip = ${ARCHIVE_BYTES} bytes (corrections.bin contributed ${CORRECTIONS_BYTES})"

# Validate archive structurally contains gradient_corrections.bin so a
# silent-skip in the inflate dispatcher can't be blamed for a flat score.
"$PYBIN" -c "
import zipfile, sys
with zipfile.ZipFile('$ARCHIVE') as z:
    names = set(z.namelist())
    required = {'renderer.bin', 'masks.mkv', 'optimized_poses.pt', 'gradient_corrections.bin'}
    missing = required - names
    if missing:
        print(f'FATAL: archive missing required entries: {missing}', file=sys.stderr)
        sys.exit(2)
    print(f'archive OK: {sorted(names)}')
"

# Strip macOS AppleDouble files from upstream/videos before auth eval
# (check 37 + feedback_canonical_remote_bootstraps). setup_full does this
# at bootstrap; this is a belt-and-suspenders re-strip in case the host
# acquired ._* files mid-session.
rm -f upstream/videos/._*.mkv

log "=== Stage 4: contest_auth_eval on Lane EC archive [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

grep -q '^RESULT_JSON' "$LOG_DIR/auth_eval.log" || {
    log "FATAL: auth_eval did not produce RESULT_JSON — invalid measurement"
    exit 2
}

log "=== LANE_EC_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "=== predicted_band=[0.85, 1.15], baseline=1.15, see $LOG_DIR/provenance.json ==="
