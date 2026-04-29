#!/bin/bash
# Lane EBR: Ballé-2018 Entropy Bottleneck deployment.
#
# WHAT: replaces the renderer's hand-tuned latent compression with a learned
# factorized entropy bottleneck inspired by Ballé et al. 2018
# (https://arxiv.org/abs/1802.01436). At training time, a per-channel
# logistic CDF prior (src/tac/entropy_bottleneck.py — EntropyBottleneck) is
# attached as a forward hook on the renderer's bottleneck activation.
# Continuous relaxation: uniform[-0.5,0.5] noise during training; hard
# torch.round() during eval. The estimated mean bits/element is added to
# the loss with weight `eb_lambda`, pushing the latent toward a
# lower-entropy distribution that compresses better post-hoc.
#
# Lane EBR's claim: by jointly optimizing reconstruction + estimated bit
# rate, the trained renderer.bin should be more compressible after FP4 +
# Brotli (the existing pipeline) than the same arch trained without the
# entropy regularizer. Rate is currently ~44% of the score wedge at the
# Lane G v3 floor (1.05) — even a modest -0.04 rate cut moves the needle.
#
# At inflate time NO entropy_bottleneck weights are loaded — they are
# stripped via _strip_t2_training_only_state (train_renderer.py:1401-1411).
# The bottleneck is a TRAINING-TIME regularizer; the rate gain comes from
# the renderer.bin packing better, not from any runtime entropy decode.
# Strict-scorer-rule preserved.
#
# BASELINE: Lane A (`submissions/baseline_dilated_h64_0_90/`) at 1.15
# [contest-CUDA] (renderer.bin + masks.mkv + optimized_poses.pt). The EBR
# profile (`ebr_dilated_h64`) inherits PROVEN_BASELINE arch (dilated h=64)
# and adds use_entropy_bottleneck=True / eb_lambda=0.01 / eb_num_channels=64.
#
# PREDICTED BAND: [0.90, 1.10] [contest-CUDA].
#   Floor 0.90: 0.04 rate reduction from EB-aware training + 0.01 byproduct
#               distortion improvement (better-conditioned latents).
#   Ceiling 1.10: 0.05 below baseline if EB only marginally compresses.
#               Worst case: eb_lambda fights reconstruction → flat or +0.05.
#   Council Quantizr: lambda=0.01 is conservative; if proxy bits don't
#                     drop after 200 epochs, kill early — don't burn 14h.
#
# COMPOSITION: Lane EBR composes with Lane EC (engineered corrections,
# 585f85d2). Both attack different score components (EBR rate; EC SegNet)
# and use the SAME baseline artifacts as anchor. After Lane EBR lands a
# compressed renderer.bin, swap into Lane EC's archive build.
#
# CLAUDE.md compliance:
#   * set -euo pipefail (zip_dep_bootstrap_trap memory)
#   * Python `zipfile.ZipFile` (NOT shell `zip` — PyTorch container has none)
#   * --device cuda everywhere (no MPS/CPU fallback)
#   * Stage 0 NVDEC probe (check 33 + feedback_vastai_nvdec_host_variation)
#   * Verified CLI flags via grep add_argument:
#       experiments/train_renderer.py:
#         --profile --tag --device --output-dir --epochs --auth-eval-on-best
#         --auth-eval-masks --auth-eval-poses --auth-eval-upstream-dir
#         --video --seed --eval-roundtrip --eval-every
#       experiments/contest_auth_eval.py:
#         --archive --inflate-sh --upstream-dir --device --keep-work-dir
#         --work-dir
#   * predicted_band metadata + [contest-CUDA] tag in completion line
#   * provenance.json + heartbeat.log (canonical_remote_bootstraps memory)
#   * macOS AppleDouble cleanup pre-eval (check 37)
#   * Container Python /opt/conda/bin/python (NOT venv)
#   * Strict-scorer-rule: NO scorer at inflate time (EB params stripped)
#
# ANCHOR_LANE_A_BASELINE=submissions/baseline_dilated_h64_0_90  # for tarball auto-discovery

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_ebr_entropy_bottleneck_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-ebr] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_ebr_entropy_bottleneck.sh',
    'lane_name': 'lane_ebr_entropy_bottleneck',
    'predicted_band': [0.90, 1.10],
    'score_tag': '[contest-CUDA]',
    'baseline_score': 1.15,
    'baseline_artifact_dir': 'submissions/baseline_dilated_h64_0_90',
    'training_profile': 'ebr_dilated_h64',
    'eb_lambda': 0.01,
    'eb_num_channels': 64,
    'output_dir': '$LOG_DIR',
    'note': 'Balle-2018 entropy bottleneck as TRAINING-TIME rate regularizer. '
            'EB params stripped before deploy (strict-scorer-rule). Rate gain '
            'comes from better post-hoc FP4+Brotli compression of the trained '
            'renderer.bin, not from any runtime entropy decode.',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=EBR gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend (check 33 + memory
# feedback_vastai_nvdec_host_variation).
log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed (exit $?). Refusing to spend GPU on a"
    log "       host that cannot run upstream/evaluate.py at the end."
    log "       Destroy this Vast.ai instance and pick a different host."
    exit 2
}

# Stage 0b: git pull so remote code matches local HEAD
# (CLAUDE.md remote-code-parity non-negotiable + feedback_remote_code_parity_required).
# CODE PARITY: launcher tarball is authoritative — do NOT git reset --hard.
# Doing so wipes local-only anchor files (archive_lane_a.zip, baseline dirs,
# etc.) that the launcher just SCP'd. The tarball IS the parity mechanism.
# (memory: feedback_git_reset_nukes_anchors_20260429)

# Stage 1: stage Lane A baseline anchor (the masks + poses Lane EBR will be
# evaluated against). The renderer.bin is what we OVERWRITE with the EBR-
# trained one in Stage 3; we keep Lane A's masks/poses unchanged so the
# delta is attributable purely to the renderer.
log "=== Stage 1: stage Lane A baseline anchor (masks + poses for archive build) ==="
BASELINE_DIR="$WORKSPACE/submissions/baseline_dilated_h64_0_90"
LANE_A_RENDERER="$BASELINE_DIR/renderer.bin"
LANE_A_MASKS="$BASELINE_DIR/masks.mkv"
LANE_A_POSES="$BASELINE_DIR/optimized_poses.pt"
GT_VIDEO="$WORKSPACE/upstream/videos/0.mkv"

for f in "$LANE_A_RENDERER" "$LANE_A_MASKS" "$LANE_A_POSES" "$GT_VIDEO"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  Lane A renderer.bin   ($(stat -c '%s' "$LANE_A_RENDERER" 2>/dev/null || stat -f '%z' "$LANE_A_RENDERER") bytes)"
log "  Lane A masks.mkv      ($(stat -c '%s' "$LANE_A_MASKS" 2>/dev/null || stat -f '%z' "$LANE_A_MASKS") bytes)"
log "  Lane A optimized_poses.pt ($(stat -c '%s' "$LANE_A_POSES" 2>/dev/null || stat -f '%z' "$LANE_A_POSES") bytes)"

# Stage 2: train renderer with the EBR profile. The profile sets
# use_entropy_bottleneck=True / eb_lambda=0.01 / eb_num_channels=64 — these
# are NOT direct CLI flags on train_renderer.py (they come from PROFILES;
# verified via grep "p.add_argument" — only the profile selector flag delivers
# them).  (Avoid the literal `--profile <word>` phrase here so the
# preflight regex `--profile[\s=]+(\w+)` does not flag this comment as a
# missing-profile reference.)
#
# eval_roundtrip=True is the train_renderer default since the 2026-04-21
# fix; we pass it explicitly so the run record proves it. masks/poses for
# the auto auth-eval-on-best are pinned to Lane A so each *BEST* checkpoint
# is scored against the same anchor we'll use for the final Stage 3 build.
log "=== Stage 2: train renderer with profile=ebr_dilated_h64 (Balle-2018 EB) ==="
log "  --profile ebr_dilated_h64  (use_entropy_bottleneck=True, eb_lambda=0.01)"
log "  --auth-eval-masks=Lane-A-masks  --auth-eval-poses=Lane-A-poses"
TRAIN_OUT="$LOG_DIR/train"
mkdir -p "$TRAIN_OUT"
"$PYBIN" -u -m tac.experiments.train_renderer \
    --profile ebr_dilated_h64 \
    --tag ebr \
    --device cuda \
    --output-dir "$TRAIN_OUT" \
    --video "$GT_VIDEO" \
    --seed 2026 \
    --eval-roundtrip \
    --auth-eval-on-best \
    --auth-eval-masks "$LANE_A_MASKS" \
    --auth-eval-poses "$LANE_A_POSES" \
    --auth-eval-upstream-dir upstream \
    2>&1 | tee "$LOG_DIR/train_renderer.log" | tail -60
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Stage 2 verifies: train_renderer.py with --auth-eval-on-best produces
# the canonical FP4A renderer.bin via export_asymmetric_checkpoint_fp4
# (train_renderer.py:3536-3543). The .bin path is out_dir/renderer_<tag>_best.bin.
# This is the SAME format Lane A's renderer.bin uses, so inflate_renderer.py's
# _inline_load_fp4a deserializer reads it without modification.
# EB params are stripped before export via _strip_t2_training_only_state
# (train_renderer.py:1401-1411), preserving strict-scorer-rule.
BEST_BIN="$TRAIN_OUT/renderer_ebr_best.bin"
if [ ! -f "$BEST_BIN" ]; then
    log "FATAL: train_renderer.py did NOT produce $BEST_BIN."
    log "       Either training failed (see train_renderer.log) or no *BEST*"
    log "       checkpoint was reached (eval_every=5 in profile; check eval"
    log "       trajectory). The auth-eval-on-best path emits this file via"
    log "       export_asymmetric_checkpoint_fp4 — without it Lane EBR has"
    log "       nothing to ship. Abort rather than silently fall back."
    exit 2
fi
BEST_BYTES=$(stat -c '%s' "$BEST_BIN" 2>/dev/null || stat -f '%z' "$BEST_BIN")
LANE_A_BYTES=$(stat -c '%s' "$LANE_A_RENDERER" 2>/dev/null || stat -f '%z' "$LANE_A_RENDERER")
log "  EBR renderer.bin    = ${BEST_BYTES} bytes (Lane A baseline = ${LANE_A_BYTES} bytes)"
log "  EBR vs Lane A delta = $((BEST_BYTES - LANE_A_BYTES)) bytes (negative = rate win)"

# Validate the new renderer.bin is loadable as FP4A so we don't ship a file
# that crashes inflate_renderer.py (the SHIRAZ-class arch-drift class).
"$PYBIN" -c "
import sys
sys.path.insert(0, '$WORKSPACE/src')
sys.path.insert(0, '$WORKSPACE/upstream')
from tac.renderer_export import load_asymmetric_checkpoint_fp4
model = load_asymmetric_checkpoint_fp4('$BEST_BIN', device='cuda')
n_params = sum(p.numel() for p in model.parameters())
print(f'EBR renderer.bin loads OK: {type(model).__name__} '
      f'with {n_params:,} parameters')
"

# Stage 3: build new archive with EBR renderer + Lane A masks + Lane A poses.
# Python zipfile.ZipFile (NOT shell zip — PyTorch container has none, per
# memory feedback_zip_dep_bootstrap_trap). Deterministic timestamps so the
# archive bytes hash is stable across rebuilds (codex R5-r6 #5).
log "=== Stage 3: build Lane EBR archive (EBR renderer + Lane A masks + Lane A poses) ==="
ARCHIVE="$LOG_DIR/archive_lane_ebr.zip"
"$PYBIN" -c "
import os, zipfile
files = [
    ('renderer.bin', '$BEST_BIN'),
    ('masks.mkv', '$LANE_A_MASKS'),
    ('optimized_poses.pt', '$LANE_A_POSES'),
]
for name, path in files:
    assert os.path.isfile(path), f'missing {path}'
dst = '$ARCHIVE'
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name, path in files:
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
log "  archive_lane_ebr.zip = ${ARCHIVE_BYTES} bytes"

# Validate archive structurally so a silent-skip in the inflate dispatcher
# can't be blamed for a flat score.
"$PYBIN" -c "
import zipfile, sys
with zipfile.ZipFile('$ARCHIVE') as z:
    names = set(z.namelist())
    required = {'renderer.bin', 'masks.mkv', 'optimized_poses.pt'}
    missing = required - names
    if missing:
        print(f'FATAL: archive missing required entries: {missing}', file=sys.stderr)
        sys.exit(2)
    print(f'archive OK: {sorted(names)}')
"

# Strip macOS AppleDouble files from upstream/videos before auth eval
# (check 37 + feedback_canonical_remote_bootstraps).
rm -f upstream/videos/._*.mkv

log "=== Stage 4: contest_auth_eval on Lane EBR archive [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
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

log "=== LANE_EBR_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "=== predicted_band=[0.90, 1.10], baseline=1.15, see $LOG_DIR/provenance.json ==="
