#!/bin/bash
# Lane SZ Phase 2 — szabolcs no-masks paradigm full pipeline.
#
# Reproduces szabolcs-cs PR#56 (auth 0.36 [contest-CUDA]) with our codebase:
#   * Block-FP weight codec at ~1.0-1.5 bits/weight (post tar.xz).
#   * tar.xz double-compressed renderer (no masks.mkv, no optimized_poses.pt).
#   * Per-frame 6-DoF affine embedding lives INSIDE the renderer state — there
#     is no separate poses file in the archive. The Gaussian-softmax LUT is
#     reconstructed in code at inflate time from CLASS_TARGETS + LUT_SIGMA.
#
# Pipeline:
#   Stage 0  — NVDEC probe + provenance.json
#   Stage 1  — train_szabolcs.py (~10-12h on RTX 4090, $3-4)
#   Stage 2  — export_szabolcs_archive.py (SZv1 binary)
#   Stage 3  — build archive.zip (renderer.bin only — szabolcs paradigm)
#   Stage 4  — contest_auth_eval.py [contest-CUDA]
#
# Strict-scorer-rule: nothing here loads PoseNet/SegNet at inflate time. The
# training stage uses CUDA pixel L1 only.
#
# CLI flag pre-flight (CLAUDE.md non-negotiable: NEVER invent CLI flags):
#   train_szabolcs.py:
#     --device                     ✓ argparse choices=["cuda"]
#     --video                      ✓ argparse default repo upstream/videos/0.mkv
#     --total-epochs               ✓ argparse
#     --lr                         ✓ argparse
#     --batch-size                 ✓ argparse
#     --output-dir                 ✓ argparse required=True
#     --tag                        ✓ argparse
#     --hidden                     ✓ argparse
#     --num-blocks                 ✓ argparse
#     --seed                       ✓ argparse
#     --max-frames                 ✓ argparse
#   export_szabolcs_archive.py:
#     --checkpoint                 ✓ argparse required=True
#     --output                     ✓ argparse
#     --block-size                 ✓ argparse
#     --clip-threshold             ✓ argparse
#   contest_auth_eval.py:
#     --archive                    ✓ existing flag (lane_i precedent)
#     --inflate-sh                 ✓ existing flag (lane_i precedent)
#     --upstream-dir               ✓ existing flag (lane_i precedent)
#     --device                     ✓ existing flag (lane_i precedent)
#     --keep-work-dir              ✓ existing flag (lane_i precedent)
#     --work-dir                   ✓ existing flag (lane_i precedent)
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_sz_phase2_results"
mkdir -p "$LOG_DIR"
TAG="lane_sz_phase2"

log() { echo "[lane-sz] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# ── Provenance + heartbeat (CLAUDE.md canonical pipeline + memory
#    feedback_canonical_remote_bootstraps). ──────────────────────────────
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
    'lane_script': 'scripts/remote_lane_sz_phase2_full.sh',
    'lane_name': 'lane_sz_phase2_szabolcs_no_masks_paradigm',
    'predicted_band': [0.30, 0.50],
    'rationale': 'szabolcs PR#56 reports auth 0.36 [contest-CUDA]. Our SzabolcsRenderer is the byte-for-byte architecture replica from /tmp/szabolcs_re/inflate.py + an encoder we wrote (the reference is decode-only). Block-FP at 1.0-1.5 bits/weight + tar.xz outer compression. Predicted band: low end matches PR#56; high end accounts for our encoder fidelity + lack of QAT/pose-TTO refinement.',
    'archive_contents': 'renderer.bin (SZv1) ONLY — no masks.mkv (LUT reconstructed from luma), no optimized_poses.pt (per-frame affine embedded).',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=SZ gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# ── Stage 0: NVDEC probe (skip if missing — Lane SZ uses pyav, not DALI). ──
log "=== Stage 0: NVDEC probe ==="
if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
        # NVDEC not strictly required — Lane SZ training uses pyav CPU decode.
        # We log the failure but continue; the auth eval Stage 4 may need it.
        log "WARN: NVDEC probe non-zero — Stage 4 auth eval may need a different host."
    }
else
    log "  probe_nvdec.sh not present, skipping."
fi

# ── Pre-flight: required input files. ──────────────────────────────────
for f in upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         experiments/train_szabolcs.py \
         experiments/export_szabolcs_archive.py \
         submissions/robust_current/inflate.sh \
         src/tac/contrib/szabolcs_renderer.py \
         src/tac/szabolcs_archive.py \
         src/tac/block_fp_codec.py; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  inputs OK"

# ── Pre-flight: dead-flag-wiring guard for both Python invocations. ────
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
def _real_flags(path):
    src = open(path).read()
    return set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', src))

train_real = _real_flags('experiments/train_szabolcs.py')
export_real = _real_flags('experiments/export_szabolcs_archive.py')
auth_real = _real_flags('experiments/contest_auth_eval.py')
script = open('scripts/remote_lane_sz_phase2_full.sh').read()

def _used_in(block):
    return set(re.findall(r'\B--([a-z][a-z0-9-]+)', block))

# Locate Stage 1 (train), Stage 2 (export), Stage 4 (auth) blocks via marker comments.
stage1 = re.search(r'# Stage 1:.*?(?=# Stage 2:)', script, re.DOTALL)
stage2 = re.search(r'# Stage 2:.*?(?=# Stage 3:)', script, re.DOTALL)
stage4 = re.search(r'# Stage 4:.*?(?=\Z|# LANE_SZ_DONE)', script, re.DOTALL)
assert stage1 and stage2 and stage4, 'missing stage markers'

for label, block, real in [
    ('train', stage1.group(0), train_real),
    ('export', stage2.group(0), export_real),
    ('auth', stage4.group(0), auth_real),
]:
    used = _used_in(block)
    bogus = used - real
    if bogus:
        print(f'INVENTED FLAGS in {label}: {sorted(bogus)} not in argparse', file=sys.stderr)
        sys.exit(3)
    print(f'OK ({label}): {len(used)} flags all real')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Stage 1: train szabolcs renderer from scratch. ─────────────────────
log "=== Stage 1: train szabolcs renderer (~10-12h on 4090) ==="
TRAIN_DIR="$LOG_DIR/train"
mkdir -p "$TRAIN_DIR"
"$PYBIN" -u experiments/train_szabolcs.py \
    --device cuda \
    --video upstream/videos/0.mkv \
    --total-epochs 1200 \
    --lr 5e-4 \
    --batch-size 4 \
    --hidden 32 \
    --num-blocks 4 \
    --seed 1234 \
    --max-frames 1200 \
    --output-dir "$TRAIN_DIR" \
    --tag "$TAG" \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[train_szabolcs\]|epoch=|DONE|FATAL" | tail -200

BEST_CKPT="$TRAIN_DIR/${TAG}_best.pt"
[ -f "$BEST_CKPT" ] || {
    echo "FATAL: train_szabolcs didn't produce $BEST_CKPT" >&2
    ls -la "$TRAIN_DIR/" >&2
    exit 2
}
log "  best checkpoint: $BEST_CKPT ($(stat -c '%s' "$BEST_CKPT" 2>/dev/null || stat -f '%z' "$BEST_CKPT") bytes)"

# ── Stage 2: SZv1 export. ──────────────────────────────────────────────
log "=== Stage 2: SZv1 export ==="
SZV1_BIN="$LOG_DIR/renderer_szabolcs.bin"
"$PYBIN" -u experiments/export_szabolcs_archive.py \
    --checkpoint "$BEST_CKPT" \
    --output "$SZV1_BIN" \
    --block-size 16 \
    --clip-threshold 0.5 \
    --predicted-band-low 0.30 \
    --predicted-band-high 0.50 \
    2>&1 | tee "$LOG_DIR/export.log"

[ -f "$SZV1_BIN" ] || { echo "FATAL: SZv1 export failed" >&2; exit 2; }
SZV1_BYTES=$(stat -c '%s' "$SZV1_BIN" 2>/dev/null || stat -f '%z' "$SZV1_BIN")
log "  SZv1 binary: $SZV1_BIN ($SZV1_BYTES bytes)"

# ── Stage 3: build archive.zip (renderer.bin ONLY). ────────────────────
log "=== Stage 3: build archive.zip (szabolcs paradigm — renderer.bin only) ==="
ARCHIVE="$LOG_DIR/archive_lane_sz.zip"
"$PYBIN" -c "
import zipfile, os
src = '$SZV1_BIN'
dst = '$ARCHIVE'
assert os.path.isfile(src), f'missing {src}'
# Use python zipfile (CLAUDE.md: never shell zip on PyTorch container).
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    z.write(src, arcname='renderer.bin')
print(f'archive {dst}: {os.path.getsize(dst)} bytes')
"
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo "FATAL: archive empty" >&2; exit 2; }
log "  archive: $ARCHIVE ($ARCHIVE_BYTES bytes — szabolcs paradigm)"
log "  contents: renderer.bin only (NO masks.mkv, NO optimized_poses.pt)"

# ── Stage 4: contest_auth_eval [contest-CUDA]. ─────────────────────────
log "=== Stage 4: contest_auth_eval on Lane SZ archive [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

# ── Validate auth eval emitted a RESULT_JSON. ──────────────────────────
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval log has no RESULT_JSON line — eval crashed."
    exit 4
fi

log "=== LANE_SZ_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
