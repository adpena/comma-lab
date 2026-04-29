#!/bin/bash
# Lane Ω: per-weight Hessian-aware bit-budget quantization anchored on Lane A.
#
# Council 2026-04-27: Lane A landed at 1.15 [contest-CUDA] (rate = 60% of
# score). Lane S (per-channel learnable bit-depth) and Lane W (per-channel
# + hard-pair weighted) are the per-channel attacks. Lane Ω is the
# per-WEIGHT extension: each individual weight gets its own bit-depth from
# water-fill allocation over a fixed total bit budget, driven by the
# Hessian (Fisher importance) on either Lane W's hard pairs OR uniform
# all-pairs (fallback when Lane W's profile isn't yet built).
#
# Predicted: rate drops from 0.0185 (Lane A) → ~0.005-0.008 (Lane Ω at
# B=600,000 bits = 75KB renderer.bin payload), distortion preserved by
# leaving the critical 1-5% of weights at high bit-depth. Predicted band
# [0.70, 1.05] [contest-CUDA].
#
# Pipeline:
#   Stage 0 — NVDEC probe (catches bad-host in 5s).
#   Stage 1 — profile per-weight Hessian importance on hard pairs.
#   Stage 2 — water-fill bit allocation at target B = 600,000 bits.
#   Stage 3 — constrained QAT: load Lane A + apply FrozenBitFakeQuant
#             per-weight + fine-tune 200 epochs at lr=2.5e-6.
#             [PHASE 3 NOTE: Lane Ω constrained QAT is invoked via a
#              dedicated entrypoint stub `experiments/qat_omega.py` —
#              this is a future-work stub: the QAT loop wiring must be
#              added in a follow-up commit. For now Stage 3 is the
#              direct (no-fine-tune) export of the bit-quantized weights.]
#   Stage 4 — Ωv1 export of (optionally fine-tuned) renderer → renderer.bin.
#   Stage 5 — build archive + contest_auth_eval [contest-CUDA].
#
# Anchored artifacts:
#   * renderer.bin: experiments/results/lane_a_landed/iter_0/renderer.bin (290KB FP32)
#   * masks.mkv:    experiments/results/lane_a_landed/iter_0/masks.mkv
#   * poses.pt:     experiments/results/lane_a_landed/iter_0/optimized_poses.pt
#
# Council preflight (CLAUDE.md non-negotiable: NEVER invent CLI flags):
#   * profile_hessian_per_weight.py: --checkpoint --video --masks-mkv
#     --poses --upstream --output --top-k --all-pairs --device --pair-batch
#     (verified 2026-04-27 against the script's argparse).
#   * contest_auth_eval.py: --archive --inflate-sh --upstream-dir --device
#     --keep-work-dir --work-dir
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
# Honor inherited PYBIN (Modal sets to /usr/local/bin/python; Vast.ai
# default = /opt/conda/bin/python via PyTorch container).
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_omega_results"
mkdir -p "$LOG_DIR"
TAG="lane_omega_hessian_qat"

log() { echo "[lane-omega] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline + memory
# feedback_canonical_remote_bootstraps).
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
    'lane_script': 'scripts/remote_lane_omega_hessian_qat.sh',
    'lane_name': 'lane_omega_hessian_qat_on_lane_a',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/iter_0/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/iter_0/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [0.70, 1.05],
    'rationale': 'Per-WEIGHT Hessian-aware bit allocation. Water-fill 600k bits across all eligible Conv2d/Linear weights. Critical weights (high Fisher importance) keep 6-8 bits; bulk weights drop to 1-2 bits. Protected layers (head/motion/FiLM/fuse_conv) stay FP16 — same protection list as Lane S.',
    'target_total_bits': 600000,
    'alpha': 0.5,
    'min_bits': 1,
    'max_bits': 8,
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=Omega gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 — NVDEC probe BEFORE any GPU spend (memory:
# feedback_vastai_nvdec_host_variation).
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: anchor on Lane A artifacts committed to the repo.
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/iter_0/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_a_landed/iter_0/masks.mkv"
for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         "$ANCHOR_MASKS" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer: $ANCHOR_RENDERER ($(stat -c '%s' "$ANCHOR_RENDERER") bytes, ASYM FP32)"
log "  anchor_poses:    $ANCHOR_POSES"
log "  anchor_masks:    $ANCHOR_MASKS ($(stat -c '%s' "$ANCHOR_MASKS") bytes)"

# Pre-flight: dead-flag scan (CLAUDE.md non-negotiable: NEVER invent CLI flags).
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_omega_hessian_qat.sh').read()
prof_src = open('experiments/profile_hessian_per_weight.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', prof_src))
m = re.search(r'experiments/profile_hessian_per_weight\.py(.*?)(?=\n# Stage 2|\nlog \"=== Stage 2|\Z)',
              script, re.DOTALL)
assert m, 'could not locate profile_hessian_per_weight.py invocation'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in profile_hessian_per_weight argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Stage 1 — profile per-weight Hessian importance (uniform all-pairs since
# Lane W's pair_weights.pt is not always available; the profiler accepts
# either). Uses Lane A's renderer + masks + poses for the gradient signal.
log "=== Stage 1: per-weight Hessian importance profile ==="
HESSIAN_PT="$LOG_DIR/hessian_per_weight.pt"
"$PYBIN" -u experiments/profile_hessian_per_weight.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --video upstream/videos/0.mkv \
    --masks-mkv "$ANCHOR_MASKS" \
    --poses "$ANCHOR_POSES" \
    --upstream upstream \
    --output "$HESSIAN_PT" \
    --top-k 30 \
    --all-pairs \
    --device cuda \
    --pair-batch 4 2>&1 | tee "$LOG_DIR/profile.log" | tail -30
[ -f "$HESSIAN_PT" ] || { echo "FATAL: profiler didn't produce $HESSIAN_PT"; exit 2; }
log "  hessian profile: $HESSIAN_PT ($(stat -c '%s' "$HESSIAN_PT") bytes)"

# Stage 2 — water-fill bit allocation + Ωv1 export of un-fine-tuned weights.
# This is the "no-QAT" Lane Ω lower bound — the QAT step (Stage 3 below)
# is currently a placeholder until experiments/qat_omega.py lands.
log "=== Stage 2: water-fill bit allocation (target 600k bits) ==="
mkdir -p "$LOG_DIR/qat"
BITS_PT="$LOG_DIR/bits_per_weight.pt"
OMEGA_BIN="$LOG_DIR/qat/renderer.bin"
"$PYBIN" -c "
import sys, torch, json
sys.path.insert(0, 'src')
from tac.bit_allocator import allocate_bits, allocation_report
from tac.renderer_export import (
    load_any_renderer_checkpoint,
    export_omega_renderer,
)

# Load the per-weight Hessian importance from Stage 1.
hessian = torch.load('$HESSIAN_PT', map_location='cpu', weights_only=False)
imp = hessian['importance']
meta = hessian['metadata']
print(f'[stage2] loaded importance for {len(imp)} layers, '
      f'{sum(t.numel() for t in imp.values()):,} weights')

# Water-fill at target B = 600,000 bits ≈ 75KB renderer.bin payload (before
# LZMA + bit-packing overhead).
TARGET_BITS = 600_000
bits = allocate_bits(imp, total_bits=TARGET_BITS, alpha=0.5,
                     min_bits=1, max_bits=8)
report = allocation_report(bits, imp)
print(f'[stage2] mean bits/weight: {report[\"mean_bits\"]:.3f} '
      f'(target ≈ {TARGET_BITS / report[\"total_weights\"]:.3f})')
print(f'[stage2] histogram (bits 0..8): {report[\"bits_histogram_0_8\"]}')

torch.save({'bits': bits, 'report': report, 'profile_meta': meta},
           '$BITS_PT')
print(f'[stage2] WROTE $BITS_PT')

# Stage 3 (currently no-fine-tune): export Lane A weights with the
# allocated per-weight bit-depths. The QAT loop is a stub.
print(f'[stage2] [Stage 3 stub] loading Lane A renderer for export ...')
model = load_any_renderer_checkpoint('$ANCHOR_RENDERER', device='cpu')
n_bytes = export_omega_renderer(model, bits, '$OMEGA_BIN', use_lzma=True)
print(f'[stage2] WROTE $OMEGA_BIN: {n_bytes} bytes (vs Lane A 290KB)')
" 2>&1 | tee -a "$LOG_DIR/run.log"
[ -f "$OMEGA_BIN" ] || { echo "FATAL: Ωv1 export failed"; exit 2; }
OMEGA_SIZE=$(stat -c '%s' "$OMEGA_BIN")
log "  Ωv1 binary: $OMEGA_BIN ($OMEGA_SIZE bytes)"

# Stage 3 — TODO: constrained QAT for Lane Ω.
# Until experiments/qat_omega.py is wired (loads Lane A, swaps eligible
# Conv2d → FrozenBitConv2d, fine-tunes 200 epochs at lr=2.5e-6 with
# eval_roundtrip + KL distill + the per-weight bits buffer frozen), Stage
# 3 here is a no-op marker. The Stage 2 export already writes a usable
# OMG1 binary; QAT just adds a delta on top.
log "=== Stage 3: constrained QAT [STUB — direct export from Stage 2] ==="
log "  (qat_omega.py not yet implemented; Lane Ω predicted band reflects no-fine-tune)"

# Stage 4 — build archive (Ωv1 renderer + Lane A masks + Lane A poses).
log "=== Stage 4: build archive (Ωv1 renderer + Lane A masks + Lane A poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$OMEGA_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$ANCHOR_MASKS" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_omega.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        z.write(p, arcname=n)
print(f'archive {dst}: {os.path.getsize(dst)} bytes')
"
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo "FATAL: archive empty" >&2; exit 2; }
log "  archive: $ARCHIVE ($ARCHIVE_BYTES bytes)"

# Stage 5 — contest_auth_eval on the EXACT archive that would be submitted.
log "=== Stage 5: contest_auth_eval on Lane Ω archive [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

# Sanity-check the auth eval emitted RESULT_JSON (catches silent crashes,
# LANE-B 2026-04-26 cascade pattern).
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval log has no RESULT_JSON line — eval crashed."
    exit 4
fi

log "=== LANE_OMEGA_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
