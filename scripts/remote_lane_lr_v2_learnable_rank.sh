#!/bin/bash
# Lane LR-V2 — LEARNABLE-rank LoRA pose adaptation TTO. Anchored on Lane A's
# 1.15 [contest-CUDA] floor.
#
# V1 oversight (memory project_posenet_rank1_discovery): rank=1 was hard-coded
# based on the 99.8%-variance heuristic. The OPTIMAL rank for the contest
# score may be 1 OR 2 OR 3 — picking it offline is a council decision the
# optimiser is better placed to make.
#
# V2 fix: max_rank=6 + per-rank gates (sigmoid of learnable scalars). Gates
# co-train with U + V; ranks with final gate < 0.1 are pruned at
# serialisation. The effective rank is data-driven.
#
# Single variable vs Lane LR-V1: --learnable-lora-max-rank 6 (replaces
# --lora-rank 1). Everything else identical to remote_lane_lr_lora_pose_tto.sh.
#
# Predicted band: [1.10, 1.16] [contest-CUDA].
#   * Floor 1.10: V2 prunes back to rank-1 (matching V1's predicted landing).
#   * Ceiling 1.16: V2 keeps rank-2 or rank-3, captures non-radial-zoom
#     pose variance V1 was missing, beats Lane A's frontier by ~0.05.
# Anchor: Lane A's 1.15 [contest-CUDA] artifacts at experiments/results/
#         lane_a_landed/.
#
# Cost paranoia: identical to V1 (~$0.50 4090 budget, 30-45 min wall clock).
# ── Bash safety ─────────────────────────────────────────────────────────
# CLAUDE.md non-negotiable: `set -euo pipefail`. The cascade trap that ate
# LANE-B 6.5h + $2 in 2026-04-26 must not reappear.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_lr_v2_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-lr-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_lr_v2_learnable_rank.sh',
    'output_dir': '$LOG_DIR',
    'lane_internal_name': 'lane_lr_v2',
    'anchor_score_baseline': 1.15,
    'anchor_artifacts': 'experiments/results/lane_a_landed/',
    'predicted_band': [1.10, 1.16],
    'delta_from_lane_a': 'learnable-rank LoRA (max_rank=6, gates pruned at 0.1)',
    'delta_from_lane_lr_v1': 'rank is LEARNABLE not FROZEN',
    'frozen_baseline_dims': 'all 6 — base is the warm-start poses',
    'learnable_lora_max_rank': 6,
    'lora_prune_threshold': 0.1,
    'lora_init_gate_logit': 0.0,
    'lora_steps': 500,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=LR-V2 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 NVDEC probe — refuse to spend GPU on a host that cannot run
# upstream/evaluate.py at the end (memory: feedback_vastai_nvdec_host_variation).
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py at the end. Destroy this"
    log "       Vast.ai instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present (Lane A's verified renderer +
# masks + warm-start poses).
for f in experiments/results/lane_a_landed/iter_0/renderer.bin \
         experiments/results/lane_a_landed/iter_0/masks.mkv \
         experiments/results/lane_a_landed/optimized_poses.pt \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Pre-flight: dead-flag-wiring guard. Every CLI flag we pass to
# experiments/optimize_poses.py MUST exist in its argparse. Catches the
# 2026-04-26 dead-auth-eval-masks bug class (memory:
# feedback_dead_flag_wiring_pattern).
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_lr_v2_learnable_rank.sh').read()
op_src = open('experiments/optimize_poses.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', op_src))
m = re.search(
    r'experiments/optimize_poses\.py(.*?)(?=\n\s*\[ -f|\Z)',
    script, re.DOTALL,
)
assert m, 'could not locate optimize_poses.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in optimize_poses argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
"

log "=== Stage 1: pose TTO with --learnable-lora-max-rank 6, warm-start from Lane A ==="
log "   --gt-poses-path = experiments/results/lane_a_landed/optimized_poses.pt"
log "   --learnable-lora-max-rank 6 (Lane LR-V2 — gates prune toward effective rank)"
log "   --lora-prune-threshold 0.1 --lora-init-gate-logit 0.0"
log "   --eval-roundtrip + --posetto-noise-std=0.5 (Fridrich C1 fixes)"
# Determinism: pin seeds + python hash (CLAUDE.md non-negotiable).
export PYTHONHASHSEED=1234
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint experiments/results/lane_a_landed/iter_0/renderer.bin \
    --masks experiments/results/lane_a_landed/iter_0/masks.mkv \
    --gt-poses-path experiments/results/lane_a_landed/optimized_poses.pt \
    --device cuda \
    --steps 500 \
    --learnable-lora-max-rank 6 \
    --lora-prune-threshold 0.1 \
    --lora-init-gate-logit 0.0 \
    --lora-steps 500 \
    --batch-pairs 8 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPESTATUS[0]}" >&2; exit "${PIPESTATUS[0]}"
    fi

# Validate: did optimize_poses produce the LoRA-V2-encoded .pt?
[ -f "$LOG_DIR/optimized_poses.pt" ] || { echo "FATAL: missing optimized_poses.pt"; exit 2; }
LORA_BYTES=$(stat -c '%s' "$LOG_DIR/optimized_poses.pt" 2>/dev/null || stat -f '%z' "$LOG_DIR/optimized_poses.pt")
log "  produced optimized_poses.pt (${LORA_BYTES} bytes — Lane LR-V2 learnable-rank)"

# Sanity check: the saved .pt MUST be a LoRA-V2 dict (format sentinel
# present) AND the materialised pose tensor MUST be (N, 6) — fail loud
# on any mismatch so we don't burn $0.20 of contest_auth_eval discovering
# a wiring bug.
"$PYBIN" -c "
import sys, torch
from tac.lora_pose_v2 import is_lora_v2_poses_dict, decode_lora_v2_poses_dict
obj = torch.load('$LOG_DIR/optimized_poses.pt', map_location='cpu', weights_only=False)
if not is_lora_v2_poses_dict(obj):
    print('FATAL: optimized_poses.pt is not a LoRA-V2 dict; got', type(obj).__name__, file=sys.stderr)
    sys.exit(2)
poses = decode_lora_v2_poses_dict(obj, pose_dim=6)
if poses.shape[0] != 600 or poses.shape[1] != 6:
    print(f'FATAL: materialised LoRA-V2 poses shape {tuple(poses.shape)} != (600, 6)', file=sys.stderr)
    sys.exit(2)
print(f'[Lane LR-V2] sanity OK — kept_rank={obj[\"rank\"]}/{obj[\"max_rank\"]}; '
      f'kept_indices={obj[\"kept_indices\"]}; '
      f'final_gate_values={obj[\"final_gate_values\"].tolist()}')
"

log "=== Stage 2: build NEW archive (Lane A renderer + Lane A masks + LoRA-V2 poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp experiments/results/lane_a_landed/iter_0/renderer.bin "$LOG_DIR/iter_0/renderer.bin"
cp experiments/results/lane_a_landed/iter_0/masks.mkv "$LOG_DIR/iter_0/masks.mkv"
cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_lr_v2.zip"
# Python zipfile (NOT shell `zip`) — PyTorch container has no `zip` binary
# (memory: feedback_zip_dep_bootstrap_trap).
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

log "=== Stage 3: contest_auth_eval on Lane LR-V2 archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPESTATUS[0]}" >&2; exit "${PIPESTATUS[0]}"
    fi

# RESULT_JSON guard — silent eval crash protection (LANE-B-style cascade).
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash. Investigate before reporting score."
    exit 4
fi

log "=== LANE_LR_V2_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "  predicted band: [1.10, 1.16] (vs Lane LR-V1 [1.10, 1.18])"
log "  anchor baseline: 1.15 [contest-CUDA] (Lane A frontier)"
