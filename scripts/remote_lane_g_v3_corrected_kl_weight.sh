#!/bin/bash
# Lane G V3 (corrected): Pose TTO warm-started from Lane A poses, with the
# Quantizr-validated SegNet KL-distillation auxiliary loss stacked on top
# of the standard scorer loss.
#
# V1 history (--kl-distill-weight 1.0):
#   * KL dominated total loss ~14000× over scorer terms (raw KL ≈ 20000,
#     scorer ≈ 1.0). Optimizer ignored the scorer signal entirely.
#     Killed after ~300 steps.
#
# V2 history (--kl-distill-weight 0.01):
#   * KL still dominated ~4000× (raw KL ≈ 20000, scorer ≈ 5.0). Same
#     failure mode at lower amplitude. Killed.
#
# V3 v2-prev (--kl-distill-weight 5e-6) — DEPRECATED, this script supersedes:
#   * 5e-6 was sized for the BUGGY raw-KL value of ~20000 (5e-6 × 20000 = 0.1).
#   * AFTER commit f17de3bb FIXED the KL-divergence reduction in
#     src/tac/losses.py (`F.kl_div(reduction="batchmean")` →
#     `F.kl_div(reduction="none").sum(dim=1).mean() * (T*T)`), the raw KL
#     value is ~2.7 instead of ~20000 (a ~7400× drop because batchmean
#     was dividing by a wrong dimension count).
#   * 5e-6 × 2.7 = 1.4e-5 — TOO SMALL to provide any auxiliary signal.
#     Effectively the KL term contributes nothing; V3-prev would have
#     reproduced Lane A exactly.
#
# V3 corrected (THIS SCRIPT — --kl-distill-weight 0.002):
#   * Council math (post-fix): with raw KL ≈ 2.7 and scorer loss ≈ 0.05,
#     weight 0.002 makes the KL contribution 0.002 × 2.7 = 5.4e-3, which
#     is ~10% of the total scorer loss (0.05). That ratio (KL ≈ 10% of
#     scorer) is the canonical Hinton 2015 auxiliary-distill regime —
#     enough signal to bias the renderer toward GT logit distributions
#     without overwhelming the primary score gradient.
#   * Same warm-start poses (Lane A's optimized_poses.pt), same 600 pairs
#     × 500 steps, same eval_roundtrip + posetto_noise=0.5 (Fridrich C1).
#
# Predicted score landing zone for V3 corrected ([1.10, 1.18]):
#   * Lane A's contest-CUDA score is 1.15. KL distill is the Quantizr
#     "secret sauce" per CLAUDE.md (T=2.0 on SegNet logits) — applied at
#     the right amplitude this should improve SegNet distortion by
#     ~1-3% while leaving PoseNet untouched (KL is SegNet-only via
#     `kl_distill_segnet_only`).
#   * Lower bound 1.10 = 4.3% improvement (best case, SegNet drops from
#     ~0.0029 to ~0.0027).
#   * Upper bound 1.18 = 2.6% regression (worst case, KL still slightly
#     pulls SegNet off the narrow plateau the scorer-only loss found).
#   * Anchor 1.15 = no change (KL is too weak to move the needle).
#
# Per CLAUDE.md non-negotiable (NEVER invent CLI flags): the flag names
# `--kl-distill-weight` / `--kl-distill-temperature` are verified by
# argparse-grep on experiments/optimize_poses.py L159-171. Loss is
# `kl_distill_segnet_only` from tac.losses (per-step log shows
# `kl=<float>` when --kl-distill-weight > 0).
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/lane_g_v3_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-g-v3] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps): every remote run must
# emit provenance.json so a fresh agent can reconstruct the experiment.
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
    'lane_script': 'scripts/remote_lane_g_v3_corrected_kl_weight.sh',
    'output_dir': '$LOG_DIR',
    'kl_distill_weight': 0.002,
    'kl_distill_temperature': 2.0,
    'predicted_band': [1.10, 1.18],
    'anchor_score_baseline': 1.15,
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/optimized_poses.pt',
    'delta_from_v3_prev': {
        'kl_distill_weight': '5e-6 -> 0.002 (400x increase)',
        'rationale': 'Post-bug-fix raw KL is ~2.7 (was ~20000 with the broken batchmean reduction in commit f17de3bb). Weight 5e-6 left KL contribution at 1.4e-5 (negligible). Weight 0.002 puts KL contribution at ~10% of scorer loss — canonical Hinton 2015 auxiliary regime.',
        'kl_bug_fix_commit': 'f17de3bb',
    },
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=G-V3 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend. Reference:
# feedback_vastai_nvdec_host_variation. --ensure-dali so a fresh container
# that hasn't run remote_setup_full.sh installs DALI rather than spuriously
# failing on a missing import.
log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py at the end. Destroy this"
    log "       Vast.ai instance and pick a different host."
    exit 2
}

# Pre-flight: required artifacts present. Anchor on Lane A (verified
# 1.15 [contest-CUDA] at experiments/results/lane_a_landed/).
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# Pre-flight: dead-flag-wiring guard — every CLI flag in the optimize_poses
# invocation below must exist in experiments/optimize_poses.py argparse.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_g_v3_corrected_kl_weight.sh').read()
op_src = open('experiments/optimize_poses.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', op_src))
m = re.search(r'experiments/optimize_poses\.py(.*?)(?=\n\s*\[\s*-f|\n\s*log\b|\Z)',
              script, re.DOTALL)
assert m, 'could not locate optimize_poses.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in optimize_poses argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
"

log "=== Stage 1: rebuild full-res masks (same as Lane A baseline) ==="
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 \
    --output "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5

# Extract just masks for optimize_poses input
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_baseline_seed.zip" 2>&1 | tail -3
cd "$WORKSPACE"

log "=== Stage 2: pose TTO (Lane A flow) + KL-distill SegNet auxiliary ==="
log "   --gt-poses-path = $ANCHOR_POSES"
log "   --eval-roundtrip + --posetto-noise-std=0.5 (Fridrich C1 fixes)"
log "   --kl-distill-weight=0.002  --kl-distill-temperature=2.0  (LANE G V3 CORRECTED — KL ≈ 10% of scorer loss)"
# Determinism: pin seeds + cublas + python hash (CLAUDE.md non-negotiable).
export PYTHONHASHSEED=1234
# 2026-04-27 launch finding (inherited from V3-prev): KL distill auxiliary
# loss adds a SECOND SegNet forward+backward pass per step. On RTX 4090 24GB
# this OOMs at --batch-pairs 8 (Lane A's setting) — peak alloc 23GB.
# Halving to 4 fits in ~13GB. Total work (600 pairs × 500 steps) is unchanged.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYBIN" -u experiments/optimize_poses.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --masks "$LOG_DIR/extracted/masks.mkv" \
    --gt-poses-path "$ANCHOR_POSES" \
    --device cuda \
    --steps 500 \
    --batch-pairs 4 \
    --eval-roundtrip \
    --posetto-noise-std 0.5 \
    --kl-distill-weight 0.002 \
    --kl-distill-temperature 2.0 \
    --output-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/optimize_poses.log" | tail -30

# Validate: did optimize_poses produce the file?
[ -f "$LOG_DIR/optimized_poses.bin" ] || { echo "FATAL: optimize_poses didn't produce optimized_poses.bin"; exit 2; }
log "  produced optimized_poses.bin ($(stat -c '%s' "$LOG_DIR/optimized_poses.bin") bytes)"

log "=== Stage 3: build NEW archive (Lane A renderer + masks + NEW poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$ANCHOR_RENDERER" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
[ -f "$LOG_DIR/optimized_poses.pt" ] && cp "$LOG_DIR/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_g_v3.zip"
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

log "=== Stage 4: contest_auth_eval on Lane G V3 archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15

log "=== LANE_G_V3_DONE -- see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
