#!/bin/bash
# Lane J-JBL — Jaccard Metric Loss + Boundary Label Smoothing distillation.
#
# Jack-from-skunkworks Cycle 1 TOP-1 SegNet attack (research file
# .omx/research/jack_skunkworks_segnet_rate_research_20260428.md §S1).
# Anchored on Lane G v3 = 1.05 [contest-CUDA] (the current frontier).
#
# Mechanism (Wang et al. NeurIPS 2023, arXiv 2302.05666):
#   * Replace the SegNet KL distill auxiliary with the JBL combined loss:
#     - Jaccard Metric Loss on student vs teacher soft labels (subsumes
#       Lovász-Softmax, +2-5 mIoU near boundaries on EfficientNet)
#     - Boundary Label Smoothing on GT cross-entropy with boundary pixels
#       weighted 3× interior pixels.
#   * The kl_distill_weight knob is repurposed as the JBL master scalar
#     so the wiring stays byte-identical to Lane G v3 except the loss
#     family.
#
# Predicted band [0.92, 1.02] per Jack §S1. Lowest cost, highest
# confidence SegNet attack: SegNet wedge = 38% of Lane G v3 = 0.401 score
# points. A conservative 20% reduction in SegNet distortion (0.004 →
# 0.0032) yields -0.08 score = 0.97.
#
# Cost cap: $1.50 / 8h on RTX 4090 @ Vast.ai. Destroy instance immediately
# at completion or on any heartbeat-stale > 30 min (per CLAUDE.md
# "Vast.ai cost paranoia" + memory feedback_per_instance_verify_pattern).
#
# Per CLAUDE.md non-negotiables:
#   * Preflight Stage 0 = NVDEC probe (--ensure-dali) before any GPU spend.
#   * Stage 1 = canonical git fetch + reset --hard origin/main to enforce remote_code_parity.
#   * Stage N+1 = contest-CUDA auth eval against the EXACT archive bytes
#     submitted (no proxy, no MPS, label every score [contest-CUDA]).
#   * Heartbeat every 60s to /tmp/heartbeat_lane_j_jbl.log.
#   * AppleDouble (._*) cleanup after rsync from macOS host.
#   * Dead-flag-wiring guard: every CLI flag below is verified against
#     train_renderer.py argparse before launch.
#
# CLI flag verification (per memory feedback_dead_flag_wiring_pattern):
#   --profile               train_renderer.py:~270  (verified)
#   --device                train_renderer.py:~250  (verified)
#   --epochs                train_renderer.py:~290  (verified)
#   --output-dir            train_renderer.py:~260  (verified)
#   --loss-mode             train_renderer.py NEW (Lane J-JBL flag, this commit)
#   --boundary-weight       train_renderer.py NEW (Lane J-JBL flag, this commit)
#   --bls-smoothing         train_renderer.py NEW (Lane J-JBL flag, this commit)
#   --kl-distill-weight     train_renderer.py:~313 (verified, repurposed scalar)
#   --kl-distill-temperature train_renderer.py:~316 (verified)
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
LANE_LABEL=lane-j-jbl
LOG_DIR="$WORKSPACE/lane_j_jbl_results"
HEARTBEAT="/tmp/heartbeat_lane_j_jbl.log"
PROVENANCE="$LOG_DIR/provenance.json"

mkdir -p "$LOG_DIR"
log() { echo "[$LANE_LABEL] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Stage 0: NVDEC probe (memory feedback_vastai_nvdec_host_variation).
# --ensure-dali installs DALI when the container hasn't yet been bootstrapped.
log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed — refusing to spend GPU on a host that"
    log "       cannot run upstream/evaluate.py at the end. Destroy this"
    log "       Vast.ai instance and pick a different host."
    exit 2
}

# Stage 1: enforce remote_code_parity (CLAUDE.md non-negotiable).
cd "$WORKSPACE"
# CODE PARITY: launcher tarball is authoritative — do NOT git reset --hard.
# Doing so wipes local-only anchor files (archive_lane_a.zip, baseline dirs,
# etc.) that the launcher just SCP'd. The tarball IS the parity mechanism.
# (memory: feedback_git_reset_nukes_anchors_20260429)
log "  git HEAD = $GIT_HASH"

# Stage 1b: AppleDouble cleanup (rsync from macOS leaves ._* sidecars
# that confuse setuptools / pytest — see memory
# feedback_remote_setup_script_correct_path_20260428).
log "=== Stage 1b: AppleDouble (._*) sidecar purge ==="
find "$WORKSPACE" -name '._*' -type f -delete 2>/dev/null || true

# Stage 2: container Python env (per CLAUDE.md: use /opt/conda Python,
# NOT a project venv on remote — see memory
# feedback_canonical_remote_bootstraps).
source "$WORKSPACE/env.sh" 2>/dev/null || true

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps): every remote run must
# emit provenance.json so a fresh agent can reconstruct the experiment.
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
    'lane_script': 'scripts/remote_lane_j_jbl_jaccard_bls.sh',
    'lane_label': '$LANE_LABEL',
    'output_dir': '$LOG_DIR',
    'profile': 'j_jbl_dilated_h64',
    'loss_mode': 'jbl',
    'boundary_weight': 3.0,
    'bls_smoothing': 0.1,
    'predicted_band': [0.92, 1.02],
    'cost_cap_usd': 1.50,
    'cost_cap_hours': 8.0,
    'anchor': 'Lane G v3 1.05 [contest-CUDA]',
    'wedge_attribution': 'SegNet 38% of 1.05 = 0.401 pts; conservative 20% reduction = -0.08 score',
    'paper': 'Wang et al. NeurIPS 2023 arXiv 2302.05666',
    'research_doc': '.omx/research/jack_skunkworks_segnet_rate_research_20260428.md',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"

# Heartbeat (CLAUDE.md: tmux session existence is NOT a heartbeat).
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=$LANE_LABEL gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 2b: dead-flag-wiring guard (memory feedback_dead_flag_wiring_pattern).
# Verify every --flag below exists in train_renderer.py argparse.
log "=== Stage 2b: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_j_jbl_jaccard_bls.sh').read()
tr_src = open('src/tac/experiments/train_renderer.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', tr_src))
m = re.search(r'train_renderer\.py(.*?)(?=\n\s*\[\s*-f|\n\s*log\b|\Z)',
              script, re.DOTALL)
assert m, 'could not locate train_renderer.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in train_renderer.py argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
"

# Stage 3: train renderer with j_jbl_dilated_h64 profile.
log "=== Stage 3: train renderer (profile=j_jbl_dilated_h64, loss_mode=jbl) ==="
log "    boundary_weight=3.0, bls_smoothing=0.1"
log "    predicted_band=[0.92, 1.02]"
export PYTHONHASHSEED=1234
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="src:upstream:$WORKSPACE"
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile j_jbl_dilated_h64 \
    --device cuda \
    --output-dir "$LOG_DIR/train" 2>&1 | tee "$LOG_DIR/train.log" | tail -50
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Locate best checkpoint.
BEST_CKPT=$(find "$LOG_DIR/train" -name 'renderer*BEST*.bin' -o -name 'renderer*best*.bin' 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(find "$LOG_DIR/train" -name 'renderer*.bin' 2>/dev/null | head -1)
fi
[ -f "$BEST_CKPT" ] || { log "FATAL: no renderer*.bin produced under $LOG_DIR/train"; exit 4; }
log "  best checkpoint = $BEST_CKPT"

# Stage 4: build archive (renderer.bin + masks.mkv + optimized_poses.pt).
# Anchor masks + poses on Lane G v3 results for fair A/B (only the
# renderer changes; mask + pose contributions are held constant so the
# auth eval delta isolates the JBL effect on SegNet distortion).
log "=== Stage 4: build archive (renderer = JBL-trained, masks/poses = Lane G v3 anchor) ==="
ANCHOR_MASKS="experiments/results/lane_g_v3_landed/iter_0/masks.mkv"
ANCHOR_POSES="experiments/results/lane_g_v3_landed/optimized_poses.pt"
for f in "$ANCHOR_MASKS" "$ANCHOR_POSES"; do
    [ -f "$f" ] || { log "FATAL: missing anchor file $f (need Lane G v3 landed dir)"; exit 5; }
done

mkdir -p "$LOG_DIR/iter_0"
cp "$BEST_CKPT" "$LOG_DIR/iter_0/renderer.bin"
cp "$ANCHOR_MASKS" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_j_jbl.zip"
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
ARCHIVE_BYTES=$(stat -c%s "$ARCHIVE" 2>/dev/null || stat -f%z "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo "FATAL: archive empty" >&2; exit 1; }
[ "$ARCHIVE_BYTES" -lt 5000000 ] || { echo "FATAL: archive >5MB ($ARCHIVE_BYTES) — composition bug" >&2; exit 1; }
log "archive size guard: $ARCHIVE_BYTES bytes (within sanity bounds)"

# Stage 5: contest-CUDA auth eval on the EXACT archive bytes submitted.
# CLAUDE.md "Auth eval EVERYWHERE" non-negotiable + "Auth eval measurement"
# (every auth eval must use the EXACT archive that will be submitted).
log "=== Stage 5: contest-CUDA auth eval [contest-CUDA] ==="
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

log "=== LANE_J_JBL_DONE -- see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "    cost cap: \$1.50 / 8h on RTX 4090; destroy Vast.ai instance immediately at completion."
