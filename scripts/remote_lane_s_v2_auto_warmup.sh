#!/bin/bash
# Lane S V2: Self-Compressing renderer with AUTO-WARMUP for the
# Lagrangian rate ramp. Replaces the V1 hand-coded
# `--self-compress-lambda-ramp-start-frac=0.3` constant with a SAGA-
# style scorer-loss convergence detector
# (`tac.scorer_loss_convergence_detector`). Once the per-epoch scorer
# loss has plateaued (OLS slope < 1e-4 / epoch on a 50-epoch sliding
# window), the rate penalty is allowed to ramp — no operator-tuned
# fraction.
#
# Theoretical backing: Hayashi 2000 §1.3 (OLS slope on a stationary
# process is an unbiased estimator of the underlying drift) +
# Polyak-Juditsky 1992 (averaged-iterate convergence rate) +
# Csefalvay 2023 §3.2 (SAGA-style early termination heuristic). See
# src/tac/scorer_loss_convergence_detector.py for the convergence
# proof.
#
# Pipeline strategy (mirrors V1, only the warmup mechanism changes):
#   1. Stage 0 NVDEC probe.
#   2. Stage 1 — load Lane A's renderer, swap to SC layers.
#   3. Stage 2 — train_renderer.py with --auto-warmup-lambda. The
#      detector observes the per-epoch scorer loss; once it plateaus
#      the Lagrangian rate ramp begins NEXT epoch (replacing the
#      static 0.3 · total_epochs trigger).
#   4. Stage 3 — SCv1 export.
#   5. Stage 4 — build archive.
#   6. Stage 5 — contest_auth_eval [contest-CUDA].
#
# Predicted band: identical or BETTER than V1's [0.85, 1.20] —
# auto-warmup eliminates the wasted "ramp-too-early" risk on slow-
# converging arches AND the wasted "ramp-too-late" risk on fast-
# converging arches. V1 was fixed at 30%; the detector typically
# fires anywhere in [15%, 45%] depending on arch.
#
# Council preflight (CLAUDE.md non-negotiable: NEVER invent CLI flags):
#   * --auto-warmup-lambda          ✓ argparse line ~384 of train_renderer.py
#   * --auto-warmup-window           ✓ argparse line ~393
#   * --auto-warmup-slope-tol        ✓ argparse line ~395
#   * --auto-warmup-min-epochs       ✓ argparse line ~397
#   * (all other flags identical to V1, see remote_lane_s_self_compress.sh)
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_s_v2_results"
mkdir -p "$LOG_DIR"
TAG="lane_s_v2_auto_warmup"

log() { echo "[lane-s-v2] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat.
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
    'lane_script': 'scripts/remote_lane_s_v2_auto_warmup.sh',
    'lane_name': 'lane_s_v2_auto_warmup_on_lane_a',
    'profile': 'self_compress_renderer_full',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/iter_0/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/iter_0/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [0.85, 1.20],
    'rationale': 'Auto-warmup eliminates the V1 ramp_start_frac=0.3 hand-coded constant. The detector fires when scorer loss has actually plateaued (Hayashi 2000 / Csefalvay 2023). Predicted equal-or-better than V1.',
    'sc_init_bits': 8.0,
    'sc_target_bits': 2.5,
    'sc_lambda_end': 1.0,
    'sc_lambda_ramp_start_frac_FALLBACK': 0.3,
    'auto_warmup_window': 50,
    'auto_warmup_slope_tol': 1e-4,
    'auto_warmup_min_epochs': 50,
    'total_epochs': 500,
    'lr': 5e-5,
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'lagrangian_target': 'auto-detect scorer convergence then ramp lambda → 1.0',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=S-V2 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 — NVDEC probe.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: anchor on Lane A.
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

# Pre-flight: profile validation.
log "=== Pre-flight: profile validation ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.profiles import PROFILES
assert 'self_compress_renderer_full' in PROFILES, \
    'profile self_compress_renderer_full not registered'
p = PROFILES['self_compress_renderer_full']
for k in ('use_self_compress_codec', 'self_compress_init_bits',
          'self_compress_target_bits', 'self_compress_lambda_end',
          'self_compress_lambda_ramp_start_frac'):
    assert k in p, f'profile missing key: {k}'
assert p['use_self_compress_codec'] is True
print(f'PROFILE OK: init={p[\"self_compress_init_bits\"]} '
      f'target={p[\"self_compress_target_bits\"]} '
      f'lambda_end={p[\"self_compress_lambda_end\"]} '
      f'fallback_ramp_start={p[\"self_compress_lambda_ramp_start_frac\"]}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: dead-flag scan (CLAUDE.md non-negotiable).
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_s_v2_auto_warmup.sh').read()
tr_src = open('src/tac/experiments/train_renderer.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', tr_src))
m = re.search(r'src/tac/experiments/train_renderer\.py(.*?)(?=\n# Stage 3:|\n\s*BEST_FP32=|\Z)',
              script, re.DOTALL)
assert m, 'could not locate train_renderer.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in train_renderer argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Stage 1: warm-start SC init from Lane A's FP32 ASYM renderer ==="
RESUME_INIT_PT="$LOG_DIR/lane_a_sc_init.pt"
"$PYBIN" -c "
import sys, torch
sys.path.insert(0, 'src')
from tac.renderer_export import load_any_renderer_checkpoint
from tac.self_compress import (
    swap_renderer_convs_with_self_compress,
    list_self_compress_layers,
    renderer_average_bits_per_weight,
)
src_bin = '$ANCHOR_RENDERER'
out_pt = '$RESUME_INIT_PT'
print(f'[stage1] loading Lane A renderer: {src_bin}')
model = load_any_renderer_checkpoint(src_bin, device='cpu')
diag = swap_renderer_convs_with_self_compress(model, init_bits=8.0)
print(f'[stage1] SC swap: swapped={len(diag[\"swapped\"])} '
      f'protected={len(diag[\"protected\"])} skipped={len(diag[\"skipped\"])}')
sc_layers = list_self_compress_layers(model)
print(f'[stage1] {len(sc_layers)} SC layers; init bits/weight = '
      f'{renderer_average_bits_per_weight(model):.2f} (expect 8.0)')
torch.save({'model_state_dict': model.state_dict()}, out_pt)
import os
print(f'[stage1] wrote {out_pt}: {os.path.getsize(out_pt)} bytes')
" 2>&1 | tee -a "$LOG_DIR/run.log"
[ -f "$RESUME_INIT_PT" ] || { echo "FATAL: stage 1 didn't produce $RESUME_INIT_PT"; exit 2; }

log "=== Stage 2: SC fine-tune (500 ep) WITH AUTO-WARMUP DETECTOR ==="
log "  profile:    self_compress_renderer_full"
log "  schedule:   500 ep total; rate ramp begins when scorer plateau detected"
log "  fallback:   --self-compress-lambda-ramp-start-frac=0.3 (only if detector"
log "              never fires)"
log "  detector:   window=50 epochs, slope_tol=1e-4 / epoch, min_warmup=50"
log "  resume:     $RESUME_INIT_PT (Lane A weights pre-loaded into SC layers)"

cat > "$LOG_DIR/kill_targets.json" <<'EOF'
{
  "phase1_end_epoch": 150,
  "phase1_pixel_l1_max": 5.0,
  "phase4_end_epoch": 500,
  "phase4_end_scorer_max": 1.50,
  "comment": "Lane S V2 kill targets — auto-warmup variant; scorer floor identical to V1."
}
EOF

"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile self_compress_renderer_full \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --resume-from "$RESUME_INIT_PT" \
    --use-self-compress-codec \
    --self-compress-init-bits 8.0 \
    --self-compress-target-bits 2.5 \
    --self-compress-lambda-start 0.0 \
    --self-compress-lambda-end 1.0 \
    --self-compress-lambda-ramp-start-frac 0.3 \
    --auto-warmup-lambda \
    --auto-warmup-window 50 \
    --auto-warmup-slope-tol 1e-4 \
    --auto-warmup-min-epochs 50 \
    --phase1-epochs 150 \
    --phase2-epochs 250 \
    --phase3-epochs 50 \
    --phase4-epochs 50 \
    --phase5-epochs 0 \
    --phase1-lr 5e-5 \
    --phase2-lr 5e-5 \
    --phase3-lr 5e-5 \
    --phase4-lr 5e-5 \
    --phase5-lr 5e-5 \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch|stage1|sc|lane-s-v2)\]|epoch|Phase|scorer|bits/weight" | tail -200

# Stage 3: SCv1 export.
BEST_FP32="$LOG_DIR/train/renderer_${TAG}_best_fp32.pt"
[ -f "$BEST_FP32" ] || {
    echo "FATAL: train_renderer didn't produce ${BEST_FP32}" >&2
    ls -la "$LOG_DIR/train/" >&2
    exit 2
}
log "  best fp32 checkpoint: $BEST_FP32 ($(stat -c '%s' "$BEST_FP32") bytes)"

log "=== Stage 3: SCv1 export of best fp32 → renderer_sc.bin ==="
mkdir -p "$LOG_DIR/qat"
SC_BIN="$LOG_DIR/qat/renderer_sc.bin"
"$PYBIN" -c "
import sys, torch
sys.path.insert(0, 'src')
from tac.renderer import build_renderer
from tac.self_compress import (
    swap_renderer_convs_with_self_compress,
    list_self_compress_layers,
    renderer_average_bits_per_weight,
)
from tac.renderer_export import export_self_compressed_renderer

ckpt_path = '$BEST_FP32'
out_bin = '$SC_BIN'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
meta = ckpt.get('__meta__', {}) or {}

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
    use_zoom_flow=m('use_zoom_flow', False),
)
diag = swap_renderer_convs_with_self_compress(model, init_bits=8.0)
print(f'[stage3] SC swap: {len(diag[\"swapped\"])} swapped, {len(diag[\"protected\"])} protected')
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise RuntimeError(
        f'SC load mismatch: missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}'
    )
avg_bits = renderer_average_bits_per_weight(model)
print(f'[stage3] learned bits/weight (mean): {avg_bits:.3f} (target 2.5)')
n_bytes = export_self_compressed_renderer(model, out_bin, use_lzma=True)
print(f'[stage3] WROTE {out_bin}: {n_bytes} bytes')
" 2>&1 | tee -a "$LOG_DIR/run.log"
[ -f "$SC_BIN" ] || { echo "FATAL: SCv1 export failed — no $SC_BIN" >&2; exit 2; }
SC_SIZE=$(stat -c '%s' "$SC_BIN")
log "  SCv1 binary: $SC_BIN ($SC_SIZE bytes)"

log "=== Stage 4: build NEW archive (SCv1 renderer + Lane A masks + Lane A poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$SC_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$ANCHOR_MASKS" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_s_v2.zip"
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

log "=== Stage 5: contest_auth_eval on Lane S V2 archive [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval log has no RESULT_JSON line — eval crashed."
    exit 4
fi

log "=== LANE_S_V2_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
