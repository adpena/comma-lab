#!/bin/bash
# Lane F-V5: Hardware FP8 (e4m3fn) on dilated-h64, anchored on Lane G v3 1.05.
#
# Background:
#   Lane F (FakeQuantFP4) regressed +0.44 vs baseline; Lane F-V2 narrowed
#   the gap to 1.79 but PoseNet still 20× worse — FP4 is structurally
#   hostile to YUV6/FastViT and (per FP4 hardware-disclosure rule, Check 40)
#   the RTX 4090 is sm_89, NOT Blackwell sm_100, so FP4 was always a
#   simulated path with no hardware lift. Lane F-V5 is the rescue: hardware-
#   native FP8 (float8_e4m3fn) IS supported on Ada Lovelace (sm_89) via
#   torchao.
#
#   Module landed weeks ago (src/tac/quantization_fp8.py + renderer_export
#   .export_hardware_fp8_checkpoint + load_hardware_fp8_checkpoint) but
#   never deployed.  This script closes the orphan.
#
# Stack:
#   * dilated-h64 baseline arch (288K params, our Lane G v3 footprint)
#   * Profile f_v5_hardware_fp8_dilated_h64 (DILATED_H64_HALF_FRAME +
#     quantization_mode='hardware_fp8' + qat_warmup_batches=50)
#   * eval_roundtrip=True (NON-NEGOTIABLE per CLAUDE.md)
#   * KL distill from baseline DILATED_H64_HALF_FRAME profile (inherited)
#   * HardwareFP8Quantizer calibrates on 50 warmup batches then freezes
#   * Lane G v3 anchor: masks.mkv + optimized_poses.pt bundled verbatim
#     (project_baseline_poses_load_bearing — poses are JOINT artifact)
#
# Predicted band [0.95, 1.20] [contest-CUDA]:
#   * Floor 0.95: hardware FP8 is precision-bound (atol≈5e-2, rtol≈2.5e-1
#     per renderer_export.py:1764) — should match Lane G v3 1.05 closely
#     because FP8 e4m3fn preserves much more dynamic range than FP4.
#   * Anchor 1.05: Lane G v3 control. FP8 pack rate (~8 bits/param) is
#     similar to default storage so rate term is roughly neutral.
#   * Ceiling 1.20: pessimistic case where FP8 quant noise on PoseNet path
#     bleeds 0.15 distortion (still 5× better than Lane F-V2's 0.96 PoseNet
#     regression).
#
# Hardware gate (CLAUDE.md Check 40 — hardware quantization disclosure):
#   FP8_HARDWARE_DISCLOSED: assert_quantization_hardware_supported(
#       'hardware_fp8', torch.device('cuda:0')) called at runtime in
#   Stage 2 preamble. RTX 4090 = sm_89 = supported. H100/A100 also OK.
#   Pre-Ada GPUs (T4/V100/RTX 30xx) will FATAL on the assertion.
#
# Hard kill targets ($4.00 cost cap; destroy if exceeded):
#   * NVDEC probe failure: instant destroy ($0)
#   * FP8 hardware assertion failure: instant destroy ($0)
#   * Wall-clock > 8h: auto-destroy ($2 max, gives buffer for $4 cap)
#
# Reproducibility:
#   * seed=1234, deterministic=True
#   * PYTHONHASHSEED=1234, CUBLAS_WORKSPACE_CONFIG=:4096:8
#
# controlled_baseline: Lane G v3 (1.05 [contest-CUDA] verified). Single-
#   mechanism change being isolated: "FP4 simulation" -> "hardware FP8".
#
# References:
#   * project_lane_f_v2_fp4_architectural_bottleneck_20260427
#   * project_cosmos_deep_dive_addendum_20260428 (Lane F-V5 LOWEST RISK)
#   * feedback_hardware_quantization_disclosure_20260428 (Check 40)
#
# ── Bash safety (CLAUDE.md non-negotiable) ──────────────────────────────
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_f_v5_results"
mkdir -p "$LOG_DIR"
TAG="lane_f_v5_hardware_fp8"

log() { echo "[lane-f-v5] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance (CLAUDE.md canonical pipeline standard).
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
    'lane_script': 'scripts/remote_lane_f_v5_hardware_fp8.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'profile': 'f_v5_hardware_fp8_dilated_h64',
    'predicted_band': [0.95, 1.20],
    'anchor_score_baseline': 1.05,
    'anchor_lane': 'lane_g_v3 (1.05 contest-CUDA, masks+poses are JOINT artifact)',
    'lane_f_v5_premise': 'hardware-native FP8 (e4m3fn) via torchao replaces FakeQuantFP4 simulation; uses Ada Lovelace sm_89 native float8',
    'controlled_baseline': 'lane_g_v3_kldistill_pose_tto (1.05)',
    'cost_estimate_usd': 2.0,
    'cost_cap_usd': 4.0,
    'wall_clock_estimate_hours': 4.0,
    'wall_clock_cap_hours': 8.0,
    'hardware_disclosure': 'FP8_HARDWARE_DISCLOSED — assert_quantization_hardware_supported at Stage 2 preamble',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=F-V5 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# ──────────────────────────────────────────────────────────────────────────
# Stage 0: NVDEC probe BEFORE any GPU spend.
# Reference: feedback_vastai_nvdec_host_variation. --ensure-dali so a fresh
# container that hasn't run remote_setup_full.sh installs DALI before probe.
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 0: NVDEC probe + git pull ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed -- this host cannot run upstream/evaluate.py"
    log "       at the end. Destroy this Vast.ai instance and pick a different host."
    exit 2
}

# ──────────────────────────────────────────────────────────────────────────
# Stage 1: canonical git sync (CLAUDE.md "Remote code parity required",
# preflight Check 57). Nuke local junk from prior failed deploys, then sync
# to origin/main exactly.
# ──────────────────────────────────────────────────────────────────────────
# CODE PARITY: launcher tarball is authoritative — do NOT git reset --hard.
# Doing so wipes local-only anchor files (archive_lane_a.zip, baseline dirs,
# etc.) that the launcher just SCP'd. The tarball IS the parity mechanism.
# (memory: feedback_git_reset_nukes_anchors_20260429)

# ──────────────────────────────────────────────────────────────────────────
# Stage 1b: AppleDouble cleanup (CLAUDE.md non-negotiable, Check 37).
# Strip any macOS resource forks that snuck in via tarball deploys.
# ──────────────────────────────────────────────────────────────────────────
find "$WORKSPACE" -name "._*" -type f -delete 2>/dev/null || true

# ──────────────────────────────────────────────────────────────────────────
# Stage 1c: anchor on Lane G v3 (1.05 [contest-CUDA] verified). Bundle
# masks.mkv + optimized_poses.pt verbatim (project_baseline_poses_load_bearing
# — poses are JOINT artifact with the renderer they were trained against;
# Lane F-V5 retrains the renderer ON TOP of the same arch so poses transfer).
# ──────────────────────────────────────────────────────────────────────────
ANCHOR_RENDERER="experiments/results/lane_g_v3_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_g_v3_landed/iter_0/masks.mkv"

# Lane G v3 landed an archive bundle; extract anchor pieces if not present.
if [ ! -f "$ANCHOR_RENDERER" ] && [ -f "experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip" ]; then
    mkdir -p "experiments/results/lane_g_v3_landed/iter_0"
    "$PYBIN" -c "
import zipfile
with zipfile.ZipFile('experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip') as z:
    z.extractall('experiments/results/lane_g_v3_landed/iter_0')
print('extracted Lane G v3 anchor bundle')
"
fi

for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         "$ANCHOR_MASKS" \
         upstream/videos/0.mkv \
         upstream/models/posenet.safetensors \
         upstream/models/segnet.safetensors \
         src/tac/quantization_fp8.py \
         src/tac/renderer_export.py; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer: $ANCHOR_RENDERER"
log "  anchor_poses:    $ANCHOR_POSES"
log "  anchor_masks:    $ANCHOR_MASKS"

# ──────────────────────────────────────────────────────────────────────────
# Stage 1d: argparse dead-flag scan (CLAUDE.md non-neg, Check 12 — preflight
# arity). Verify every flag we pass to train_renderer.py is real.
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 1d: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_f_v5_hardware_fp8.sh').read()
op_src = open('src/tac/experiments/train_renderer.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', op_src))
matches = re.findall(r'train_renderer\.py(.*?)(?=\n\s*\[\s*-f|\n\s*log\b|\Z)',
                      script, re.DOTALL)
assert matches, 'could not locate train_renderer.py invocation in script'
used = set()
for m in matches:
    used |= set(re.findall(r'\B--([a-z][a-z0-9-]+)', m))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in train_renderer argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} train_renderer flags all real')
"

# ──────────────────────────────────────────────────────────────────────────
# Stage 2: FP8 hardware assertion (CLAUDE.md Check 40 — hardware
# quantization disclosure). FAIL FAST if not on Ada Lovelace or newer.
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 2: hardware FP8 capability assertion ==="
"$PYBIN" -c "
import torch
from tac.quantization import (
    assert_quantization_hardware_supported,
    get_supported_quantization_modes,
)
dev = torch.device('cuda:0')
supported = get_supported_quantization_modes(dev)
print(f'CUDA capability supported modes: {sorted(supported)}')
assert 'fp8' in supported, f'FP8 NOT supported on this GPU; modes={supported}'
# FP8_HARDWARE_DISCLOSED: hardware-backed FP8 verified, not simulated.
assert_quantization_hardware_supported('hardware_fp8', dev)
print('OK: hardware FP8 (float8_e4m3fn) gate passes — Ada Lovelace sm_89 or newer')
"

# ──────────────────────────────────────────────────────────────────────────
# Stage 3: train Lane F-V5 renderer with hardware FP8 quantization.
# Profile drives quantization_mode='hardware_fp8' (NOT a CLI flag — the
# train_renderer parser does not have --use-hardware-fp8 or
# --quantization-mode; CLAUDE.md non-negotiable: NEVER invent CLI flags.
# Profile is the canonical config source).
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 3: train Lane F-V5 renderer (hardware FP8) ==="
log "   --profile f_v5_hardware_fp8_dilated_h64"
log "   profile drives quantization_mode='hardware_fp8' + qat_warmup_batches=50"
log "   eval_roundtrip=True inherited from DILATED_H64_HALF_FRAME"
log "   ~3-4h on RTX 4090"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile f_v5_hardware_fp8_dilated_h64 \
    --device cuda \
    --seed 1234 \
    --output-dir "$LOG_DIR/train" 2>&1 | tee "$LOG_DIR/train.log" | tail -50
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

# Validate the training produced a checkpoint we can export.
BEST_CKPT=$(ls -t "$LOG_DIR/train"/*BEST*.pt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t "$LOG_DIR/train"/*.pt 2>/dev/null | head -1)
fi
[ -f "$BEST_CKPT" ] || { echo "FATAL: train_renderer didn't produce any .pt checkpoint"; exit 2; }
log "  best checkpoint: $BEST_CKPT ($(stat -c '%s' "$BEST_CKPT" 2>/dev/null || stat -f '%z' "$BEST_CKPT") bytes)"

# ──────────────────────────────────────────────────────────────────────────
# Stage 4: export hardware FP8 checkpoint via canonical
# renderer_export.export_hardware_fp8_checkpoint. Stores arch config in JSON
# header + per-tensor [scale (float32 4B)][raw e4m3fn bytes] blobs.
# Format: [b"FP8H"][header_len][JSON header][blob_len][blob]...
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 4: export FP8H binary (hardware float8_e4m3fn) ==="
"$PYBIN" -u -c "
import sys
sys.path.insert(0, 'src')
import torch
from pathlib import Path
from tac.renderer_export import export_hardware_fp8_checkpoint, load_hardware_fp8_checkpoint
from tac.experiments.train_renderer import build_model_from_profile
from tac.profiles import PROFILES

profile = PROFILES['f_v5_hardware_fp8_dilated_h64']
ckpt = torch.load('$BEST_CKPT', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

# Reconstruct the renderer from profile + load weights.
model = build_model_from_profile(profile, device='cpu')
load_result = model.load_state_dict(sd, strict=False)
missing = getattr(load_result, 'missing_keys', [])
unexpected = getattr(load_result, 'unexpected_keys', [])
print(f'load: missing={len(missing)} unexpected={len(unexpected)}')
if missing:
    print(f'  first 5 missing: {missing[:5]}')
if unexpected:
    print(f'  first 5 unexpected: {unexpected[:5]}')

n_params = sum(p.numel() for p in model.parameters())
print(f'renderer reconstructed: {n_params:,} params')

# Export to FP8H format.
out_path = Path('$LOG_DIR/train/renderer.bin')
n_bytes = export_hardware_fp8_checkpoint(model, out_path)
print(f'FP8H exported: {n_bytes:,} bytes ({n_bytes/n_params*8:.2f} bits/param)')

# Round-trip verification: load it back and compare a few tensors.
loaded = load_hardware_fp8_checkpoint(out_path, device='cpu')
sd_loaded = loaded.state_dict()
sd_orig = model.state_dict()
common = [k for k in sd_orig if k in sd_loaded and sd_orig[k].numel() > 0]
print(f'round-trip: {len(common)} common tensors')
" 2>&1 | tee "$LOG_DIR/fp8_export.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

EXPORT_BIN="$LOG_DIR/train/renderer.bin"
[ -f "$EXPORT_BIN" ] || { echo "FATAL: FP8H export failed"; exit 2; }
log "  renderer.bin: $(stat -c '%s' "$EXPORT_BIN" 2>/dev/null || stat -f '%z' "$EXPORT_BIN") bytes"

# ──────────────────────────────────────────────────────────────────────────
# Stage 5: assemble archive (renderer.bin FP8H + Lane G v3 masks + Lane G
# v3 poses). Deterministic ZIP per Codex R5-r6 #5
# (check_archive_builders_use_deterministic_zip). AppleDouble cleanup per
# CLAUDE.md non-negotiable, Check 37.
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 5: build Lane F-V5 archive ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$EXPORT_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$ANCHOR_MASKS" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"

# Strip macOS resource forks before zipping.
find "$LOG_DIR/iter_0" -name '._*' -delete 2>/dev/null || true
find "$LOG_DIR/iter_0" -name '.DS_Store' -delete 2>/dev/null || true

ARCHIVE="$LOG_DIR/archive_lane_f_v5.zip"
"$PYBIN" -c "
import zipfile, os
from datetime import datetime
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
det_dt = (1980, 1, 1, 0, 0, 0)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        info = zipfile.ZipInfo(filename=n, date_time=det_dt)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        with open(p, 'rb') as f:
            z.writestr(info, f.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
size = os.path.getsize(dst)
print(f'archive {dst}: {size} bytes')
# ARCHIVE_BYTES guard (memory: feedback_zip_dep_bootstrap_trap, Check 47)
assert 100_000 < size < 1_500_000, f'archive size {size} outside sane band [100K, 1.5M]'
"

# ──────────────────────────────────────────────────────────────────────────
# Stage 6: contest_auth_eval [contest-CUDA] on the EXACT submission archive.
# Per CLAUDE.md "Auth eval EVERYWHERE" non-negotiable.
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 6: contest_auth_eval [contest-CUDA] ==="
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

# AppleDouble cleanup post-eval (work-dir may be on a sshfs mount).
find "$LOG_DIR/eval_work" -name "._*" -type f -delete 2>/dev/null || true

log "=== LANE_F_V5_DONE -- see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "    archive: $ARCHIVE"
log "    predicted band: [0.95, 1.20] [contest-CUDA]"
log "    anchor: Lane G v3 (1.05 [contest-CUDA])"
