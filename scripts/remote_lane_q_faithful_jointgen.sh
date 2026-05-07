#!/bin/bash
# Lane Q-FAITHFUL: TRUE 1:1 Quantizr PR #55 architecture replica.
#
# Background:
#   The Sherlock audit (.omx/research/quantizr_replica_audit_20260428.md)
#   proved that ALL prior Lane V / Lane K / "DSConv Quantizr-killer"
#   attempts kept the wrong architectural family — they retained our
#   warp-based AsymmetricPairGenerator + dual-mask + MotionPredictor.
#   PR #55 EXPLICITLY removes the motion module: "dropping optical flow
#   and using Feature-wise Linear Modulation on pose vectors instead of
#   using both masks. As a result the mask video only needs to encode
#   half as many frames."
#
#   Lane Q-FAITHFUL is the corrected rebuild. Architecture lives in
#   src/tac/quantizr_faithful_renderer.py (87,836 params, NO motion,
#   NO warp, single-mask + FiLM-on-pose dual-head). Profile:
#   q_faithful_dilated_88k. Variant dispatch in train_renderer.py
#   selects the new builder via variant="quantizr_faithful".
#
# Stack:
#   * 87,836 params (target 88K) — DSConv + GroupNorm + SiLU
#   * Single-mask trunk (SharedMaskDecoder c1=56 c2=64)
#   * Frame2StaticHead (UNCONDITIONAL frame2 decoder)
#   * FiLMFrameHead (FiLM-conditioned frame1, cond_dim=48)
#   * pose_mlp = Linear(6, 48) -> SiLU -> Linear(48, 48)
#   * KL distill T=2.0 weight=0.002 (Hinton 10% regime, post-fix math)
#   * eval_roundtrip=True (NON-NEGOTIABLE)
#   * 5-stage QAT pipeline (anchor / finetune / joint / QAT / final)
#   * Pose warm-start from Lane A's optimized_poses.pt (scorer-measured,
#     load-bearing per project_baseline_poses_load_bearing memory).
#     Lane A poses bundled into Q-FAITHFUL archive verbatim.
#
# Schedule (matches profile q_faithful_dilated_88k):
#   Phase 1 (600ep)   ~2.5h  pixel L1 + scorer warmup
#   Phase 2 (1500ep)  ~6h    + KL distill on SegNet logits
#   Phase 3 (400ep)   ~1.5h  hard-pair fine-tune
#   Phase 4 (400ep)   ~1.5h  QAT FakeQuantFP4
#   Phase 5 (100ep)   ~0.5h  final consolidation
#   TOTAL: ~12h on RTX 4090 ($3.00 @ $0.25/hr); + ~30min QFAI export +
#   ~15min auth-eval = $4-5 end-to-end.
#
# Predicted band [0.40, 0.80] [contest-CUDA]:
#   * Floor 0.40: Quantizr ships at 0.33; we use Lane A poses (load-bearing)
#     + DSConv + KL distill T=2.0 + 5-stage QAT (our advantage over his
#     vanilla quantization).
#   * Anchor 0.55: matches Quantizr 0.33 ± 0.5 lane variance.
#   * Ceiling 0.80: Lane A 1.15 minus rate gain. 88K renderer (~64KB FP4)
#     saves ~225KB vs Lane A's 290KB renderer; at 25 bits/byte rate factor
#     that's a 0.30 rate reduction. Even with PoseNet/SegNet matching
#     Lane A exactly, archive shrinks; ceiling is rate-floored.
#
# Hard kill targets ($8 cost cap; destroy if exceeded):
#   * Phase 1 end (~2.5h, ep600):  pixel L1 < 14
#   * Phase 2 end (~8h, ep2100):   scorer < 4.0  (must show learning signal)
#   * Phase 4 end (~11h, ep2900):  scorer < 1.5  (target standalone 0.40-0.80)
#   * If wall-clock exceeds 32h: auto-destroy (Vast.ai cost cap $8).
#
# Reproducibility:
#   * seed=1234, deterministic=True
#   * PYTHONHASHSEED=1234, CUBLAS_WORKSPACE_CONFIG=:4096:8
#
# controlled_baseline: Lane V (quantizr_replica_88k_halfframe). Lane V kept
#   the warp-based architecture; Q-FAITHFUL drops it. The single-mechanism
#   change being isolated is "motion module presence" (Lane V has it,
#   Q-FAITHFUL does not — per Quantizr's PR-#55 architectural removal).
#
# ── Bash safety (CLAUDE.md non-negotiable) ──────────────────────────────
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_q_faithful_results"
mkdir -p "$LOG_DIR"
TAG="lane_q_faithful_jointgen_88k"

log() { echo "[lane-q-faithful] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_q_faithful_jointgen.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'profile': 'q_faithful_dilated_88k',
    'predicted_band': [0.40, 0.80],
    'anchor_score_baseline': 1.15,
    'anchor_lane': 'lane_a (1.15 contest-CUDA, poses are load-bearing)',
    'lane_q_faithful_premise': 'TRUE 1:1 Quantizr PR-#55 — JointFrameGenerator, NO motion module, NO warp, single-mask + FiLM-on-pose dual-head',
    'controlled_baseline': 'lane_v_quantizr_replica_88k_halfframe (kept the wrong arch)',
    'cost_estimate_usd': 5.0,
    'cost_cap_usd': 8.0,
    'wall_clock_estimate_hours': 12.5,
    'wall_clock_cap_hours': 32.0,
    'param_count_target': 88000,
    'audit_reference': '.omx/research/quantizr_replica_audit_20260428.md',
    'pr_reference': 'https://github.com/commaai/comma_video_compression_challenge/pull/55',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=Q-FAITHFUL gpu=$GPU" >> "$HEARTBEAT"
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

# Code parity (CLAUDE.md "Remote code parity required"). Pull latest before
# starting work; if uncommitted local changes exist on remote, abort.
# CODE PARITY: launcher tarball is authoritative — do NOT git reset --hard.
# Doing so wipes local-only anchor files (archive_lane_a.zip, baseline dirs,
# etc.) that the launcher just SCP'd. The tarball IS the parity mechanism.
# (memory: feedback_git_reset_nukes_anchors_20260429)

# Pre-flight: required artifacts present. Anchor on Lane A poses (verified
# 1.15 [contest-CUDA] + scorer-measured + load-bearing).
ANCHOR_LANE_A_POSES="experiments/results/lane_a_landed/optimized_poses.pt"
for f in "$ANCHOR_LANE_A_POSES" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         src/tac/quantizr_faithful_renderer.py \
         src/tac/quantizr_faithful_export.py; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  all required artifacts present"

# Pre-flight: argparse dead-flag scan for the train_renderer invocation.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_q_faithful_jointgen.sh').read()
op_src = open('src/tac/experiments/train_renderer.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', op_src))
# Find every 'train_renderer.py' invocation in this script
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
# Stage 1: extract Lane G v3 anchor poses + masks (the load-bearing artifact)
# Lane Q-FAITHFUL builds a NEW renderer from scratch — but it ships Lane A's
# scorer-measured poses verbatim in the archive (per
# project_baseline_poses_load_bearing: poses are joint with the renderer they
# were trained against, but for ARCHIVE PACKAGING we just need the canonical
# 600 pose-6 vectors — Q-FAITHFUL's FiLM head learns the
# pose -> frame mapping during training).
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 1: extract / build masks + reuse Lane A poses ==="
"$PYBIN" experiments/build_baseline_archive.py \
    --device cuda --crf 50 --half-frame \
    --output "$LOG_DIR/archive_masks_seed.zip" 2>&1 | tee "$LOG_DIR/build_masks.log" | tail -5
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi
mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/archive_masks_seed.zip" 2>&1 | tail -3
cd "$WORKSPACE"
log "  masks.mkv extracted ($(stat -c '%s' "$LOG_DIR/extracted/masks.mkv" 2>/dev/null || stat -f '%z' "$LOG_DIR/extracted/masks.mkv") bytes)"
log "  Lane A poses to bundle: $ANCHOR_LANE_A_POSES"

# ──────────────────────────────────────────────────────────────────────────
# Stage 2: train Q-FAITHFUL JointFrameGenerator (auto-resume on preemption).
# Dispatch to variant="quantizr_faithful" via the q_faithful_dilated_88k
# profile. The 5-stage QAT happens inside train_renderer.py natively.
#
# AUTO-RESUME ENGINEERING (2026-05-07): preemption fix per
# feedback_q_faithful_4090_preemption_redeploy_h100_20260501. Vast/Modal can
# preempt mid-training (instance 35959478 was lost at ep 810/3000). Without
# resume the next dispatch starts from epoch 0 → 800 epochs of GPU spend wasted.
# Engineering fix: detect prior training_state_q_faithful_modal.pt under
# $LOG_DIR/train/ (or override via RESUME_FROM env) and pass --resume-from
# automatically. train_renderer.py already supports the flag — only the
# dispatch wrapper was missing it.
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 2: train Q-FAITHFUL JointFrameGenerator (5-stage QAT) ==="
log "   --profile q_faithful_dilated_88k (variant=quantizr_faithful)"
log "   --eval-roundtrip + --kl-distill-weight 0.002 + --kl-distill-temperature 2.0"
log "   total epochs ~3000; ~12h on RTX 4090"

# Auto-resume: env override > local-disk auto-detect > start fresh.
RESUME_ARGS=()
RESUME_FROM="${RESUME_FROM:-}"
if [ -z "$RESUME_FROM" ]; then
    # Auto-detect newest training_state checkpoint from prior preempted run.
    AUTO_RESUME=""
    if [ -f "$LOG_DIR/train/training_state_q_faithful_modal.pt" ]; then
        AUTO_RESUME="$LOG_DIR/train/training_state_q_faithful_modal.pt"
    fi
    if [ -n "$AUTO_RESUME" ]; then
        RESUME_FROM="$AUTO_RESUME"
        log "  AUTO-RESUME: detected $RESUME_FROM ($(stat -c '%s' "$RESUME_FROM" 2>/dev/null || stat -f '%z' "$RESUME_FROM") bytes)"
    fi
fi
if [ -n "$RESUME_FROM" ]; then
    [ -f "$RESUME_FROM" ] || { echo "FATAL: RESUME_FROM=$RESUME_FROM not a file" >&2; exit 3; }
    RESUME_ARGS=(--resume-from "$RESUME_FROM")
    log "  RESUMING from $RESUME_FROM (preempted run survives via auto-resume)"
else
    log "  NO resume checkpoint detected — starting fresh from epoch 0"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile q_faithful_dilated_88k \
    --device cuda \
    --seed 1234 \
    --tag q_faithful_modal \
    --qfaithful-training-poses "$ANCHOR_LANE_A_POSES" \
    --no-auth-eval-on-best \
    "${RESUME_ARGS[@]}" \
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
# Stage 3 + 4: export Q-FAITHFUL state_dict to QFAI binary, brotli-compress.
# QFAI format = [b"QFAI"][header_len][JSON header][torch.save(state_dict)].
# V1 keeps weights at FP32; future variant flips header.fp4_packed=True.
# Brotli q=11 is what Quantizr uses for his model.pt.br.
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 3+4: export QFAI binary + brotli compress ==="
"$PYBIN" -u -c "
import sys
sys.path.insert(0, 'src')
import torch, brotli, json, os
from pathlib import Path
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_faithful_export import save_qfai

# Load the training checkpoint. Training stores either {state_dict:...} or
# the raw state_dict — handle both. The training-side shim wraps the gen
# in _QuantizrFaithfulShim so we strip the 'gen.' prefix here.
ckpt = torch.load('$BEST_CKPT', map_location='cpu', weights_only=False)
sd_raw = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
# Strip the shim's 'gen.' prefix if present.
sd = {}
for k, v in sd_raw.items():
    if k.startswith('gen.'):
        sd[k[len('gen.'):]] = v
    else:
        sd[k] = v
# Reconstruct the bare JointFrameGenerator and load.
gen = build_quantizr_faithful_renderer()
gen.load_state_dict(sd, strict=True)
gen.eval()
n_params = sum(p.numel() for p in gen.parameters())
print(f'JointFrameGenerator loaded: {n_params:,} params')

# Save raw QFAI as renderer.bin because inflate_renderer.py dispatches QFAI/QZS3 by file magic.
training_pose_contract = None
for key in ('qfaithful_training_pose_contract', 'training_pose_contract'):
    value = ckpt.get(key)
    if isinstance(value, dict):
        training_pose_contract = value
        break
if training_pose_contract is None:
    meta = ckpt.get('__meta__') or ckpt.get('arch_meta') or {}
    if isinstance(meta, dict):
        for key in ('qfaithful_training_pose_contract', 'training_pose_contract'):
            value = meta.get(key)
            if isinstance(value, dict):
                training_pose_contract = value
                break
if not isinstance(training_pose_contract, dict) or training_pose_contract.get('training_pose_contract_promotable') is not True:
    raise SystemExit('FATAL: checkpoint missing promotable Q-FAITHFUL training_pose_contract')
qfai_path = Path('$LOG_DIR/train/renderer.bin')
n_bytes = save_qfai(gen, qfai_path, extra_meta={'training_pose_contract': training_pose_contract})
print(f'QFAI raw renderer.bin: {n_bytes:,} bytes')

# Brotli sidecar is for byte research only; deploy raw renderer.bin so magic dispatch works.
raw = qfai_path.read_bytes()
br = brotli.compress(raw, quality=11)
br_path = Path('$LOG_DIR/train/renderer.qfai.bin.br')
br_path.write_bytes(br)
print(f'QFAI brotli sidecar q=11: {len(br):,} bytes ({100*len(br)/len(raw):.1f}% of raw)')
"

EXPORT_BIN="$LOG_DIR/train/renderer.bin"
[ -f "$EXPORT_BIN" ] || { echo "FATAL: QFAI export failed"; exit 2; }
log "  renderer.bin: $(stat -c '%s' "$EXPORT_BIN" 2>/dev/null || stat -f '%z' "$EXPORT_BIN") bytes"

# ──────────────────────────────────────────────────────────────────────────
# Stage 5: assemble archive (renderer.bin + masks.mkv + Lane A poses).
# Determined ZIP dating per Codex R5-r6 #5 (check_archive_builders_use_deterministic_zip).
# AppleDouble cleanup per CLAUDE.md non-negotiable.
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 5: build Lane Q-FAITHFUL archive ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$EXPORT_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_LANE_A_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"

# Strip macOS resource forks before zipping.
find "$LOG_DIR/iter_0" -name '._*' -delete 2>/dev/null || true
find "$LOG_DIR/iter_0" -name '.DS_Store' -delete 2>/dev/null || true

ARCHIVE="$LOG_DIR/archive_lane_q_faithful.zip"
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
# ARCHIVE_BYTES guard (memory: feedback_zip_dep_bootstrap_trap)
assert 100_000 < size < 1_500_000, f'archive size {size} outside sane band [100K, 1.5M]'
"

# ──────────────────────────────────────────────────────────────────────────
# Stage 6: contest_auth_eval [contest-CUDA] on the EXACT submission archive.
# ──────────────────────────────────────────────────────────────────────────
log "=== Stage 6: contest_auth_eval [contest-CUDA] ==="
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

log "=== LANE_Q_FAITHFUL_DONE -- see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "    archive: $ARCHIVE"
log "    predicted band: [0.40, 0.80] [contest-CUDA]"
log "    audit reference: .omx/research/quantizr_replica_audit_20260428.md"
