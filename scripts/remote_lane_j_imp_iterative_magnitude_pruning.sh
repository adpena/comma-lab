#!/bin/bash
# Lane J-IMP — 10-cycle Iterative Magnitude Pruning (Frankle-Carbin LTH +
# Frankle-2019 weight-rewinding-to-early-epoch stabilization).
#
# DEPLOY-WORTHY (was RESEARCH-PARK at $25 budget): with the post-2026-04-28
# new $200-500 budget the full 10-cycle IMP at $25 / 60h is now in budget.
# Jack-from-skunkworks ranked Lane J-IMP TOP-3 highest-EV
# ($25 / 60h, predicted_band [0.85, 1.00]).
#
# MECHANISM (per arXiv 1803.03635 + 1912.05671 + 2406.01820):
#   for cycle in 0..9:
#     - Train current sparse network to convergence
#     - Globally prune lowest-magnitude 20% of SURVIVING conv weights
#     - Rewind survivors to a snapshot taken at ~1% through cycle 0
#       (NOT to init — Frankle 2019 stabilization fix; init-rewinding
#       fails at scale, early-epoch-rewinding succeeds at ResNet-50).
#   Final cumulative sparsity = 1 - 0.8^10 = 89.3%.
#
# COMPOSITION TARGET: pair with Lane Ω-V2 per-element 4-bit quantization on
# the ~9,400 surviving weights → renderer.bin shrinks to ~5KB (vs 170KB
# Lane G v3 baseline) → rate term drops by ~0.10 score points.
#
# SPARSE-CSR BREAKEVEN (from src/tac/iterative_magnitude_pruning.py):
#   dense FP4: 88K × 4 / 8 = 44KB
#   sparse-CSR: nnz × 2.5B (uint16 idx + FP4 val)
#   beats dense iff nnz < 17_600 ⇔ sparsity > 80.0%
#   At our 89% target: nnz ≈ 9_400 → sparse-CSR ≈ 23.5KB (saves ~21KB).
#
# COST CAP: $25 hard cap. With $200-500 budget headroom this is acceptable
# (still <12% of low-band budget). DESTROY THE INSTANCE when LANE_J_IMP_DONE
# fires to avoid burn — `vastai destroy instance $INSTANCE_ID`.
#
# SAFETY: cycle 0 fine-tune is the most expensive (~10h on RTX 4090);
# cycles 1-9 are ~5h each. Smoke-test cycles 0+1 locally with --epochs 4
# before launching the full 10-cycle remote run.
#
# CRITICAL (CLAUDE.md non-negotiable): the FINAL stage runs CUDA
# auth_eval on the 90%-sparse renderer wrapped in a contest-compliant
# archive. No score is reported without [contest-CUDA] tag.

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="$WORKSPACE/lane_j_imp_results"
mkdir -p "$LOG_DIR"

log() { echo "[lane-j-imp] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# AppleDouble cleanup (memory: feedback_remote_setup_script_correct_path —
# macOS-tarred bundles can leak ._* files into the workspace and break
# Python imports under torch.load).
find "$WORKSPACE" -name "._*" -type f -delete 2>/dev/null || true

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard).
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
    'lane_script': 'scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh',
    'lane_name': 'lane_j_imp_iterative_magnitude_pruning',
    'paper_refs': ['arXiv:1803.03635', 'arXiv:1912.05671', 'arXiv:2406.01820'],
    'mechanism': '10-cycle IMP @ 20%/cycle, rewind-to-early-epoch (Frankle 2019)',
    'final_sparsity_target': 0.893,
    'predicted_band': [0.85, 1.00],
    'anchor_score_baseline': 1.05,
    'anchor_renderer': 'experiments/results/lane_g_v3_landed/iter_0/renderer.bin',
    'anchor_archive': 'experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip',
    'cost_cap_usd': 25,
    'output_dir': '$LOG_DIR',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=J-IMP gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend.
log "=== Stage 0: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed — destroy this Vast.ai instance."
    exit 2
}

# Stage 0b: canonical git sync (CLAUDE.md non-negotiable: deployed code parity).
log "=== Stage 0b: canonical git sync (fetch + reset --hard origin/main) ==="
# Nuke local junk from prior failed deploys, then sync to origin/main exactly.
git -C "$WORKSPACE" fetch origin main && git -C "$WORKSPACE" reset --hard origin/main 2>&1 | tail -3 || {
    log "WARN: git fetch/reset failed — continuing with current HEAD ($GIT_HASH)"
}

# Pre-flight: anchor on Lane G v3 (1.05 [contest-CUDA]).
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
         "$ANCHOR_MASKS" \
         upstream/videos/0.mkv \
         upstream/models/posenet.safetensors \
         upstream/models/segnet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer: $ANCHOR_RENDERER"
log "  anchor_masks:    $ANCHOR_MASKS"

# Pre-flight: dead-flag scan for train_imp_cycle.py (CLAUDE.md
# non-negotiable: NEVER invent CLI flags).
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh').read()
op_src = open('experiments/train_imp_cycle.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', op_src))
m = re.search(r'experiments/train_imp_cycle\.py(.*?)(?=\n\s*\[\s*-f|\n\s*log\b|\Z)',
              script, re.DOTALL)
assert m, 'could not locate train_imp_cycle.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in train_imp_cycle argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
"

# 10-cycle IMP loop. Each cycle's output feeds the next cycle's input.
# Cycle 0 starts from the Lane G v3 anchor renderer; cycle N reads
# cycle N-1's renderer.pt + mask.pt + early_epoch_snapshot.pt.
TARGET_SPARSITY_PER_CYCLE=0.20
FINAL_TARGET=0.893
EPOCHS_PER_CYCLE=200

# Council Lane-17 design 2026-04-30 (Q3 7/10 vote, Q4 9/10 vote):
#   Q3: per-cycle CUDA auth eval at cycles 0, 2, 4, 6, 8, 9 (6 evals = $1.80).
#   Q4: revert-on-regression — kill if cycle_N_score > 1.10 × min(cycle_0..N-1).
# Variables defined here drive the in-loop hooks. To run a quick variant
# (5-cycle), pass IMP_QUICK_VARIANT=1 to the launcher.
IMP_AUTH_EVAL_CYCLES=${IMP_AUTH_EVAL_CYCLES:-"0 2 4 6 8 9"}
IMP_REGRESSION_THRESHOLD=${IMP_REGRESSION_THRESHOLD:-1.10}
# Council Round 1 M1 fix (2026-04-30): pre-populate BEST_CYCLE_SCORE with the
# Lane G v3 anchor's known [contest-CUDA] score (1.05). This anchors the
# regression check to the DENSE baseline, not whichever cycle's smoke happens
# to land first (failure mode: cycle 0 auth eval crashes → cycle 2 becomes
# baseline → 36% sparse network masks the dense-vs-sparse regression).
BEST_CYCLE_SCORE="${IMP_BASELINE_SCORE:-1.05}"
BEST_CYCLE_IDX="lane_g_v3_anchor"
CYCLE_SCORE_FLOOR="$BEST_CYCLE_SCORE"  # Council kill-criterion sentinel (Check 94 token)

log "=== Stage 1: 10-cycle IMP (target_per_cycle=$TARGET_SPARSITY_PER_CYCLE, final=$FINAL_TARGET) ==="
log "  Council kill-criterion: revert-on-regression at threshold ×${IMP_REGRESSION_THRESHOLD}"
log "  Auth-eval-on-cycles: $IMP_AUTH_EVAL_CYCLES"
log "  cumulative sparsity schedule:"
log "    cycle 0 → 0.200    cycle 5 → 0.738"
log "    cycle 1 → 0.360    cycle 6 → 0.790"
log "    cycle 2 → 0.488    cycle 7 → 0.832"
log "    cycle 3 → 0.590    cycle 8 → 0.866"
log "    cycle 4 → 0.672    cycle 9 → 0.893"

PREV_RENDER=""
PREV_MASK=""
PREV_SNAPSHOT=""

for i in 0 1 2 3 4 5 6 7 8 9; do
    CYC_DIR="$LOG_DIR/cycle_${i}"
    mkdir -p "$CYC_DIR"
    log "--- Cycle $i → $CYC_DIR ---"
    if [ "$i" -eq 0 ]; then
        # Cycle 0: input = Lane G v3 anchor renderer.bin (no mask, no snapshot yet)
        "$PYBIN" -u experiments/train_imp_cycle.py \
            --cycle 0 \
            --checkpoint "$ANCHOR_RENDERER" \
            --output-dir "$CYC_DIR" \
            --target-sparsity "$TARGET_SPARSITY_PER_CYCLE" \
            --final-sparsity-target "$FINAL_TARGET" \
            --profile imp_cycle_dilated_h64 \
            --base-ch 36 --mid-ch 60 --motion-hidden 32 \
            --depth 1 --embed-dim 6 --pose-dim 6 \
            --padding-mode zeros \
            --epochs "$EPOCHS_PER_CYCLE" \
            --lr 1e-4 --batch-size 4 \
            --device cuda 2>&1 | tee "$CYC_DIR/cycle.log" | tail -10
    else
        "$PYBIN" -u experiments/train_imp_cycle.py \
            --cycle "$i" \
            --checkpoint "$PREV_RENDER" \
            --mask-from "$PREV_MASK" \
            --early-epoch-weights "$PREV_SNAPSHOT" \
            --output-dir "$CYC_DIR" \
            --target-sparsity "$TARGET_SPARSITY_PER_CYCLE" \
            --final-sparsity-target "$FINAL_TARGET" \
            --profile imp_cycle_dilated_h64 \
            --base-ch 36 --mid-ch 60 --motion-hidden 32 \
            --depth 1 --embed-dim 6 --pose-dim 6 \
            --padding-mode zeros \
            --epochs "$EPOCHS_PER_CYCLE" \
            --lr 1e-4 --batch-size 4 \
            --device cuda 2>&1 | tee "$CYC_DIR/cycle.log" | tail -10
    fi

    # Validate the cycle produced expected artifacts.
    for art in renderer.pt mask.pt early_epoch_snapshot.pt stats.json; do
        [ -f "$CYC_DIR/$art" ] || {
            log "FATAL: cycle $i missing $art"
            exit 5
        }
    done

    SPARSITY=$("$PYBIN" -c "import json; print(json.load(open('$CYC_DIR/stats.json'))['sparsity_after_rewind'])")
    log "  cycle $i complete: sparsity=$SPARSITY"

    # Council Q3+Q4 (2026-04-30): per-cycle CUDA auth eval at scheduled
    # cycles + revert-on-regression. The auth eval runs on a contest archive
    # built by re-exporting the cycle's renderer.pt to FP4A and packaging
    # alongside the Lane G v3 anchor masks/poses. All cost: ~$0.30/eval ×
    # 6 evals = $1.80 — cheap insurance against burning $25 on a regressing
    # 10-cycle run.
    if echo " $IMP_AUTH_EVAL_CYCLES " | grep -q " $i "; then
        log "  Stage 1.5: per-cycle CUDA auth eval (cycle $i)"
        SMOKE_DIR="$CYC_DIR/auth_smoke"
        mkdir -p "$SMOKE_DIR/iter_0"
        # Re-export FP4A bytes for this cycle's renderer.pt → smoke archive.
        "$PYBIN" -c "
import sys, torch
sys.path.insert(0, 'src'); sys.path.insert(0, 'upstream')
from tac.renderer import build_renderer
from tac.renderer_export import export_asymmetric_checkpoint_fp4
ckpt = torch.load('$CYC_DIR/renderer.pt', map_location='cpu', weights_only=False)
state = ckpt.get('model_state_dict', ckpt)
m = build_renderer(num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
                   motion_hidden=32, depth=1, pose_dim=6,
                   use_zoom_flow=False, padding_mode='zeros')
m.load_state_dict(state, strict=False); m.eval()
with open('$SMOKE_DIR/iter_0/renderer.bin', 'wb') as f:
    f.write(export_asymmetric_checkpoint_fp4(m))
" 2>&1 | tee -a "$CYC_DIR/cycle.log" | tail -3
        cp "$ANCHOR_MASKS" "$SMOKE_DIR/iter_0/masks.mkv"
        [ -f "$ANCHOR_POSES" ] && cp "$ANCHOR_POSES" "$SMOKE_DIR/iter_0/optimized_poses.pt" || true
        SMOKE_ARCHIVE="$SMOKE_DIR/archive_cycle_${i}.zip"
        "$PYBIN" -c "
import zipfile, os
src = '$SMOKE_DIR/iter_0'
with zipfile.ZipFile('$SMOKE_ARCHIVE', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        if os.path.isfile(p):
            z.write(p, arcname=n)
"
        # Run auth eval. Don't abort the dispatcher on a single eval crash —
        # log and continue (Council resilience principle).
        "$PYBIN" -u experiments/contest_auth_eval.py \
            --archive "$SMOKE_ARCHIVE" \
            --inflate-sh submissions/robust_current/inflate.sh \
            --upstream-dir upstream \
            --device cuda \
            --keep-work-dir \
            --work-dir "$SMOKE_DIR/eval_work" \
            2>&1 | tee "$SMOKE_DIR/auth_eval.log" | tail -8 || \
            log "  WARN: cycle $i auth eval crashed; continuing (no revert decision possible)"

        if grep -q "RESULT_JSON" "$SMOKE_DIR/auth_eval.log"; then
            CYC_SCORE=$("$PYBIN" -c "
import json, re
log = open('$SMOKE_DIR/auth_eval.log').read()
m = re.search(r'RESULT_JSON\s*[:=]\s*(\{[^}]*\})', log)
if not m:
    print('NaN'); raise SystemExit(0)
try:
    j = json.loads(m.group(1))
    print(j.get('score', j.get('total', 'NaN')))
except Exception:
    print('NaN')
")
            log "  cycle $i [contest-CUDA] score = $CYC_SCORE"
            # Revert-on-regression — Council Q4 9/10 (the cycle_score_floor
            # token is the Check 94 STRICT detection sentinel).
            # BEST_CYCLE_SCORE is pre-populated to the Lane G v3 anchor
            # baseline (1.05) per Round 1 M1 fix, so the regression check
            # is always against the canonical [contest-CUDA] anchor.
            IS_BETTER=$("$PYBIN" -c "
try:
    a, b = float('$CYC_SCORE'), float('$BEST_CYCLE_SCORE')
    print('1' if a < b else '0')
except: print('0')
")
            IS_REGRESSION=$("$PYBIN" -c "
try:
    a, b, t = float('$CYC_SCORE'), float('$BEST_CYCLE_SCORE'), float('$IMP_REGRESSION_THRESHOLD')
    print('1' if a > b * t else '0')
except: print('0')
")
            if [ "$IS_BETTER" = "1" ]; then
                BEST_CYCLE_SCORE="$CYC_SCORE"
                BEST_CYCLE_IDX="$i"
                CYCLE_SCORE_FLOOR="$CYC_SCORE"
                log "    NEW BEST: BEST_CYCLE_SCORE=$BEST_CYCLE_SCORE @ cycle $i"
            fi
            if [ "$IS_REGRESSION" = "1" ]; then
                log "  REVERT_ON_REGRESSION: cycle $i score $CYC_SCORE > "
                log "    threshold ($IMP_REGRESSION_THRESHOLD × $BEST_CYCLE_SCORE = "
                log "    $("$PYBIN" -c "print(float('$BEST_CYCLE_SCORE') * float('$IMP_REGRESSION_THRESHOLD'))"))."
                log "    REVERT to cycle $BEST_CYCLE_IDX as the lane's "
                log "    final result and STOP per Council Q4."
                # Mark the dispatcher's final-cycle pointer to the BEST.
                echo "$BEST_CYCLE_IDX" > "$LOG_DIR/REVERT_TO_CYCLE.txt"
                break
            fi
        else
            log "  WARN: cycle $i auth eval log missing RESULT_JSON; "
            log "    cannot decide revert. Continuing — final Stage 4 will catch."
        fi
    fi

    PREV_RENDER="$CYC_DIR/renderer.pt"
    PREV_MASK="$CYC_DIR/mask.pt"
    PREV_SNAPSHOT="$CYC_DIR/early_epoch_snapshot.pt"
done

# If revert-on-regression fired, swap the Stage-2 final pointer to the
# best cycle. Otherwise it's the last cycle (9 in normal flow).
# Special case (Round 1 M1 / Round 2 followup): BEST_CYCLE_IDX may be
# "lane_g_v3_anchor" if EVERY cycle regressed past the threshold from the
# DENSE baseline. In that case, NO IMP cycle survives — the lane's final
# result IS the Lane G v3 anchor (sparse-CSR shipped no benefit). The
# dispatcher emits a special "ALL_CYCLES_REGRESSED" exit code (8) so the
# launcher / harvester can distinguish this from a normal "best-of-the-
# pruned-cycles" revert.
if [ -f "$LOG_DIR/REVERT_TO_CYCLE.txt" ]; then
    REVERT_IDX=$(cat "$LOG_DIR/REVERT_TO_CYCLE.txt")
    log "=== REVERTING to $REVERT_IDX (Council Q4 kill-criterion fired) ==="
    if [ "$REVERT_IDX" = "lane_g_v3_anchor" ]; then
        log "ALL_CYCLES_REGRESSED: every IMP cycle regressed >${IMP_REGRESSION_THRESHOLD}x "
        log "  from Lane G v3 anchor (1.05). Lane 17 ships NO new artifact; "
        log "  user should re-evaluate the LTH-at-88K hypothesis."
        echo "ALL_CYCLES_REGRESSED" > "$LOG_DIR/LANE_17_VERDICT.txt"
        exit 8
    fi
    FINAL_CYCLE_DIR="$LOG_DIR/cycle_${REVERT_IDX}"
else
    FINAL_CYCLE_DIR="$LOG_DIR/cycle_9"
fi

FINAL_RENDERER_PT="$FINAL_CYCLE_DIR/renderer.pt"
FINAL_MASK="$FINAL_CYCLE_DIR/mask.pt"
log "=== Stage 2: re-export final renderer to ASYM .bin (Lane G v3 mask + poses) ==="
log "    sourcing from $FINAL_CYCLE_DIR (BEST cycle per kill-criterion gate)"
# The IMP runs save .pt; the contest archive requires renderer.bin.
# We re-export from the final FP32 weights so the on-disk bytes match
# the inflate-time format. Sparse-CSR export is OPTIONAL post-stage and
# only beats dense FP4 above ~80% sparsity (we hit ~89%, so it pays).
"$PYBIN" -c "
import sys, torch, json
sys.path.insert(0, 'src'); sys.path.insert(0, 'upstream')
from tac.renderer import build_renderer
from tac.renderer_export import export_asymmetric_checkpoint, export_asymmetric_checkpoint_fp4
from tac.iterative_magnitude_pruning import compute_actual_sparsity
ckpt = torch.load('$FINAL_RENDERER_PT', map_location='cpu', weights_only=False)
state = ckpt.get('model_state_dict', ckpt)
mask = torch.load('$FINAL_MASK', map_location='cpu', weights_only=False)
m = mask.get('mask', mask) if isinstance(mask, dict) else mask
model = build_renderer(num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
                       motion_hidden=32, depth=1, pose_dim=6,
                       use_zoom_flow=False, padding_mode='zeros')
model.load_state_dict(state, strict=False)
model.eval()
asym_bytes = export_asymmetric_checkpoint(model)
fp4_bytes = export_asymmetric_checkpoint_fp4(model)
with open('$LOG_DIR/renderer.bin', 'wb') as f:
    f.write(fp4_bytes)
sparsity = compute_actual_sparsity(model)
stats = {
    'asym_fp32_bytes': len(asym_bytes),
    'fp4_bytes': len(fp4_bytes),
    'final_sparsity': sparsity,
    'theoretical_sparse_csr_bytes': int((sum(p.numel() for n, p in model.named_parameters()
                                              if 'conv.weight' in n or n.endswith('.weight') and p.ndim == 4)
                                          * (1 - sparsity) * 2.5)),
}
print('export stats:', json.dumps(stats, indent=2))
with open('$LOG_DIR/export_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
" 2>&1 | tee "$LOG_DIR/export.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

[ -f "$LOG_DIR/renderer.bin" ] || { log "FATAL: re-export didn't produce renderer.bin"; exit 6; }
RENDERER_SIZE=$(stat -c '%s' "$LOG_DIR/renderer.bin")
log "  Lane J-IMP renderer.bin = $RENDERER_SIZE bytes"

log "=== Stage 3: build contest archive (IMP renderer + Lane G v3 masks + poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$LOG_DIR/renderer.bin" "$LOG_DIR/iter_0/renderer.bin"
cp "$ANCHOR_MASKS" "$LOG_DIR/iter_0/masks.mkv"
[ -f "$ANCHOR_POSES" ] && cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt" || \
    log "  no anchor poses — archive will rely on inflate.sh defaults"
ARCHIVE="$LOG_DIR/archive_lane_j_imp.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        if os.path.isfile(p):
            z.write(p, arcname=n)
        else:
            print(f'skip absent {p}')
print(f'archive {dst}: {os.path.getsize(dst)} bytes')
"

log "=== Stage 3b: archive-size assertion (Lane B-class disaster guard) ==="
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo 'FATAL: archive empty'; exit 2; }
log "  archive bytes = $ARCHIVE_BYTES (must include renderer + masks + poses)"

log "=== Stage 4: contest_auth_eval [contest-CUDA] on Lane J-IMP archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -15
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval log has no RESULT_JSON — eval crashed."
    exit 7
fi

log "=== LANE_J_IMP_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "Cost reminder: destroy the Vast.ai instance now to stop \$0.25/hr accrual."
