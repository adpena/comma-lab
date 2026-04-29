#!/bin/bash
# Lane Ω-V3: explicit rate-distortion FRONTIER sweep over Lagrangian
# bits-per-weight targets. Replaces the V1/V2 single-target heuristic
# (`--target-bits 2.5`, equivalent to ~600,000 total bits for the
# 240K-weight ASYM renderer) with an `ε-constraint` parametric sweep
# over {1.25, 1.875, 2.5, 3.125, 3.75} bits/weight (≈ {300K, 450K,
# 600K, 750K, 900K} total bits). The OPERATOR picks the budget that
# minimises the contest score post-hoc.
#
# Lagrangian framing (Boyd & Vandenberghe §4.7.5): each sub-run solves
# the inner Lagrangian dual `min_θ D(θ) + λ·max(0, R(θ) - B_k)` with
# its own λ-anneal schedule. The OUTER loop is the parametric sweep
# over budget B_k, tracing the Pareto rate-distortion frontier — the
# canonical scalarisation of the multi-objective problem. V1/V2
# committed to a single B_k a priori; V3 derives B_k* from the
# measured frontier instead.
#
# Pipeline:
#   Stage 0 — NVDEC probe (5s sanity, catches bad-host before $)
#   Stage 1 — sweep_omega_rate_frontier.py: runs Lagrangian QAT at
#             each of 5 budgets, writes frontier.csv
#   Stage 2 — for EACH budget, build archive + auth_eval [contest-CUDA]
#   Stage 3 — pick best budget by lowest contest score, log it
#
# Predicted band: [0.55, 1.10] [contest-CUDA] — wider than V2's
# [0.65, 1.05] because we expect the frontier to expose a budget BELOW
# 2.5 bpw that beats the hand-picked target (council Quantizr argues
# 1.875 bpw is the inflection point; Yousfi argues 3.125 to keep
# PoseNet stable).
#
# Council preflight (CLAUDE.md non-negotiable: NEVER invent CLI flags).
# Verified 2026-04-27 against argparse:
#   * sweep_omega_rate_frontier.py: --checkpoint --video --masks-mkv
#     --poses --upstream --output-dir --target-bits-per-weight
#     --total-epochs --lr --bits-lr-scale --noise-std --seg-weight
#     --pose-weight --lambda-start --lambda-end --lambda-ramp-start-frac
#     --init-bits --device --seed --log-every --python --dry-run
#   * qat_omega_lagrangian.py (used internally): see Lane Ω-V2 script.
#   * contest_auth_eval.py: --archive --inflate-sh --upstream-dir
#     --device --keep-work-dir --work-dir
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_omega_v3_results"
mkdir -p "$LOG_DIR"
TAG="lane_omega_v3_rate_frontier"

log() { echo "[lane-omega-v3] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

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
    'lane_script': 'scripts/remote_lane_omega_v3_rate_frontier.sh',
    'lane_name': 'lane_omega_v3_rate_frontier_on_lane_a',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/iter_0/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/iter_0/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [0.55, 1.10],
    'rationale': 'Replace V2 single-target hand-picked B_k=600K with a 5-budget rate-distortion frontier sweep (epsilon-constraint scalarisation, Boyd & Vandenberghe §4.7.5). The operator picks the B_k that minimises the contest score from the measured frontier instead of pre-committing.',
    'budgets_bits_per_weight': [1.25, 1.875, 2.5, 3.125, 3.75],
    'lambda_start': 0.0,
    'lambda_end': 1.0,
    'lambda_ramp_start_frac': 0.3,
    'init_bits': 8.0,
    'total_epochs_per_subrun': 200,
    'lr': 2.5e-6,
    'bits_lr_scale': 0.1,
    'noise_std': 0.5,
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'lagrangian_target': 'epsilon-constraint frontier sweep',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=Omega-V3 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0 — NVDEC probe BEFORE any GPU spend.
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

# Pre-flight: dead-flag scan for the orchestrator AND the embedded QAT
# invocation. CLAUDE.md non-negotiable: NEVER invent CLI flags.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_omega_v3_rate_frontier.sh').read()
sweep_src = open('experiments/sweep_omega_rate_frontier.py').read()
qat_src = open('experiments/qat_omega_lagrangian.py').read()
sweep_real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', sweep_src))
qat_real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', qat_src))
m = re.search(r'experiments/sweep_omega_rate_frontier\.py(.*?)(?=\n# Stage|\nlog \"===|\Z)',
              script, re.DOTALL)
assert m, 'could not locate sweep_omega_rate_frontier.py invocation'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - sweep_real
if invented:
    print(f'INVENTED FLAGS (sweep): {sorted(invented)}', file=sys.stderr); sys.exit(3)
print(f'OK: sweep={len(used)} flags all real (qat real_count={len(qat_real)})')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Stage 1 — sweep over budgets.
log "=== Stage 1: rate-distortion frontier sweep (5 budgets) ==="
"$PYBIN" -u experiments/sweep_omega_rate_frontier.py \
    --checkpoint "$ANCHOR_RENDERER" \
    --video upstream/videos/0.mkv \
    --masks-mkv "$ANCHOR_MASKS" \
    --poses "$ANCHOR_POSES" \
    --upstream upstream \
    --output-dir "$LOG_DIR/sweep" \
    --target-bits-per-weight "1.25,1.875,2.5,3.125,3.75" \
    --total-epochs 200 \
    --lr 2.5e-6 \
    --bits-lr-scale 0.1 \
    --noise-std 0.5 \
    --seg-weight 100.0 \
    --pose-weight 10.0 \
    --lambda-start 0.0 \
    --lambda-end 1.0 \
    --lambda-ramp-start-frac 0.3 \
    --init-bits 8.0 \
    --device cuda \
    --seed 1234 \
    --log-every 10 2>&1 | tee "$LOG_DIR/sweep.log" | tail -40

[ -f "$LOG_DIR/sweep/frontier.csv" ] || { echo "FATAL: sweep didn't produce frontier.csv"; exit 2; }
log "  frontier.csv at $LOG_DIR/sweep/frontier.csv"

# Stage 2 — for each budget that produced a renderer.bin, build archive
# + auth_eval [contest-CUDA]. Skip budgets whose sub-run failed.
log "=== Stage 2: per-budget contest_auth_eval [contest-CUDA] ==="
SCORES_CSV="$LOG_DIR/scores.csv"
echo "target_bits_per_weight,renderer_bytes,archive_bytes,auth_score" > "$SCORES_CSV"
for SUBDIR in "$LOG_DIR"/sweep/budget_bpw_*/; do
    BPW=$(basename "$SUBDIR" | sed 's/^budget_bpw_//')
    OMEGA_BIN="$SUBDIR/renderer.bin"
    if [ ! -f "$OMEGA_BIN" ]; then
        log "  SKIP bpw=$BPW (no renderer.bin)"
        echo "$BPW,,,SKIP_NO_RENDERER" >> "$SCORES_CSV"
        continue
    fi
    OMEGA_SIZE=$(stat -c '%s' "$OMEGA_BIN")
    mkdir -p "$SUBDIR/iter_0"
    cp "$OMEGA_BIN" "$SUBDIR/iter_0/renderer.bin"
    cp "$ANCHOR_MASKS" "$SUBDIR/iter_0/masks.mkv"
    cp "$ANCHOR_POSES" "$SUBDIR/iter_0/optimized_poses.pt"
    ARCHIVE="$SUBDIR/archive.zip"
    "$PYBIN" -c "
import zipfile, os
src = '$SUBDIR/iter_0'
dst = '$ARCHIVE'
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        z.write(p, arcname=n)
print(f'archive {dst}: {os.path.getsize(dst)} bytes')
"
    ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")
    rm -rf "$SUBDIR/eval_work"
    "$PYBIN" -u experiments/contest_auth_eval.py \
        --archive "$ARCHIVE" \
        --inflate-sh submissions/robust_current/inflate.sh \
        --upstream-dir upstream \
        --device "${AUTH_EVAL_DEVICE:-cuda}" \
        --keep-work-dir \
        --work-dir "$SUBDIR/eval_work" 2>&1 | tee "$SUBDIR/auth_eval.log" | tail -15
    if ! grep -q "RESULT_JSON" "$SUBDIR/auth_eval.log"; then
        log "  bpw=$BPW: auth_eval crashed (no RESULT_JSON)"
        echo "$BPW,$OMEGA_SIZE,$ARCHIVE_BYTES,EVAL_CRASH" >> "$SCORES_CSV"
        continue
    fi
    # codex Round 2 finding #3 fix: auth_eval emits 'final_score', not
    # 'score'/'total_score'. Try canonical key first; fail-loud if missing.
    SCORE=$(grep "RESULT_JSON" "$SUBDIR/auth_eval.log" | tail -1 | "$PYBIN" -c "
import sys, json, re
line = sys.stdin.read()
m = re.search(r'RESULT_JSON\s*[:=]?\s*({.*})', line)
if not m:
    print('PARSE_FAIL'); sys.exit(0)
try:
    d = json.loads(m.group(1))
    # Canonical contest_auth_eval key is 'final_score'.
    score = d.get('final_score') or d.get('score') or d.get('total_score')
    if score is None:
        print(f'NO_SCORE_KEY({list(d.keys())[:5]})')
    else:
        print(score)
except Exception as e:
    print(f'PARSE_FAIL_{e}')
" 2>/dev/null || echo "PARSE_ERR")
    log "  bpw=$BPW: renderer=$OMEGA_SIZE archive=$ARCHIVE_BYTES auth=$SCORE [contest-CUDA]"
    echo "$BPW,$OMEGA_SIZE,$ARCHIVE_BYTES,$SCORE" >> "$SCORES_CSV"
done

# Stage 3 — pick best budget.
log "=== Stage 3: best-budget summary ==="
"$PYBIN" -c "
import csv, sys
with open('$SCORES_CSV') as f:
    rows = list(csv.DictReader(f))
print('Frontier (target_bits_per_weight | archive_bytes | auth_score):')
best = None
for r in rows:
    sc = r.get('auth_score', '')
    print(f\"  bpw={r['target_bits_per_weight']:>6} archive={r.get('archive_bytes',''):>8} auth={sc}\")
    try:
        s = float(sc)
        if best is None or s < best[0]:
            best = (s, r)
    except (ValueError, TypeError):
        continue
if best:
    print(f\"BEST_BUDGET bpw={best[1]['target_bits_per_weight']} auth={best[0]} archive={best[1].get('archive_bytes','')}\")
else:
    print('BEST_BUDGET: no successful sub-run')
"

log "=== LANE_OMEGA_V3_DONE [contest-CUDA] -- see $SCORES_CSV ==="
