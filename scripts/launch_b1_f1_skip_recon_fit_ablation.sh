#!/bin/bash
# F1 recon-fit CAPACITY ablation — does the PR95 bilinear-skip + terminal HF-refine
# break the skip-free 21.74 dB PSNR plateau? (deep_hinerv_snerv_fidelity_review H1/H4)
#
# CONTROL (already running, PID 24818): skip-OFF, w=30, N=600 -> plateau 21.74 dB.
# This launcher adds two skip-ON arms SEQUENTIALLY (one new MLX job at a time, to
# bound GPU contention with the live clean-PR95 baseline + the OFF control):
#   Arm A: skip ON, w=30  -> isolates H1 (does the residual path alone help at our w?)
#   Arm B: skip ON, w=1   -> PR95-faithful (skip + implicit w~1; tests H4 alias trap)
# PSNR plateaus by ~ep100 for the skip-free carrier, so ep800 is ample headroom.
#
# [macOS-MLX research-signal] — pure-recon PSNR is NOT a contest score (promotable=False).
# It isolates Mistake-A (architecture) from Mistake-B (recon-MSE-only objective): the
# recon-fit probe is ALWAYS MSE, so a PSNR break here is purely the residual-path effect.
#
# Run detached: nohup bash scripts/launch_b1_f1_skip_recon_fit_ablation.sh </dev/null >/dev/null 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TIER=/Volumes/VertigoDataTier/pact
N=600
EPOCHS=800
TS="$(date -u +%Y%m%dT%H%M%SZ)"

run_arm() {
  local label="$1"; shift
  local work="${TIER}/recon_fit_f1_${label}_${TS}"
  mkdir -p "$work"
  echo "[$(date -u +%H:%M:%SZ)] START arm=${label} -> ${work}"
  "$PY" tools/run_hi_nerv_recon_fit_capacity.py \
    --num-pairs "$N" --epochs "$EPOCHS" --batch-pairs 16 \
    --eval-every-epochs 50 --eval-sample-pairs 32 \
    "$@" \
    --work-dir "$work" \
    --out "${work}/recon_fit_f1_${label}.json" \
    > "${work}/stdout.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] DONE  arm=${label} rc=$? -> ${work}/recon_fit_f1_${label}.json"
}

# Arm A: skip ON, w=30 (single-variable skip add at the current frequency).
run_arm "skipON_w30" --use-bilinear-skip
# Arm B: skip ON, w=1 (PR95-faithful coherent-carrier-inside-sin form).
run_arm "skipON_w1" --use-bilinear-skip --sin-frequency 1.0
echo "[$(date -u +%H:%M:%SZ)] F1 ablation complete (control=skipOFF_w30 N600 plateau 21.74 dB)"
