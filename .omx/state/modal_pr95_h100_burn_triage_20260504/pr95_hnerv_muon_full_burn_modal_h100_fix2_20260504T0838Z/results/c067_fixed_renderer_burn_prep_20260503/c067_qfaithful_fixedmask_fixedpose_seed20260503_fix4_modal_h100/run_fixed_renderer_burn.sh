#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=1234
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Run the exporter beside training so long burns produce checkpoint
# archives before provider timeout or operator stop. The snapshot loop
# waits for checkpoints and stays non-scoring (`--eval-mode none`).
('.venv/bin/python' '-u' 'scripts/q_faithful_snapshot_loop.py' '--workspace' "$PWD" '--python-bin' '.venv/bin/python' '--checkpoint-dir' 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/train' '--checkpoint-glob' 'training_state_*.pt' '--masks-mkv' 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/inputs/masks.mkv' '--mask-frame-contract' 'auto' '--poses-pt' 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/inputs/optimized_poses.bin' '--output-root' 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/snapshots' '--min-checkpoint-age-seconds' '60' '--poll-seconds' '120' '--max-idle-polls' '720' '--profile' 'q_faithful_dilated_88k' '--state-source' 'ema_shadow' '--renderer-codec' 'qzs3' '--qzs3-block-size' '32' '--submission-layout' 'pr64_mask_first_single_blob' '--pose-codec' 'raw' '--brotli-quality' '11' '--eval-mode' 'none' '--dispatch-claim-mode' 'none' 2>&1 | tee 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/logs/q_faithful_snapshot_loop.log') &
SNAPSHOT_PID=$!
cleanup_snapshot_loop() { kill ${SNAPSHOT_PID} 2>/dev/null || true; }
trap cleanup_snapshot_loop EXIT
set +e
'.venv/bin/python' '-u' 'src/tac/experiments/train_renderer.py' '--profile' 'q_faithful_dilated_88k' '--video' 'upstream/videos/0.mkv' '--device' 'cuda' '--seed' '20260503' '--deterministic' '--tag' 'c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100' '--output-dir' 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/train' '--qfaithful-training-poses' 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/inputs/optimized_poses.bin' '--mask-noise-mkv' 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/inputs/masks.mkv' '--mask-noise-prob' '1.0' '--auth-eval-masks' 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/inputs/masks.mkv' '--auth-eval-poses' 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/inputs/optimized_poses.bin' '--no-auth-eval-on-best' '--wall-clock-timeout' '43200' 2>&1 | tee 'experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503_fix4_modal_h100/logs/train_renderer.log'
TRAIN_STATUS=${PIPESTATUS[0]}
if kill -0 ${SNAPSHOT_PID} 2>/dev/null; then
  kill ${SNAPSHOT_PID} 2>/dev/null || true
  wait ${SNAPSHOT_PID} 2>/dev/null || true
  SNAPSHOT_STATUS=0
else
  wait ${SNAPSHOT_PID}
  SNAPSHOT_STATUS=$?
fi
trap - EXIT
set -e
if [[ ${TRAIN_STATUS} -ne 0 ]]; then exit ${TRAIN_STATUS}; fi
if [[ ${SNAPSHOT_STATUS} -ne 0 ]]; then exit ${SNAPSHOT_STATUS}; fi
