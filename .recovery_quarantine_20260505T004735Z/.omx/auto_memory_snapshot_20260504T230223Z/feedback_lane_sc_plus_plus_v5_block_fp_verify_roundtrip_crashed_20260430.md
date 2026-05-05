# Lane SC++ V5 (kl_distill) crashed Stage 3 — block_fp_codec.verify_roundtrip tolerance failure

**Date**: 2026-04-30 ~12:30 UTC
**Instance**: 35885106 (Vast.ai RTX 4090) — destroyed after harvest
**Lane**: lane_sc_plus_plus_v5_post_round10_a1
**Spend**: ~$1.58 (5.3h @ $0.2731/hr)
**Result**: NO contest-CUDA score. Stages 0-2 succeeded; Stage 3 crashed; Stages 4-5 never ran.

## What happened

- Stages 0 (NVDEC probe), 1 (anchor checks), 2 (train SegMap 600 epochs, 4h32m, final epoch=599 loss=1.722 seg=0.014463 pose=0.009920) all completed cleanly.
- Stage 3 (`pack inference state via block_fp_codec`) crashed at `tools/.../inline-script` line 7:
  ```
  File "/workspace/pact/src/tac/block_fp_codec.py", line 799, in verify_roundtrip
      raise AssertionError(...)
  AssertionError: verify_roundtrip: layer_in.weight MSE 0.000313327 > tol 1e-06
  ```
- Tolerance `1e-06` is **313× tighter** than observed reconstruction error `3.13e-04`.
- Process exited rc=1; instance idled at 0% GPU for ~46 min before investigation/destruction.

## Why this matters

- The training itself succeeded (final seg=0.014463 is competitive with prior SC++ landings) — only the **packing step** failed.
- This is the same architectural family that landed Lane SC++ v3+v4. Either:
  - (a) **block_fp_codec was tightened recently** and the existing trained checkpoints would also fail this assertion, OR
  - (b) **kl_distill variant produces weights with different distribution** that the codec's tolerance was never calibrated for.

## Harvested artifacts

`experiments/results/lane_sc_plus_plus_kl_distill_failed_landed/`:
- `train/segmap_train.pt` (764K) — full state-dict
- `train/segmap_inference.pt` (380K) — inference weights (the file that failed roundtrip)
- `train/segmap_train.json` (123K) — full epoch history
- `segmap_weights.tar.xz` (49K) — partial pack output
- `train.log` + `run.log` + `heartbeat.log` + `provenance.json`

## Required follow-up

1. Investigate `src/tac/block_fp_codec.py:799` — is `tol=1e-06` a sane verify_roundtrip tolerance for block-FP at this bit width? Lane SC++ v3 (1.04 score) packed successfully, so something changed.
2. Reproduce locally: load `experiments/results/lane_sc_plus_plus_kl_distill_failed_landed/train/segmap_inference.pt`, call `block_fp_codec.encode(...)` then `verify_roundtrip(...)`. If it fails locally too, the codec needs to expose a `tolerance=` kwarg or the lane script needs to pass an appropriate one.
3. Add a STRICT preflight: every lane that calls `verify_roundtrip` must specify or be tested-against the actual achievable tolerance for its weight distribution.
4. Once block_fp_codec issue resolved, can re-pack from harvested `segmap_inference.pt` WITHOUT re-running the 4.5h training — this $1.58 is recoverable as a 30-min $0.13 re-run (Stage 3-5 only).

## Decision

KILL: instance destroyed. Artifacts harvested. Bug is in our packer, not the GPU instance — no value in keeping the instance running.
