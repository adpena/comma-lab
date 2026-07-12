# V9 CGauge MLX micro-batch unlock implementation spec

Date: 2026-07-12
Status: implementation contract
Lane: `lane_micro_batch_v9_unlock_mlx_20260712`
Authority: operator directive `max_throughput_over_bit_identity_operator_override_20260712.md`

## Objective

Make the canonical `compile_v9_cgauge_432_launch_config()` argv valid and
functionally faithful with `--micro-batch-pairs 2`. Training-loop bit identity
is waived. The admission bar is per-lever functional loss/gradient parity plus
a measured wall-clock receipt. This landing has no score authority.

## Settled blocker inventory

The trainer has six actual micro-batch refusal sites: margin reachability,
spike reweight, subpixel residual, phase advection, chroma boundary, and
normalized StEik. The canonical V9 argv activates only phase advection and
chroma boundary among those explicit refusals. It also activates silently
omitted temporal screw, area constraint, and birth-completion-dependent
logit/amplify/persistence state. Analytic lane render-band is already inside
the render composition for every frame; it requires a parity test, not a
duplicate loss term.

## Implementation boundary

1. `src/tac/boundary_math/levelset_micro_batch_loss.py`
   - Extend `LeverConfig` with the V9 providers and live-state references.
   - Preserve serial semantics: normalize each pair independently, then mean
     over pairs.
   - Route chroma, phase, temporal screw, area constraint, dynamic logit
     offset, ramped amplify, and ramped persistence.
   - Use one batched frame-0 scorer call for temporal screw.
2. `src/tac/local_acceleration/metal_micro_batch_v9_levers.py`
   - Supply fused `mx.fast.metal_kernel` paths for the batched chroma/phase/
     temporal map work and their theta-bearing VJPs, with a pure-MLX
     reference/fallback and post-evaluation backend custody.
   - Never evaluate pixel-map math or dispatch kernels per pair. Provider rows may be host-packed
     from the trainer's legacy list store, then all pixel math executes in one batched dispatch.
3. `experiments/train_levelset_witness_realized_through_R_mlx.py`
   - Wire provider stacks and live gate/ramp cells into `_micro_batch_lc`.
   - Remove phase/chroma refusals only after their twins exist.
   - Validate `micro_batch_pairs >= 1` and correct stale capability text.
4. Tests/probe/DSL/triality
   - Extend `test_levelset_micro_batch_loss.py` for K=2/K=4 loss and gradient
     parity, gate-off/zero-mask behavior, dynamic ramp mutation, lane compose,
     Metal-reference parity, and the mixed V9 lever stack.
   - Update `micro_batch_bit_identity_probe.py` so bit drift remains diagnostic
     while functional parity and speedup control training admission.
   - Update the existing `MicroBatch` DSL definition and tests; append DAG and
     equation receipts without overwriting unrelated work.

## Acceptance criteria

- Mechanical DSL compile and trainer parse identify the complete active set.
- Every canonical-V9-active semantic has a real batched twin or is proven to
  execute in the shared render path.
- Focused tests pass on Apple MLX, including loss and gradient parity at a
  documented functional tolerance.
- The canonical launcher dry-run with `--micro-batch-pairs 2` exits zero and
  emits exactly one micro-batch flag without spawning training.
- A bounded faithful-scale benchmark records B=1 versus B=2 speed class. No
  heavy n600 run is launched; n600 validation remains explicitly owed.
- Final landing memo is `.omx/research/micro_batch_v9_unlock_20260712.md`.

## Do not touch

- `experiments/results/v9_cgauge_432_coherent_arm_20260711/` or its process.
- `src/tac/cuda_levelset_training.py`.
- `experiments/train_levelset_witness_realized_through_R_torch.py`.
- Any Torch/CUDA-port tests owned by the sister lane.
- The contest scorer or any score/frontier pointer.

## Verification commands

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_levelset_micro_batch_loss.py
.venv/bin/python -m pytest -q src/tac/tests/test_witness_curriculum_dsl.py
.venv/bin/python tools/launch_witness_run.py --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 --config v9_cgauge_432 --extra-trainer-flags "--micro-batch-pairs 2" --out-dir experiments/results/v9_cgauge_micro_batch_dryrun_20260712 --dry-run --skip-throughput-gate
```

Local disposition: configuration/DSL/static verification is green. The managed session has no
initializable Metal device, so the installed tiny and faithful 384x512 forward/VJP runtime gates
correctly REFUSE here; real-device full-V9 parity/timing and n600 validation remain owed.
