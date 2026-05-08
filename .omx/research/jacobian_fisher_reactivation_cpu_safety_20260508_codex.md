# Jacobian/Fisher Reactivation CPU Safety Prototype - 2026-05-08

Scope: CPU-only score-lowering design/prototype after the
Beta/Jacobian-Fisher Path-B CPU candidate. No GPU jobs were launched and no
dispatch claim was made.

## Code Prototype

Touched surfaces:

- `tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py`
- `src/tac/tests/test_build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py`

The no-dead-K builder now has two explicit selected-K safety controls:

- `--selected-Ks-additive-baseline-cap N`: for external selected-K manifests,
  cap any tensor more aggressive than the default no-dead-K baseline at
  `baseline_K + N`.
- `--selected-Ks-max-fp32-smoke-rel-err`: selected-K builds whose aggregate
  fp32 smoke rel_err exceeds this threshold are marked
  `cuda_eval_worth_testing=false` and receive
  `selected_Ks_fp32_smoke_rel_err_above_guard`.

This separates wire-format correctness from candidate safety. A selected-K
vector can decode correctly while still being rejected because its aggregate
fp32 smoke rel_err is outside the CPU trust region.

## Scratch CPU Blend Sweep

Inputs:

- Default no-dead-K baseline Ks:
  `[2,1,5,1,5,1,5,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]`
- Raw beta/Jacobian-Fisher selected Ks:
  `[10,24,3,10,3,10,5,10,2,1,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]`
- Source manifest:
  `experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/jacobian_fisher_allocation_manifest.json`

CPU scratch metrics, no persisted candidate archive:

| variant | archive bytes | int8 rel_err | fp32 smoke rel_err | max tensor fp32 rel_err | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| default no-dead-K | 153,671 | 0.0415379678 | 0.0361681970 | 0.0657398548 | baseline |
| raw beta selected-Ks | 147,285 | 0.0580904314 | 0.0873994010 | 0.1929084483 | rejected by guard |
| additive cap 3 blend | 153,378 | 0.0426281093 | 0.0512570250 | 0.0865895014 | safer CPU byte-lower prototype |

Additive cap 3 blended Ks:

```text
[5,4,3,4,3,4,5,4,2,1,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
```

Interpretation:

- Raw beta selected-Ks are byte-lower but rel_err-risky and should not route
  toward CUDA as-is.
- The additive-cap-3 blend is a safer CPU candidate than raw beta:
  it keeps a small byte win versus default no-dead-K (`-293` archive bytes)
  while cutting aggregate fp32 smoke rel_err from `0.0873994010` to
  `0.0512570250`.
- This is still CPU-build evidence only. It is not a score claim, not a
  promotion, and not dispatch-ready.

## Remaining Blockers

- CPU fp32 smoke rel_err is not scorer authority.
- The beta/Jacobian importance source remains diagnostic/proxy sensitivity, not
  CUDA pixel-Jacobian/Fisher pullback custody.
- Exact CUDA auth eval, static pre-submission compliance, and the inherited
  `apogee_int6_contest_cuda_anchor_required_first` blocker remain required
  before any score/rank/promotion decision.

## Verification

Command:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py \
  src/tac/tests/test_jacobian_fisher_importance_allocator.py -q
```

Result:

```text
28 passed in 0.51s
```
