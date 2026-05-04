# C067 Fixed-Renderer Burn Audited Override

## 2026-05-03T04:58Z

Status: audited operator override for one paid training dispatch.

This does not create `.omx/state/lane12_nerv_l2_clearance.json` and does not
clear Lane 12 / Alpha NeRV retraining. The existing readiness report remains
fail-closed:

- `.omx/state/lane12_nerv_l2_clearance.json` is missing.
- no passing Alpha-Geo promotion-threshold geometry JSON exists.
- usable pose-regeneration provenance is missing.

Reason for override:

- The user explicitly requested maximum-aggression, no-expense-spared parallel
  work including unsafe/not-time-safe lanes under contest-faithful custody.
- A read-only Grand Council strategy review recommended a fresh C067
  fixed-mask/fixed-pose renderer self-compression burn as the only current
  standalone sub-0.30-class path under the deadline.
- This dispatch is not Lane 12 / Alpha mask-codec retraining. It trains a
  renderer against exact charged C067 masks and poses and writes no score claim.

Scope:

- Run exactly the prepared deterministic command packet:
  `experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503/run_fixed_renderer_burn.sh`
- Inputs are fixed C067 logical runtime members recorded in:
  `experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503/fixed_c067_renderer_burn_manifest.json`
- Export snapshots with `--eval-mode none`.
- Do not promote or rank any checkpoint, renderer, or archive until a later
  byte-closed archive is built and exact CUDA auth-evaluated on the canonical
  path.

Required post-burn gates:

1. Run trained-renderer transplant preflight.
2. Run pose-safety preflight against the selected archive SHA.
3. Exact CUDA diagnostic only if byte math is plausible.
4. T4/equivalent A++ promotion only for identical bytes with archive SHA,
   manifest, component trace, and adjudicated JSON.

## 2026-05-03T05:38Z Retry Override Addendum

Status: audited operator override extended to bounded retry dispatches after the
first two provider jobs failed before useful training artifacts.

Failure classification:

- `train_c067_fixed_renderer_burn_qfaithful_rtxpro_20260503T0458Z`: failed
  before useful logs; no renderer or score evidence.
- `train_c067_fixed_renderer_burn_qfaithful_rtxpro_fix1_20260503T0529Z`:
  failed before useful logs; local postmortem found the generic Lightning
  command embedded a malformed Python heredoc that `bash -n` did not catch.

Permanent guardrail:

- `LightningBatchJobSpec.validate()` now compiles Python stdin heredocs before
  provider submission, so this bug class fails locally instead of after paid
  GPU allocation.

Retry scope:

- Launch bounded `fix2` training retries from freshly staged source manifest
  `.omx/state/renderer_burn_fix2_h100_inputs_20260503T0536Z_manifest.json`.
- Use distinct run directories and lane claims for each hardware attempt.
- Keep `score_claim=false`; export snapshots only.
- Do not promote any checkpoint until trained-renderer transplant preflight,
  pose-safety preflight, byte-closed archive build, and exact CUDA auth eval all
  pass.

## 2026-05-03T07:25Z Stale Lightning Retry Cleanup

The older Lightning `fix2` H100/A100 retries remained pending at zero cost
after the half-frame mask-index training bug was found and fixed. They were
stopped with the bounded `scripts/launch_lightning_batch_job.py stop` command
so they cannot later start from a stale pre-fix source snapshot:

- `train_c067_fixed_renderer_burn_qfaithful_fix2_h100p5_20260503T0544Z`:
  stopped, cost `0.0`.
- `train_c067_fixed_renderer_burn_qfaithful_fix2_a100p4d_20260503T0544Z`:
  stopped, cost `0.0`.

Their dispatch claims are closed as
`stopped_stale_pre_half_frame_fix_superseded`. The active renderer training
spend is now the patched Modal `fix3` H100/A100/A10G set, still with
`score_claim=false` until transplant preflight, pose-safety preflight, and
exact CUDA/T4 archive eval pass.
