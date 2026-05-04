# C091 Fixed-Renderer Burn Sub-0.31 Readiness - 2026-05-03 Worker

Scope: local readiness artifact only. No remote GPU dispatch was started from
this worker slice. The lane is Block-FP/Selfcomp-style JointFrameGenerator
renderer self-compression, gated on terminal fixed-mask/fixed-pose burn
exports.

## Parent Custody

Selected parent is the current local best exact CUDA candidate found in the
Lightning batch artifacts:

- Archive:
  `experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/archive.zip`
- Eval JSON:
  `experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/contest_auth_eval.adjudicated.json`
- Archive bytes: `276485`
- Archive SHA-256:
  `79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8`
- Recomputed score: `0.31514430182167497`
- SegNet: `0.00060804`
- PoseNet: `0.00049337`
- Samples: `600`

## Local Artifact

Generated:

- JSON:
  `experiments/results/c091_fixed_renderer_burn_sub031_readiness_20260503_worker/handoff_manifest.json`
- Markdown:
  `experiments/results/c091_fixed_renderer_burn_sub031_readiness_20260503_worker/handoff_manifest.md`

The artifact is empirical dry-run planning evidence only. It records
`remote_gpu_dispatch_performed=false`, `score_claim=false`, and
`promotion_eligible=false`.

## Break-Even

For strict sub-`0.31` at unchanged components:

- Score gap: `0.005144301821674968`
- Rate score per archive byte: `6.658589531221714e-7`
- Required byte savings: `7726`
- Max archive bytes for byte-only crossing: `268759`

This is a credible high-EV target only if a terminal trained renderer export
both preserves the fixed C091/C067 mask-pose basin and compresses substantially
better than the source JFG renderer. Existing byte-only QBF1/global-reblock,
SJ-KL, IMP, and multimask artifacts do not satisfy that gate.

## Current Readiness

Dispatch readiness is false. The blocking conditions are:

- no terminal Modal renderer export is available locally;
- the active Modal calls must be recovered first;
- no transplant candidate archive exists yet;
- no matching local pose-safety JSON exists for the exact source/candidate SHA
  pair.

Active export recovery commands are in the handoff manifest for:

- `fc-01KQP9K42CAWJH7XEV4KC0V28M`
- `fc-01KQP9T1VD14785MG63H7JM5VK`
- `fc-01KQP9T19Y7PMDETDN99WDMF2W`

## Patch

While generating the handoff, a recipe bug was found and fixed in
`experiments/prepare_trained_renderer_transplant_dispatch.py`: exact-eval
command templates now derive `--baseline-score` and
`--baseline-archive-bytes` from the verified parent eval JSON instead of a
hard-coded older source. The focused regression test now asserts the dynamic
baseline wiring.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_prepare_trained_renderer_transplant_dispatch.py -q
```

Result: `6 passed in 0.09s`.

## Dispatch Boundary

Do not exact-eval from this lane until all local gates are true:

1. Recover a terminal Modal renderer export.
2. Build local candidates with `experiments/build_renderer_shrink_candidate.py`
   against the C091 parent above.
3. Run `experiments/preflight_renderer_transplant_pose_safety.py` against the
   exact source/candidate archive SHA pair.
4. Confirm the selected archive is byte-closed, sidecar-free, not source
   identical, and either `<=268759` bytes at unchanged-distortion hypothesis or
   backed by a concrete component-improvement reason.
5. Claim the lane with `tools/claim_lane_dispatch.py claim ...` before any
   Lightning exact-eval submission.

The handoff manifest already emits the corrected claim and dry-run command
shape, but it remains blocked until a terminal export and pose-safety report
exist.
