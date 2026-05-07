# Apogee int6 scorer-basin parity evidence - 2026-05-07

## Scope

This ledger records a local predispatch readiness probe for
`apogee_int6_archive.zip`. It is not a contest-CUDA score and must not be used
as a rank, kill, or paper-score claim.

Evidence semantics: `scorer_basin_parity_gate`

Evidence tag: `[scorer-basin-parity:CPU]`

## Inputs

- Candidate archive:
  `experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip`
- Candidate SHA-256:
  `0176a2691a4daf5991170404d30a304ae30389621c0fc54914628414aef39ff1`
- Lossless reference archive:
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- Lossless reference SHA-256:
  `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- Raw local evidence directory:
  `experiments/results/apogee_int6_basin_parity_20260507_claude/`

## Result

The 10-probe / 4-Hutchinson-sample parity run passed:

- `ready_for_exact_eval_dispatch`: `true`
- `scorer_basin_parity_status`: `pass`
- `pose_dist_delta`: `+1.0792740795295686e-04`
- `pose_threshold`: `1.0e-03`
- `seg_dist_delta`: `+9.618123876862228e-04`
- `seg_threshold`: `5.0e-03`
- `hessian_trace_lossless`: `2.7495672440185547e+06`
- `hessian_trace_quantized`: `2.8226146886596680e+06`
- `hessian_log_ratio`: `+0.011387251887827765`
- `hessian_log_ratio_tolerance`: `1.0`

Interpretation: this is positive local basin-geometry readiness evidence for
apogee_int6. It clears the local non-proxy readiness blocker but does not create
an exact score claim. The next score truth remains exact CUDA auth eval on the
exact archive bytes through `archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Command

```bash
.venv/bin/python tools/build_scorer_basin_parity_evidence.py \
  --candidate-archive experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip \
  --lossless-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
  --output-json experiments/results/apogee_int6_basin_parity_20260507_claude/parity_evidence.json \
  --device cpu \
  --n-probes 10 \
  --n-hessian-samples 4
```
