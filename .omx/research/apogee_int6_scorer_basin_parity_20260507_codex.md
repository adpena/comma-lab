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

## Predispatch Gate Narrowing

After the parity evidence landed, the local distortion proxy was run with:

- `archive_bytes`: `170450`
- `rel_err_pct`: `1.55`
- `n_layers`: `13`
- `evidence_semantics`: `local_distortion_proxy`
- `tag`: `[distortion-proxy:local]`

Artifact:
`experiments/results/apogee_int6_basin_parity_20260507_claude/distortion_proxy_local.json`

Re-running `tools/predispatch_sanity.py` with both `--distortion-proxy-ran` and
the scorer-basin parity JSON narrowed the refusal to one remaining blocker:

- `sanity_lossy_vs_lossless`: predicted band `[0.190, 0.204]` is below the
  lossless PR106 baseline `0.2095`, so the current policy treats the band as
  incoherent for a lossy repack.

Cleared gates in that run:

- `anchors_sufficient`
- `distortion_model_gate`
- `hazard_scan`
- `lane_registry_consistent`
- `apogee_evidence_semantics`

Current status: evidence-complete locally, still not dispatch-ready under the
strict sanity policy unless that policy is revised for rate-only lossy repacks
or an explicit operator override is used and recorded.

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

Predispatch check:

```bash
.venv/bin/python tools/predispatch_sanity.py \
  --archive experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip \
  --predicted-low 0.190 \
  --predicted-high 0.204 \
  --rel-err-pct 1.55 \
  --lane-class apogee_intN \
  --distortion-proxy-ran \
  --readiness-evidence-json experiments/results/apogee_int6_basin_parity_20260507_claude/parity_evidence.json \
  --json
```
