# Component Response Producer Round 2 - 2026-04-30

Worker scope: official component response CUDA producer design/implementation
only. Lightning and NWCS files were not touched.

This is not a score ledger. No CUDA auth eval was run, no archive was scored,
and no component sensitivity artifact from this work is promotion evidence.

## Files Inspected

- `experiments/profile_component_sensitivity.py`
- `experiments/build_component_sensitivity_manifest.py`
- `src/tac/component_sensitivity_artifact.py`
- `src/tac/tests/test_profile_component_sensitivity.py`
- `src/tac/tests/test_build_component_sensitivity_manifest.py`
- `src/tac/tests/test_component_sensitivity_artifact.py`
- `.omx/research/component_response_producer_next_patch_20260430_worker.md`
- `.omx/research/component_sensitivity_producer_grand_council_review_20260430_agent.md`
- `.omx/research/component_sensitivity_owv3_nwcs_execution_plan_20260430_codex.md`

## Patch Landed

Changed files:

- `experiments/profile_component_sensitivity.py`
- `src/tac/tests/test_profile_component_sensitivity.py`
- `.omx/research/component_response_producer_round2_20260430_worker.md`

The profiler remains fail-closed and non-promotable. Fisher maps still write
diagnostic metadata and cannot assemble a promotable
`component_sensitivity_v1` manifest.

Producer-side improvements:

- Response-curve evaluation now uses official-style local component readouts:
  PoseNet pose MSE and SegNet argmax disagreement, with combined contribution
  recomputed as `100 * seg + sqrt(10 * pose)` with no rate term.
- Default response epsilons are now symmetric and include zero:
  `[-0.002, -0.001, -0.0005, 0, 0.0005, 0.001, 0.002]`.
- Response JSON now records official readout metadata, symmetric/directional
  epsilon coverage, finite/zero-repro gate diagnostics, gate specs, component
  units, and explicit non-promotable blockers.
- Stability JSON now includes CV, top-k overlap, Pearson/Spearman rank
  correlation, top 1%/5%/10% overlap, counts, thresholds, per-component pass
  booleans, and aggregate `passed`.

Important caveat: CUDA response curves from this profiler may now be official
local component-response diagnostics, but the artifact still cannot promote
because map-to-response prediction-error calibration is not implemented and the
maps remain empirical Fisher proxies.

## Tests Added

- Global combined formula from mean PoseNet/SegNet distortions.
- Symmetric/default epsilon metadata and directional metadata.
- Response JSON official metadata, gate specs, blockers, and zero-repro
  diagnostics.
- Official SegNet argmax disagreement path using toy scorer fixtures.
- Stability JSON thresholds, pass booleans, rank metadata, top-fraction
  overlap, and counts.

## Remaining Producer Gaps

Next bounded patch should add a perturbation-basis artifact and prediction-error
calibration:

1. Write `perturbation_basis_v1.json` with atom IDs, ordering, normalization,
   epsilon units, sign convention, split hash, pair IDs, and input custody.
2. Add predicted deltas for each response point from the map/basis contract.
3. Compute holdout prediction error, sign accuracy, and rank/top-k agreement
   between predicted and observed official component damage.
4. Set `passed=true` only when finite values, zero repro, response coverage,
   official SegNet argmax readout, prediction-error threshold, and stability
   thresholds all pass.
5. Keep exact archive CUDA auth eval as the only score truth after any map-built
   archive is produced.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile \
  experiments/profile_component_sensitivity.py \
  experiments/build_component_sensitivity_manifest.py \
  src/tac/component_sensitivity_artifact.py \
  src/tac/tests/test_profile_component_sensitivity.py

.venv/bin/python -m pytest \
  src/tac/tests/test_profile_component_sensitivity.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  src/tac/tests/test_component_sensitivity_artifact.py \
  -q

git diff --check
```

Result:

```text
54 passed in 1.40s
```

`git diff --check` passed. No shell scripts were touched. No CUDA auth eval was
run.
