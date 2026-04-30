# Component Response Producer Next Patch - 2026-04-30

Worker scope: component sensitivity official-response producer design and
bounded implementation only. Lightning and NWCS scripts were not edited.

This is not a score ledger. No CUDA eval was run and no sensitivity artifact is
promotion evidence from this work.

## Files Inspected

- `experiments/profile_component_sensitivity.py`
- `experiments/build_component_sensitivity_manifest.py`
- `src/tac/component_sensitivity_artifact.py`
- `src/tac/tests/test_profile_component_sensitivity.py`
- `src/tac/tests/test_build_component_sensitivity_manifest.py`
- `src/tac/tests/test_component_sensitivity_artifact.py`
- Adjacent research ledgers for component-sensitivity producer design and
  OWV3/NWCS readiness.

## Findings

The current profiler is correctly fenced as diagnostic Fisher-proxy output:
it writes `promotion_eligible=false`, `official_component_response=false`, and
mathematical blockers. It still does not produce official finite-difference
PoseNet/SegNet response maps.

The manifest assembler and validator were the smallest low-conflict place to
land a rigorous next patch. Before this patch, a promotable manifest could
carry response-curve files with only `count` and `holdout_error` metadata. That
checked custody, but not the official-response gate required by the research
protocol.

The profiler also parsed negative epsilons for future symmetric curves, but
`apply_channel_perturbation()` rejected negative values. That was a small local
inconsistency, not an official producer implementation.

## Bounded Patch Landed

Changed files:

- `src/tac/component_sensitivity_artifact.py`
  - Promotion validation now requires every response curve to declare:
    `official_component_response=true`, `passed=true`, finite numeric
    `gate_spec`, an official component readout, and either symmetric
    `-eps/0/+eps` coverage or explicit directional-action metadata.
  - Promotion validation now requires `stability.passed=true` and finite
    numeric `stability.thresholds`.
  - SegNet promotion curves must identify an official argmax-disagreement
    readout; CE/proxy readouts do not pass.

- `experiments/build_component_sensitivity_manifest.py`
  - Response-curve metadata handoff now preserves official gate fields from
    the curve JSON into the manifest: `official_component_response`, `passed`,
    `gate_spec`, readout, response kind, epsilon ladder, and directional
    metadata.
  - If no explicit epsilon ladder is present, it derives one from curve points
    for manifest validation.

- `experiments/profile_component_sensitivity.py`
  - Negative perturbation epsilons are now accepted when finite, allowing
    diagnostic symmetric probes to run. The profiler remains non-promotable.

- Tests updated under `src/tac/tests/` to cover:
  - official response metadata required for promotion,
  - failed/missing response gates rejected,
  - SegNet CE/proxy readout rejected for promotion,
  - missing symmetric coverage rejected,
  - missing stability thresholds rejected,
  - assembler preservation of official response metadata,
  - negative epsilon perturbation support.

## Exact Next Producer Spec

The next code patch should add a new official response producer path rather than
promoting the Fisher profiler output.

Minimum producer behavior:

1. Build `perturbation_basis_v1.json` with deterministic atom IDs, atom family,
   normalization, sign convention, epsilon units, clamp/roundtrip policy, full
   pair universe, calibration/holdout split, split hash, and input hashes.
2. Evaluate official CUDA component values at `eps=0`, `-eps`, `+eps`, and
   normally `-2eps`, `+2eps` for the validation atom subset.
3. Use official PoseNet pose MSE and official SegNet argmax disagreement for
   response curves. Combined deltas must be recomputed as:

   ```text
   100 * (seg_eps - seg_0)
     + sqrt(10 * pose_eps) - sqrt(10 * pose_0)
   ```

4. Emit response JSON with raw component values, observed deltas, predicted
   deltas, per-point errors, rank/top-k metrics, asymmetry/nonmonotone flags,
   `gate_spec`, `passed`, and `official_component_response=true` only when all
   gates pass.
5. Emit stability JSON with CV, Spearman/Pearson, top 1%/5%/10% overlap,
   per-layer summaries, zero/nonfinite/negative/excluded counts, thresholds,
   and `passed`.
6. Assemble `component_sensitivity_v1` only after exact CUDA
   `contest_auth_eval.json` custody exists for the anchor archive.

Non-goals for the next patch:

- Do not edit Lightning or NWCS scripts.
- Do not make a score claim from local component curves.
- Do not use CE-only SegNet curves as official response evidence.
- Do not promote top-k hard-pair subsets without an explicit scoped diagnostic
  blocker.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile \
  experiments/profile_component_sensitivity.py \
  experiments/build_component_sensitivity_manifest.py \
  src/tac/component_sensitivity_artifact.py

.venv/bin/python -m pytest \
  src/tac/tests/test_component_sensitivity_artifact.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  src/tac/tests/test_profile_component_sensitivity.py \
  -q
```

Result:

```text
48 passed in 1.50s
```

Also passed:

```bash
git diff --check
```

No CUDA auth eval was run.
