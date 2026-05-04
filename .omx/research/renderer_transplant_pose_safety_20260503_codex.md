# Renderer Transplant Pose-Safety Preflight - 2026-05-03 Codex

## Scope

Local repair for the trained renderer self-compression / Block-FP transplant
gate. No remote dispatch was performed. This is empirical local preflight
evidence only, not score evidence and not promotion evidence.

## Trigger

The exact CUDA diagnostic
`experiments/results/lightning_batch/exact_eval_trained_qbf1_b0512_g7e_20260503T0142Z/contest_auth_eval.json`
reported:

- archive bytes: `283432`
- archive SHA-256:
  `6174424516a93aa35852cd4dc22b9132516fc38f824b77562b3f91b3f55f58dc`
- avg SegNet distance: `0.0026408`
- avg PoseNet distance: `29.82484055`
- recomputed score: `17.72267562501643`
- failure: PoseNet gate collapse versus C067 reference

The prior transplant preflight proved byte closure and QBF1 CPU load, but it
did not compare rendered outputs against the source C067 runtime. That missing
gate allowed a byte-valid renderer transplant to spend GPU despite being far
outside the source geometry/output basin.

## Infrastructure Added

Added:

`experiments/preflight_renderer_transplant_pose_safety.py`

Focused test:

`src/tac/tests/test_preflight_renderer_transplant_pose_safety.py`

The preflight:

- extracts direct or packed renderer payload archives through the existing
  runtime-member parser;
- enforces safe ZIP names, exactly one packed payload member for packed
  archives, and logical members exactly
  `renderer.bin`, `masks.mkv`, and `optimized_poses.bin`;
- requires transplanted archives to keep source `masks.mkv` and
  `optimized_poses.bin` byte-identical while changing only `renderer.bin`;
- loads the source and candidate renderers through the actual
  `submissions/robust_current/inflate_renderer.py::_load_renderer` runtime
  path;
- decodes archive masks and optimized poses locally;
- samples deterministic pair indices across the contest window;
- records output hashes and summary statistics for source and candidate
  native renderer outputs; and
- fails closed with `score_claim=false`, `promotion_eligible=false`, and
  `remote_gpu_dispatch_performed=false` when output parity is unsafe.

## Local Artifact

Command:

```bash
.venv/bin/python experiments/preflight_renderer_transplant_pose_safety.py \
  --source-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --candidate-archive experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146/trained_qbf1_b0512/archive.zip \
  --output-json experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146/trained_qbf1_b0512/pose_safety_preflight.json \
  --max-pairs 5
```

The command intentionally exited `2` because the candidate failed closed.

Artifact:

`experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146/trained_qbf1_b0512/pose_safety_preflight.json`

Key result:

- `safe_for_exact_eval_dispatch=false`
- failure class: `renderer_transplant_pose_safety_failed`
- fail-closed reason: `render_output_parity_unsafe`
- source archive SHA-256:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- candidate archive SHA-256:
  `6174424516a93aa35852cd4dc22b9132516fc38f824b77562b3f91b3f55f58dc`
- masks SHA-256 matched:
  `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
- optimized poses SHA-256 matched:
  `5236cf75cc95f6a9731b35f4e2fb1211aa99bfcc69e5964869edd19a7af3df9f`
- source renderer SHA-256:
  `5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb`
- candidate renderer SHA-256:
  `d5942c380582f4825810323faf72dee7a1b60d57666468d1a041589ae6402025`
- sampled pair indices: `0`, `150`, `300`, `449`, `599`
- aggregate source-vs-candidate mean absolute output delta:
  `72.17086791992188`
- aggregate RMS output delta: `87.61087004921963`
- aggregate max absolute output delta: `254.8052520751953`
- source output uint8 SHA-256:
  `78b73e1ed8957004568790011d1c2c1af5a4f4caca7fd3fd15673c5105e78630`
- candidate output uint8 SHA-256:
  `01e86906759e502d2433a4fd6dec454cb91e324e5b7ab00812f500299617a2fc`

Interpretation:

This is not a packaging bug: the candidate archive has the expected single
payload member and preserves source masks/poses byte-for-byte. The bug class is
semantic renderer transplant / output-geometry incompatibility. The
transplanted renderer produces native frames far outside the C067 source
runtime output basin before any scorer is involved, explaining the exact CUDA
PoseNet collapse.

## Smallest Actionable Fix

Do not exact-eval another non-surrogate trained renderer archive until it
passes this local output-parity gate against the exact source archive it is
transplanting into.

The smallest score-safe export path is:

1. Train or distill the non-surrogate renderer against the source runtime
   outputs under the exact charged `masks.mkv` and `optimized_poses.bin` that
   will ship in the archive.
2. Run this preflight on the raw trained export archive before QBF1
   self-compression.
3. Only after raw export parity passes, run QBF1/Block-FP self-compression and
   rerun this same preflight on the compressed archive.
4. Dispatch exact CUDA auth eval only if both raw-export and compressed-export
   parity pass and the archive remains byte-closed.

The failed `trained_qbf1_b0512` candidate should remain an A-negative diagnostic
artifact for this implementation/config only. It should not be used to retire
the broader trained-renderer or self-compression family.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile experiments/preflight_renderer_transplant_pose_safety.py
.venv/bin/python -m pytest src/tac/tests/test_preflight_renderer_transplant_pose_safety.py -q
```
